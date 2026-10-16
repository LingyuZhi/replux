"""Provider-independent reply events and streaming protocol adapters."""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Self, TypeVar, cast

from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic, DefaultAsyncHttpxClient, NOT_GIVEN, omit
from anthropic.types import ContentBlock, MessageParam
import httpx2 as httpx
import openai
from openai.types.responses import Response
from openai.types.responses.response_input_param import ResponseInputItemParam

from replux.config import DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS, ProviderConfig


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ReplyCompleted:
    """The final Reply is complete and committed, so the Turn is complete."""

    stop_reason: str
    input_tokens: int | None
    output_tokens: int | None


StreamEvent = TextDelta | ThinkingDelta | ReplyCompleted

ContextItem = TypeVar("ContextItem")


@dataclass(frozen=True)
class ModelResponse(Generic[ContextItem]):
    """One complete model response, kept in native form for replay."""

    context_items: tuple[ContextItem, ...]
    stop_reason: str
    input_tokens: int | None
    output_tokens: int | None


ResponseEvent = TextDelta | ThinkingDelta | ModelResponse[ContextItem]


class LLMError(RuntimeError):
    """A known request failure, including an incomplete reply."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(LLMError):
    """The service rejected the credentials."""


class RateLimitError(LLMError):
    """The service rate limit was reached; retry_after is informational only."""

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class NetworkError(LLMError):
    """The connection failed or a network operation timed out."""


class IncompleteReplyError(LLMError):
    """The stream ended without a supported, complete reply."""


def _request_error(
    error: APIError | openai.APIError | httpx.HTTPError, *, thinking: bool = False,
) -> LLMError:
    status_code: int | None = getattr(error, "status_code", None)
    if isinstance(error, (APITimeoutError, openai.APITimeoutError, httpx.TimeoutException)):
        return NetworkError("Network request timed out. Check your connection or try again later.", status_code=status_code)
    if isinstance(error, (APIConnectionError, openai.APIConnectionError, httpx.RequestError)):
        return NetworkError("Network connection failed. Check your connection or try again later.", status_code=status_code)
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else None
    code = details.get("code") or details.get("type") if isinstance(details, dict) else None
    response = getattr(error, "response", None)
    return _response_error(
        status_code=status_code, code=code if isinstance(code, str) else None,
        retry_after=response.headers.get("retry-after") if response is not None else None,
        thinking=thinking,
    )


def _response_error(
    *, status_code: int | None, code: str | None = None,
    retry_after: str | None = None, thinking: bool = False,
) -> LLMError:
    # Interpret known codes, but never display server bodies or invent HTTP status.
    if status_code == 401 or code in {"authentication_error", "invalid_api_key"}:
        return AuthenticationError("Authentication failed. Check api_key.", status_code=status_code)
    if status_code == 429 or code in {"rate_limit_error", "rate_limit_exceeded"}:
        return RateLimitError("Rate limit reached. Try again later.", status_code=status_code, retry_after=retry_after)
    if status_code == 400:
        message = (
            "The model rejected the request. Check the thinking configuration and other request parameters." if thinking
            else "The model rejected the request. Check the request parameters or conversation context length."
        )
        return LLMError(message, status_code=status_code)
    return LLMError("Model request failed. Check your connection settings or the service status.", status_code=status_code)


def _raise_if_cancelled() -> None:
    # A transport can finish an awaited read after cancellation was requested.
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


class _ResponseStream(httpx.AsyncByteStream):
    """Keep actual response cleanup alive when the SDK closes implicitly."""

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._stream.__aiter__()

    async def aclose(self) -> None:
        closing = asyncio.create_task(self._stream.aclose())
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            # Cancellation must not release the client while the response is open.
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    pass
            closing.result()
            raise


async def _protect_response_cleanup(response: httpx.Response) -> None:
    # HTTPX marks a response closed before awaiting the underlying stream. Guard
    # that stream so a later SDK close cannot skip unfinished network cleanup.
    assert isinstance(response.stream, httpx.AsyncByteStream)
    response.stream = _ResponseStream(response.stream)


class LLMClient(ABC):
    @abstractmethod
    def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        """Reply using completed turns; the caller must close an abandoned generator."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the client's network resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


class ProtocolAdapter(ABC, Generic[ContextItem]):
    """Make single model requests in one protocol's native context format."""

    @abstractmethod
    def user_message(self, user_input: str) -> ContextItem:
        """Return the native context item for one user input."""

    @abstractmethod
    def stream_response(self, context: list[ContextItem]) -> AsyncGenerator[ResponseEvent[ContextItem], None]:
        """Yield deltas, then ModelResponse last, only after the stream is complete and closed."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the adapter's network resources."""


class Conversation(LLMClient, Generic[ContextItem]):
    """Run Turns and keep the Conversation Context of completed Turns."""

    def __init__(self, adapter: ProtocolAdapter[ContextItem]) -> None:
        self._adapter = adapter
        self._active = False
        self._context: list[ContextItem] = []

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        if self._active:
            raise RuntimeError("A reply is already in progress.")
        self._active = True
        try:
            user_message = self._adapter.user_message(user_input)
            response: ModelResponse[ContextItem] | None = None
            async with aclosing(self._adapter.stream_response([*self._context, user_message])) as events:
                async for event in events:
                    assert response is None
                    if isinstance(event, ModelResponse):
                        response = event
                    else:
                        yield event
            assert response is not None
            # No await between committing both sides of the Turn and publishing success.
            self._context.extend([user_message, *response.context_items])
            yield ReplyCompleted(response.stop_reason, response.input_tokens, response.output_tokens)
        finally:
            self._active = False

    async def aclose(self) -> None:
        await self._adapter.aclose()


class AnthropicAdapter(ProtocolAdapter[MessageParam]):
    def __init__(
        self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._sdk = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds if config.timeout_seconds is not None else NOT_GIVEN,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(
                transport=transport, event_hooks={"response": [_protect_response_cleanup]},
            ),
        )

    def user_message(self, user_input: str) -> MessageParam:
        return {"role": "user", "content": user_input}

    async def stream_response(self, context: list[MessageParam]) -> AsyncGenerator[ResponseEvent[MessageParam], None]:
        try:
            content: list[ContentBlock] = []
            input_tokens: int | None = None
            output_tokens: int | None = None
            stop_reason: str | None = None
            started = False
            stopped = False
            async with await self._sdk.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_output_tokens or DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS,
                messages=context,
                thinking=self._config.thinking if self._config.thinking is not None else omit,
                output_config={"effort": self._config.thinking_effort} if self._config.thinking_effort is not None else omit,
                stream=True,
            ) as stream:
                async for event in stream:
                    _raise_if_cancelled()
                    if event.type == "content_block_delta":
                        block = content[event.index]
                        if event.delta.type == "text_delta":
                            assert block.type == "text"
                            block.text += event.delta.text
                            yield TextDelta(event.delta.text)
                        elif event.delta.type == "thinking_delta":
                            assert block.type == "thinking"
                            block.thinking += event.delta.thinking
                            yield ThinkingDelta(event.delta.thinking)
                        elif event.delta.type == "signature_delta":
                            assert block.type == "thinking"
                            block.signature = event.delta.signature
                        elif event.delta.type == "citations_delta":
                            assert block.type == "text"
                            block.citations = [*(block.citations or []), event.delta.citation]
                    elif event.type == "message_start":
                        started = True
                        content = list(event.message.content)
                        usage = getattr(event.message, "usage", None)
                        input_tokens = getattr(usage, "input_tokens", None)
                    elif event.type == "content_block_start":
                        if event.content_block.type in {"tool_use", "server_tool_use"}:
                            raise IncompleteReplyError("Reply incomplete: tool requests are not supported.")
                        content.append(event.content_block)
                    elif event.type == "message_delta":
                        if event.delta.stop_reason is not None:
                            stop_reason = event.delta.stop_reason
                        usage = getattr(event, "usage", None)
                        if (count := getattr(usage, "input_tokens", None)) is not None:
                            input_tokens = count
                        output_tokens = getattr(usage, "output_tokens", None)
                    elif event.type == "message_stop":
                        stopped = True
                        break
            _raise_if_cancelled()
            if not started or not stopped or stop_reason is None:
                raise IncompleteReplyError("Reply incomplete: the final completion signal is missing.")
            if stop_reason == "tool_use":
                raise IncompleteReplyError("Reply incomplete: tool requests are not supported.")
            if stop_reason not in {"end_turn", "stop_sequence", "refusal"}:
                raise IncompleteReplyError("Reply incomplete: the output limit was reached or the stop reason is not supported.")
            yield ModelResponse(({"role": "assistant", "content": content},), stop_reason, input_tokens, output_tokens)
        except (APIError, httpx.HTTPError) as error:
            _raise_if_cancelled()
            raise _request_error(error, thinking=self._config.thinking is not None or self._config.thinking_effort is not None) from error

    async def aclose(self) -> None:
        await self._sdk.close()


class OpenAIAdapter(ProtocolAdapter[ResponseInputItemParam]):
    def __init__(
        self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._sdk = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds if config.timeout_seconds is not None else openai.NOT_GIVEN,
            max_retries=0,
            http_client=openai.DefaultAsyncHttpxClient(
                transport=transport, event_hooks={"response": [_protect_response_cleanup]},
            ),
        )

    def user_message(self, user_input: str) -> ResponseInputItemParam:
        return {"role": "user", "content": user_input}

    async def stream_response(
        self, context: list[ResponseInputItemParam],
    ) -> AsyncGenerator[ResponseEvent[ResponseInputItemParam], None]:
        try:
            final: Response | None = None
            async with await self._sdk.responses.create(
                model=self._config.model,
                max_output_tokens=self._config.max_output_tokens if self._config.max_output_tokens is not None else openai.omit,
                reasoning={"effort": self._config.thinking_effort} if self._config.thinking_effort is not None else openai.omit,
                input=context,
                store=False,
                stream=True,
            ) as stream:
                async for event in stream:
                    _raise_if_cancelled()
                    if event.type in {"response.output_text.delta", "response.refusal.delta"}:
                        yield TextDelta(event.delta)
                    elif event.type == "response.completed":
                        final = event.response
                        break
                    elif event.type == "response.incomplete":
                        raise IncompleteReplyError("Reply incomplete: the output limit was reached or the response was interrupted.")
                    elif event.type in {"response.failed", "error"}:
                        code = event.code if event.type == "error" else (
                            event.response.error.code if event.response.error is not None else None
                        )
                        raise _response_error(
                            status_code=stream.response.status_code, code=code,
                            retry_after=stream.response.headers.get("retry-after"),
                        )
                    elif event.type in {"response.output_item.added", "response.output_item.done"}:
                        if event.item.type not in {"message", "reasoning"}:
                            raise IncompleteReplyError("Reply incomplete: tool requests are not supported.")
            _raise_if_cancelled()
            if final is None or final.status != "completed":
                raise IncompleteReplyError("Reply incomplete: the final completion signal is missing.")
            for item in final.output:
                if item.type not in {"message", "reasoning"}:
                    raise IncompleteReplyError("Reply incomplete: tool requests are not supported.")
                if item.type == "message" and item.status != "completed":
                    raise IncompleteReplyError("Reply incomplete: the reply text is unfinished.")
            usage = final.usage
            yield ModelResponse(
                # Only replay output items; response metadata is not conversation input.
                tuple(cast(ResponseInputItemParam, item.to_dict()) for item in final.output),
                final.status,
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
            )
        except (openai.APIError, httpx.HTTPError) as error:
            _raise_if_cancelled()
            raise _request_error(error, thinking=self._config.thinking_effort is not None) from error

    async def aclose(self) -> None:
        await self._sdk.close()


def create_client(
    config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> LLMClient:
    if config.protocol == "anthropic":
        return Conversation(AnthropicAdapter(config, transport=transport))
    if config.protocol == "openai":
        return Conversation(OpenAIAdapter(config, transport=transport))
    raise ValueError("Unsupported protocol; expected anthropic or openai.")
