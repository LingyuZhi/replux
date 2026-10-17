"""Provider-independent reply events and streaming protocol adapters."""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import TracebackType
from typing import Generic, Self, TypeVar, cast

from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic, DefaultAsyncHttpxClient, NOT_GIVEN, omit
from anthropic.types import ContentBlock, MessageParam, ToolParam
import httpx2 as httpx
import openai
from openai.types.responses import FunctionToolParam, Response
from openai.types.responses.response_input_param import ResponseInputItemParam

from replux.config import DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS, ProviderConfig
from replux.tools import ToolDefinition, ToolRegistry, ToolResult, create_default_registry, summarize_arguments


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    name: str
    summary: str


@dataclass(frozen=True)
class ToolFinished:
    """A Tool Call has its final Tool Result; skipped calls never ran and have no duration."""

    call_id: str
    name: str
    summary: str
    result: ToolResult
    duration_seconds: float | None


@dataclass(frozen=True)
class ReplyCompleted:
    """The final Reply is complete and committed, so the Turn is complete."""

    stop_reason: str
    input_tokens: int | None
    output_tokens: int | None


StreamEvent = TextDelta | ThinkingDelta | ToolStarted | ToolFinished | ReplyCompleted

ContextItem = TypeVar("ContextItem")


@dataclass(frozen=True)
class ToolCall:
    """One Tool Call from a model response, with its raw JSON arguments."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse(Generic[ContextItem]):
    """One complete model response, kept in native form for replay."""

    context_items: tuple[ContextItem, ...]
    stop_reason: str
    input_tokens: int | None
    output_tokens: int | None
    tool_calls: tuple[ToolCall, ...] = ()


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


class ToolBatchLimitError(LLMError):
    """The Turn stopped because the model requested tools after its Tool Batch limit."""


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
    def stream_response(
        self, context: list[ContextItem], tools: Sequence[ToolDefinition], *, tools_allowed: bool,
    ) -> AsyncGenerator[ResponseEvent[ContextItem], None]:
        """Yield deltas, then ModelResponse last, only after the stream is complete and closed.

        Tool definitions stay in every request; tools_allowed controls whether the model may call them.
        """

    @abstractmethod
    def tool_results(self, results: Sequence[tuple[ToolCall, ToolResult]]) -> list[ContextItem]:
        """Return native context items that pair each Tool Result with its Tool Call."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the adapter's network resources."""


class Conversation(LLMClient, Generic[ContextItem]):
    """Run Turns and keep the Conversation Context.

    A Turn may run one Tool Batch, then requests a summary with tools disabled.
    """

    def __init__(self, adapter: ProtocolAdapter[ContextItem], tools: ToolRegistry) -> None:
        self._adapter = adapter
        self._tools = tools
        self._active = False
        self._context: list[ContextItem] = []

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        if self._active:
            raise RuntimeError("A reply is already in progress.")
        self._active = True
        try:
            uncommitted = [self._adapter.user_message(user_input)]
            tools_allowed = True
            while True:
                response: ModelResponse[ContextItem] | None = None
                request = self._adapter.stream_response(
                    [*self._context, *uncommitted], self._tools.definitions(), tools_allowed=tools_allowed,
                )
                async with aclosing(request) as events:
                    async for event in events:
                        assert response is None
                        if isinstance(event, ModelResponse):
                            response = event
                        else:
                            yield event
                assert response is not None
                if not response.tool_calls:
                    # No await between committing the Turn and publishing success.
                    self._context.extend([*uncommitted, *response.context_items])
                    yield ReplyCompleted(response.stop_reason, response.input_tokens, response.output_tokens)
                    return
                if not tools_allowed:
                    raise ToolBatchLimitError("Turn stopped: the model requested more tools after this Turn's tool batch.")
                ids = [call.id for call in response.tool_calls]
                if not all(ids) or len(set(ids)) != len(ids):
                    raise IncompleteReplyError("Reply incomplete: tool calls have missing or duplicate IDs.")
                uncommitted.extend(response.context_items)
                results: list[tuple[ToolCall, ToolResult]] = []
                durations: dict[str, float] = {}
                shown = 0
                try:
                    batch = self._run_batch(response.tool_calls, results, durations)
                    async with aclosing(batch) as tool_events:
                        async for tool_event in tool_events:
                            if isinstance(tool_event, ToolFinished):
                                shown += 1
                            yield tool_event
                except (asyncio.CancelledError, GeneratorExit) as stop:
                    # Started tool work stays in the Context even when the Turn ends early.
                    for call in response.tool_calls[len(results):]:
                        results.append((call, ToolResult("skipped", "Not run because the user cancelled the Turn.")))
                    self._context.extend([*uncommitted, *self._adapter.tool_results(results)])
                    # Commit before publishing. The caller may stop as soon as cancellation
                    # resumes it, and these results must already be in the Context.
                    if isinstance(stop, asyncio.CancelledError):
                        for call, result in results[shown:]:
                            yield ToolFinished(
                                call.id, call.name, summarize_arguments(call.arguments), result, durations.get(call.id),
                            )
                    raise
                self._context.extend([*uncommitted, *self._adapter.tool_results(results)])
                uncommitted = []
                tools_allowed = False
        finally:
            self._active = False

    async def _run_batch(
        self,
        calls: Sequence[ToolCall],
        results: list[tuple[ToolCall, ToolResult]],
        durations: dict[str, float],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run calls in order, appending each result; stop after the first unsuccessful one."""
        for call in calls:
            summary = summarize_arguments(call.arguments)
            if results and results[-1][1].status != "success":
                result = ToolResult("skipped", "Not run because an earlier tool call in this batch did not succeed.")
                results.append((call, result))
                yield ToolFinished(call.id, call.name, summary, result, None)
                continue
            yield ToolStarted(call.id, call.name, summary)
            started = time.monotonic()
            try:
                result = await self._tools.execute(call.name, call.arguments)
            except asyncio.CancelledError:
                durations[call.id] = time.monotonic() - started
                results.append((call, ToolResult("cancelled", "Cancelled by the user before the tool call finished.")))
                raise
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                # The tool returned its real result after cancellation was requested.
                durations[call.id] = time.monotonic() - started
                results.append((call, result))
                raise asyncio.CancelledError
            results.append((call, result))
            yield ToolFinished(call.id, call.name, summary, result, time.monotonic() - started)

    async def aclose(self) -> None:
        try:
            await self._adapter.aclose()
        finally:
            if self._tools.session is not None:
                self._tools.session.cleanup()


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

    async def stream_response(
        self, context: list[MessageParam], tools: Sequence[ToolDefinition], *, tools_allowed: bool,
    ) -> AsyncGenerator[ResponseEvent[MessageParam], None]:
        try:
            content: list[ContentBlock] = []
            arguments: dict[int, list[str]] = {}
            input_tokens: int | None = None
            output_tokens: int | None = None
            stop_reason: str | None = None
            started = False
            stopped = False
            definitions: list[ToolParam] = [
                {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema} for tool in tools
            ]
            async with await self._sdk.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_output_tokens or DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS,
                messages=context,
                thinking=self._config.thinking if self._config.thinking is not None else omit,
                output_config={"effort": self._config.thinking_effort} if self._config.thinking_effort is not None else omit,
                tools=definitions or omit,
                tool_choice={"type": "none"} if definitions and not tools_allowed else omit,
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
                        elif event.delta.type == "input_json_delta":
                            assert block.type == "tool_use"
                            arguments[event.index].append(event.delta.partial_json)
                    elif event.type == "message_start":
                        started = True
                        content = list(event.message.content)
                        usage = getattr(event.message, "usage", None)
                        input_tokens = getattr(usage, "input_tokens", None)
                    elif event.type == "content_block_start":
                        if event.content_block.type == "server_tool_use":
                            raise IncompleteReplyError("Reply incomplete: server tool requests are not supported.")
                        if event.content_block.type == "tool_use":
                            arguments[event.index] = []
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
            if stop_reason not in {"end_turn", "stop_sequence", "refusal", "tool_use"}:
                raise IncompleteReplyError("Reply incomplete: the output limit was reached or the stop reason is not supported.")
            if (stop_reason == "tool_use") != bool(arguments):
                raise IncompleteReplyError("Reply incomplete: the tool request is incomplete.")
            calls: list[ToolCall] = []
            for index, pieces in arguments.items():
                block = content[index]
                assert block.type == "tool_use"
                raw = "".join(pieces) or json.dumps(block.input)
                try:
                    block.input = json.loads(raw)
                except json.JSONDecodeError:
                    raise IncompleteReplyError("Reply incomplete: tool call arguments are not valid JSON.") from None
                if not isinstance(block.input, dict):
                    raise IncompleteReplyError("Reply incomplete: tool call arguments are not a JSON object.")
                calls.append(ToolCall(block.id, block.name, raw))
            yield ModelResponse(
                ({"role": "assistant", "content": content},), stop_reason, input_tokens, output_tokens, tuple(calls),
            )
        except (APIError, httpx.HTTPError) as error:
            _raise_if_cancelled()
            raise _request_error(error, thinking=self._config.thinking is not None or self._config.thinking_effort is not None) from error

    def tool_results(self, results: Sequence[tuple[ToolCall, ToolResult]]) -> list[MessageParam]:
        # The status inside the JSON text is authoritative; is_error only marks non-success for the API.
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": result.to_json(), "is_error": result.status != "success"}
            for call, result in results
        ]}]

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
        self, context: list[ResponseInputItemParam], tools: Sequence[ToolDefinition], *, tools_allowed: bool,
    ) -> AsyncGenerator[ResponseEvent[ResponseInputItemParam], None]:
        try:
            arguments: dict[int, list[str]] = {}
            final: Response | None = None
            definitions: list[FunctionToolParam] = [
                {
                    "type": "function", "name": tool.name, "description": tool.description,
                    "parameters": tool.input_schema, "strict": False,
                }
                for tool in tools
            ]
            async with await self._sdk.responses.create(
                model=self._config.model,
                max_output_tokens=self._config.max_output_tokens if self._config.max_output_tokens is not None else openai.omit,
                reasoning={"effort": self._config.thinking_effort} if self._config.thinking_effort is not None else openai.omit,
                input=context,
                tools=definitions or openai.omit,
                tool_choice="none" if definitions and not tools_allowed else openai.omit,
                store=False,
                stream=True,
            ) as stream:
                async for event in stream:
                    _raise_if_cancelled()
                    if event.type in {"response.output_text.delta", "response.refusal.delta"}:
                        yield TextDelta(event.delta)
                    elif event.type == "response.function_call_arguments.delta":
                        arguments.setdefault(event.output_index, []).append(event.delta)
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
                        if event.item.type not in {"message", "reasoning", "function_call"}:
                            raise IncompleteReplyError("Reply incomplete: tool requests are not supported.")
            _raise_if_cancelled()
            if final is None or final.status != "completed":
                raise IncompleteReplyError("Reply incomplete: the final completion signal is missing.")
            calls: list[ToolCall] = []
            for index, item in enumerate(final.output):
                if item.type == "function_call":
                    if item.status not in {None, "completed"}:
                        raise IncompleteReplyError("Reply incomplete: the tool request is unfinished.")
                    pieces = arguments.get(index, [])
                    # Pieces belong to this output item. call_id pairs the result; the item id does not.
                    raw = "".join(pieces) if pieces else item.arguments
                    item.arguments = raw
                    calls.append(ToolCall(item.call_id, item.name, raw))
                elif item.type == "message":
                    if item.status != "completed":
                        raise IncompleteReplyError("Reply incomplete: the reply text is unfinished.")
                elif item.type != "reasoning":
                    raise IncompleteReplyError("Reply incomplete: tool requests are not supported.")
            usage = final.usage
            yield ModelResponse(
                # Only replay output items; response metadata is not conversation input.
                tuple(cast(ResponseInputItemParam, item.to_dict()) for item in final.output),
                final.status,
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                tuple(calls),
            )
        except (openai.APIError, httpx.HTTPError) as error:
            _raise_if_cancelled()
            raise _request_error(error, thinking=self._config.thinking_effort is not None) from error

    def tool_results(self, results: Sequence[tuple[ToolCall, ToolResult]]) -> list[ResponseInputItemParam]:
        return [
            {"type": "function_call_output", "call_id": call.id, "output": result.to_json()}
            for call, result in results
        ]

    async def aclose(self) -> None:
        await self._sdk.close()


def create_client(
    config: ProviderConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    bash: str | None = None,
    ripgrep: str | None = None,
) -> LLMClient:
    """Assemble a Conversation; relative tool paths start at the current (startup) directory."""
    registry = create_default_registry(Path.cwd(), bash=bash, ripgrep=ripgrep)
    if config.protocol == "anthropic":
        return Conversation(AnthropicAdapter(config, transport=transport), registry)
    if config.protocol == "openai":
        return Conversation(OpenAIAdapter(config, transport=transport), registry)
    raise ValueError("Unsupported protocol; expected anthropic or openai.")
