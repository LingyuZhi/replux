"""Provider-independent reply events and streaming protocol adapters."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import TracebackType
from typing import Self, cast

from anthropic import APIError, APIStatusError, AsyncAnthropic, DefaultAsyncHttpxClient, omit
from anthropic.types import ContentBlock, MessageParam
import httpx2 as httpx
import openai
from openai.types.responses import Response, ResponseInputParam
from openai.types.responses.response_input_param import ResponseInputItemParam

from replux.config import ProviderConfig


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ReplyCompleted:
    stop_reason: str
    input_tokens: int | None
    output_tokens: int | None


StreamEvent = TextDelta | ThinkingDelta | ReplyCompleted


class LLMError(RuntimeError):
    """A known request failure, including an incomplete reply."""


class IncompleteReplyError(LLMError):
    """The stream ended without a supported, complete reply."""


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


class AnthropicClient(LLMClient):
    def __init__(
        self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._active = False
        self._history: list[MessageParam] = []
        self._sdk = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=transport) if transport else None,
        )

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        if self._active:
            raise RuntimeError("已有进行中的回复。")
        self._active = True
        try:
            user_message: MessageParam = {"role": "user", "content": user_input}
            content: list[ContentBlock] = []
            input_tokens: int | None = None
            output_tokens: int | None = None
            stop_reason: str | None = None
            started = False
            stopped = False
            async with await self._sdk.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_output_tokens,
                messages=[*self._history, user_message],
                thinking=self._config.thinking if self._config.thinking is not None else omit,
                stream=True,
            ) as stream:
                async for event in stream:
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
                            raise IncompleteReplyError("回复未完成：当前版本不支持工具请求。")
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
            if not started or not stopped or stop_reason is None:
                raise IncompleteReplyError("回复未完成：缺少完整的结束信号。")
            if stop_reason == "tool_use":
                raise IncompleteReplyError("回复未完成：当前版本不支持工具请求。")
            if stop_reason not in {"end_turn", "stop_sequence", "refusal"}:
                raise IncompleteReplyError("回复未完成：输出达到上限，或返回了不支持的结束状态。")
            # No await between committing both sides of the turn and publishing success.
            self._history.extend([user_message, {"role": "assistant", "content": content}])
            yield ReplyCompleted(stop_reason, input_tokens, output_tokens)
        except (APIError, httpx.HTTPError) as error:
            # Keep the cause for callers, but never display a server's error body.
            if isinstance(error, APIStatusError) and error.status_code == 400 and self._config.thinking is not None:
                raise LLMError("模型拒绝请求，请检查 thinking 配置及其他请求参数。") from error
            raise LLMError("模型请求失败，请检查连接配置或服务状态。") from error
        finally:
            self._active = False

    async def aclose(self) -> None:
        await self._sdk.close()


class OpenAIClient(LLMClient):
    def __init__(
        self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._active = False
        self._history: ResponseInputParam = []
        self._sdk = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=openai.DefaultAsyncHttpxClient(transport=transport) if transport else None,
        )

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        if self._active:
            raise RuntimeError("已有进行中的回复。")
        self._active = True
        try:
            user_message: ResponseInputItemParam = {"role": "user", "content": user_input}
            final: Response | None = None
            async with await self._sdk.responses.create(
                model=self._config.model,
                max_output_tokens=self._config.max_output_tokens,
                input=[*self._history, user_message],
                store=False,
                stream=True,
            ) as stream:
                async for event in stream:
                    if event.type in {"response.output_text.delta", "response.refusal.delta"}:
                        yield TextDelta(event.delta)
                    elif event.type == "response.completed":
                        final = event.response
                        break
                    elif event.type == "response.incomplete":
                        raise IncompleteReplyError("回复未完成：输出达到上限或响应被中断。")
                    elif event.type in {"response.failed", "error"}:
                        raise LLMError("模型请求失败，服务端未完成回复。")
                    elif event.type in {"response.output_item.added", "response.output_item.done"}:
                        if event.item.type not in {"message", "reasoning"}:
                            raise IncompleteReplyError("回复未完成：当前版本不支持工具请求。")
            if final is None or final.status != "completed":
                raise IncompleteReplyError("回复未完成：缺少完整的结束信号。")
            for item in final.output:
                if item.type not in {"message", "reasoning"}:
                    raise IncompleteReplyError("回复未完成：当前版本不支持工具请求。")
                if item.type == "message" and item.status != "completed":
                    raise IncompleteReplyError("回复未完成：正文尚未完成。")
            usage = final.usage
            output = [cast(ResponseInputItemParam, item.to_dict()) for item in final.output]
            # Only replay output items; response metadata is not conversation input.
            self._history.extend([user_message, *output])
            yield ReplyCompleted(
                final.status,
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
            )
        except (openai.APIError, httpx.HTTPError) as error:
            # Keep the cause for callers, but never display a server's error body.
            raise LLMError("模型请求失败，请检查连接配置或服务状态。") from error
        finally:
            self._active = False

    async def aclose(self) -> None:
        await self._sdk.close()


def create_client(
    config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> LLMClient:
    if config.protocol == "anthropic":
        return AnthropicClient(config, transport=transport)
    if config.protocol == "openai":
        return OpenAIClient(config, transport=transport)
    raise ValueError("Unsupported protocol; expected anthropic or openai.")
