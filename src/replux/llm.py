"""Provider-independent reply events and the Anthropic adapter."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from anthropic import APIError, AsyncAnthropic, DefaultAsyncHttpxClient
import httpx2 as httpx

from replux.config import ProviderConfig


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReplyCompleted:
    stop_reason: str
    input_tokens: int | None
    output_tokens: int | None


StreamEvent = TextDelta | ReplyCompleted


class LLMError(RuntimeError):
    """A known request failure, including an incomplete reply."""


class LLMClient(ABC):
    @abstractmethod
    def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream one reply; the caller must close an abandoned generator."""

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
            input_tokens: int | None = None
            output_tokens: int | None = None
            stop_reason: str | None = None
            started = False
            stopped = False
            async with await self._sdk.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_output_tokens,
                messages=[{"role": "user", "content": user_input}],
                stream=True,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield TextDelta(event.delta.text)
                    elif event.type == "message_start":
                        started = True
                        usage = getattr(event.message, "usage", None)
                        input_tokens = getattr(usage, "input_tokens", None)
                    elif event.type == "content_block_start" and event.content_block.type in {
                        "tool_use", "server_tool_use"
                    }:
                        raise LLMError("当前版本不支持工具请求。")
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
                raise LLMError("回复未完成：缺少完整的结束信号。")
            if stop_reason not in {"end_turn", "stop_sequence", "refusal"}:
                raise LLMError("回复未完成：输出达到上限，或返回了不支持的结束状态。")
            yield ReplyCompleted(stop_reason, input_tokens, output_tokens)
        except (APIError, httpx.HTTPError) as error:
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
    raise ValueError("Unsupported protocol; this version supports anthropic.")
