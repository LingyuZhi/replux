"""Conversation ownership, Turn orchestration, and application assembly."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from pathlib import Path
import time
from typing import Generic, Literal

from anthropic.types import MessageParam
import httpx2 as httpx
from openai.types.responses.response_input_param import ResponseInputItemParam

from replux.config import ProviderConfig
from replux.llm import (
    AnthropicAdapter, ContextItem, IncompleteReplyError, LLMClient, ModelResponse, OpenAIAdapter,
    ProtocolAdapter, ReplyCompleted, StreamEvent, ToolBatchLimitError, ToolCall, ToolFinished, ToolStarted,
)
from replux.tools import ToolRegistry, ToolResult, create_default_registry, summarize_arguments


AgentEvent = StreamEvent


class Agent(LLMClient, Generic[ContextItem]):
    """Own one Conversation Context, its mode, and its active Turn.

    A Turn may run one Tool Batch, then requests a summary with tools disabled.
    """

    def __init__(self, adapter: ProtocolAdapter[ContextItem], tools: ToolRegistry) -> None:
        self._adapter = adapter
        self._tools = tools
        self._active = False
        self._mode: Literal["work"] = "work"
        self._context: list[ContextItem] = []

    @property
    def mode(self) -> Literal["work"]:
        return self._mode

    async def stream_reply(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
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
                                call.arguments,
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
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run calls in order, appending each result; stop after the first unsuccessful one."""
        for call in calls:
            summary = summarize_arguments(call.arguments)
            if results and results[-1][1].status != "success":
                result = ToolResult("skipped", "Not run because an earlier tool call in this batch did not succeed.")
                results.append((call, result))
                yield ToolFinished(call.id, call.name, summary, result, None, call.arguments)
                continue
            yield ToolStarted(call.id, call.name, summary, call.arguments)
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
            yield ToolFinished(call.id, call.name, summary, result, time.monotonic() - started, call.arguments)

    async def aclose(self) -> None:
        try:
            await self._adapter.aclose()
        finally:
            if self._tools.session is not None:
                self._tools.session.cleanup()


def create_agent(
    config: ProviderConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    bash: str | None = None,
    ripgrep: str | None = None,
) -> Agent[MessageParam] | Agent[ResponseInputItemParam]:
    """Assemble an Agent; relative tool paths start at the current (startup) directory."""
    registry = create_default_registry(Path.cwd(), bash=bash, ripgrep=ripgrep)
    if config.protocol == "anthropic":
        return Agent(AnthropicAdapter(config, transport=transport), registry)
    if config.protocol == "openai":
        return Agent(OpenAIAdapter(config, transport=transport), registry)
    raise ValueError("Unsupported protocol; expected anthropic or openai.")
