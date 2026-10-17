"""Controlled HTTP responses at the SDK's transport boundary."""

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any

import httpx2 as httpx

from replux.config import ProviderConfig


def provider_config(protocol: str = "anthropic") -> ProviderConfig:
    return ProviderConfig(
        protocol=protocol,
        model="custom-model-20261001",
        base_url="https://gateway.example/anthropic/v2",
        api_key="${LITERAL_KEY}",
    )


def native_reply(protocol: str, *, refusal: bool = False) -> list[dict[str, Any]]:
    if protocol == "openai":
        return list(reply_events(protocol, refusal=refusal)[-1]["response"]["output"])
    text = "抱歉，无法回答。\n" if refusal else "你好，世界\n"
    return [{"role": "assistant", "content": [{"type": "text", "text": text}]}]


def reply_events(protocol: str = "anthropic", *, refusal: bool = False) -> list[dict[str, Any]]:
    if protocol == "openai":
        return responses_events(refusal=refusal)
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "custom-model-20261001",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "抱歉，" if refusal else "你好"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "无法回答。\n" if refusal else "，世界\n"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "refusal" if refusal else "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]


def responses_events(*, refusal: bool = False) -> list[dict[str, Any]]:
    kind = "refusal" if refusal else "output_text"
    text_key = "refusal" if refusal else "text"
    chunks = ["抱歉，", "无法回答。\n"] if refusal else ["你好", "，世界\n"]
    text_fields: dict[str, Any] = {} if refusal else {"annotations": [], "logprobs": []}
    part = {"type": kind, text_key: "".join(chunks), **text_fields}
    message = {
        "id": "msg_offline", "type": "message", "role": "assistant",
        "status": "completed", "content": [part],
    }
    response = {
        "id": "resp_offline", "object": "response", "created_at": 1,
        "status": "completed", "model": "custom-model-20261001",
        "output": [message], "error": None, "incomplete_details": None,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {
            "input_tokens": 11, "output_tokens": 5, "total_tokens": 16,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    position = {"item_id": "msg_offline", "output_index": 0, "content_index": 0}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": [], "usage": None}},
        {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": [], "usage": None}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**message, "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", **position, "part": {**part, text_key: ""}},
        {"type": f"response.{kind}.delta", **position, "delta": chunks[0], **({} if refusal else {"logprobs": []})},
        {"type": f"response.{kind}.delta", **position, "delta": chunks[1], **({} if refusal else {"logprobs": []})},
        {"type": f"response.{kind}.done", **position, text_key: "".join(chunks), **({} if refusal else {"logprobs": []})},
        {"type": "response.content_part.done", **position, "part": part},
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": response},
    ]
    for sequence_number, event in enumerate(events):
        event["sequence_number"] = sequence_number
    return events


def thinking_events(
    chunks: tuple[str, ...] = ("先分析\n", "再回答"), *, redacted_only: bool = False,
) -> list[dict[str, Any]]:
    events = reply_events()
    text_index = 1 if redacted_only else 2
    for event in events:
        if "index" in event:
            event["index"] = text_index
    thinking = [] if redacted_only else [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        *[
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": chunk}}
            for chunk in chunks
        ],
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "opaque-signature+/="}},
        {"type": "content_block_stop", "index": 0},
    ]
    events[1:1] = [
        *thinking,
        {"type": "content_block_start", "index": text_index - 1, "content_block": {"type": "redacted_thinking", "data": "opaque-redacted+/="}},
        {"type": "content_block_stop", "index": text_index - 1},
    ]
    return events


def tool_batch_events(
    calls: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("toolu_read", "read_file", ('{"path": ', '"notes.txt"}')),
    ),
    *,
    stop_reason: str = "tool_use",
    decoy_input: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Anthropic response with Thinking, text, and tool calls whose argument pieces are interleaved.

    Pieces for different blocks share the stream but keep their own block index.
    decoy_input is the object carried on each tool_use start; execution must use the pieces.
    """
    start = reply_events()[0]
    events: list[dict[str, Any]] = [
        start,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "先看文件"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "tool-signature+/="}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "我先读文件。"}},
        {"type": "content_block_stop", "index": 1},
    ]
    base = 2
    for offset, (call_id, name, _) in enumerate(calls):
        events.append({
            "type": "content_block_start", "index": base + offset,
            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {} if decoy_input is None else decoy_input},
        })
    piece_count = max(len(pieces) for _, _, pieces in calls)
    for piece_index in range(piece_count):
        for offset, (_, _, pieces) in enumerate(calls):
            if piece_index < len(pieces):
                events.append({
                    "type": "content_block_delta", "index": base + offset,
                    "delta": {"type": "input_json_delta", "partial_json": pieces[piece_index]},
                })
    events.extend(
        {"type": "content_block_stop", "index": base + offset} for offset in range(len(calls))
    )
    events.append({
        "type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 20},
    })
    events.append({"type": "message_stop"})
    return events


def tool_use_events(
    arguments: tuple[str, ...] = ('{"path": ', '"notes.txt"}'), *,
    name: str = "read_file", call_id: str = "toolu_read", stop_reason: str = "tool_use",
) -> list[dict[str, Any]]:
    """An Anthropic response with Thinking, text, and one tool_use whose arguments arrive in pieces."""
    return tool_batch_events(((call_id, name, arguments),), stop_reason=stop_reason)


def function_call_events(
    calls: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
        ("fc_notes", "call_notes", "read_file", ('{"path": ', '"notes.txt"}')),
    ),
    *,
    completed_arguments: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Responses SSE with reasoning, text, and function calls whose arguments arrive in pieces.

    Pieces for different output items are interleaved. completed_arguments replaces only the
    arguments on the finished output items, so a test can show execution uses the pieces.
    """
    stored = tuple("".join(pieces) for _, _, _, pieces in calls) if completed_arguments is None else completed_arguments
    reasoning = {
        "id": "rs_offline", "type": "reasoning", "status": "completed",
        "summary": [{"type": "summary_text", "text": "offline reasoning summary"}],
        "encrypted_content": "opaque-offline-data",
    }
    text = "我先读文件。"
    part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
    message = {
        "id": "msg_offline", "type": "message", "role": "assistant", "status": "completed",
        "phase": "commentary", "content": [part],
    }
    function_items = [
        {
            "id": item_id, "type": "function_call", "call_id": call_id, "name": name,
            "arguments": arguments, "status": "completed",
        }
        for (item_id, call_id, name, _), arguments in zip(calls, stored, strict=True)
    ]
    response = {
        "id": "resp_offline", "object": "response", "created_at": 1, "status": "completed",
        "model": "custom-model-20261001", "output": [reasoning, message, *function_items],
        "error": None, "incomplete_details": None, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [],
        "usage": {
            "input_tokens": 11, "output_tokens": 5, "total_tokens": 16,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    position = {"item_id": "msg_offline", "output_index": 1, "content_index": 0}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": [], "usage": None}},
        {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": [], "usage": None}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning, "status": "in_progress", "summary": [], "encrypted_content": None}},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {"type": "response.output_item.added", "output_index": 1, "item": {**message, "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", **position, "part": {**part, "text": ""}},
        {"type": "response.output_text.delta", **position, "delta": text, "logprobs": []},
        {"type": "response.output_text.done", **position, "text": text, "logprobs": []},
        {"type": "response.content_part.done", **position, "part": part},
        {"type": "response.output_item.done", "output_index": 1, "item": message},
    ]
    for index, item in enumerate(function_items):
        events.append({
            "type": "response.output_item.added", "output_index": index + 2,
            "item": {**item, "status": "in_progress", "arguments": ""},
        })
    piece_count = max(len(pieces) for _, _, _, pieces in calls)
    for piece_index in range(piece_count):
        for index, (item_id, _, _, pieces) in enumerate(calls):
            if piece_index < len(pieces):
                events.append({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id, "output_index": index + 2, "delta": pieces[piece_index],
                })
    for index, item in enumerate(function_items):
        events.append({"type": "response.output_item.done", "output_index": index + 2, "item": item})
    events.append({"type": "response.completed", "response": response})
    for sequence_number, event in enumerate(events):
        event["sequence_number"] = sequence_number
    return events


def native_tool_use(arguments: dict[str, Any], *, name: str = "read_file", call_id: str = "toolu_read") -> dict[str, Any]:
    return {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "先看文件", "signature": "tool-signature+/="},
        {"type": "text", "text": "我先读文件。"},
        {"type": "tool_use", "id": call_id, "name": name, "input": arguments},
    ]}


def unfinished_events(protocol: str, ending: str) -> list[dict[str, Any]]:
    events = reply_events(protocol)
    if ending == "eof":
        return events[:-1]
    if ending == "content_end":
        return events[:5] if protocol == "anthropic" else events[:8]
    if protocol == "anthropic":
        reasons = {"output_limit": "max_tokens", "incomplete": "pause_turn", "tool_terminal": "tool_use"}
        events[-2]["delta"]["stop_reason"] = reasons[ending]
    elif ending == "tool_request":
        tool = {"id": "fc_offline", "type": "function_call", "call_id": "call_offline", "name": "read_file", "arguments": "", "status": "in_progress"}
        done = {**tool, "arguments": '{"path": "notes.txt"}', "status": "completed"}
        events[-1:-1] = [
            {"type": "response.output_item.added", "output_index": 1, "item": tool},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_offline", "output_index": 1, "delta": '{"path": "notes.txt"}'},
            {"type": "response.output_item.done", "output_index": 1, "item": done},
        ]
        events.pop()
    elif ending == "tool_terminal":
        tool = {
            "id": "fc_offline", "type": "function_call", "call_id": "call_offline",
            "name": "read_file", "arguments": "{}", "status": "incomplete",
        }
        events[-1]["response"]["output"].append(tool)
    else:
        reasons = {"output_limit": "max_output_tokens", "incomplete": "content_filter"}
        events[-1]["type"] = "response.incomplete"
        events[-1]["response"]["status"] = "incomplete"
        events[-1]["response"]["incomplete_details"] = {"reason": reasons[ending]}
    if protocol == "openai":
        for number, event in enumerate(events):
            event["sequence_number"] = number
    return events


class ControlledSSE(httpx.AsyncByteStream):
    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        *,
        gate_after: int | None = None,
        close_error: Exception | None = None,
        suppress_cancel: bool = False,
        read_error: Exception | None = None,
    ) -> None:
        self.events = reply_events() if events is None else events
        self.gate_after = gate_after
        self.close_error = close_error
        self.suppress_cancel = suppress_cancel
        self.read_error = read_error
        self.cancelled = asyncio.Event()
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.closing = asyncio.Event()
        self.release_close = asyncio.Event()
        self.release_close.set()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, event in enumerate(self.events):
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            if index == self.gate_after:
                self.blocked.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    if not self.suppress_cancel:
                        raise
                    await self.release.wait()
        if self.read_error is not None:
            raise self.read_error

    async def aclose(self) -> None:
        self.closing.set()
        await self.release_close.wait()
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class RecordingTransport(httpx.MockTransport):
    def __init__(self, stream: ControlledSSE | list[ControlledSSE], status_code: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.closed = False
        streams = iter(stream if isinstance(stream, list) else [stream])

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                status_code,
                headers={"content-type": "text/event-stream"},
                stream=next(streams),
            )

        super().__init__(respond)

    async def aclose(self) -> None:
        self.closed = True
