"""Controlled HTTP responses at the SDK's transport boundary."""

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any

import httpx2 as httpx


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


class ControlledSSE(httpx.AsyncByteStream):
    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        *,
        gate_after: int | None = None,
    ) -> None:
        self.events = reply_events() if events is None else events
        self.gate_after = gate_after
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
                await self.release.wait()

    async def aclose(self) -> None:
        self.closing.set()
        await self.release_close.wait()
        self.closed = True


class RecordingTransport(httpx.MockTransport):
    def __init__(self, stream: ControlledSSE, status_code: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.closed = False

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                status_code,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )

        super().__init__(respond)

    async def aclose(self) -> None:
        self.closed = True
