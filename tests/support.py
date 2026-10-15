"""Controlled HTTP responses at the SDK's transport boundary."""

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any

import httpx2 as httpx


def reply_events() -> list[dict[str, Any]]:
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
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你好"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "，世界\n"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]


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
