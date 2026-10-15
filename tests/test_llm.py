import asyncio
from contextlib import aclosing
from dataclasses import replace
import json
from typing import Any

import pytest
from anthropic import APIStatusError, APITimeoutError
import httpx2 as httpx

from replux.config import ProviderConfig
from replux.llm import AnthropicClient, LLMClient, LLMError, ReplyCompleted, TextDelta, create_client
from tests.support import ControlledSSE, RecordingTransport, reply_events


def provider_config() -> ProviderConfig:
    return ProviderConfig(
        protocol="anthropic",
        model="custom-model-20261001",
        base_url="https://gateway.example/anthropic/v2",
        api_key="${LITERAL_KEY}",
    )


async def test_text_arrives_before_completion_and_completion_waits_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-environment-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-environment-token")
    stream = ControlledSSE(gate_after=5)
    stream.release_close.clear()
    transport = RecordingTransport(stream)
    client = create_client(provider_config(), transport=transport)
    assert isinstance(client, LLMClient)
    assert isinstance(client, AnthropicClient)

    async with client:
        reply = client.stream_reply("  你好，Replux  ")
        assert await anext(reply) == TextDelta("你好")
        assert await anext(reply) == TextDelta("，世界\n")
        completed = asyncio.create_task(anext(reply))
        await asyncio.wait_for(stream.blocked.wait(), timeout=2)
        assert not completed.done()
        stream.release.set()
        await asyncio.wait_for(stream.closing.wait(), timeout=2)
        assert not completed.done()
        stream.release_close.set()
        assert await asyncio.wait_for(completed, timeout=2) == ReplyCompleted("end_turn", 11, 5)
        assert stream.closed
        assert [event async for event in reply] == []

    assert transport.closed
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == "https://gateway.example/anthropic/v2/v1/messages"
    assert request.headers["x-api-key"] == "${LITERAL_KEY}"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "custom-model-20261001",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "  你好，Replux  "}],
        "stream": True,
    }
    assert request.extensions["timeout"] == {
        "connect": 120, "read": 120, "write": 120, "pool": 120
    }


@pytest.mark.parametrize(
    ("initial_usage", "final_usage", "expected"),
    [
        ({"input_tokens": 11, "output_tokens": 1}, {"output_tokens": 5}, (11, 5)),
        ({"input_tokens": 11, "output_tokens": 1}, {"input_tokens": 13, "output_tokens": 5}, (13, 5)),
        (None, None, (None, None)),
        ({"input_tokens": 11, "output_tokens": 1}, None, (11, None)),
        (None, {"output_tokens": 0}, (None, 0)),
        ({"input_tokens": 0}, {"output_tokens": 0}, (0, 0)),
    ],
)
async def test_usage_is_final_cumulative_data_with_missing_values_unknown(
    initial_usage: dict[str, Any] | None,
    final_usage: dict[str, Any] | None,
    expected: tuple[int | None, int | None],
) -> None:
    events = reply_events()
    if initial_usage is None:
        del events[0]["message"]["usage"]
    else:
        events[0]["message"]["usage"] = initial_usage
    if final_usage is None:
        del events[5]["usage"]
    else:
        events[5]["usage"] = final_usage
    events.insert(5, {"type": "message_delta", "delta": {"stop_reason": None}, "usage": {"output_tokens": 2}})
    transport = RecordingTransport(ControlledSSE(events))
    async with create_client(provider_config(), transport=transport) as client:
        received = [event async for event in client.stream_reply("hello")]

    assert received == [TextDelta("你好"), TextDelta("，世界\n"), ReplyCompleted("end_turn", *expected)]


@pytest.mark.parametrize(
    "failure", ["eof", "block_stop", "missing_start", "missing_reason", "max_tokens", "tool_use", "pause_turn", "tool_block"]
)
async def test_incomplete_or_unsupported_response_never_publishes_completion(failure: str) -> None:
    events = reply_events()
    if failure == "eof":
        events.pop()
    elif failure == "block_stop":
        events = events[:5]
    elif failure == "missing_start":
        events.pop(0)
    elif failure == "tool_block":
        events[1]["content_block"] = {"type": "tool_use", "id": "tool_1", "name": "unavailable", "input": {}}
    else:
        events[5]["delta"]["stop_reason"] = None if failure == "missing_reason" else failure
    stream = ControlledSSE(events)
    received = []
    async with create_client(provider_config(), transport=RecordingTransport(stream)) as client:
        with pytest.raises(RuntimeError):
            async for event in client.stream_reply("hello"):
                received.append(event)
    assert not any(isinstance(event, ReplyCompleted) for event in received)
    assert stream.closed


async def test_concurrent_reply_is_rejected_before_another_request() -> None:
    stream = ControlledSSE()
    transport = RecordingTransport(stream)
    async with create_client(provider_config(), transport=transport) as client:
        async with aclosing(client.stream_reply("first")) as first:
            assert await anext(first) == TextDelta("你好")
            async with aclosing(client.stream_reply("second")) as second:
                with pytest.raises(RuntimeError, match="进行"):
                    await anext(second)
    assert stream.closed
    assert len(transport.requests) == 1


def test_factory_rejects_unknown_protocol_without_fallback() -> None:
    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(ValueError, match="protocol"):
        create_client(replace(provider_config(), protocol="unknown"), transport=transport)
    assert transport.requests == []


@pytest.mark.parametrize("failure", [429, 500, "timeout"])
async def test_sdk_request_failure_has_no_automatic_retry(failure: int | str) -> None:
    requests = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        assert isinstance(failure, int)
        return httpx.Response(failure, json={"error": {"type": "api_error", "message": "offline error"}})

    async with create_client(provider_config(), transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(LLMError) as caught:
            await anext(client.stream_reply("hello"))
    assert isinstance(caught.value.__cause__, (APIStatusError, APITimeoutError))
    assert len(requests) == 1
    assert requests[0].headers["x-stainless-retry-count"] == "0"
