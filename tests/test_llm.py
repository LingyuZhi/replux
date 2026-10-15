import asyncio
from contextlib import aclosing
from dataclasses import replace
import json
from typing import Any

import pytest
from anthropic import APIStatusError, APITimeoutError
from anthropic.types import ThinkingConfigParam
import httpx2 as httpx
import openai

from replux.llm import AnthropicClient, LLMClient, LLMError, OpenAIClient, ReplyCompleted, TextDelta, create_client
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events


@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "adaptive", "display": "summarized"},
        {"type": "enabled", "budget_tokens": 1024, "display": "summarized"},
        {"type": "disabled"},
    ],
)
async def test_rejected_thinking_configuration_is_reported_without_fallback(
    thinking: ThinkingConfigParam,
) -> None:
    requests = []

    def reject(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json={"error": {
            "type": "invalid_request_error",
            "message": "thinking is unsupported for this model; test-secret-never-display",
        }})

    config = replace(provider_config(), thinking=thinking)
    async with create_client(config, transport=httpx.MockTransport(reject)) as client:
        with pytest.raises(LLMError, match="thinking") as caught:
            await anext(client.stream_reply("hello"))

    assert isinstance(caught.value.__cause__, APIStatusError)
    assert caught.value.__cause__.status_code == 400
    assert "test-secret-never-display" not in str(caught.value)
    assert len(requests) == 1
    request = json.loads(requests[0].content)
    assert request["thinking"] == thinking
    assert request["max_tokens"] == 8192
    assert request["model"] == "custom-model-20261001"
    assert requests[0].headers["x-stainless-retry-count"] == "0"


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_text_arrives_before_completion_and_completion_waits_for_cleanup(
    monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-environment-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-environment-token")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-environment-key")
    events = reply_events(protocol)
    stream = ControlledSSE(events, gate_after=len(events) - 2)
    stream.release_close.clear()
    transport = RecordingTransport(stream)
    client = create_client(provider_config(protocol), transport=transport)
    assert isinstance(client, LLMClient)
    assert isinstance(client, AnthropicClient if protocol == "anthropic" else OpenAIClient)

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
        stop_reason = "end_turn" if protocol == "anthropic" else "completed"
        assert await asyncio.wait_for(completed, timeout=2) == ReplyCompleted(stop_reason, 11, 5)
        assert stream.closed
        assert [event async for event in reply] == []

    assert transport.closed
    assert len(transport.requests) == 1
    request = transport.requests[0]
    if protocol == "anthropic":
        assert str(request.url) == "https://gateway.example/anthropic/v2/v1/messages"
        assert request.headers["x-api-key"] == "${LITERAL_KEY}"
        assert "authorization" not in request.headers
        assert json.loads(request.content) == {
            "model": "custom-model-20261001", "max_tokens": 8192,
            "messages": [{"role": "user", "content": "  你好，Replux  "}], "stream": True,
        }
    else:
        assert str(request.url) == "https://gateway.example/anthropic/v2/responses"
        assert request.headers["authorization"] == "Bearer ${LITERAL_KEY}"
        assert json.loads(request.content) == {
            "model": "custom-model-20261001", "max_output_tokens": 8192,
            "input": [{"role": "user", "content": "  你好，Replux  "}],
            "stream": True, "store": False,
        }
    assert request.extensions["timeout"] == {
        "connect": 120, "read": 120, "write": 120, "pool": 120
    }


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_completed_refusal_is_visible_reply_text(protocol: str) -> None:
    stream = ControlledSSE(reply_events(protocol, refusal=True))
    transport = RecordingTransport(stream)
    async with create_client(provider_config(protocol), transport=transport) as client:
        received = [event async for event in client.stream_reply("hello")]

    stop_reason = "refusal" if protocol == "anthropic" else "completed"
    assert received == [TextDelta("抱歉，"), TextDelta("无法回答。\n"), ReplyCompleted(stop_reason, 11, 5)]
    assert stream.closed
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize(
    ("initial_usage", "final_usage", "expected", "openai_expected"),
    [
        ({"input_tokens": 11, "output_tokens": 1}, {"output_tokens": 5}, (11, 5), (None, 5)),
        ({"input_tokens": 11, "output_tokens": 1}, {"input_tokens": 13, "output_tokens": 5}, (13, 5), (13, 5)),
        (None, None, (None, None), (None, None)),
        ({"input_tokens": 11, "output_tokens": 1}, None, (11, None), (None, None)),
        (None, {"output_tokens": 0}, (None, 0), (None, 0)),
        ({"input_tokens": 0}, {"input_tokens": 0, "output_tokens": 0}, (0, 0), (0, 0)),
        (None, {"input_tokens": 11}, (11, None), (11, None)),
    ],
)
async def test_usage_is_final_cumulative_data_with_missing_values_unknown(
    protocol: str,
    initial_usage: dict[str, Any] | None,
    final_usage: dict[str, Any] | None,
    expected: tuple[int | None, int | None],
    openai_expected: tuple[int | None, int | None],
) -> None:
    events = reply_events(protocol)
    initial = events[0]["message"] if protocol == "anthropic" else events[0]["response"]
    final = events[5] if protocol == "anthropic" else events[-1]["response"]
    if initial_usage is None:
        del initial["usage"]
    else:
        initial["usage"] = initial_usage
    if final_usage is None:
        if initial_usage is None:
            del final["usage"]
        else:
            final["usage"] = None
    else:
        final["usage"] = final_usage
    if protocol == "anthropic":
        events.insert(5, {"type": "message_delta", "delta": {"stop_reason": None}, "usage": {"output_tokens": 2}})
    transport = RecordingTransport(ControlledSSE(events))
    async with create_client(provider_config(protocol), transport=transport) as client:
        received = [event async for event in client.stream_reply("hello")]

    stop_reason = "end_turn" if protocol == "anthropic" else "completed"
    tokens = expected if protocol == "anthropic" else openai_expected
    assert received == [TextDelta("你好"), TextDelta("，世界\n"), ReplyCompleted(stop_reason, *tokens)]


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


@pytest.mark.parametrize("terminal", ["response.incomplete", "response.failed", "error"])
async def test_openai_terminal_failure_closes_without_waiting_for_eof(terminal: str) -> None:
    events = reply_events("openai")
    final = events[-1]
    final["type"] = terminal
    if terminal == "error":
        final.clear()
        final.update(type="error", code="server_error", message="test-secret-never-display", sequence_number=9)
    else:
        final["response"]["status"] = terminal.removeprefix("response.")
        if terminal == "response.incomplete":
            final["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
        else:
            final["response"]["error"] = {"code": "server_error", "message": "test-secret-never-display"}
    stream = ControlledSSE(events, gate_after=len(events) - 1)
    transport = RecordingTransport(stream)
    received = []
    async with create_client(provider_config("openai"), transport=transport) as client:
        async def consume() -> None:
            async for event in client.stream_reply("hello"):
                received.append(event)

        with pytest.raises(LLMError) as caught:
            await asyncio.wait_for(consume(), timeout=2)
    assert received == [TextDelta("你好"), TextDelta("，世界\n")]
    assert "test-secret-never-display" not in str(caught.value)
    assert stream.closed
    assert transport.closed


@pytest.mark.parametrize("failure", ["eof", "text_done", "status", "tool_item", "tool_final", "incomplete_message"])
async def test_openai_requires_a_completed_supported_response(failure: str) -> None:
    events = reply_events("openai")
    tool = {"id": "fc_offline", "type": "function_call", "call_id": "call_offline", "name": "unavailable", "arguments": "{}", "status": "completed"}
    if failure == "eof":
        events.pop()
    elif failure == "text_done":
        events = events[:7]
    elif failure == "status":
        events[-1]["response"]["status"] = "incomplete"
    elif failure == "tool_item":
        events[2]["item"] = tool
    elif failure == "tool_final":
        events[-1]["response"]["output"] = [tool]
    else:
        events[-1]["response"]["output"][0]["status"] = "incomplete"
    stream = ControlledSSE(events)
    transport = RecordingTransport(stream)
    received = []
    async with create_client(provider_config("openai"), transport=transport) as client:
        with pytest.raises(LLMError):
            async for event in client.stream_reply("hello"):
                received.append(event)
    assert not any(isinstance(event, ReplyCompleted) for event in received)
    assert stream.closed
    assert transport.closed


async def test_openai_replays_complete_output_without_displaying_reasoning() -> None:
    events = reply_events("openai")
    summary = {"type": "summary_text", "text": "offline reasoning summary"}
    reasoning = {
        "id": "rs_offline", "type": "reasoning", "summary": [summary],
        "encrypted_content": "opaque-offline-data", "status": "completed",
    }
    position = {"item_id": "rs_offline", "output_index": 0, "summary_index": 0}
    for event in events:
        if "output_index" in event:
            event["output_index"] += 1
    events[-1]["response"]["output"].insert(0, reasoning)
    events[-1]["response"]["output"][1]["phase"] = "final_answer"
    events[2:2] = [
        {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning, "status": "in_progress", "summary": [], "encrypted_content": None}},
        {"type": "response.reasoning_summary_part.added", **position, "part": {**summary, "text": ""}},
        {"type": "response.reasoning_summary_text.delta", **position, "delta": "offline reasoning summary"},
        {"type": "response.reasoning_summary_text.done", **position, "text": "offline reasoning summary"},
        {"type": "response.reasoning_summary_part.done", **position, "part": summary},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
    ]
    for number, event in enumerate(events):
        event["sequence_number"] = number
    stream = ControlledSSE(events)
    follow_up = ControlledSSE(reply_events("openai"))
    transport = RecordingTransport([stream, follow_up])
    async with create_client(provider_config("openai"), transport=transport) as client:
        received = [event async for event in client.stream_reply("hello")]
        _ = [event async for event in client.stream_reply("follow-up")]
    assert received == [TextDelta("你好"), TextDelta("，世界\n"), ReplyCompleted("completed", 11, 5)]
    request = json.loads(transport.requests[1].content)
    assert request["input"] == [
        {"role": "user", "content": "hello"},
        *events[-1]["response"]["output"],
        {"role": "user", "content": "follow-up"},
    ]
    assert set(request) == {"model", "input", "max_output_tokens", "stream", "store"}
    assert request["store"] is False
    assert stream.closed
    assert follow_up.closed
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_concurrent_reply_is_rejected_before_another_request(protocol: str) -> None:
    stream = ControlledSSE(reply_events(protocol))
    transport = RecordingTransport(stream)
    async with create_client(provider_config(protocol), transport=transport) as client:
        async with aclosing(client.stream_reply("first")) as first:
            assert await anext(first) == TextDelta("你好")
            async with aclosing(client.stream_reply("second")) as second:
                with pytest.raises(RuntimeError, match="in progress"):
                    await anext(second)
    assert stream.closed
    assert len(transport.requests) == 1


def test_factory_rejects_unknown_protocol_without_fallback() -> None:
    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(ValueError, match="protocol"):
        create_client(replace(provider_config(), protocol="unknown"), transport=transport)
    assert transport.requests == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("failure", [429, 500, "timeout"])
async def test_sdk_request_failure_has_no_automatic_retry(failure: int | str, protocol: str) -> None:
    requests = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        assert isinstance(failure, int)
        return httpx.Response(failure, json={"error": {"type": "api_error", "message": "offline error"}})

    async with create_client(provider_config(protocol), transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(LLMError) as caught:
            await anext(client.stream_reply("hello"))
    errors = (APIStatusError, APITimeoutError) if protocol == "anthropic" else (openai.APIStatusError, openai.APITimeoutError)
    assert isinstance(caught.value.__cause__, errors)
    assert len(requests) == 1
    assert requests[0].headers["x-stainless-retry-count"] == "0"
