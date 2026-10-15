import asyncio
import json
from typing import Any

import pytest
import httpx2 as httpx

from replux.llm import IncompleteReplyError, LLMError, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta, create_client
from tests.support import ControlledSSE, RecordingTransport, native_reply, provider_config, reply_events, thinking_events, unfinished_events


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("refusal", [False, True], ids=["text", "refusal"])
async def test_next_request_replays_each_completed_turn(protocol: str, refusal: bool) -> None:
    streams = [ControlledSSE(reply_events(protocol, refusal=refusal)) for _ in range(3)]
    transport = RecordingTransport(streams)
    expected: list[dict[str, Any]] = []
    async with create_client(provider_config(protocol), transport=transport) as client:
        for user_input in ["  第一问  ", "为什么？", "再说明一次"]:
            received = [event async for event in client.stream_reply(user_input)]
            expected.append({"role": "user", "content": user_input})
            field = "messages" if protocol == "anthropic" else "input"
            request = json.loads(transport.requests[-1].content)
            assert request[field] == expected
            assert set(request) == (
                {"model", "messages", "max_tokens", "stream"} if protocol == "anthropic"
                else {"model", "input", "max_output_tokens", "stream", "store"}
            )
            stop_reason = ("refusal" if refusal else "end_turn") if protocol == "anthropic" else "completed"
            chunks = ["抱歉，", "无法回答。\n"] if refusal else ["你好", "，世界\n"]
            assert received == [TextDelta(chunks[0]), TextDelta(chunks[1]), ReplyCompleted(stop_reason, 11, 5)]
            expected.extend(native_reply(protocol, refusal=refusal))
    assert len(transport.requests) == 3
    assert all(stream.closed for stream in streams)
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("cleanup_failure", [False, True], ids=["closed", "cleanup-failed"])
async def test_turn_commit_requires_final_success_and_finished_cleanup(
    protocol: str, cleanup_failure: bool,
) -> None:
    events = reply_events(protocol)
    pending = ControlledSSE(
        events, gate_after=len(events) - 2,
        close_error=httpx.ReadError("offline cleanup failure") if cleanup_failure else None,
    )
    pending.release_close.clear()
    transport = RecordingTransport([
        ControlledSSE(reply_events(protocol)), pending, ControlledSSE(reply_events(protocol)),
    ])
    received: list[StreamEvent] = []
    async with create_client(provider_config(protocol), transport=transport) as client:
        _ = [event async for event in client.stream_reply("first")]

        async def consume() -> None:
            async for event in client.stream_reply("pending"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(pending.blocked.wait(), timeout=2)
            assert received == [TextDelta("你好"), TextDelta("，世界\n")]
            assert not consuming.done()
            pending.release.set()
            await asyncio.wait_for(pending.closing.wait(), timeout=2)
            assert received == [TextDelta("你好"), TextDelta("，世界\n")]
            assert not consuming.done()
            with pytest.raises(RuntimeError, match="进行"):
                await anext(client.stream_reply("too soon"))
            assert len(transport.requests) == 2
        finally:
            pending.release.set()
            pending.release_close.set()
            if cleanup_failure:
                with pytest.raises(LLMError):
                    await asyncio.wait_for(consuming, timeout=2)
            else:
                await asyncio.wait_for(consuming, timeout=2)
        assert pending.closed
        completions = [event for event in received if isinstance(event, ReplyCompleted)]
        assert len(completions) == (0 if cleanup_failure else 1)
        _ = [event async for event in client.stream_reply("after cleanup")]
    expected = [{"role": "user", "content": "first"}, *native_reply(protocol)]
    if not cleanup_failure:
        expected.extend([{"role": "user", "content": "pending"}, *native_reply(protocol)])
    expected.append({"role": "user", "content": "after cleanup"})
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[2].content)[field] == expected
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["eof", "content_end", "output_limit", "incomplete", "tool_request", "tool_terminal"])
async def test_unfinished_turn_preserves_previously_completed_history(protocol: str, ending: str) -> None:
    streams = [
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(unfinished_events(protocol, ending)),
        ControlledSSE(reply_events(protocol)),
    ]
    transport = RecordingTransport(streams)
    async with create_client(provider_config(protocol), transport=transport) as client:
        _ = [event async for event in client.stream_reply("first")]
        received: list[StreamEvent] = []
        with pytest.raises(IncompleteReplyError) as caught:
            async for event in client.stream_reply("unfinished"):
                received.append(event)
        if ending.startswith("tool_"):
            assert "不支持工具" in str(caught.value)
        assert received == [TextDelta("你好"), TextDelta("，世界\n")]
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 3
    assert json.loads(transport.requests[2].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    assert all(stream.closed for stream in streams)
    assert transport.closed


async def test_anthropic_replays_native_thinking_signature_redacted_and_citations() -> None:
    events = thinking_events()
    citation = {
        "type": "char_location", "cited_text": "世界", "document_index": 0,
        "document_title": "Reference", "start_char_index": 0, "end_char_index": 2,
    }
    events.insert(-3, {"type": "content_block_delta", "index": 2, "delta": {"type": "citations_delta", "citation": citation}})
    streams = [ControlledSSE(events), ControlledSSE()]
    transport = RecordingTransport(streams)
    async with create_client(provider_config(), transport=transport) as client:
        received = [event async for event in client.stream_reply("first")]
        assert received == [
            ThinkingDelta("先分析\n"), ThinkingDelta("再回答"),
            TextDelta("你好"), TextDelta("，世界\n"), ReplyCompleted("end_turn", 11, 5),
        ]
        _ = [event async for event in client.stream_reply("follow-up")]
    assert json.loads(transport.requests[1].content)["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "先分析\n再回答", "signature": "opaque-signature+/="},
            {"type": "redacted_thinking", "data": "opaque-redacted+/="},
            {"type": "text", "text": "你好，世界\n", "citations": [citation]},
        ]},
        {"role": "user", "content": "follow-up"},
    ]
    assert all(stream.closed for stream in streams)
    assert transport.closed
