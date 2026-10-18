import asyncio
import json
from typing import Any

import anthropic
import httpx2 as httpx
import openai
import pytest
from textual.widgets import Static, TextArea

from replux.llm import AuthenticationError, LLMError, NetworkError, RateLimitError, StreamEvent, TextDelta, create_client
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, native_reply, provider_config, reply_events
from tests.tui_support import screen_lines, screen_text


@pytest.mark.parametrize(
    "protocol, terminal, code, error_type",
    [
        ("anthropic", "error", "authentication_error", AuthenticationError),
        ("anthropic", "error", "rate_limit_error", RateLimitError),
        ("openai", "response.failed", "rate_limit_exceeded", RateLimitError),
        ("openai", "error", "rate_limit_exceeded", RateLimitError),
        ("openai", "error", "invalid_api_key", AuthenticationError),
    ],
)
async def test_stream_error_codes_keep_their_category_and_actual_http_status(
    protocol: str, terminal: str, code: str, error_type: type[LLMError],
) -> None:
    config = provider_config(protocol)
    events = reply_events(protocol)
    partial = events[:3 if protocol == "anthropic" else 5]
    if protocol == "anthropic":
        partial.append({"type": "error", "error": {"type": code, "message": config.api_key}})
    elif terminal == "response.failed":
        final = events[-1]
        final["type"] = terminal
        final["response"]["status"] = "failed"
        final["response"]["error"] = {"code": code, "message": config.api_key}
        partial.append(final)
    else:
        partial.append({"type": "error", "code": code, "message": config.api_key, "param": None, "sequence_number": 5})
    failed = ControlledSSE(partial)
    transport = RecordingTransport([failed, ControlledSSE(reply_events(protocol))])
    received: list[StreamEvent] = []
    async with create_client(config, transport=transport) as client:
        with pytest.raises(error_type) as caught:
            async for event in client.stream_reply("failed"):
                received.append(event)
        assert type(caught.value) is error_type
        assert caught.value.status_code == 200
        assert config.api_key not in str(caught.value)
        if protocol == "anthropic":
            assert isinstance(caught.value.__cause__, anthropic.APIStatusError)
        if isinstance(caught.value, RateLimitError):
            assert caught.value.retry_after is None
        assert failed.closed
        assert received == [TextDelta("你好")]
        assert len(transport.requests) == 1
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 2
    assert json.loads(transport.requests[-1].content)[field] == [{"role": "user", "content": "follow-up"}]
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize(
    "failure, error_type, hint",
    [
        (401, AuthenticationError, "Authentication"),
        (429, RateLimitError, "Rate limit"),
        (400, LLMError, "parameters"),
        (500, LLMError, "service"),
        ("connection", NetworkError, "connection"),
        ("timeout", NetworkError, "timed out"),
    ],
)
async def test_request_errors_keep_cause_and_status_without_retry_or_history_changes(
    protocol: str, failure: int | str, error_type: type[LLMError], hint: str,
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []
    streams: list[ControlledSSE] = []
    config = provider_config(protocol)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 2:
            if failure == "connection":
                raise httpx.ConnectError(config.api_key, request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout(config.api_key, request=request)
            assert isinstance(failure, int)
            return httpx.Response(failure, headers={"retry-after": "60"}, json={"error": {
                "type": "invalid_request_error", "message": config.api_key,
            }})
        stream = ControlledSSE(reply_events(protocol))
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with create_client(config, transport=httpx.MockTransport(respond)) as client:
        _ = [event async for event in client.stream_reply("first")]
        with pytest.raises(error_type, match=hint) as caught:
            _ = [event async for event in client.stream_reply("failed")]
        error = caught.value
        assert type(error) is error_type
        assert error.status_code == (failure if isinstance(failure, int) else None)
        sdk: Any = anthropic if protocol == "anthropic" else openai
        cause_type = sdk.APIStatusError if isinstance(failure, int) else sdk.APIConnectionError
        assert isinstance(error.__cause__, cause_type)
        if isinstance(error, RateLimitError):
            assert error.retry_after == "60"
        assert config.api_key not in str(error)
        assert len(requests) == 2
        _ = [event async for event in client.stream_reply("follow-up")]

    field = "messages" if protocol == "anthropic" else "input"
    assert len(requests) == 3
    assert json.loads(requests[-1].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    assert all(json.loads(request.content)["model"] == config.model for request in requests)
    assert all(request.headers["x-stainless-retry-count"] == "0" for request in requests)
    assert all(stream.closed for stream in streams)
    captured = capsys.readouterr()
    assert config.api_key not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("partial", [False, True], ids=["before-text", "partial-text"])
@pytest.mark.parametrize("late_events", [False, True], ids=["cooperative", "late-events"])
async def test_cancel_closes_stream_and_discards_late_events_and_unfinished_turn(
    protocol: str, partial: bool, late_events: bool,
) -> None:
    gate = (2 if protocol == "anthropic" else 4) if partial else 0
    pending = ControlledSSE(reply_events(protocol), gate_after=gate, suppress_cancel=late_events)
    pending.release_close.clear()
    streams = [ControlledSSE(reply_events(protocol)), pending, ControlledSSE(reply_events(protocol))]
    transport = RecordingTransport(streams)
    received: list[StreamEvent] = []
    async with create_client(provider_config(protocol), transport=transport) as client:
        _ = [event async for event in client.stream_reply("first")]

        async def consume() -> None:
            async for event in client.stream_reply("cancelled"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(pending.blocked.wait(), timeout=2)
            expected = [TextDelta("你好")] if partial else []
            assert received == expected
            consuming.cancel()
            await asyncio.wait_for(pending.cancelled.wait(), timeout=2)
            pending.release.set()
            await asyncio.wait_for(pending.closing.wait(), timeout=2)
            assert not consuming.done()
            assert not pending.closed
            assert received == expected
            with pytest.raises(RuntimeError, match="in progress"):
                await anext(client.stream_reply("too soon"))
            assert len(transport.requests) == 2
        finally:
            pending.release.set()
            pending.release_close.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consuming, timeout=2)
        assert pending.closed
        assert received == expected
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 3
    assert json.loads(transport.requests[-1].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    assert all(stream.closed for stream in streams)
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("failure", ["timeout", "connection", "api_error"])
async def test_partial_failure_keeps_visible_text_and_draft_then_uses_only_completed_history(
    protocol: str, failure: str, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    config = provider_config(protocol)
    events = reply_events(protocol)
    gate = 2 if protocol == "anthropic" else 4
    partial = events[:gate + 1]
    if failure == "api_error":
        if protocol == "anthropic":
            partial.append({"type": "error", "error": {"type": "api_error", "message": config.api_key}})
        else:
            final = events[-1]
            final["type"] = "response.failed"
            final["response"]["status"] = "failed"
            final["response"]["error"] = {"code": "server_error", "message": config.api_key}
            partial.append(final)
    pending = ControlledSSE(
        partial, gate_after=gate,
        read_error=None if failure == "api_error" else (
            httpx.ReadTimeout(config.api_key) if failure == "timeout" else httpx.ReadError(config.api_key)
        ),
    )
    streams = [ControlledSSE(reply_events(protocol)), pending, ControlledSSE(reply_events(protocol))]
    transport = RecordingTransport(streams)
    app = RepluxApp(create_client(config, transport=transport), model=config.model)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "first"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.text = "failed"
        await pilot.press("enter")
        await asyncio.wait_for(pending.blocked.wait(), timeout=2)
        prompt.text = "follow-up"
        await pilot.pause()
        assert len(app.query(".reply")) == 2
        assert any(line.strip() == "● 你好" for line in screen_lines(app))
        pending.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert pending.closed
        assert app.is_running
        assert prompt.text == "follow-up"
        assert app.query_one("#status", Static).content == "Ready"
        assert "Request failed" in str(app.query_one(".turn-status", Static).content)
        assert config.api_key not in app.export_screenshot()
        assert len(transport.requests) == 2
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert len(app.query(".reply")) == 3
        assert "你好，世界" in screen_text(app)
        assert "Request failed" in str(app.query_one(".turn-status", Static).content)
        await pilot.press("ctrl+c")
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 3
    assert json.loads(transport.requests[-1].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    if protocol == "anthropic":
        assert all(json.loads(request.content)["max_tokens"] == 128000 for request in transport.requests)
    else:
        assert all("max_output_tokens" not in json.loads(request.content) for request in transport.requests)
    assert all(json.loads(request.content)["model"] == config.model for request in transport.requests)
    assert all(stream.closed for stream in streams)
    assert transport.closed
    captured = capsys.readouterr()
    assert config.api_key not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["completed", "eof", "read_error", "api_error"])
async def test_cancel_during_stream_cleanup_waits_before_releasing_client(protocol: str, ending: str) -> None:
    events = reply_events(protocol)
    if ending != "completed":
        events = events[:-1]
    if ending == "api_error":
        if protocol == "anthropic":
            events.append({"type": "error", "error": {"type": "api_error", "message": "offline error"}})
        else:
            final = reply_events(protocol)[-1]
            final["type"] = "response.failed"
            final["response"]["status"] = "failed"
            final["response"]["error"] = {"code": "server_error", "message": "offline error"}
            events.append(final)
    pending = ControlledSSE(events, read_error=httpx.ReadError("offline read error") if ending == "read_error" else None)
    pending.release_close.clear()
    transport = RecordingTransport([pending, ControlledSSE(reply_events(protocol))])
    received: list[StreamEvent] = []
    async with create_client(provider_config(protocol), transport=transport) as client:
        async def consume() -> None:
            async for event in client.stream_reply("cancelled during cleanup"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(pending.closing.wait(), timeout=2)
            consuming.cancel()
            # A barrier on the task proves cancellation has been handled, without
            # assuming how many event loop ticks cleanup needs.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(consuming), timeout=0.05)
            assert not pending.closed
            with pytest.raises(RuntimeError, match="in progress"):
                await anext(client.stream_reply("too soon"))
        finally:
            pending.release_close.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consuming, timeout=2)
        assert pending.closed
        assert received == [TextDelta("你好"), TextDelta("，世界\n")]
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 2
    assert json.loads(transport.requests[-1].content)[field] == [{"role": "user", "content": "follow-up"}]
    assert transport.closed
