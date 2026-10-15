import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
import subprocess
import sys

import pytest
import httpx2 as httpx
from textual.worker import WorkerCancelled, WorkerFailed
from textual.widgets import Collapsible, Input, Static

from replux.cli import create_app
from replux.llm import LLMClient, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta, create_client
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, native_reply, provider_config, reply_events, thinking_events, unfinished_events
from tests.test_config import VALID_YAML


@pytest.mark.parametrize("empty_text", ["", " \n"], ids=["empty", "whitespace"])
async def test_thinking_toggle_preserves_streaming_content_and_stays_local_to_each_turn(empty_text: str) -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.reached = [asyncio.Event() for _ in range(3)]
            self.release = [asyncio.Event() for _ in range(3)]
            self.closed = False

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            for index, text in enumerate([empty_text, "[bold]先分析[/bold]\n", "再回答"]):
                yield ThinkingDelta(text)
                self.reached[index].set()
                await self.release[index].wait()
            yield TextDelta("正文正常显示")
            yield ReplyCompleted("end_turn", 11, 5)

        async def aclose(self) -> None:
            self.closed = True

    client = ScriptedClient()
    app = RepluxApp(client)
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(client.reached[0].wait(), timeout=2)
        assert not app.query(Collapsible)

        client.release[0].set()
        await asyncio.wait_for(client.reached[1].wait(), timeout=2)
        await pilot.pause()
        thinking = app.query_one(Collapsible)
        title = thinking.query_one("CollapsibleTitle", Static)
        assert thinking.collapsed
        assert str(title.content) == "+ Thinking"
        assert "先分析" not in app.export_screenshot()
        prompt.value = "next draft"

        await pilot.click(title, offset=(1, 0))
        assert not thinking.collapsed
        assert str(title.content) == "- Thinking"
        assert "[bold]先分析[/bold]" in app.export_screenshot()
        client.release[1].set()
        await asyncio.wait_for(client.reached[2].wait(), timeout=2)
        await pilot.pause()
        assert "再回答" in app.export_screenshot()
        assert app.query_one("#status", Static).content == "Generating..."
        assert not thinking.collapsed

        await pilot.click(title, offset=(1, 0))
        assert thinking.collapsed
        assert "先分析" not in app.export_screenshot()
        prompt.focus()
        await pilot.press("shift+tab")
        assert app.focused is title
        await pilot.press("enter")
        assert not thinking.collapsed
        assert "再回答" in app.export_screenshot()
        assert client.inputs == ["first"]
        assert prompt.value == "next draft"

        client.release[2].set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert app.query_one(".reply", Static).content == "正文正常显示"
        await pilot.press("enter")
        assert thinking.collapsed
        assert client.inputs == ["first"]
        assert prompt.value == "next draft"
        await pilot.press("enter")
        prompt.focus()
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        regions = list(app.query(Collapsible))
        assert len(regions) == 2
        assert not regions[0].collapsed
        assert regions[1].collapsed
        assert client.inputs == ["first", "next draft"]
    assert client.closed


@pytest.mark.parametrize("configured", [False, True], ids=["model-default", "summarized"])
async def test_yaml_to_streaming_thinking_keeps_toggles_local_and_native_blocks_intact(
    tmp_path: Path, configured: bool,
) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(
        VALID_YAML + ("thinking: {type: adaptive, display: summarized}\n" if configured else ""),
        encoding="utf-8",
    )
    stream = ControlledSSE(thinking_events(), gate_after=2)
    transport = RecordingTransport([stream, ControlledSSE()])
    app = create_app(["--config", str(path)], transport=transport)

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(stream.blocked.wait(), timeout=2)
        await pilot.pause()
        thinking = app.query_one(Collapsible)
        title = thinking.query_one("CollapsibleTitle", Static)
        assert thinking.collapsed
        assert "先分析" not in app.export_screenshot()
        assert app.query_one(".reply", Static).content == ""
        first_request = json.loads(transport.requests[0].content)
        if configured:
            assert first_request["thinking"] == {"type": "adaptive", "display": "summarized"}
        else:
            assert "thinking" not in first_request

        await pilot.click(title, offset=(1, 0))
        assert "先分析" in app.export_screenshot()
        stream.release_close.clear()
        stream.release.set()
        try:
            await asyncio.wait_for(stream.closing.wait(), timeout=2)
            await pilot.pause()
            assert "再回答" in app.export_screenshot()
            assert "opaque-" not in app.export_screenshot()
            assert app.query_one(".thinking-content", Static).content == "先分析\n再回答"
            assert app.query_one(".reply", Static).content == "你好，世界\n"
            assert app.query_one("#status", Static).content == "Generating..."
            assert not stream.closed
            await pilot.click(title, offset=(1, 0))
            assert "先分析" not in app.export_screenshot()
            assert "你好，世界" in app.export_screenshot()
            title.focus()
            await pilot.press("enter")
            assert not thinking.collapsed
            assert len(transport.requests) == 1
        finally:
            stream.release_close.set()

        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.focus()
        prompt.value = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert app.query_one("#status", Static).content == "Ready"

    assert len(transport.requests) == 2
    second_request = json.loads(transport.requests[1].content)
    assert {key: value for key, value in second_request.items() if key != "messages"} == {
        key: value for key, value in first_request.items() if key != "messages"
    }
    assert second_request["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "先分析\n再回答", "signature": "opaque-signature+/="},
            {"type": "redacted_thinking", "data": "opaque-redacted+/="},
            {"type": "text", "text": "你好，世界\n"},
        ]},
        {"role": "user", "content": "follow-up"},
    ]
    assert stream.closed
    assert transport.closed


@pytest.mark.parametrize("content", ["none", "empty", "redacted-only"])
async def test_no_thinking_control_without_displayable_api_text(tmp_path: Path, content: str) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(VALID_YAML + "thinking: {type: adaptive, display: summarized}\n", encoding="utf-8")
    events = reply_events() if content == "none" else thinking_events(("",), redacted_only=content == "redacted-only")
    stream = ControlledSSE(events, gate_after=len(events) - 2)
    transport = RecordingTransport(stream)
    app = create_app(["--config", str(path)], transport=transport)

    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "hello"
        await pilot.press("enter")
        await asyncio.wait_for(stream.blocked.wait(), timeout=2)
        await pilot.pause()
        assert not app.query(Collapsible)
        assert app.query_one(".reply", Static).content == "你好，世界\n"
        assert "opaque-" not in app.export_screenshot()
        stream.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert not app.query(Collapsible)
    assert len(transport.requests) == 1
    assert stream.closed
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("refusal", [False, True], ids=["text", "refusal"])
async def test_yaml_to_sdk_to_visible_reply_before_terminal_event(
    tmp_path: Path, protocol: str, refusal: bool
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    events = reply_events(protocol, refusal=refusal)
    stream = ControlledSSE(events, gate_after=len(events) - 2)
    transport = RecordingTransport(stream)
    app = create_app(["--config", str(config)], transport=transport)

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", Input)
        assert prompt.region.bottom == 24
        assert app.query_one("#status", Static).region.bottom <= prompt.region.y
        prompt.value = "  "
        await pilot.press("enter")
        assert transport.requests == []
        assert not app.query(".reply")

        prompt.value = "你好，Replux"
        await pilot.press("enter")
        await asyncio.wait_for(stream.blocked.wait(), timeout=2)
        await pilot.pause()
        expected = "抱歉，无法回答。\n" if refusal else "你好，世界\n"
        assert app.query_one(".reply", Static).content == expected
        assert app.query_one("#status", Static).content == "Generating..."
        assert not stream.closed
        assert len(transport.requests) == 1

        prompt.value = "下一条草稿"
        await pilot.press("enter")
        assert len(transport.requests) == 1
        assert prompt.value == "下一条草稿"

        stream.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert stream.closed
        assert app.query_one("#status", Static).content == "Ready"
        assert app.query_one(".reply", Static).content == expected
        await pilot.press("ctrl+c")

    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("explicit", [False, True])
async def test_selected_yaml_controls_actual_request_until_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool, protocol: str
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".replux" / "config.yaml"
    default.parent.mkdir()
    default.write_text("invalid: default-must-not-be-merged\n", encoding="utf-8")
    selected = tmp_path / "explicit.yaml" if explicit else default
    selected.write_text(
        f"protocol: {protocol}\n"
        "model: selected-full-model-id\n"
        "base_url: https://selected.example/custom-root\n"
        "api_key: ${LITERAL_SELECTED_KEY}\n"
        "max_output_tokens: 2048\n"
        "timeout_seconds: 7.5\n",
        encoding="utf-8",
    )
    transport = RecordingTransport(ControlledSSE(reply_events(protocol)))
    argv = ["--config", str(selected)] if explicit else []
    app = create_app(argv, transport=transport)
    selected.write_text("protocol: changed-after-startup\n", encoding="utf-8")

    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "one question"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert app.query_one(".reply", Static).content == "你好，世界\n"
        await pilot.press("ctrl+c")

    assert len(transport.requests) == 1
    request = transport.requests[0]
    endpoint = "v1/messages" if protocol == "anthropic" else "responses"
    assert str(request.url) == f"https://selected.example/custom-root/{endpoint}"
    if protocol == "anthropic":
        assert request.headers["x-api-key"] == "${LITERAL_SELECTED_KEY}"
    else:
        assert request.headers["authorization"] == "Bearer ${LITERAL_SELECTED_KEY}"
    assert json.loads(request.content)["model"] == "selected-full-model-id"
    limit_field = "max_tokens" if protocol == "anthropic" else "max_output_tokens"
    assert json.loads(request.content)[limit_field] == 2048
    assert request.extensions["timeout"] == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}
    assert transport.closed

    with pytest.raises(SystemExit):
        create_app(argv, transport=transport)


async def test_tui_accepts_a_scripted_llm_client_and_preserves_input() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.closed = False

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            yield TextDelta("[bold]这仍是正文[/bold]\n")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            self.closed = True

    client = ScriptedClient()
    app = RepluxApp(client)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "   "
        await pilot.press("enter")
        assert client.inputs == []
        prompt.value = "  [red]原样提问[/red]  "
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert client.inputs == ["  [red]原样提问[/red]  "]
        assert app.query_one(".reply", Static).content == "[bold]这仍是正文[/bold]\n"
        await pilot.press("ctrl+c")
    assert client.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("status_code", [401, 429, 400, 500])
async def test_request_error_preserves_transcript_and_draft_without_rendering_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
    protocol: str, status_code: int,
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    requests: list[httpx.Request] = []
    streams: list[ControlledSSE] = []
    failing = asyncio.Event()
    release_failure = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 2:
            failing.set()
            await release_failure.wait()
            return httpx.Response(status_code, headers={"retry-after": "60"}, json={"error": {
                "type": "invalid_request_error", "message": "context too long; test-secret-never-display",
            }})
        stream = ControlledSSE(reply_events(protocol))
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    app = create_app(["--config", str(config)], transport=httpx.MockTransport(respond))
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.value = "failed"
        await pilot.press("enter")
        await asyncio.wait_for(failing.wait(), timeout=2)
        prompt.value = "follow-up"
        release_failure.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert app.is_running
        assert "Request failed" in str(app.query_one(".turn-status", Static).content)
        assert app.query_one("#status", Static).content == "Ready"
        assert prompt.value == "follow-up"
        assert len(requests) == 2
        assert "test-secret-never-display" not in app.export_screenshot()
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert [user.content for user in app.query(".user").results(Static)] == ["first", "failed", "follow-up"]
        assert [reply.content for reply in app.query(".reply").results(Static)] == ["你好，世界\n", "", "你好，世界\n"]
        await pilot.press("ctrl+c")
    field = "messages" if protocol == "anthropic" else "input"
    assert len(requests) == 3
    assert json.loads(requests[-1].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    assert all(stream.closed for stream in streams)
    assert app.return_code == 0
    captured = capsys.readouterr()
    assert "test-secret-never-display" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("partial", [False, True], ids=["before-text", "partial-text"])
async def test_ctrl_c_waits_for_stream_cleanup_then_allows_follow_up(protocol: str, partial: bool) -> None:
    gate = (2 if protocol == "anthropic" else 4) if partial else 0
    pending = ControlledSSE(reply_events(protocol), gate_after=gate)
    pending.release_close.clear()
    streams = [ControlledSSE(reply_events(protocol)), pending, ControlledSSE(reply_events(protocol))]
    transport = RecordingTransport(streams)
    app = RepluxApp(create_client(provider_config(protocol), transport=transport))
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.value = "cancelled"
        await pilot.press("enter")
        await asyncio.wait_for(pending.blocked.wait(), timeout=2)
        worker = next(iter(app.workers))
        prompt.value = "follow-up"
        try:
            await pilot.press("ctrl+c")
            await asyncio.wait_for(pending.closing.wait(), timeout=2)
            assert app.is_running
            assert not pending.closed
            assert app.query_one("#status", Static).content == "Cancelling..."
            assert prompt.value == "follow-up"
            await pilot.press("enter", "ctrl+c")
            assert len(transport.requests) == 2
            assert not pending.closed
        finally:
            pending.release_close.set()
            with pytest.raises(WorkerCancelled):
                await asyncio.wait_for(worker.wait(), timeout=2)
        assert pending.closed
        assert app.is_running
        assert app.query_one("#status", Static).content == "Ready"
        assert "Cancelled" in str(app.query_one(".turn-status", Static).content)
        assert [reply.content for reply in app.query(".reply").results(Static)] == ["你好，世界\n", "你好" if partial else ""]
        assert prompt.value == "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert len(app.query(".reply")) == 3
        await pilot.press("ctrl+c")
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 3
    assert json.loads(transport.requests[-1].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    assert all(stream.closed for stream in streams)
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("cancel_first", [False, True], ids=["exit", "exit-during-cancellation"])
async def test_exit_waits_for_in_progress_stream_cleanup(
    tmp_path: Path, protocol: str, cancel_first: bool,
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    stream = ControlledSSE(reply_events(protocol), gate_after=0)
    stream.release_close.clear()
    transport = RecordingTransport(stream)
    app = create_app(["--config", str(config)], transport=transport)

    async def session() -> None:
        async with app.run_test() as pilot:
            app.query_one("#prompt", Input).value = "hello"
            await pilot.press("enter")
            await asyncio.wait_for(stream.blocked.wait(), timeout=2)
            if cancel_first:
                await pilot.press("ctrl+c")
                await asyncio.wait_for(stream.closing.wait(), timeout=2)
            app.exit()

    exiting = asyncio.create_task(session())
    try:
        await asyncio.wait_for(stream.closing.wait(), timeout=2)
        assert not stream.closed
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(exiting), timeout=0.1)
    finally:
        stream.release_close.set()
        await asyncio.wait_for(exiting, timeout=2)
    assert stream.closed
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_programming_error_terminates_chat_and_releases_resources(protocol: str) -> None:
    events = reply_events(protocol)
    stream = ControlledSSE(events[:3 if protocol == "anthropic" else 5], read_error=RuntimeError("injected defect"))
    transport = RecordingTransport(stream)
    app = RepluxApp(create_client(provider_config(protocol), transport=transport))
    with pytest.raises(WorkerFailed, match="injected defect"):
        async with app.run_test() as pilot:
            app.query_one("#prompt", Input).value = "hello"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert app.return_code == 1
    assert stream.closed
    assert transport.closed
    assert len(transport.requests) == 1


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_immediate_exit_does_not_leave_an_unobserved_reply_task(protocol: str) -> None:
    # A fresh loop and process expose unawaited coroutines and unobserved task
    # exceptions even when Textual has already removed a cancelled worker.
    result = subprocess.run(
        [sys.executable, "-c", """
import asyncio
import gc
import sys
from textual.widgets import Input
from replux.llm import create_client
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events

class ExitOnSubmitApp(RepluxApp):
    def on_input_submitted(self, event):
        super().on_input_submitted(event)
        self.exit()

async def session():
    protocol = sys.argv[1]
    stream = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport(stream)
    app = ExitOnSubmitApp(create_client(provider_config(protocol), transport=transport))
    async with app.run_test() as pilot:
        app.query_one('#prompt', Input).value = 'hello'
        await pilot.press('enter')
    print(app.return_code, transport.closed, len(transport.requests))

asyncio.run(session())
gc.collect()
""", protocol],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 True 0"
    assert result.stderr == ""


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["eof", "content_end", "output_limit", "incomplete"])
async def test_incomplete_turn_stays_visible_and_out_of_next_request(
    tmp_path: Path, protocol: str, ending: str,
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    incomplete = ControlledSSE(unfinished_events(protocol, ending))
    transport = RecordingTransport([
        ControlledSSE(reply_events(protocol)), incomplete, ControlledSSE(reply_events(protocol)),
    ])
    app = create_app(["--config", str(config)], transport=transport)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        for user_input in ["first", "unfinished"]:
            prompt.value = user_input
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert incomplete.closed
        assert [reply.content for reply in app.query(".reply").results(Static)] == ["你好，世界\n", "你好，世界\n"]
        assert "Reply incomplete" in str(app.query_one(".turn-status", Static).content)
        assert app.query_one("#status", Static).content == "Ready"
        prompt.value = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert len(app.query(".reply")) == 3
        assert "Reply incomplete" in str(app.query_one(".turn-status", Static).content)
        assert [user.content for user in app.query(".user").results(Static)] == ["first", "unfinished", "follow-up"]
    field = "messages" if protocol == "anthropic" else "input"
    assert len(transport.requests) == 3
    assert json.loads(transport.requests[2].content)[field] == [
        {"role": "user", "content": "first"}, *native_reply(protocol),
        {"role": "user", "content": "follow-up"},
    ]
    assert transport.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_recreated_app_starts_with_empty_conversation(tmp_path: Path, protocol: str) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    field = "messages" if protocol == "anthropic" else "input"
    for questions in [["remember this", "follow-up"], ["new conversation"]]:
        streams = [ControlledSSE(reply_events(protocol)) for _ in questions]
        transport = RecordingTransport(streams)
        app = create_app(["--config", str(config)], transport=transport)
        async with app.run_test() as pilot:
            assert not app.query(".user, .reply")
            expected = []
            for question in questions:
                app.query_one("#prompt", Input).value = question
                await pilot.press("enter")
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
                expected.append({"role": "user", "content": question})
                assert json.loads(transport.requests[-1].content)[field] == expected
                expected.extend(native_reply(protocol))
            assert [reply.content for reply in app.query(".reply").results(Static)] == ["你好，世界\n"] * len(questions)
            await pilot.press("ctrl+c")
        assert len(transport.requests) == len(questions)
        assert all(stream.closed for stream in streams)
        assert transport.closed
