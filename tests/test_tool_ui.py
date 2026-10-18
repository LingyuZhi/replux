import asyncio
from collections.abc import AsyncGenerator
from html import unescape
import json
from pathlib import Path

import pytest
import httpx2 as httpx
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled, WorkerFailed

from replux.cli import create_app
from replux.llm import LLMClient, ReplyCompleted, StreamEvent, TextDelta, ToolFinished, ToolStarted
from replux.tools import ToolRegistry, ToolResult
from replux.tui import RepluxApp, ToolCallView
from tests.support import ControlledSSE, RecordingTransport, function_call_events, provider_config, reply_events, tool_use_events
from tests.test_config import VALID_YAML
from tests.test_tool_batch import (
    BoomRead, GatedRead, batch_events, open_client, read_call, result_pairs, text_gate,
)


EIGHT_LINES = "\n".join(f"{number}\tline {number}" for number in range(1, 9))


def visible_text(app: RepluxApp) -> str:
    return unescape(app.export_screenshot()).replace("\xa0", " ")


async def test_running_tool_stays_static_while_bottom_activity_animates() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.reached = [asyncio.Event(), asyncio.Event()]
            self.release = [asyncio.Event(), asyncio.Event()]

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            yield TextDelta("我先读文件。")
            yield ToolStarted("call_1", "read_file", "path=notes.txt", '{"path": "notes.txt"}')
            self.reached[0].set()
            await self.release[0].wait()
            result = ToolResult("success", EIGHT_LINES, details={"start_line": 1, "end_line": 8})
            yield ToolFinished("call_1", "read_file", "path=notes.txt", result, 0.25, '{"path": "notes.txt"}')
            self.reached[1].set()
            await self.release[1].wait()
            yield TextDelta("文件有八行。")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 40)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "读一下 notes.txt"
        await pilot.press("enter")
        await asyncio.wait_for(client.reached[0].wait(), timeout=2)
        await pilot.pause()
        view = app.query_one(ToolCallView)
        dot = view.query_one(".tool-dot", Static)
        assert dot.has_class("running")
        assert view.query_one(".tool-header", Static).content == "Read notes.txt"
        assert view.query_one(".tool-status", Static).content == "Running"
        assert app.query_one("#status", Static).content == "Read notes.txt…"
        activity = app.query_one("#activity-dot", Static)
        first_frame = activity.content
        await pilot.pause(0.6)
        assert activity.content != first_frame
        assert dot.styles.text_opacity == 1.0
        await pilot.press(*"草稿", "enter")
        assert client.inputs == ["读一下 notes.txt"]
        assert prompt.text == "草稿"

        client.release[0].set()
        await asyncio.wait_for(client.reached[1].wait(), timeout=2)
        await pilot.pause()
        assert not dot.has_class("running")
        assert dot.styles.text_opacity == 1.0
        assert view.query_one(".tool-status", Static).content == "Success"
        preview = view.query_one(".tool-preview", Static)
        assert preview.content == "└ Lines 1–8"
        output = view.query_one(".tool-details")
        assert not output.display
        assert view.query_one(".tool-arrow", Static).content == "▸"
        assert "line 8" not in visible_text(app)
        assert app.query_one("#status", Static).content == "Working…"

        await pilot.click(view.query_one(".tool-heading"))
        assert output.display
        assert output.query_one(".tool-output", Static).content == EIGHT_LINES
        assert not preview.display
        assert "line 8" in visible_text(app)

        client.release[1].set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        last_frame = activity.content
        await pilot.pause(0.15)
        assert activity.content == last_frame
        assert not activity.display
        user = app.query_one(".user", Static)
        before, after = app.query(".reply").results(Static)
        assert user.content == "读一下 notes.txt"
        assert before.content == "我先读文件。"
        assert after.content == "文件有八行。"
        assert user.region.bottom < before.region.y < view.region.y < after.region.y
        assert prompt.text == "草稿"


async def test_failed_and_skipped_calls_keep_their_states() -> None:
    class ScriptedClient(LLMClient):
        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield ToolStarted("call_1", "read_file", "path=absent.txt", '{"path": "absent.txt"}')
            missing = ToolResult("error", "File not found: absent.txt.", "file_not_found")
            yield ToolFinished("call_1", "read_file", "path=absent.txt", missing, 0.012, '{"path": "absent.txt"}')
            skipped = ToolResult("skipped", "Not run because an earlier tool call in this batch did not succeed.")
            yield ToolFinished("call_2", "read_file", "path=b.txt", skipped, None, '{"path": "b.txt"}')
            yield TextDelta("没有找到文件。")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    app = RepluxApp(ScriptedClient(), model="scripted-model")
    async with app.run_test(size=(80, 30)) as pilot:
        app.query_one("#prompt", TextArea).text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        failed, skipped = app.query(ToolCallView).results(ToolCallView)
        assert failed.query_one(".tool-status", Static).content == "Failed"
        assert failed.query_one(".tool-preview", Static).content == "└ File not found: absent.txt."
        assert failed.query_one(".tool-dot", Static).has_class("error")
        assert skipped.query_one(".tool-header", Static).content == "Read b.txt"
        assert skipped.query_one(".tool-status", Static).content == "Skipped"
        assert not skipped.query_one(".tool-dot", Static).has_class("running")
        assert not failed.query_one(".tool-details").display
        assert not skipped.query_one(".tool-details").display
        assert app.query_one("#status", Static).content == "Ready"


async def test_cancelling_a_running_tool_keeps_its_status_and_ends_bottom_activity() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield ToolStarted("call_1", "read_file", "path=notes.txt", '{"path": "notes.txt"}')
            self.started.set()
            await asyncio.Event().wait()
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 30)) as pilot:
        app.query_one("#prompt", TextArea).text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(client.started.wait(), timeout=2)
        await pilot.pause()
        worker = next(iter(app.workers))
        await pilot.press("ctrl+c")
        with pytest.raises(WorkerCancelled):
            await asyncio.wait_for(worker.wait(), timeout=2)
        await pilot.pause()
        view = app.query_one(ToolCallView)
        assert view.query_one(".tool-status", Static).content == "Cancelled"
        dot = view.query_one(".tool-dot", Static)
        assert not dot.has_class("running")
        assert dot.styles.text_opacity == 1.0
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Cancelled after ")
        assert app.query_one("#status", Static).content == "Ready"


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_assembled_app_reads_a_file_and_stays_busy_until_the_summary_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("".join(f"line {number}\n" for number in range(1, 9)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    tool_events = tool_use_events() if protocol == "anthropic" else function_call_events()
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([ControlledSSE(tool_events), summary])
    app = create_app(["--config", str(config)], transport=transport)
    async with app.run_test(size=(80, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "读一下 notes.txt"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        view = app.query_one(ToolCallView)
        assert view.query_one(".tool-header", Static).content == "Read notes.txt"
        assert view.query_one(".tool-status", Static).content == "Success"
        assert view.query_one(".tool-preview", Static).content == "└ Lines 1–8"
        assert app.query_one("#status", Static).content == "Working…"
        assert not app.query(".turn-outcome")
        output = view.query_one(".tool-details")
        await pilot.click(view.query_one(".tool-heading"))
        assert output.query_one(".tool-output", Static).content == EIGHT_LINES
        assert len(transport.requests) == 2
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        assert [reply.content for reply in app.query(".reply").results(Static)] == ["我先读文件。", "你好，世界\n"]
    assert len(transport.requests) == 2
    assert transport.closed
    first, summary_request = (json.loads(request.content) for request in transport.requests)
    endpoint = "v1/messages" if protocol == "anthropic" else "responses"
    assert str(transport.requests[0].url).endswith("/" + endpoint)
    assert str(transport.requests[1].url).endswith("/" + endpoint)
    assert first["tools"][0]["name"] == "read_file"
    if protocol == "anthropic":
        assert first["tools"][0]["input_schema"]["properties"]["path"]["minLength"] == 1
        assert "tool_choice" not in first
        assert summary_request["tool_choice"] == {"type": "none"}
    else:
        assert first["tools"][0]["type"] == "function"
        assert first["tools"][0]["parameters"]["properties"]["path"]["minLength"] == 1
        assert first["tools"][0]["strict"] is False
        assert "tool_choice" not in first
        assert summary_request["tool_choice"] == "none"
        assert summary_request["store"] is False


def provider_file(tmp_path: Path, protocol: str) -> Path:
    path = tmp_path / "provider.yaml"
    path.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    return path


def statuses(app: RepluxApp) -> list[str]:
    return [str(view.query_one(".tool-status", Static).content) for view in app.query(ToolCallView).results(ToolCallView)]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_a_failed_batch_shows_each_result_and_finishes_after_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = (read_call("call_ok", "ok.txt"), read_call("call_missing", "missing.txt"), read_call("call_later", "later.txt"))
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([ControlledSSE(batch_events(protocol, calls)), summary])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "read three"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        assert statuses(app)[0].startswith("Success")
        assert statuses(app)[1].startswith("Failed")
        assert statuses(app)[2] == "Skipped"
        views = list(app.query(ToolCallView).results(ToolCallView))
        assert views[1].query_one(".tool-dot", Static).has_class("error")
        assert not views[2].query_one(".tool-dot", Static).has_class("running")
        assert "should-not" not in visible_text(app)
        assert app.query_one("#status", Static).content == "Working…"
        assert not app.query(".turn-outcome")
        if protocol == "anthropic":
            assert "先看文件" in str(app.query_one(".thinking-content", Static).content)
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        assert str(views[1].query_one(".tool-status", Static).content).startswith("Failed")
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Finished in ")
        assert [reply.content for reply in app.query(".reply").results(Static)][-1] == "你好，世界\n"
    assert "should-not" not in transport.requests[1].content.decode()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_unpairable_tool_calls_report_an_error_and_accept_another_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = (read_call("call_ok", "ok.txt"), read_call("call_ok", "ok.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 30)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert not app.query(ToolCallView)
        assert "missing or duplicate" in str(app.query_one(".turn-status", Static).content)
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Failed after ")
        assert app.query_one("#status", Static).content == "Ready"
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cleanup_failure_shows_an_error_and_accepts_another_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    broken = ControlledSSE(
        batch_events(protocol, (read_call("call_ok", "ok.txt"),)), close_error=httpx.ReadError("offline cleanup failure"),
    )
    transport = RecordingTransport([broken, ControlledSSE(reply_events(protocol))])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 30)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert not app.query(ToolCallView)
        assert "Request failed" in str(app.query_one(".turn-status", Static).content)
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Failed after ")
        assert app.query_one("#status", Static).content == "Ready"
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert broken.closed
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_a_second_tool_batch_stops_the_turn_and_keeps_the_finished_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (read_call("call_ok", "ok.txt"),))),
        ControlledSSE(batch_events(protocol, (read_call("call_again", "later.txt"),))),
        ControlledSSE(reply_events(protocol)),
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert len(statuses(app)) == 1
        assert statuses(app)[0].startswith("Success")
        assert "tool batch" in str(app.query_one(".turn-status", Static).content)
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Failed after ")
        assert "should-not" not in visible_text(app)
        assert app.query_one("#status", Static).content == "Ready"
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    body = transport.requests[2].content.decode()
    assert "call_ok" in body
    assert "call_again" not in body
    assert "should-not" not in body


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancelling_the_summary_keeps_tool_results_and_restores_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    summary = ControlledSSE(reply_events(protocol), gate_after=text_gate(protocol))
    summary.release_close.clear()
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (read_call("call_ok", "ok.txt"),))), summary,
        ControlledSSE(reply_events(protocol)),
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        assert statuses(app)[0].startswith("Success")
        assert app.query_one("#status", Static).content == "Writing reply…"
        worker = next(iter(app.workers))
        try:
            await pilot.press("ctrl+c")
            await asyncio.wait_for(summary.closing.wait(), timeout=2)
            assert app.query_one("#status", Static).content == "Cancelling…"
            assert not app.query(".turn-outcome")
            assert statuses(app)[0].startswith("Success")
            prompt.text = "草稿"
            await pilot.press("enter", "ctrl+c")
            assert prompt.text == "草稿"
            assert len(transport.requests) == 2
            assert not summary.closed
        finally:
            summary.release_close.set()
            with pytest.raises(WorkerCancelled):
                await asyncio.wait_for(worker.wait(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Cancelled after ")
        assert statuses(app)[0].startswith("Success")
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    history = json.dumps(result_pairs(protocol, transport.requests[2]), ensure_ascii=False)
    assert "call_ok" in transport.requests[2].content.decode()
    assert '"status": "success"' in history
    assert "你好" not in transport.requests[2].content.decode()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_tool_cleanup_blocks_submit_then_shows_cancelled_and_skipped(
    tmp_path: Path, protocol: str,
) -> None:
    side_effect = tmp_path / "ran.txt"
    tool = GatedRead(side_effect)
    registry = ToolRegistry()
    registry.register(tool)
    calls = (read_call("call_run", "ok.txt"), read_call("call_next", "later.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "read them"
        await pilot.press("enter")
        await asyncio.wait_for(tool.entered.wait(), timeout=2)
        await pilot.pause()
        assert statuses(app) == ["Running"]
        assert app.query_one("#status", Static).content == "Read ok.txt…"
        if protocol == "anthropic":
            assert "先看文件" in str(app.query_one(".thinking-content", Static).content)
        prompt.text = "草稿"
        await pilot.press("enter")
        assert prompt.text == "草稿"
        assert len(transport.requests) == 1
        worker = next(iter(app.workers))
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Cancelling…"
        assert not app.query(".turn-outcome")
        assert app.query_one(".tool-dot", Static).has_class("running")
        prompt.text = "草稿仍可编辑"
        await pilot.press("enter", "ctrl+c")
        assert app.query_one("#status", Static).content == "Cancelling…"
        assert not app.query(".turn-outcome")
        assert not tool.stopped.is_set()
        assert not side_effect.exists()
        tool.release.set()
        with pytest.raises(WorkerCancelled):
            await asyncio.wait_for(worker.wait(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        assert [status.split(" · ", 1)[0] for status in statuses(app)] == ["Cancelled", "Skipped"]
        assert str(app.query_one(".turn-outcome", Static).content).startswith("Cancelled after ")
        assert prompt.text == "草稿仍可编辑"
        if protocol == "anthropic":
            assert "先看文件" in str(app.query_one(".thinking-content", Static).content)
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert tool.runs == 1
    assert not side_effect.exists()
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == ["cancelled", "skipped"]
    assert "你好" not in transport.requests[1].content.decode()
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancel_while_showing_a_tool_keeps_unstarted_calls_skipped(
    tmp_path: Path, protocol: str,
) -> None:
    tool = GatedRead(tmp_path / "ran.txt")
    registry = ToolRegistry()
    registry.register(tool)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (read_call("call_run", "ok.txt"), read_call("call_next", "later.txt")))),
        ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one("#transcript-content")
        mount = transcript.mount
        seen = False

        async def cancel_first_tool(*children: object, **kwargs: object) -> None:
            nonlocal seen
            await mount(*children, **kwargs)  # type: ignore[arg-type]
            if seen or not any(isinstance(child, ToolCallView) for child in children):
                return
            seen = True
            raise asyncio.CancelledError

        transcript.mount = cancel_first_tool  # type: ignore[assignment]
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "read them"
        await pilot.press("enter")
        await pilot.pause()
        assert [status.split(" · ", 1)[0] for status in statuses(app)] == ["Skipped", "Skipped"]
        assert app.query_one("#status", Static).content == "Ready"
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert tool.runs == 0
    assert not (tmp_path / "ran.txt").exists()
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == ["skipped", "skipped"]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_an_internal_tool_error_terminates_the_chat_after_cleanup(protocol: str) -> None:
    tool = BoomRead()
    registry = ToolRegistry()
    registry.register(tool)
    stream = ControlledSSE(batch_events(protocol, (read_call("call_boom", "ok.txt"), read_call("call_later", "later.txt"))))
    transport = RecordingTransport([stream])
    app = RepluxApp(
        open_client(protocol, transport, registry), model=provider_config(protocol).model,
    )
    with pytest.raises(WorkerFailed, match="injected defect"):
        async with app.run_test() as pilot:
            app.query_one("#prompt", TextArea).text = "read"
            await pilot.press("enter")
            await asyncio.wait_for(tool.entered.wait(), timeout=2)
            await pilot.pause()
            assert app.query_one("#status", Static).content == "Read ok.txt…"
            tool.release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert app.return_code == 1
    assert tool.runs == 1
    assert stream.closed
    assert transport.closed
    assert len(transport.requests) == 1
