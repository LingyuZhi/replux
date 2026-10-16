import asyncio
from collections.abc import AsyncGenerator
from html import unescape
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Collapsible, Static, TextArea
from textual.worker import WorkerCancelled

from replux.cli import create_app
from replux.llm import LLMClient, ReplyCompleted, StreamEvent, TextDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp, ToolCallView
from tests.support import ControlledSSE, RecordingTransport, reply_events, tool_use_events
from tests.test_config import VALID_YAML


EIGHT_LINES = "\n".join(f"{number}\tline {number}" for number in range(1, 9))


def visible_text(app: RepluxApp) -> str:
    return unescape(app.export_screenshot()).replace("\xa0", " ")


async def wait_for_opacity_change(pilot: Pilot[None], dot: Static) -> None:
    initial = dot.styles.text_opacity
    for _ in range(40):
        await pilot.pause(0.05)
        if dot.styles.text_opacity != initial:
            return
    raise AssertionError("the running dot did not breathe")


async def test_running_tool_breathes_then_shows_status_preview_and_expandable_result() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.reached = [asyncio.Event(), asyncio.Event()]
            self.release = [asyncio.Event(), asyncio.Event()]

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            yield TextDelta("我先读文件。")
            yield ToolStarted("call_1", "read_file", "path=notes.txt")
            self.reached[0].set()
            await self.release[0].wait()
            yield ToolFinished("call_1", "read_file", "path=notes.txt", ToolResult("success", EIGHT_LINES), 0.25)
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
        assert view.query_one(".tool-header", Static).content == "read_file path=notes.txt"
        assert view.query_one(".tool-status", Static).content == "running"
        assert app.query_one("#status", Static).content == "Running read_file..."
        await wait_for_opacity_change(pilot, dot)
        await pilot.press(*"草稿", "enter")
        assert client.inputs == ["读一下 notes.txt"]
        assert prompt.text == "草稿"

        client.release[0].set()
        await asyncio.wait_for(client.reached[1].wait(), timeout=2)
        await pilot.pause()
        assert not dot.has_class("running")
        assert dot.styles.text_opacity == 1.0
        assert view.query_one(".tool-status", Static).content == "success · 0.25s"
        preview = view.query_one(".tool-preview", Static)
        assert preview.content == "\n".join(EIGHT_LINES.split("\n")[:5])
        output = view.query_one(Collapsible)
        assert output.collapsed
        assert str(output.query_one("CollapsibleTitle", Static).content) == "+ Show all 8 lines"
        assert "line 8" not in visible_text(app)
        assert app.query_one("#status", Static).content == "Generating..."
        await pilot.pause(0.6)
        assert dot.styles.text_opacity == 1.0

        await pilot.click(output.query_one("CollapsibleTitle", Static), offset=(1, 0))
        assert not output.collapsed
        assert output.query_one(".tool-output", Static).content == EIGHT_LINES
        assert not preview.display
        assert "line 8" in visible_text(app)

        client.release[1].set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        order = [
            child.content if isinstance(child, Static) else type(child).__name__
            for child in app.query_one("#transcript-content").children
        ][1:]
        assert order == ["读一下 notes.txt", "我先读文件。", "ToolCallView", "文件有八行。"]
        assert prompt.text == "草稿"


async def test_failed_and_skipped_calls_keep_their_states() -> None:
    class ScriptedClient(LLMClient):
        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield ToolStarted("call_1", "read_file", "path=absent.txt")
            missing = ToolResult("error", "File not found: absent.txt.", "file_not_found")
            yield ToolFinished("call_1", "read_file", "path=absent.txt", missing, 0.012)
            skipped = ToolResult("skipped", "Not run because an earlier tool call in this batch did not succeed.")
            yield ToolFinished("call_2", "read_file", "path=b.txt", skipped, None)
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
        assert failed.query_one(".tool-status", Static).content == "error (file_not_found) · 0.01s"
        assert failed.query_one(".tool-preview", Static).content == "File not found: absent.txt."
        assert failed.query_one(".tool-dot", Static).has_class("error")
        assert skipped.query_one(".tool-header", Static).content == "read_file path=b.txt"
        assert skipped.query_one(".tool-status", Static).content == "skipped"
        assert not skipped.query_one(".tool-dot", Static).has_class("running")
        assert not app.query(".tool-call Collapsible")
        assert app.query_one("#status", Static).content == "Ready"


async def test_cancelling_a_running_tool_stops_its_animation() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield ToolStarted("call_1", "read_file", "path=notes.txt")
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
        assert view.query_one(".tool-status", Static).content == "cancelled"
        dot = view.query_one(".tool-dot", Static)
        assert not dot.has_class("running")
        assert dot.styles.text_opacity == 1.0
        assert app.query_one(".turn-status", Static).content == "Cancelled"
        assert app.query_one("#status", Static).content == "Ready"


async def test_assembled_app_reads_a_file_and_stays_busy_until_the_summary_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("".join(f"line {number}\n" for number in range(1, 9)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    summary = ControlledSSE(reply_events(), gate_after=0)
    transport = RecordingTransport([ControlledSSE(tool_use_events()), summary])
    app = create_app(["--config", str(config)], transport=transport)
    async with app.run_test(size=(80, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "读一下 notes.txt"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        view = app.query_one(ToolCallView)
        assert view.query_one(".tool-header", Static).content == "read_file path=notes.txt"
        assert str(view.query_one(".tool-status", Static).content).startswith("success · ")
        assert view.query_one(".tool-preview", Static).content == "\n".join(f"{n}\tline {n}" for n in range(1, 6))
        assert app.query_one("#status", Static).content == "Generating..."
        output = view.query_one(Collapsible)
        await pilot.click(output.query_one("CollapsibleTitle", Static), offset=(1, 0))
        assert output.query_one(".tool-output", Static).content == EIGHT_LINES
        assert len(transport.requests) == 2
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
        assert [reply.content for reply in app.query(".reply").results(Static)] == ["我先读文件。", "你好，世界\n"]
    assert len(transport.requests) == 2
    assert transport.closed
