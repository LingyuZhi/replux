import asyncio
from collections.abc import AsyncGenerator
import json
import re

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled

from replux.llm import LLMClient, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp
from tests.test_reply_markdown import GatedClient
from tests.tui_support import screen_text, text_position


@pytest.mark.parametrize(("name", "arguments", "title", "target"), [
    ("read_file", {"path": "notes.txt"}, "Read notes.txt", "notes.txt"),
    ("write_file", {"path": "new.txt", "content": "new bytes"}, "Write new.txt", "new.txt"),
    ("edit_file", {"path": "[red]literal[/red].txt", "edits": []}, "Edit [red]literal[/red].txt", "[red]literal[/red].txt"),
    ("find_files", {"pattern": "*.py", "path": "src"}, "Find *.py in src", "*.py"),
    ("search_content", {"pattern": "TODO", "path": "src"}, "Search TODO in src", "TODO"),
    ("run_command", {"command": "printf done", "cwd": "src"}, "Run printf done in src", "printf done"),
])
async def test_tool_actions_use_soft_orange_and_keep_targets_and_summaries_plain(
    monkeypatch: pytest.MonkeyPatch, name: str, arguments: dict[str, object], title: str, target: str,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    raw = json.dumps(arguments)
    client = GatedClient([
        ToolStarted("call", name, "decoy", raw),
        ToolFinished("call", name, "decoy", ToolResult("success", "Operation complete."), 0.1, raw),
    ])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            app.query_one("#prompt", TextArea).text = "use tool"
            await pilot.press("enter")
            for stage in range(2):
                await asyncio.wait_for(client.reached[stage].wait(), timeout=2)
                await pilot.pause()
                x, y = text_position(app, title)
                action = app.screen.get_style_at(x, y).color
                assert action is not None and action.triplet is not None
                red, green, blue = action.triplet
                assert red > green > blue and red - blue >= 60
                target_color = app.screen.get_style_at(*text_position(app, target)).color
                assert target_color != action
                assert target_color is not None and target_color.triplet is not None
                assert max(target_color.triplet) - min(target_color.triplet) <= 10
                dot = app.screen.get_style_at(x - 2, y).color
                assert dot is not None and dot.triplet is not None
                assert dot.triplet.red > dot.triplet.green > dot.triplet.blue
                if stage == 1:
                    assert dot == action
                    assert "Success" in screen_text(app)
                    summary = app.screen.get_style_at(*text_position(app, "Operation complete.")).color
                    assert summary != action
                client.release[stage].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        finally:
            for release in client.release:
                release.set()


async def test_cancelling_keeps_both_animations_until_each_cleanup_boundary_and_preserves_the_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    now = 300.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)

    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.running = asyncio.Event()
            self.closing = asyncio.Event()
            self.finish_tool = asyncio.Event()
            self.results_shown = asyncio.Event()
            self.finish_stream = asyncio.Event()
            self.closed = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            if len(self.inputs) > 1:
                yield TextDelta("Next turn completed.")
                yield ReplyCompleted("end_turn", None, None)
                return
            try:
                yield ToolStarted("run", "run_command", "decoy", '{"command": "sleep 30"}')
                self.running.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.closing.set()
                await self.finish_tool.wait()
                yield ToolFinished("run", "run_command", "decoy", ToolResult("cancelled", "Process stopped."), 2.0, '{"command": "sleep 30"}')
                yield ToolFinished("next", "read_file", "decoy", ToolResult("skipped", "Not run."), None, '{"path": "later.txt"}')
                self.results_shown.set()
                await self.finish_stream.wait()
                raise
            finally:
                self.closed.set()

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "run"
            await pilot.press("enter")
            await asyncio.wait_for(client.running.wait(), timeout=2)
            worker = next(iter(app.workers))
            await pilot.press("ctrl+c")
            await asyncio.wait_for(client.closing.wait(), timeout=2)
            await pilot.press(*"draft", "ctrl+j", *"second line", "ctrl+c", "enter")
            frames = []
            bottom_frames = []
            for elapsed in (0.5, 0.9, 1.3):
                now = 300.0 + elapsed
                await pilot.pause(0.12)
                x, y = text_position(app, "Run sleep 30")
                frames.append(app.screen.get_style_at(x - 2, y).color)
                bottom_frames.append(app.query_one("#activity-dot", Static).content)
                assert "Running" in screen_text(app)
                assert app.query_one("#status", Static).content == "Cancelling…"
                assert app.focused is prompt and prompt.text == "draft\nsecond line"
                assert not app.query(".turn-outcome")
            assert len(set(frames)) >= 2 and len(set(bottom_frames)) >= 2
            now = 302.0
            client.finish_tool.set()
            await asyncio.wait_for(client.results_shown.wait(), timeout=2)
            terminal_frames = []
            for elapsed in (2.0, 2.4, 2.8):
                now = 300.0 + elapsed
                await pilot.pause(0.12)
                colors = []
                for title in ("Run sleep 30", "Read later.txt"):
                    x, y = text_position(app, title)
                    colors.append(app.screen.get_style_at(x - 2, y).color)
                terminal_frames.append(tuple(colors))
                assert [status.content for status in app.query(".tool-status").results(Static)] == ["Cancelled", "Skipped"]
                assert app.query_one("#status", Static).content == "Cancelling…"
                assert not app.query(".turn-outcome")
            assert len(set(terminal_frames)) == 1
            await pilot.press("enter", "ctrl+c", "ctrl+c")
            assert client.inputs == ["run"] and not client.closed.is_set()
            now = 304.0
            client.finish_stream.set()
            with pytest.raises(WorkerCancelled):
                await asyncio.wait_for(worker.wait(), timeout=2)
            await pilot.pause()
            assert client.closed.is_set()
            assert app.query_one(".turn-outcome", Static).content == "Cancelled after 4.0s"
            assert not app.query_one("#activity-dot").display
            assert prompt.text == "draft\nsecond line"
            now = 310.0
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert client.inputs == ["run", "draft\nsecond line"]
            assert [outcome.content for outcome in app.query(".turn-outcome").results(Static)] == [
                "Cancelled after 4.0s", "Finished in 0.0s",
            ]
            for title, color in zip(("Run sleep 30", "Read later.txt"), terminal_frames[0]):
                x, y = text_position(app, title)
                assert app.screen.get_style_at(x - 2, y).color == color
            assert app.focused is prompt and prompt.text == ""
        finally:
            client.finish_tool.set()
            client.finish_stream.set()


async def test_activity_keeps_history_in_place_when_the_running_tool_is_offscreen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 400.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)
    raw = '{"path": "notes.txt"}'
    client = GatedClient([
        TextDelta("\n".join(f"History line {i:03d}  " for i in range(80))),
        ToolStarted("call", "read_file", "decoy", raw),
        ToolFinished("call", "read_file", "decoy", ToolResult("success", "Done."), 0.1, raw),
    ])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "read history"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            transcript = app.query_one("#transcript", VerticalScroll)
            transcript.focus()
            await pilot.press("pageup", "pageup")
            await pilot.wait_for_scheduled_animations()
            prompt.focus()
            prompt.text = "Draft line\n" * 10
            await pilot.pause()
            landmark = re.findall(r"History line \d{3}", screen_text(app))[0]
            position = text_position(app, landmark)
            frames = []
            for elapsed in (0.3, 0.6, 0.9):
                now = 400.0 + elapsed
                await pilot.pause(0.12)
                frames.append(app.query_one("#activity-dot", Static).content)
                assert "Read notes.txt" not in screen_text(app)
                assert "Reading… notes.txt" in screen_text(app)
                assert text_position(app, landmark) == position
                assert app.focused is prompt and prompt.text == "Draft line\n" * 10
            assert len(set(frames)) >= 2
            client.release[1].set()
            await asyncio.wait_for(client.reached[2].wait(), timeout=2)
            await pilot.pause()
            assert text_position(app, landmark) == position
            transcript.focus()
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            prompt.focus()
            assert "Read notes.txt" in screen_text(app) and "Success" in screen_text(app)
            client.release[2].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            assert app.focused is prompt and prompt.text == "Draft line\n" * 10
        finally:
            for release in client.release:
                release.set()


async def test_fast_consecutive_tools_and_blank_deltas_do_not_invent_thinking_or_delay_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 500.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)
    events: list[StreamEvent] = [ThinkingDelta(" \n"), TextDelta("\n ")]
    for index, name in enumerate(("read_file", "write_file", "edit_file", "find_files", "search_content", "run_command")):
        raw = json.dumps({"path": f"file-{index}.txt", "pattern": "TODO", "command": "printf done"})
        events.append(ToolStarted(str(index), name, "decoy", raw))
        events.append(ToolFinished(str(index), name, "decoy", ToolResult("success", "Done."), 0.001, raw))
    client = GatedClient(events)
    for release in client.release[2:]:
        release.set()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(160, 50)) as pilot:
        try:
            app.query_one("#prompt", TextArea).text = "quick tools"
            await pilot.press("enter")
            for index in range(2):
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                await pilot.pause(0.12)
                assert app.query_one("#status", Static).content == "Working…"
                assert not app.query(".thinking, .reply")
                client.release[index].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert [status.content for status in app.query(".tool-status").results(Static)] == ["Success"] * 6
            assert app.query_one(".turn-outcome", Static).content == "Finished in 0.0s"
            dots = [dot.styles.text_opacity for dot in app.query(".tool-dot").results(Static)]
            assert dots == [1.0] * 6
            now = 504.625
            await pilot.pause(0.15)
            assert [dot.styles.text_opacity for dot in app.query(".tool-dot").results(Static)] == dots
            assert app.query_one(".turn-outcome", Static).content == "Finished in 0.0s"
            assert app.query_one("#status", Static).content == "Ready"
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize(("result", "label"), [
    (ToolResult("success", "Read complete."), "Success"),
    (ToolResult("error", "Permission denied.", "permission_denied"), "Failed"),
    (ToolResult("cancelled", "Stopped by the user."), "Cancelled"),
    (ToolResult("skipped", "Not run after an earlier failure."), "Skipped"),
])
async def test_terminal_tools_stay_static_and_keep_their_status_after_the_turn_finishes(
    monkeypatch: pytest.MonkeyPatch, result: ToolResult, label: str,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    now = 200.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)
    raw = '{"path": "terminal.txt"}'
    events: list[StreamEvent] = []
    if result.status != "skipped":
        events.append(ToolStarted("call", "read_file", "decoy", raw))
    events.append(ToolFinished("call", "read_file", "decoy", result, None, raw))
    client = GatedClient(events)
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            app.query_one("#prompt", TextArea).text = "read"
            await pilot.press("enter")
            if result.status != "skipped":
                await asyncio.wait_for(client.reached[0].wait(), timeout=2)
                now = 200.75
                await pilot.pause(0.12)
                client.release[0].set()
            await asyncio.wait_for(client.reached[-1].wait(), timeout=2)
            frames = []
            for elapsed in (1.0, 1.4, 1.8):
                now = 200.0 + elapsed
                await pilot.pause(0.12)
                x, y = text_position(app, "Read terminal.txt")
                action = app.screen.get_style_at(x, y).color
                dot = app.screen.get_style_at(x - 2, y).color
                assert dot == action
                frames.append(dot)
                assert label in screen_text(app)
                assert result.output in screen_text(app)
                if result.status == "error":
                    status_color = app.screen.get_style_at(*text_position(app, "Failed")).color
                    reason_color = app.screen.get_style_at(*text_position(app, "Permission denied.")).color
                    assert action == status_color == reason_color
                    assert action is not None and action.triplet is not None
                    assert action.triplet.red > action.triplet.green
                    assert action.triplet.blue > action.triplet.green
                else:
                    assert action is not None and action.triplet is not None
                    assert action.triplet.red > action.triplet.green > action.triplet.blue
                assert app.query_one("#status", Static).content == "Working…"
                assert not app.query(".turn-outcome")
            assert len(set(frames)) == 1
            now = 202.5
            client.release[-1].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert app.query_one(".turn-outcome", Static).content == "Finished in 2.5s"
            assert label in screen_text(app)
            now = 206.0
            await pilot.pause(0.12)
            x, y = text_position(app, "Read terminal.txt")
            assert app.screen.get_style_at(x - 2, y).color == frames[0]
            assert app.query_one("#status", Static).content == "Ready"
        finally:
            for release in client.release:
                release.set()


async def test_tool_and_bottom_animate_together_and_stop_at_their_own_real_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    now = 100.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)
    raw = '{"path": "notes.txt"}'
    client = GatedClient([
        ToolStarted("call", "read_file", "decoy", raw),
        ToolFinished("call", "read_file", "decoy", ToolResult("success", "Read complete."), 0.01, raw),
    ])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "read"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.press(*"draft", "ctrl+j", *"second line", "enter")
            assert client.inputs == ["read"]
            frames = []
            bottom_frames = []
            positions = []
            action_colors = []
            for elapsed in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
                now = 100.0 + elapsed
                await pilot.pause(0.12)
                x, y = text_position(app, "Read notes.txt")
                color = app.screen.get_style_at(x - 2, y).color
                assert color is not None and color.triplet is not None
                frames.append(color.triplet)
                bottom_frames.append(app.query_one("#activity-dot", Static).content)
                positions.append((x, y))
                action_colors.append(app.screen.get_style_at(x, y).color)
                assert app.focused is prompt
                assert prompt.text == "draft\nsecond line"
            assert len(set(frames)) >= 4
            assert len(set(bottom_frames)) >= 3
            assert len(set(positions)) == len(set(action_colors)) == 1
            assert max(color.red for color in frames) - min(color.red for color in frames) >= 20
            assert all(max(abs(a - b) for a, b in zip(first, second)) < 60 for first, second in zip(frames, frames[1:]))

            now = 102.0
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            stopped_frames = []
            bottom_frames = []
            for elapsed in (2.0, 2.3, 2.6):
                now = 100.0 + elapsed
                await pilot.pause(0.12)
                x, y = text_position(app, "Read notes.txt")
                stopped_frames.append(app.screen.get_style_at(x - 2, y).color)
                bottom_frames.append(app.query_one("#activity-dot", Static).content)
                assert "Success" in screen_text(app)
                assert app.query_one("#status", Static).content == "Working…"
                assert not app.query(".turn-outcome")
            assert len(set(stopped_frames)) == 1
            assert len(set(bottom_frames)) >= 2
            assert stopped_frames[0] == action_colors[0]

            now = 104.0
            client.release[1].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert app.query_one(".turn-outcome", Static).content == "Finished in 4.0s"
            bottom = app.query_one("#activity-dot", Static)
            last_frame = bottom.content
            now = 108.5
            await pilot.pause(0.15)
            x, y = text_position(app, "Read notes.txt")
            assert app.screen.get_style_at(x - 2, y).color == stopped_frames[0]
            assert bottom.content == last_frame and not bottom.display
            assert app.query_one(".turn-outcome", Static).content == "Finished in 4.0s"
            assert app.focused is prompt and prompt.text == "draft\nsecond line"
        finally:
            for release in client.release:
                release.set()
