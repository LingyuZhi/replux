import asyncio
from collections.abc import AsyncGenerator
import json
from xml.etree import ElementTree

import pytest
from textual.containers import VerticalScroll
from textual.events import Paste
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled

from replux.llm import IncompleteReplyError, LLMClient, LLMError, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp


def screen_text(app: RepluxApp) -> str:
    root = ElementTree.fromstring(app.export_screenshot())
    return " ".join("".join(node.itertext()) for node in root.iter("{http://www.w3.org/2000/svg}text")).replace("\xa0", " ")


async def test_activity_tracks_real_phases_and_total_local_time_until_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now, raising=False)

    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.reached = [asyncio.Event() for _ in range(6)]
            self.release = [asyncio.Event() for _ in range(6)]
            self.closing = asyncio.Event()
            self.closed = asyncio.Event()
            self.finish_cleanup = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            events: list[StreamEvent | None] = [
                None, ThinkingDelta("Check the file."),
                ToolStarted("read", "read_file", "unreliable summary", '{"path": "notes.txt"}'),
                ToolFinished("read", "read_file", "unreliable summary", ToolResult("success", "1\talpha"), 0.2, '{"path": "notes.txt"}'),
                TextDelta("The file contains alpha."), ReplyCompleted("end_turn", 999999, 888888),
            ]
            try:
                for index, event in enumerate(events):
                    if event is not None:
                        yield event
                    self.reached[index].set()
                    await self.release[index].wait()
            finally:
                self.closing.set()
                await self.finish_cleanup.wait()
                self.closed.set()

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            assert app.query_one("#status", Static).content == "Ready"
            prompt.text = "inspect"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            assert app.query_one("#status", Static).content == "Working…"
            assert app.query_one("#activity-time", Static).content == "0.0s"
            assert app.query_one("#activity-hint", Static).content == "Ctrl+C to cancel"
            assert not app.query(".thinking, .turn-outcome")

            for index, (action, elapsed) in enumerate([
                ("Thinking…", "2.0s"), ("Reading… notes.txt", "4.0s"),
                ("Working…", "6.0s"), ("Writing reply…", "8.0s"),
                ("Writing reply…", "10.0s"),
            ], start=1):
                now = 100.0 + 2 * index
                client.release[index - 1].set()
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                await pilot.pause(0.15)
                assert app.query_one("#status", Static).content == action
                assert app.query_one("#activity-time", Static).content == elapsed
                assert not app.query(".turn-outcome")
                assert app.focused is prompt

            client.release[5].set()
            await asyncio.wait_for(client.closing.wait(), timeout=2)
            await pilot.pause()
            assert not client.closed.is_set()
            assert not app.query(".turn-outcome")
            assert app.query_one("#status", Static).content == "Writing reply…"
            now = 112.5
            client.finish_cleanup.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert app.query_one(".turn-outcome", Static).content == "Finished in 12.5s"
            assert app.query_one("#status", Static).content == "Ready"
            assert not app.query_one("#activity-time").display
            assert not app.query_one("#activity-hint").display
            assert not app.query_one("#activity-dot").display
            assert "999999" not in screen_text(app)
            assert "888888" not in screen_text(app)
        finally:
            for release in client.release:
                release.set()
            client.finish_cleanup.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_composer_uses_a_top_rule_and_marker_and_preserves_multiline_paste(
    size: tuple[int, int],
) -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            yield TextDelta("Recorded.")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        marker = app.query_one("#prompt-marker", Static)
        assert marker.content == "›"
        assert marker.region.width == 2
        assert marker.region.right == prompt.region.x
        assert marker.region.y == prompt.region.y
        assert prompt.styles.border_top[0] == ""
        assert prompt.styles.border_bottom[0] == ""
        assert "─" in screen_text(app)
        initial_height = prompt.region.height
        await pilot.press("space", "ctrl+j", "space", "enter")
        assert client.inputs == []
        assert not app.query(".user, .turn-outcome")
        prompt.clear()
        await pilot.press(*"中文问题", "ctrl+j")
        app.post_message(Paste("    indented\n\tlast line\n\n  "))
        await pilot.pause()
        expected = "中文问题\n    indented\n\tlast line\n\n  "
        assert prompt.text == expected
        assert prompt.region.height > initial_height
        assert prompt.region.bottom == size[1]
        assert app.focused is prompt
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert client.inputs == [expected]
        assert app.query_one(".user", Static).content == expected
        assert prompt.text == ""
        assert prompt.region.height == initial_height
        assert app.focused is prompt


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_repeated_cancel_keeps_cleanup_busy_and_preserves_draft_for_the_next_turn(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int],
) -> None:
    now = 500.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)

    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.running = asyncio.Event()
            self.closing = asyncio.Event()
            self.finish_cleanup = asyncio.Event()
            self.closed = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            if len(self.inputs) == 1:
                try:
                    yield ToolStarted("run", "run_command", "decoy", '{"command": "sleep 30"}')
                    self.running.set()
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.closing.set()
                    await self.finish_cleanup.wait()
                    yield ToolFinished("run", "run_command", "decoy", ToolResult("cancelled", "Process group stopped."), 13.5, '{"command": "sleep 30"}')
                    yield ToolFinished("next", "read_file", "decoy", ToolResult("skipped", "Not run because the user cancelled."), None, '{"path": "later.txt"}')
                    raise
                finally:
                    self.closed.set()
            yield TextDelta("Next turn completed.")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "run"
            await pilot.press("enter")
            await asyncio.wait_for(client.running.wait(), timeout=2)
            worker = next(iter(app.workers))
            now = 507.0
            await pilot.press("ctrl+c")
            await asyncio.wait_for(client.closing.wait(), timeout=2)
            await pilot.press(*"草稿", "ctrl+j")
            app.post_message(Paste("    keep indentation"))
            await pilot.pause()
            now = 511.0
            await pilot.press("enter", "ctrl+c", "ctrl+c", "enter")
            await pilot.pause(0.15)
            assert app.query_one("#status", Static).content == "Cancelling…"
            assert app.query_one("#activity-time", Static).content == "11.0s"
            assert not app.query(".turn-outcome")
            assert not client.closed.is_set()
            assert client.inputs == ["run"]
            assert prompt.text == "草稿\n    keep indentation"
            assert app.focused is prompt
            assert "Cancelling…" in screen_text(app)

            now = 513.5
            client.finish_cleanup.set()
            with pytest.raises(WorkerCancelled):
                await asyncio.wait_for(worker.wait(), timeout=2)
            await pilot.pause()
            assert client.closed.is_set()
            assert app.query_one(".turn-outcome", Static).content == "Cancelled after 13.5s"
            assert [status.content for status in app.query(".tool-status").results(Static)] == ["Cancelled", "Skipped"]
            assert app.query_one("#status", Static).content == "Ready"
            assert not app.query_one("#activity-dot").display
            assert prompt.text == "草稿\n    keep indentation"
            now = 520.0
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            assert client.inputs == ["run", "草稿\n    keep indentation"]
            assert [outcome.content for outcome in app.query(".turn-outcome").results(Static)] == [
                "Cancelled after 13.5s", "Finished in 0.0s",
            ]
            assert prompt.text == ""
            assert app.focused is prompt
        finally:
            client.finish_cleanup.set()


@pytest.mark.parametrize("ending", ["request-error", "incomplete", "cleanup-error"])
async def test_failure_records_total_time_only_after_cleanup_and_allows_a_follow_up(
    monkeypatch: pytest.MonkeyPatch, ending: str,
) -> None:
    now = 1000.0
    monkeypatch.setattr("replux.tui.monotonic", lambda: now)

    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.reached = asyncio.Event()
            self.release = asyncio.Event()
            self.closing = asyncio.Event()
            self.finish_cleanup = asyncio.Event()
            self.closed = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            if user_input == "follow-up":
                yield TextDelta("Recovered.")
                yield ReplyCompleted("end_turn", None, None)
                return
            try:
                yield TextDelta("Partial reply.")
                self.reached.set()
                await self.release.wait()
                if ending == "request-error":
                    raise LLMError("Connection lost.")
                if ending == "incomplete":
                    raise IncompleteReplyError("Reply incomplete: output limit reached.")
                yield ReplyCompleted("end_turn", None, None)
            finally:
                self.closing.set()
                await self.finish_cleanup.wait()
                self.closed.set()
                if ending == "cleanup-error":
                    raise LLMError("Stream close failed.")

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "first"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached.wait(), timeout=2)
            prompt.text = "follow-up"
            now = 1020.0
            client.release.set()
            await asyncio.wait_for(client.closing.wait(), timeout=2)
            await pilot.press("enter")
            assert not app.query(".turn-status, .turn-outcome")
            assert app.query_one("#status", Static).content == "Writing reply…"
            assert not client.closed.is_set()
            assert prompt.text == "follow-up"
            now = 1035.0
            client.finish_cleanup.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert client.closed.is_set()
            assert app.query_one(".turn-outcome", Static).content == "Failed after 35.0s"
            message = {
                "request-error": "Request failed: Connection lost.",
                "incomplete": "Reply incomplete: output limit reached.",
                "cleanup-error": "Request failed: Stream close failed.",
            }[ending]
            assert app.query_one(".turn-status", Static).content == message
            assert "Partial reply." in screen_text(app)
            assert app.query_one(".turn-outcome", Static).styles.color == app.query_one("#activity").styles.color
            assert app.query_one("#status", Static).content == "Ready"
            assert not app.query_one("#activity-dot").display
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            assert [outcome.content for outcome in app.query(".turn-outcome").results(Static)] == [
                "Failed after 35.0s", "Finished in 0.0s",
            ]
            assert app.focused is prompt
        finally:
            client.release.set()
            client.finish_cleanup.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_six_tool_actions_use_raw_arguments_and_leave_time_and_cancel_hint_visible(
    size: tuple[int, int],
) -> None:
    target = "very-long-directory/" * 20 + "notes.txt"
    pattern = "long-pattern|" * 30 + "TODO"
    command = "printf 'long command' " * 30 + "\n  printf done"
    cases: list[tuple[str, dict[str, object], str]] = [
        ("read_file", {"path": target}, f"Reading… {target}"),
        ("write_file", {"path": target, "content": "new bytes"}, f"Writing… {target}"),
        ("edit_file", {"path": target, "edits": []}, f"Editing… {target}"),
        ("find_files", {"pattern": pattern, "path": "src"}, f"Finding… {pattern} in src"),
        ("search_content", {"pattern": pattern, "path": "src"}, f"Searching… {pattern} in src"),
        ("run_command", {"command": command, "cwd": "/work"}, "Running… " + "printf 'long command' " * 30 + "printf done in /work"),
    ]

    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.reached = [asyncio.Event() for _ in cases]
            self.release = [asyncio.Event() for _ in cases]

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            for index, (name, arguments, _) in enumerate(cases):
                raw = json.dumps(arguments)
                yield ToolStarted(str(index), name, "decoy: never use this summary", raw)
                self.reached[index].set()
                await self.release[index].wait()
                yield ToolFinished(str(index), name, "decoy", ToolResult("success", "Done."), 0.1, raw)
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "use tools"
            await pilot.press("enter")
            prompt.text = "Draft line\n" * 10
            for index, (_, _, expected) in enumerate(cases):
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                await pilot.pause()
                status = app.query_one("#status", Static)
                elapsed = app.query_one("#activity-time", Static)
                hint = app.query_one("#activity-hint", Static)
                assert status.content == expected
                assert status.region.height == elapsed.region.height == hint.region.height == 1
                assert status.region.right <= elapsed.region.x < elapsed.region.right <= hint.region.x
                assert hint.region.right <= size[0]
                visible = screen_text(app)
                assert "Ctrl+C to cancel" in visible
                assert "…" in visible
                assert "decoy" not in visible
                assert app.focused is prompt
                client.release[index].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            assert len(app.query(".turn-outcome")) == 1
        finally:
            for release in client.release:
                release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)


async def test_composer_growth_activity_changes_and_outcomes_keep_focus_and_transcript_position() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.reached = [asyncio.Event() for _ in range(3)]
            self.release = [asyncio.Event() for _ in range(3)]

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield TextDelta("".join(f"Old reply line {line}\n" for line in range(80)))
            self.reached[0].set()
            await self.release[0].wait()
            yield ToolStarted("read", "read_file", "decoy", '{"path": "notes.txt"}')
            self.reached[1].set()
            await self.release[1].wait()
            yield ToolFinished("read", "read_file", "decoy", ToolResult("success", "Done."), 0.1, '{"path": "notes.txt"}')
            yield ThinkingDelta("Check the result.")
            yield TextDelta("More reply content\n" * 20)
            self.reached[2].set()
            await self.release[2].wait()
            yield TextDelta("Final reply line.")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            transcript = app.query_one("#transcript", VerticalScroll)
            prompt.text = "first"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            prompt.text = "Draft line\n" * 5
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            transcript.focus()
            await pilot.press("pageup")
            await pilot.wait_for_scheduled_animations()
            position = transcript.scroll_y
            assert not transcript.is_vertical_scroll_end
            prompt.focus()
            draft = "Draft line\n" * 20
            prompt.text = draft
            await pilot.pause()
            assert 1 < prompt.region.height <= 8
            assert transcript.scroll_y == position
            for index in [1, 2]:
                client.release[index - 1].set()
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                await pilot.pause()
                assert transcript.scroll_y == position
                assert app.focused is prompt
                assert prompt.text == draft
            transcript.focus()
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            assert transcript.is_vertical_scroll_end
            prompt.focus()
            client.release[2].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            assert app.focused is prompt
            assert prompt.text == draft
            assert "Finished in" in screen_text(app)
        finally:
            for release in client.release:
                release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
