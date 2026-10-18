import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea

from replux.cli import create_app
from replux.llm import LLMClient, ReplyCompleted, StreamEvent, TextDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp, ToolCallView
from tests.support import ControlledSSE, RecordingTransport, reply_events
from tests.test_tool_batch import batch_events, result_pairs
from tests.test_tool_ui import provider_file


class ResultClient(LLMClient):
    def __init__(self, name: str, arguments: str, result: ToolResult, duration: float | None = 0.125) -> None:
        self.name = name
        self.arguments = arguments
        self.result = result
        self.duration = duration
        self.inputs: list[str] = []

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        self.inputs.append(user_input)
        if self.result.status != "skipped":
            yield ToolStarted("call", self.name, "deliberately incomplete…", self.arguments)
        yield ToolFinished("call", self.name, "deliberately incomplete…", self.result, self.duration, self.arguments)
        yield ReplyCompleted("end_turn", None, None)

    async def aclose(self) -> None:
        pass


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_raw_arguments_reach_each_call_including_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = "long-" + "target-" * 20 + "notes.txt"
    write_arguments = json.dumps({"path": target, "content": "第一行\n  second line\n"}, ensure_ascii=False, indent=2)
    skipped_arguments = json.dumps({"command": "printf 'first line\\n'\n  printf '第二行\\n'"}, ensure_ascii=False, indent=2)
    calls = (
        ("write", "write_file", (write_arguments,)),
        ("missing", "read_file", ('{"path": "missing.txt"}',)),
        ("skipped", "run_command", (skipped_arguments,)),
    )
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "write then read"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        views = list(app.query(ToolCallView).results(ToolCallView))
        assert views[0].query_one(".tool-header", Static).content == f"Write {target}"
        assert views[2].query_one(".tool-header", Static).content == "Run printf 'first line\\n' printf '第二行\\n'"
        assert views[2].query_one(".tool-status", Static).content == "Skipped"
        for view, (_, name, (arguments,)) in zip(views, calls, strict=True):
            heading = view.query_one(".tool-heading")
            heading.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click(heading)
            assert view.query_one(".tool-arguments", Static).content == arguments
            assert view.query_one(".tool-details").display
            assert f"Tool: {name}" in str(view.query_one(".tool-metadata", Static).content)
            await pilot.press("enter")
            assert not view.query_one(".tool-details").display
        assert "Duration:" not in str(views[2].query_one(".tool-metadata", Static).content)
        assert len(transport.requests) == 2
    request = json.loads(transport.requests[1].content)
    if protocol == "anthropic":
        tool_calls = [block for block in request["messages"][1]["content"] if block["type"] == "tool_use"]
        assert tool_calls[0]["input"] == json.loads(write_arguments)
        assert tool_calls[2]["input"] == json.loads(skipped_arguments)
    else:
        tool_calls = [item for item in request["input"] if item.get("type") == "function_call"]
        assert tool_calls[0]["arguments"] == write_arguments
        assert tool_calls[2]["arguments"] == skipped_arguments


@pytest.mark.parametrize(("output", "details", "expected"), [
    ("10\talpha\n11\tbeta\n12\tgamma", {"start_line": 10, "end_line": 12}, "Lines 10–12"),
    ("4\tlast", {"start_line": 4, "end_line": 4}, "Lines 4–4"),
    ("The file is empty.", {"path": "notes.txt"}, "The file is empty."),
    ("Range not provided.\n1\talpha", {}, "Range not provided."),
    ("10\talpha\n11\tbeta\n[Use offset=12 to continue.]",
     {"start_line": 10, "end_line": 11, "next_offset": 12, "truncated_by": "lines"},
     "Truncated · Lines 10–11"),
    ("Line 7 is 60000 bytes and does not fit in the 50 KiB read limit.",
     {"line": 7, "line_bytes": 60000, "truncated_by": "bytes"},
     "Truncated · Line 7 is 60000 bytes and does not fit in the 50 KiB read limit."),
])
async def test_read_summary_uses_only_returned_ranges(
    output: str, details: dict[str, object], expected: str,
) -> None:
    result = ToolResult("success", output, details=details)
    client = ResultClient("read_file", '{"path": "notes.txt", "offset": 10, "limit": 100}', result)
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "read"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        view = app.query_one(ToolCallView)
        assert view.query_one(".tool-preview", Static).content == f"└ {expected}"
        assert view.query_one(".tool-preview").region.height == 1
        await pilot.click(view.query_one(".tool-heading"))
        assert view.query_one(".tool-output", Static).content == output
        metadata = str(view.query_one(".tool-metadata", Static).content)
        assert "Status: Success (success)" in metadata
        assert "Duration: 0.12s" in metadata
        if details:
            assert json.loads(str(view.query_one(".tool-result-details", Static).content)) == details
        await pilot.press("enter")
        assert view.query_one(".tool-preview").display
        assert client.inputs == ["read"]


@pytest.mark.parametrize(("name", "details", "expected"), [
    ("write_file", {"bytes": 9}, "Wrote 9 bytes"),
    ("write_file", {"bytes": 0}, "Wrote 0 bytes"),
    ("write_file", {}, "Write completed."),
    ("edit_file", {"edits": 2, "bytes": 5000}, "Applied 2 replacement operations"),
    ("edit_file", {"bytes": 5000}, "Write completed."),
])
async def test_write_and_edit_summaries_do_not_invent_change_statistics(
    name: str, details: dict[str, object], expected: str,
) -> None:
    client = ResultClient(name, '{"path": "notes.txt", "content": "a\\nb\\nc"}', ToolResult("success", "Write completed.", details=details))
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "change file"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one(".tool-preview", Static).content == f"└ {expected}"


@pytest.mark.parametrize(("name", "details", "expected"), [
    ("find_files", {"total": 12, "count": 2, "truncated_by": "lines"}, "Truncated · Found 12 files · Returned 2"),
    ("find_files", {"total": 2, "count": 2}, "Found 2 files · Returned 2"),
    ("find_files", {"total": 0, "count": 0}, "Found 0 files · Returned 0"),
    ("find_files", {"total": 12}, "Found 12 files"),
    ("find_files", {"count": 2}, "Returned 2 files"),
    ("find_files", {}, "Search completed."),
    ("search_content", {"count": 2, "truncated_by": "bytes"}, "Truncated · Returned 2 matching lines"),
    ("search_content", {"count": 0}, "Returned 0 matching lines"),
    ("search_content", {}, "Search completed."),
    ("search_content", {"count": 1}, "Returned 1 matching line"),
])
async def test_find_and_search_summaries_distinguish_total_and_returned_counts(
    name: str, details: dict[str, object], expected: str,
) -> None:
    client = ResultClient(name, '{"pattern": "alpha|beta", "path": "src"}', ToolResult("success", "Search completed.", details=details))
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "search"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one(".tool-preview", Static).content == f"└ {expected}"
        action = "Find" if name == "find_files" else "Search"
        assert app.query_one(".tool-header", Static).content == f"{action} alpha|beta in src"


@pytest.mark.parametrize(("result", "label", "expected"), [
    (ToolResult("success", "Command finished.", details={"exit_code": 0}), "Success", "Exited with code 0"),
    (ToolResult("success", "Command finished."), "Success", "Command finished."),
    (ToolResult("success", "Command finished.", details={"exit_code": None}), "Success", "Command finished."),
    (ToolResult("success", "Command finished.\nkept tail", details={"exit_code": 0, "truncated_by": "lines"}),
     "Success", "Truncated · Exited with code 0"),
    (ToolResult("error", "Cannot run the command: missing Bash.", "command_failed"),
     "Failed", "Command failed · Cannot run the command: missing Bash."),
    (ToolResult("error", "Command failed with exit code 7.", "command_failed", {"exit_code": 7}),
     "Failed", "Command failed with exit code 7."),
    (ToolResult("error", "Command timed out after 5 seconds.", "timeout", {"exit_code": -15, "side_effects": "kept"}),
     "Failed", "Timeout · Command timed out after 5 seconds."),
    (ToolResult("cancelled", "Command was cancelled and its process group was stopped.", details={"exit_code": -15, "side_effects": "kept"}),
     "Cancelled", "Command was cancelled and its process group was stopped."),
    (ToolResult("skipped", "Not run because an earlier call failed."), "Skipped", "Not run because an earlier call failed."),
    (ToolResult("error", "Command stopped because captured output reached the log limit.", "output_limit_exceeded", {"exit_code": -15}),
     "Failed", "Truncated · Output limit exceeded · Command stopped because captured output reached the log limit."),
    (ToolResult("error", "", "timeout"), "Failed", "Failed (timeout)"),
])
async def test_command_summaries_keep_real_terminal_states_and_keyboard_details(
    result: ToolResult, label: str, expected: str,
) -> None:
    duration = None if result.status == "skipped" else 0.125
    client = ResultClient("run_command", '{"command": "printf hello", "cwd": "src"}', result, duration)
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "run"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        prompt.text = "next draft"
        view = app.query_one(ToolCallView)
        heading = view.query_one(".tool-heading")
        assert view.query_one(".tool-header", Static).content == "Run printf hello in src"
        assert view.query_one(".tool-status", Static).content == label
        assert view.query_one(".tool-preview", Static).content == f"└ {expected}"
        assert not view.query_one(".tool-details").display
        for _ in range(4):
            await pilot.press("tab")
            if app.focused == heading:
                break
        assert app.focused == heading
        await pilot.press("enter")
        assert view.query_one(".tool-details").display
        assert view.query_one(".tool-output", Static).content == result.output
        if result.details:
            assert json.loads(str(view.query_one(".tool-result-details", Static).content)) == result.details
        metadata = str(view.query_one(".tool-metadata", Static).content)
        assert ("Duration:" in metadata) == (duration is not None)
        if result.error_code:
            assert f"Error code: {result.error_code}" in metadata
        await pilot.press("enter")
        assert view.query_one(".tool-preview").display
        assert prompt.text == "next draft"
        assert client.inputs == ["run"]


async def test_search_summary_marks_a_truncated_match_line_without_guessing_more_matches() -> None:
    output = "notes.txt:1:alpha... [line truncated]"
    client = ResultClient("search_content", '{"pattern": "alpha"}', ToolResult("success", output, details={"count": 1}))
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "search"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one(".tool-preview", Static).content == "└ Truncated · Returned 1 matching line"


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_all_six_real_tools_keep_bounded_results_and_context_when_details_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("alpha alpha\nbeta\n", encoding="utf-8")
    calls = (
        ("read", "read_file", ('{"path": "notes.txt", "limit": 1}',)),
        ("write", "write_file", ('{"path": "new.txt", "content": "hi\\n"}',)),
        ("edit", "edit_file", ('{"path": "new.txt", "edits": [{"old_text": "hi", "new_text": "hello"}]}',)),
        ("find", "find_files", ('{"pattern": "*.txt", "path": ".", "limit": 1}',)),
        ("search", "search_content", ('{"pattern": "alpha", "path": "notes.txt"}',)),
        ("run", "run_command", ('{"command": "printf done"}',)),
    )
    expected = [
        "└ Truncated · Lines 1–1", "└ Wrote 3 bytes", "└ Applied 1 replacement operation",
        "└ Truncated · Found 2 files · Returned 1", "└ Returned 1 matching line", "└ Exited with code 0",
    ]
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), summary, ControlledSSE(reply_events(protocol)),
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(160, 50)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "use six tools"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        prompt.text = "next draft"
        views = list(app.query(ToolCallView).results(ToolCallView))
        assert len(views) == 6
        results = result_pairs(protocol, transport.requests[1])
        for view, (_, name, (arguments,)), (_, result), preview in zip(views, calls, results, expected, strict=True):
            assert view.query_one(".tool-preview", Static).content == preview
            assert view.query_one(".tool-status", Static).content == "Success"
            assert not view.query_one(".tool-details").display
            heading = view.query_one(".tool-heading")
            heading.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.click(heading)
            assert view.query_one(".tool-output", Static).content == result["output"]
            assert view.query_one(".tool-arguments", Static).content == arguments
            assert f"Tool: {name}" in str(view.query_one(".tool-metadata", Static).content)
            assert json.loads(str(view.query_one(".tool-result-details", Static).content)) == result["details"]
            await pilot.press("enter")
            assert view.query_one(".tool-preview").display
            assert app.focused is heading
        assert len(transport.requests) == 2
        assert prompt.text == "next draft"
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.focus()
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert len(transport.requests) == 3
        assert result_pairs(protocol, transport.requests[2]) == results
    bodies = [json.loads(request.content) for request in transport.requests]
    assert bodies[0]["tools"] == bodies[1]["tools"] == bodies[2]["tools"]


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
@pytest.mark.parametrize(("name", "arguments"), [
    ("read_file", json.dumps({"path": "长路径/" * 60 + "notes.txt"}, ensure_ascii=False)),
    ("search_content", json.dumps({"pattern": "long-pattern|" * 60, "path": "src"})),
    ("run_command", json.dumps({"command": "printf 'long command with spaces' " * 60})),
])
async def test_long_targets_leave_status_arrow_and_focus_visible(
    size: tuple[int, int], name: str, arguments: str,
) -> None:
    result = ToolResult("error", "Output limit reached. " * 40, "output_limit_exceeded", {"truncated_by": "bytes"})
    app = RepluxApp(ResultClient(name, arguments, result), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        app.query_one("#prompt", TextArea).text = "show long target"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        status = app.query_one(".tool-status", Static)
        arrow = app.query_one(".tool-arrow", Static)
        header = app.query_one(".tool-header", Static)
        assert header.region.height == status.region.height == arrow.region.height == 1
        assert header.region.right <= status.region.x < status.region.right <= arrow.region.x
        assert arrow.region.right <= size[0]
        assert app.query_one(".tool-preview").region.height == 1
        screenshot = app.export_screenshot()
        assert "Failed" in screenshot and "▸" in screenshot and "Truncated" in screenshot and "…" in screenshot
        heading = app.query_one(".tool-heading")
        for _ in range(4):
            await pilot.press("tab")
            if app.focused is heading:
                break
        assert app.focused is heading
        assert heading.styles.background.a > 0
        await pilot.press("enter")
        assert app.query_one(".tool-arguments", Static).content == arguments
        assert app.query_one(".tool-output", Static).content == result.output


async def test_each_call_keeps_its_own_details_while_running_finishing_and_starting_another_turn() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            if user_input == "first turn":
                result = ToolResult("success", "1\talpha", details={"start_line": 1, "end_line": 1})
                yield ToolStarted("first", "read_file", "path=notes.txt", '{"path": "notes.txt"}')
                yield ToolFinished("first", "read_file", "path=notes.txt", result, 0.1, '{"path": "notes.txt"}')
                yield ToolStarted("second", "read_file", "path=notes.txt", '{"path": "notes.txt"}')
                self.started.set()
                await self.release.wait()
                yield ToolFinished("second", "read_file", "path=notes.txt", result, 0.2, '{"path": "notes.txt"}')
            yield TextDelta("Done.")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "first turn"
            await pilot.press("enter")
            await asyncio.wait_for(client.started.wait(), timeout=2)
            await pilot.pause()
            first, second = app.query(ToolCallView).results(ToolCallView)
            await pilot.click(first.query_one(".tool-heading"))
            assert first.query_one(".tool-details").display
            assert not second.query_one(".tool-details").display
            await pilot.click(second.query_one(".tool-heading"))
            assert second.query_one(".tool-details").display
            assert "Status: Running" in str(second.query_one(".tool-metadata", Static).content)
            assert second.query_one(".tool-arguments", Static).content == '{"path": "notes.txt"}'
            focus = app.focused
            prompt.text = "next draft"
            client.release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert first.query_one(".tool-details").display
            assert second.query_one(".tool-details").display
            assert "Status: Success" in str(second.query_one(".tool-metadata", Static).content)
            assert second.query_one(".tool-output", Static).content == "1\talpha"
            assert app.focused is focus
            await pilot.click(first.query_one(".tool-heading"))
            assert not first.query_one(".tool-details").display
            assert second.query_one(".tool-details").display
            assert prompt.text == "next draft"
            prompt.focus()
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            assert not first.query_one(".tool-details").display
            assert second.query_one(".tool-details").display
        finally:
            client.release.set()


async def test_expanding_tool_details_while_reading_keeps_position_focus_and_bottom_following() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.reached = [asyncio.Event(), asyncio.Event()]
            self.release = [asyncio.Event(), asyncio.Event()]

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            output = "".join(f"{line}\tfile content\n" for line in range(1, 61))
            yield ToolFinished("read", "read_file", "path=notes.txt", ToolResult("success", output), 0.1, '{"path": "notes.txt"}')
            yield TextDelta("Old reply\n" * 60)
            self.reached[0].set()
            await self.release[0].wait()
            yield TextDelta("Reply appended while reading")
            self.reached[1].set()
            await self.release[1].wait()
            yield TextDelta("New reply after returning to the bottom")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            transcript = app.query_one("#transcript", VerticalScroll)
            prompt.text = "read and reply"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            transcript.focus()
            await pilot.press("home")
            await pilot.wait_for_scheduled_animations()
            position = transcript.scroll_y
            heading = app.query_one(".tool-heading")
            title_y = heading.region.y
            await pilot.click(heading)
            await pilot.wait_for_scheduled_animations()
            assert app.query_one(".tool-details").display
            assert heading.region.y == title_y
            assert transcript.scroll_y == position
            assert app.focused is heading
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert transcript.scroll_y == position
            assert app.focused is heading
            prompt.text = "unsent draft"
            transcript.focus()
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            prompt.focus()
            client.release[1].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            assert prompt.text == "unsent draft"
            assert app.focused is prompt
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize("name", ["write_file", "edit_file"])
@pytest.mark.parametrize("status", ["error", "cancelled"])
async def test_file_failures_keep_actual_file_state_and_side_effects(name: str, status: str) -> None:
    result = (
        ToolResult("error", "Timed out. The file contains the new bytes.", "timeout", {"file_state": "written"})
        if status == "error" else
        ToolResult("cancelled", "Cancelled. The file contains the new bytes.", details={"file_state": "written"})
    )
    app = RepluxApp(ResultClient(name, '{"path": "notes.txt"}', result), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "change file"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        prefix = "Timeout · " if status == "error" else ""
        assert app.query_one(".tool-preview", Static).content == f"└ {prefix}{result.output}"
        assert app.query_one(".tool-status", Static).content == ("Failed" if status == "error" else "Cancelled")
        await pilot.click(app.query_one(".tool-heading"))
        assert json.loads(str(app.query_one(".tool-result-details", Static).content)) == {"file_state": "written"}
        assert app.query_one(".tool-output", Static).content == result.output


@pytest.mark.parametrize(("name", "arguments", "details", "expected"), [
    ("find_files", '{"pattern": "*.py"}', {"path": "/work/src", "count": 0, "total": 0}, "Find *.py in /work/src"),
    ("search_content", '{"pattern": "TODO"}', {"path": "/work/src", "count": 0}, "Search TODO in /work/src"),
    ("run_command", '{"command": "pwd"}', {"cwd": "/work/src", "exit_code": 0}, "Run pwd in /work/src"),
])
async def test_titles_include_reported_scope_when_arguments_use_the_default(
    name: str, arguments: str, details: dict[str, object], expected: str,
) -> None:
    app = RepluxApp(ResultClient(name, arguments, ToolResult("success", "Done.", details=details)), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "use default scope"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one(".tool-header", Static).content == expected


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
@pytest.mark.parametrize(("error_code", "reason"), [("timeout", "Timeout"), ("not_a_file", "Not a file")])
async def test_failure_reason_stays_visible_before_a_long_path(
    size: tuple[int, int], error_code: str, reason: str,
) -> None:
    target = "/work/" + "long-directory/" * 30 + "notes.txt"
    output = (
        f"Reading {target} did not finish within 10 seconds."
        if error_code == "timeout" else f"{target} is not a regular file."
    )
    result = ToolResult("error", output, error_code, {"path": target})
    app = RepluxApp(ResultClient("read_file", json.dumps({"path": target}), result), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        app.query_one("#prompt", TextArea).text = "read long path"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert str(app.query_one(".tool-preview", Static).content).startswith(f"└ {reason} · ")
        screenshot = app.export_screenshot()
        assert reason.replace(" ", "&#160;") in screenshot
        await pilot.click(app.query_one(".tool-heading"))
        assert app.query_one(".tool-output", Static).content == output
        assert f"Error code: {error_code}" in str(app.query_one(".tool-metadata", Static).content)
