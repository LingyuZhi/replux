import asyncio
import re

import pytest
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.worker import WorkerCancelled
from textual.widgets import TextArea

from replux.llm import TextDelta, ThinkingDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp
from tests.test_reply_markdown import GatedClient, ScriptedClient
from tests.tui_support import screen_text, text_position


async def read_right_until(app: RepluxApp, pilot: Pilot[None], text: str) -> list[str]:
    frames = [screen_text(app)]
    for _ in range(24):
        if text in frames[-1]:
            return frames
        await pilot.press(*(["right"] * 8))
        frames.append(screen_text(app))
    raise AssertionError(f"Cannot read {text!r}\n{frames[-1]}")


@pytest.mark.parametrize("kind", ["code", "table"])
async def test_long_content_with_tabs_reaches_the_last_character(kind: str) -> None:
    value = "\tpath = '/workspace/" + "中文-part/" * 18 + "TAB_END'"
    source = f"```python\n{value}\n```" if kind == "code" else f"| Value |\n| --- |\n| `{value}` |"
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "Read tab-indented code"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        await pilot.press("tab", "tab")
        await read_right_until(app, pilot, "TAB_END'")


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_keyboard_reads_long_code_without_wrapping_or_moving_the_reply(
    size: tuple[int, int],
) -> None:
    code = "    path = '/workspace/" + "中文-mixed/" * 18 + "CODE_END'"
    source = f"Before code\n\n```python\n{code}\n\n    second = 42\n```\n\nAfter code"
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read long code"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.focused is prompt
        await pilot.press("tab", "tab")
        region = app.screen.focused
        assert region is not None and region is not prompt
        assert region is not app.query_one("#transcript", VerticalScroll)
        assert region.show_horizontal_scrollbar
        assert region.scrollbar_size_horizontal == 1
        assert text_position(app, "path =")[0] == 9
        assert text_position(app, "second = 42")[1] == text_position(app, "path =")[1] + 2
        before = text_position(app, "Before code")
        after = text_position(app, "After code")
        assert "CODE_END" not in screen_text(app)
        await pilot.press(*(["right"] * 64))
        await pilot.wait_for_scheduled_animations()
        assert "CODE_END'" in screen_text(app)
        assert text_position(app, "Before code") == before
        assert text_position(app, "After code") == after
        await pilot.press(*(["left"] * 64))
        await pilot.wait_for_scheduled_animations()
        assert text_position(app, "path =")[0] == 9
        await pilot.press("tab")
        assert app.focused is prompt
        await pilot.press("shift+tab")
        assert app.focused is region


@pytest.mark.parametrize("kind", ["code", "table"])
async def test_streaming_keeps_horizontal_focus_position_and_editable_draft(kind: str) -> None:
    long = "segment/" * 24 + "MIDDLE_READING"
    if kind == "code":
        initial = f"Before region\n\n```python\n    value = '{long}"
        addition = "_EXTENDED_END'\n    next_value = 42\n```\n"
    else:
        initial = f"Before region\n\n| Name | Value |\n| --- | --- |\n| first-row | {long}"
        addition = "_EXTENDED_END |\n| second | new row |\n"
    client = GatedClient([TextDelta(initial), TextDelta(addition), TextDelta("\nAfter streaming\n")])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "Stream horizontal content"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            await pilot.press("tab", "tab")
            region = app.screen.focused
            assert region is not None and region is not prompt
            await read_right_until(app, pilot, "MIDDLE_READING")
            position = text_position(app, "MIDDLE_READING")
            offset = region.scroll_x
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert app.focused is region
            assert region.scroll_x == offset
            assert text_position(app, "MIDDLE_READING") == position
            await read_right_until(app, pilot, "EXTENDED_END")
            await pilot.press("tab", *"draft", "ctrl+j", *"second", "enter")
            assert app.focused is prompt
            assert prompt.text == "draft\nsecond"
            client.release[1].set()
            await asyncio.wait_for(client.reached[2].wait(), timeout=2)
            await pilot.pause()
            assert app.focused is prompt
            assert prompt.text == "draft\nsecond"
            assert client.inputs == ["Stream horizontal content"]
            await pilot.press("shift+tab")
            assert app.focused is region
            assert "EXTENDED_END" in screen_text(app)
            worker = next(iter(app.workers))
            await pilot.press("ctrl+c")
            with pytest.raises(WorkerCancelled):
                await asyncio.wait_for(worker.wait(), timeout=2)
            await pilot.pause()
            assert "Cancelled after" in screen_text(app)
            assert prompt.text == "draft\nsecond"
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_keyboard_reads_complete_table_cells_and_preserves_rows(
    size: tuple[int, int],
) -> None:
    path = "/workspace/" + "中文-mixed/" * 12 + "PATH_END"
    token = "abcdefghijklmnopqrstuvwxyz" * 5 + "TOKEN_END"
    number = "987654321012345678909876543210123456789"
    source = (
        "Table introduction\n\n| Path | Token | Number | Result |\n| --- | --- | --- | --- |\n"
        f"| {path} | {token} | {number} | ROW_ONE_END |\n"
        "| second path | second token | 42 | ROW_TWO_END |\n\nAfter table"
    )
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read wide table"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        await pilot.press("tab", "tab")
        region = app.screen.focused
        assert region is not None and region is not prompt
        assert region is not app.query_one("#transcript", VerticalScroll)
        assert region.show_horizontal_scrollbar
        assert region.scrollbar_size_horizontal == 1
        intro = text_position(app, "Table introduction")
        after = text_position(app, "After table")
        first_y = text_position(app, "/workspace/")[1]
        second_y = text_position(app, "second path")[1]
        frames = await read_right_until(app, pilot, "ROW_TWO_END")
        for expected in ("PATH_END", "TOKEN_END", number, "ROW_ONE_END", "ROW_TWO_END"):
            assert any(expected in frame for frame in frames), expected
        assert all("…" not in frame for frame in frames)
        assert text_position(app, "ROW_ONE_END")[1] == first_y
        assert text_position(app, "ROW_TWO_END")[1] == second_y
        assert text_position(app, "Table introduction") == intro
        assert text_position(app, "After table") == after
        await pilot.press("tab")
        assert app.focused is prompt
        await pilot.press("shift+tab")
        assert app.focused is region


@pytest.mark.parametrize("kind", ["code", "table"])
async def test_resize_preserves_valid_horizontal_position_and_updates_overflow(kind: str) -> None:
    value = "begin_" + "a" * 60 + "READING_HERE_" + "b" * 100 + "RESIZE_END"
    source = f"```\n{value}\n```" if kind == "code" else f"| Value |\n| --- |\n| {value} |"
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Resize while reading"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        await pilot.press("tab", "tab", *(["right"] * 8))
        region = app.screen.focused
        assert region is not None and region is not prompt
        offset = region.scroll_x
        assert offset > 0
        reading_x = text_position(app, "READING_HERE_")[0]
        for width, height in ((120, 40), (160, 50), (80, 24)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            assert app.focused is region
            assert region.scroll_x == offset
            assert text_position(app, "READING_HERE_")[0] == reading_x
            assert region.show_horizontal_scrollbar
        await read_right_until(app, pilot, "RESIZE_END")
        await pilot.resize_terminal(160, 50)
        await pilot.pause()
        assert "RESIZE_END" in screen_text(app)
        await pilot.resize_terminal(240, 50)
        await pilot.pause()
        assert not region.show_horizontal_scrollbar
        assert region.scrollbar_size_horizontal == 0
        assert "begin_" in screen_text(app) and "RESIZE_END" in screen_text(app)
        await pilot.press("tab")
        assert app.focused is not region
        prompt.focus()
        await pilot.press("tab", "tab")
        assert app.focused is prompt
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.focused is prompt
        await pilot.press("tab", "tab")
        assert app.focused is region
        assert region.show_horizontal_scrollbar
        await read_right_until(app, pilot, "RESIZE_END")


@pytest.mark.parametrize("kind", ["code", "table"])
async def test_horizontal_reading_coexists_with_vertical_history_and_bottom_follow(kind: str) -> None:
    values = ["path/" * 32 + f"LANDMARK_{i:03}" for i in range(90)]
    if kind == "code":
        initial = "```\n" + "\n".join(values) + "\n"
        addition = "path/" * 40 + "NEW_ROW_END\n```\n"
    else:
        initial = "| Value |\n| --- |\n" + "".join(f"| {value} |\n" for value in values)
        addition = "| " + "path/" * 40 + "NEW_ROW_END |\n"
    client = GatedClient([TextDelta(initial), TextDelta(addition), TextDelta("\n## Bottom followed\n")])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            app.query_one("#prompt", TextArea).text = "Read history horizontally"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=3)
            await pilot.pause()
            await pilot.press("tab", "tab")
            region = app.screen.focused
            assert region is not None and region.show_horizontal_scrollbar
            await read_right_until(app, pilot, "LANDMARK_")
            await pilot.press("end", "pageup")
            await pilot.wait_for_scheduled_animations()
            transcript = app.query_one("#transcript", VerticalScroll)
            assert not transcript.is_vertical_scroll_end
            landmark = re.findall(r"LANDMARK_\d{3}", screen_text(app))[0]
            position = text_position(app, landmark)
            offset = region.scroll_x
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=3)
            await pilot.pause()
            assert app.focused is region
            assert region.scroll_x == offset
            assert text_position(app, landmark) == position
            assert not transcript.is_vertical_scroll_end
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            assert transcript.is_vertical_scroll_end
            await read_right_until(app, pilot, "NEW_ROW_END")
            client.release[1].set()
            await asyncio.wait_for(client.reached[2].wait(), timeout=3)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            assert "Bottom followed" in screen_text(app)
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_only_overflowing_regions_join_thinking_tools_and_prompt_in_focus_order(
    size: tuple[int, int],
) -> None:
    arguments = '{"path":"notes.txt"}'
    before_tool = (
        "Short code\n\n```\nshort_code = 1\n```\n\nAfter short code\n\n"
        "| Item | Value |\n| --- | --- |\n| 中文 | 42 |\n\n"
        "```unknown-language\n" + "混合-path/" * 24 + "CODE_FOCUS_END\n```\n"
    )
    after_tool = "| Path |\n| --- |\n| " + "中文-long/" * 24 + "TABLE_FOCUS_END |\n"
    client = GatedClient([
        ThinkingDelta("Literal **thought**"), TextDelta(before_tool),
        ToolStarted("read", "read_file", "path=notes.txt", arguments),
        ToolFinished("read", "read_file", "path=notes.txt", ToolResult("success", "# RAW_TOOL_OUTPUT"), 0.1, arguments),
        TextDelta(after_tool),
    ])
    for release in client.release:
        release.set()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read, think and show tables"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert len([widget for widget in app.query("*") if widget.show_horizontal_scrollbar]) == 2
        await pilot.press("tab", "tab")
        thinking = app.screen.focused
        assert thinking is not None and thinking is not prompt
        await pilot.press("enter")
        assert "Literal **thought**" in screen_text(app)
        assert text_position(app, "After short code")[1] == text_position(app, "short_code = 1")[1] + 2
        await pilot.press("tab")
        code = app.screen.focused
        assert code is not None and code.show_horizontal_scrollbar
        await read_right_until(app, pilot, "CODE_FOCUS_END")
        await pilot.press("tab")
        tool = app.screen.focused
        assert tool is not None and not tool.show_horizontal_scrollbar
        await pilot.press("enter")
        if "# RAW_TOOL_OUTPUT" not in screen_text(app):
            await pilot.press("pagedown")
            await pilot.wait_for_scheduled_animations()
        assert "# RAW_TOOL_OUTPUT" in screen_text(app)
        await pilot.press("tab")
        table = app.screen.focused
        assert table is not None and table.show_horizontal_scrollbar
        await read_right_until(app, pilot, "TABLE_FOCUS_END")
        await pilot.press("tab")
        assert app.focused is prompt
        for expected in (table, tool, code, thinking, app.query_one("#transcript"), prompt):
            await pilot.press("shift+tab")
            assert app.focused is expected


@pytest.mark.parametrize("container", ["quote", "list"])
@pytest.mark.parametrize("kind", ["code", "table"])
async def test_streaming_nested_regions_retains_focus_when_sibling_content_arrives(
    container: str, kind: str,
) -> None:
    value = "nested-path/" * 15 + "NESTED_END"
    content = f"```\n{value}\n```\n" if kind == "code" else f"| Value |\n| --- |\n| {value} |\n"
    if container == "quote":
        initial = "".join(f"> {line}\n" for line in content.splitlines())
        addition = ">\n> A new quoted paragraph\n"
    else:
        initial = "- Nested example\n\n" + "".join(f"  {line}\n" for line in content.splitlines())
        addition = "\n  A new list paragraph\n"
    client = GatedClient([TextDelta(initial), TextDelta(addition)])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            app.query_one("#prompt", TextArea).text = "Nested horizontal reading"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            await pilot.press("tab", "tab")
            region = app.screen.focused
            assert region is not None and region.show_horizontal_scrollbar
            await read_right_until(app, pilot, "NESTED_END")
            offset = region.scroll_x
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert app.focused is region
            assert region.scroll_x == offset
            assert "NESTED_END" in screen_text(app)
            assert ("A new quoted paragraph" if container == "quote" else "A new list paragraph") in screen_text(app)
        finally:
            for release in client.release:
                release.set()
