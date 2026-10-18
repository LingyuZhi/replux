import asyncio
from collections.abc import AsyncGenerator
from html import unescape

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static, TextArea
from textual.worker import WorkerCancelled

from replux.llm import LLMClient, LLMError, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp


async def test_tool_first_turn_has_no_empty_reply_or_spacing_before_the_tool() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield TextDelta("")
            yield TextDelta(" \n")
            yield ToolStarted("read", "read_file", "path=notes.txt", '{"path": "notes.txt"}')
            self.started.set()
            await self.release.wait()
            yield ToolFinished("read", "read_file", "path=notes.txt", ToolResult("success", "1\talpha"), 0.1, '{"path": "notes.txt"}')
            yield TextDelta("[bold]Plain reply[/bold]\n  continued")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read notes.txt"
        await pilot.press("enter")
        await asyncio.wait_for(client.started.wait(), timeout=2)
        await pilot.pause()
        user = app.query_one(".user", Static)
        heading = app.query_one(".tool-header", Static)
        assert not [reply for reply in app.query(".reply").results(Static) if reply.display]
        assert heading.region.y == user.region.bottom + 1
        assert heading.region.x == user.region.x
        assert app.focused is prompt

        client.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        reply = app.query_one(".reply", Static)
        assert reply.content == "[bold]Plain reply[/bold]\n  continued"
        assert reply.region.x == user.region.x
        assert reply.region.y == app.query_one(".tool-preview").region.bottom + 1
        assert "[bold]Plain reply[/bold]" in unescape(app.export_screenshot()).replace("\xa0", " ")


async def test_late_thinking_follows_its_input_even_when_reply_and_tools_are_already_visible() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.ready = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield TextDelta("Before the tool")
            yield ToolFinished("read", "read_file", "path=notes.txt", ToolResult("skipped", "Not run."), None, '{"path": "notes.txt"}')
            yield ThinkingDelta(" \n")
            self.ready.set()
            await self.release.wait()
            yield ThinkingDelta("Late thought")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "Think and read"
            await pilot.press("enter")
            await asyncio.wait_for(client.ready.wait(), timeout=2)
            await pilot.pause()
            assert not app.query(".thinking")
            user = app.query_one(".user", Static)
            reply = app.query_one(".reply", Static)
            heading = app.query_one(".tool-header", Static)
            assert user.region.bottom < reply.region.y < heading.region.y
            prompt.text = "unsent draft"
            client.release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            thinking = app.query_one(".thinking", Collapsible)
            assert thinking.title == "Thoughts"
            assert thinking.region.height == 1
            assert thinking.region.y == user.region.bottom + 1
            assert reply.region.y == thinking.region.bottom + 1
            assert heading.region.y == reply.region.bottom + 1
            assert app.focused is prompt
            assert prompt.text == "unsent draft"
        finally:
            client.release.set()


async def test_expanding_thinking_while_reading_keeps_position_focus_and_bottom_following() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.reached = [asyncio.Event(), asyncio.Event()]
            self.release = [asyncio.Event(), asyncio.Event()]

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield ThinkingDelta("".join(f"Thought {line}\n" for line in range(60)))
            yield TextDelta("Old reply\n" * 60)
            self.reached[0].set()
            await self.release[0].wait()
            yield ThinkingDelta("Additional thought\n")
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
            prompt.text = "Think and reply"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            transcript.focus()
            await pilot.press("home")
            await pilot.wait_for_scheduled_animations()
            reading_position = transcript.scroll_y
            thinking = app.query_one(".thinking", Collapsible)
            title_y = thinking.region.y
            await pilot.click(offset=(thinking.region.x + 1, title_y))
            await pilot.wait_for_scheduled_animations()
            title = app.focused
            assert not thinking.collapsed
            assert transcript.scroll_y == reading_position
            assert thinking.region.y == title_y
            assert not transcript.is_vertical_scroll_end

            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert transcript.scroll_y == reading_position
            assert app.focused is title
            prompt.text = "unsent draft"
            transcript.focus()
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            prompt.focus()
            client.release[1].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            assert app.focused is prompt
            assert prompt.text == "unsent draft"
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize("ending", ["success", "error", "cancel"])
async def test_thinking_is_one_line_and_becomes_thoughts_only_after_turn_cleanup(ending: str) -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.reached = [asyncio.Event(), asyncio.Event()]
            self.release = [asyncio.Event(), asyncio.Event()]
            self.closing = asyncio.Event()
            self.finish_cleanup = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            try:
                yield ThinkingDelta("[bold]First thought[/bold]\n")
                self.reached[0].set()
                await self.release[0].wait()
                yield ThinkingDelta("  Second thought")
                self.reached[1].set()
                await self.release[1].wait()
                if ending == "error":
                    raise LLMError("offline failure")
                yield ReplyCompleted("end_turn", None, None)
            finally:
                self.closing.set()
                await self.finish_cleanup.wait()

        async def aclose(self) -> None:
            pass

    client = ScriptedClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "Think first"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            thinking = app.query_one(".thinking", Collapsible)
            user = app.query_one(".user", Static)
            assert thinking.region.height == 1
            assert thinking.region.x == user.region.x
            assert thinking.region.y == user.region.bottom + 1
            assert app.screen.get_style_at(thinking.region.x, thinking.region.y).bgcolor == app.screen.get_style_at(0, 0).bgcolor
            assert app.focused is prompt
            prompt.text = "unsent draft"
            await pilot.press("shift+tab", "enter")
            title = app.focused
            assert isinstance(title, Static)
            assert str(title.content) == "- Thinking"
            assert not thinking.collapsed
            assert "[bold]First thought[/bold]" in unescape(app.export_screenshot()).replace("\xa0", " ")

            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert app.focused is title
            assert not thinking.collapsed
            assert app.query_one(".thinking-content", Static).content == "[bold]First thought[/bold]\n  Second thought"
            assert not app.query(".reply")
            worker = next(iter(app.workers))
            if ending == "cancel":
                await pilot.press("ctrl+c")
            else:
                client.release[1].set()
            await asyncio.wait_for(client.closing.wait(), timeout=2)
            await pilot.pause()
            assert thinking.title == "Thinking"
            assert not thinking.collapsed
            assert prompt.text == "unsent draft"
            assert client.inputs == ["Think first"]

            client.finish_cleanup.set()
            if ending == "cancel":
                with pytest.raises(WorkerCancelled):
                    await asyncio.wait_for(worker.wait(), timeout=2)
            else:
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert thinking.title == "Thoughts"
            assert str(title.content) == "- Thoughts"
            assert not thinking.collapsed
            assert app.focused is title
            await pilot.press("enter")
            assert thinking.collapsed
            assert thinking.region.height == 1
            assert prompt.text == "unsent draft"
            assert client.inputs == ["Think first"]
        finally:
            for release in client.release:
                release.set()
            client.finish_cleanup.set()


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_multiline_input_background_markers_and_body_columns_at_agreed_sizes(size: tuple[int, int]) -> None:
    user_input = "首行 [bold]literal[/bold]\n  second line\n" + "wrap " * 40
    reply_text = "[bold]Plain reply[/bold]\n" + "reply " * 30

    class ScriptedClient(LLMClient):
        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield ThinkingDelta("Visible reasoning")
            yield ToolStarted("read", "read_file", "path=notes.txt", '{"path": "notes.txt"}')
            yield ToolFinished("read", "read_file", "path=notes.txt", ToolResult("success", "1\talpha"), 0.1, '{"path": "notes.txt"}')
            yield TextDelta(reply_text)
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            pass

    app = RepluxApp(ScriptedClient(), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = user_input
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        user = app.query_one(".user", Static)
        reply = app.query_one(".reply", Static)
        heading = app.query_one(".tool-header", Static)
        thinking = app.query_one(".thinking", Collapsible)
        assert user.content == user_input
        assert user.region.height >= 4
        assert reply.content == reply_text
        assert reply.region.height >= 3
        assert user.region.x == reply.region.x == heading.region.x == thinking.region.x
        assert thinking.region.height == 1
        assert thinking.region.y == user.region.bottom + 1
        assert heading.region.y == thinking.region.bottom + 1
        assert reply.region.y == app.query_one(".tool-preview").region.bottom + 1

        background = app.screen.get_style_at(user.region.x, user.region.y).bgcolor
        foreground = app.screen.get_style_at(user.region.x, user.region.y).color
        assert background != app.screen.get_style_at(0, 0).bgcolor
        assert foreground == app.screen.get_style_at(reply.region.x, reply.region.y).color
        for y in range(user.region.y, user.region.bottom):
            for x in [user.region.x - 2, user.region.x, user.region.right - 1]:
                assert app.screen.get_style_at(x, y).bgcolor == background
        assert app.screen.get_style_at(reply.region.x - 2, reply.region.y).color == foreground
        assert app.screen.get_style_at(heading.region.x - 2, heading.region.y).color != foreground
        screenshot = unescape(app.export_screenshot()).replace("\xa0", " ")
        assert screenshot.count("›") == 2
        assert screenshot.count("●") == 2
        assert "[bold]literal[/bold]" in screenshot
        assert "[bold]Plain reply[/bold]" in screenshot
        assert app.focused is prompt
