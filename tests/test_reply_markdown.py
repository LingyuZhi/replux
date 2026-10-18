import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
import re

import pytest
from textual.containers import VerticalScroll
from textual.worker import WorkerCancelled
from textual.widgets import TextArea

from replux.cli import create_app
from replux.llm import LLMClient, LLMError, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta, ToolFinished, ToolStarted
from replux.tools import ToolResult
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, reply_events
from tests.test_config import VALID_YAML
from tests.tui_support import screen_lines, screen_text, text_position


MARKDOWN = """# Compact heading

普通段落 mixed 中文。 **Strong** and *emphasis* with `inline_code`.

## Second heading

- First bullet
- Second bullet

1. First step
2. Second step

> Quoted words

[Web reference](https://example.com/docs)

```python
def greet():
    return "你好"
```

```
    plain code
```

```unknown-language
    unknown code
```

| Name | Value |
| --- | --- |
| 中文 | 42 |
"""


class ScriptedClient(LLMClient):
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.inputs: list[str] = []

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        self.inputs.append(user_input)
        for chunk in self.chunks:
            yield TextDelta(chunk)
        yield ReplyCompleted("end_turn", None, None)

    async def aclose(self) -> None:
        pass


class GatedClient(LLMClient):
    def __init__(self, events: list[StreamEvent], *, error: Exception | None = None) -> None:
        self.events = events
        self.reached = [asyncio.Event() for _ in events]
        self.release = [asyncio.Event() for _ in events]
        self.inputs: list[str] = []
        self.error = error

    async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        self.inputs.append(user_input)
        for index, event in enumerate(self.events):
            yield event
            self.reached[index].set()
            await self.release[index].wait()
        if self.error is not None:
            raise self.error
        yield ReplyCompleted("end_turn", None, None)

    async def aclose(self) -> None:
        pass


async def test_reply_renders_compact_markdown_and_preserves_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = RepluxApp(ScriptedClient([MARKDOWN]), model="scripted-model")
    async with app.run_test(size=(160, 50)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "**User input stays literal**"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()

        output = screen_text(app)
        assert "**User input stays literal**" in output
        assert "# Compact heading" not in output
        assert "**Strong**" not in output
        assert "*emphasis*" not in output
        assert "`inline_code`" not in output
        assert "```" not in output
        for text in ("普通段落 mixed 中文。", "First bullet", "Second bullet", "First step", "Second step",
                     "Quoted words", "Web reference", "def greet():", '    return "你好"',
                     "    plain code", "    unknown code", "Name", "Value", "42"):
            assert text in output
        assert "https://example.com/docs" not in output
        x, y = text_position(app, "Compact heading")
        assert x == 4
        assert app.screen.get_style_at(x, y).bold
        assert screen_lines(app)[y][2] == "●"
        assert app.screen.get_style_at(x, y).color != app.screen.get_style_at(*text_position(app, "Second heading")).color
        assert text_position(app, "普通段落")[1] == y + 2
        for text, attribute in (("Strong", "bold"), ("emphasis", "italic"), ("Web reference", "underline")):
            x, y = text_position(app, text)
            assert getattr(app.screen.get_style_at(x, y), attribute)
        link_color = app.screen.get_style_at(*text_position(app, "Web reference")).color
        assert link_color is not None and link_color.triplet == (198, 184, 235)
        x, y = text_position(app, "def greet():")
        assert app.screen.get_style_at(x, y).color != app.screen.get_style_at(x + 4, y).color
        assert app.screen.get_style_at(x, y).bgcolor != app.screen.get_style_at(0, 0).bgcolor
        assert app.focused is prompt


async def test_reply_code_keeps_indentation_and_blank_lines() -> None:
    source = "```python\n    first = 1  \n\n  \n    second = 2\n```"
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "Code whitespace"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        x, y = text_position(app, "first = 1")
        assert x == 9
        assert text_position(app, "second = 2") == (9, y + 3)


async def test_reply_code_highlighting_keeps_token_positions_after_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = RepluxApp(ScriptedClient(["```python\n\treturn 'x'\n```"]), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "Code with a tab"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        x, y = text_position(app, "return 'x'")
        keyword = app.screen.get_style_at(x, y).color
        string = app.screen.get_style_at(x + 8, y).color
        assert keyword is not None and keyword.triplet == (198, 184, 235)
        assert string is not None and string.triplet == (183, 201, 177)


async def test_only_web_links_open_the_default_browser_on_click(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url, *args, **kwargs: opened.append(url))
    source = """[Secure web](https://example.com/a?q=1#part) [Plain web](http://example.org/docs)

[Local path](src/replux/tui.py) [File URI](file:///tmp/notes.txt) [Email](mailto:hello@example.com)

| Reference |
| --- |
| [Table web](https://example.com/table) |
"""
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Links"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        prompt.text = "unsent draft"
        for label in ("Secure web", "Plain web", "Local path", "File URI", "Email", "Table web"):
            await pilot.click(offset=text_position(app, label))
        assert opened == ["https://example.com/a?q=1#part", "http://example.org/docs", "https://example.com/table"]
        assert prompt.text == "unsent draft"


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_markdown_reflow_keeps_the_visible_content_at_the_same_screen_position(
    size: tuple[int, int],
) -> None:
    reference = "ref" * 200
    source = f"Before [short label][{reference}]  \n" + "\n".join(
        f"Reading marker {number:03}  " for number in range(140)
    ) + "\n\n"
    client = GatedClient([
        TextDelta(source),
        TextDelta(f"[{reference}]: https://example.com/reference\n"),
        TextDelta("\n## Followed at bottom\n"),
    ])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            transcript = app.query_one("#transcript", VerticalScroll)
            prompt.text = "Reflow"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            assert transcript.is_vertical_scroll_end
            transcript.focus()
            await pilot.press("pageup")
            await pilot.wait_for_scheduled_animations()
            marker = next(line.strip() for line in screen_lines(app)[2:8] if "Reading marker" in line)
            position = text_position(app, marker)
            prompt.focus()
            await pilot.press(*"draft", "ctrl+j", *"第二行", "enter")
            assert client.inputs == ["Reflow"]
            assert text_position(app, marker) == position
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert text_position(app, marker) == position
            assert app.focused is prompt
            assert prompt.text == "draft\n第二行"
            assert not transcript.is_vertical_scroll_end

            transcript.focus()
            await pilot.press("end")
            await pilot.wait_for_scheduled_animations()
            client.release[1].set()
            await asyncio.wait_for(client.reached[2].wait(), timeout=2)
            await pilot.pause()
            assert "Followed at bottom" in screen_text(app)
            assert transcript.is_vertical_scroll_end
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_split_markdown_stream_matches_complete_source_at_the_application_boundary(
    size: tuple[int, int],
) -> None:
    chunks = [
        "## Stre", "am\n\n**str", "ong** [s", "ite](https://exa", "mple.com)\n\n- it",
        "em\n\n`", "``py", "thon\n    pr", "int('中')\n", "``", "`\n\n| A | B |\n| -",
        "-- | --- |\n| 一 | 2 |",
    ]
    client = GatedClient([TextDelta(chunk) for chunk in chunks])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "Stream"
            await pilot.press("enter")
            for index in range(len(chunks)):
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                await pilot.pause()
                assert app.focused is prompt
                assert "Finished in" not in screen_text(app)
                if index == 1:
                    assert "**str" in screen_text(app)
                if index == 4:
                    assert "strong site" in screen_text(app)
                    assert "https://" not in screen_text(app)
                if index == 7:
                    assert "    pr" in screen_text(app)
                if index == 8:
                    assert "    print('中')" in screen_text(app)
                client.release[index].set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            streamed = [re.sub(r"Finished in \d+\.\ds", "Finished in <elapsed>", line).rstrip() for line in screen_lines(app)]
        finally:
            for release in client.release:
                release.set()
    app = RepluxApp(ScriptedClient(["".join(chunks)]), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        app.query_one("#prompt", TextArea).text = "Stream"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        complete = [re.sub(r"Finished in \d+\.\ds", "Finished in <elapsed>", line).rstrip() for line in screen_lines(app)]
        assert streamed == complete
        assert "    print('中')" in screen_text(app)
        assert "```" not in screen_text(app)


@pytest.mark.parametrize("unclosed", ["```python\n    return 1\n", "**unclosed emphasis", "[unclosed link](https://example.com"])
async def test_tool_boundaries_keep_reply_formatting_independent_and_other_content_literal(unclosed: str) -> None:
    arguments = '{"path":"**literal**.txt"}'
    client = GatedClient([
        TextDelta(unclosed),
        ToolStarted("read", "read_file", "path=**literal**.txt", arguments),
        ToolFinished("read", "read_file", "path=**literal**.txt", ToolResult("success", "# Raw result\n**raw output**"), 0.1, arguments),
        TextDelta("## Independent heading\n\n**After the tool**"),
        ThinkingDelta("# Raw thinking\n**raw thought**"),
    ])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "# Raw user\n**raw input**"
            await pilot.press("enter")
            for index, release in enumerate(client.release):
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                await pilot.pause()
                release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert "Independent heading" in screen_text(app)
            assert "## Independent heading" not in screen_text(app)
            assert "**After the tool**" not in screen_text(app)
            x, y = text_position(app, "Independent heading")
            assert app.screen.get_style_at(x, y).bold
            assert text_position(app, "return 1" if unclosed.startswith("```") else unclosed)[1] < text_position(app, "Read **literal**.txt")[1] < y
            await pilot.click(offset=text_position(app, "Read **literal**.txt"))
            await pilot.press("shift+tab", "enter")
            await pilot.pause()
            output = screen_text(app)
            for text in ("# Raw user", "**raw input**", "# Raw thinking", "**raw thought**", arguments, "# Raw result", "**raw output**"):
                assert text in output
        finally:
            for release in client.release:
                release.set()


@pytest.mark.parametrize("ending", ["cancel", "error"])
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_interrupted_markdown_keeps_partial_reply_and_waits_for_real_cleanup(
    ending: str, size: tuple[int, int],
) -> None:
    source = "**Unfinished emphasis\n\n```python\n    print('partial')\n"

    class CleanupClient(GatedClient):
        def __init__(self) -> None:
            super().__init__([TextDelta(source)], error=LLMError("Connection lost."))
            self.cleaning = asyncio.Event()
            self.finish_cleanup = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            try:
                async for event in super().stream_reply(user_input):
                    yield event
            finally:
                self.cleaning.set()
                await self.finish_cleanup.wait()

    client = CleanupClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=size) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "Interrupt"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            assert "    print('partial')" in screen_text(app)
            if ending == "cancel":
                await pilot.press("ctrl+c")
            else:
                client.release[0].set()
            await asyncio.wait_for(client.cleaning.wait(), timeout=2)
            await pilot.press(*"draft", "ctrl+j", *"下一行", "enter")
            if ending == "cancel":
                await pilot.press("ctrl+c")
                assert "Cancelling…" in screen_text(app)
            assert "Finished in" not in screen_text(app)
            assert "Cancelled after" not in screen_text(app)
            assert "Failed after" not in screen_text(app)
            assert client.inputs == ["Interrupt"]
            client.finish_cleanup.set()
            if ending == "cancel":
                with pytest.raises(WorkerCancelled):
                    await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            else:
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            output = screen_text(app)
            assert "**Unfinished emphasis" in output
            assert "    print('partial')" in output
            assert ("Cancelled after" if ending == "cancel" else "Failed after") in output
            assert ("Request failed: Connection lost." in output) == (ending == "error")
            assert "Finished in" not in output
            assert prompt.text == "draft\n下一行"
            assert app.focused is prompt
        finally:
            client.release[0].set()
            client.finish_cleanup.set()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_markdown_display_preserves_the_original_reply_in_the_next_model_request(
    tmp_path: Path, protocol: str,
) -> None:
    source = "## Raw heading\n\n**Original** [web](https://example.com)\n\n```python\n    print('原文')  \n\n```\n"
    events = reply_events(protocol)
    if protocol == "anthropic":
        events[2]["delta"]["text"] = source[:23]
        events[3]["delta"]["text"] = source[23:]
        native = [{"role": "assistant", "content": [{"type": "text", "text": source}]}]
    else:
        events[4]["delta"] = source[:23]
        events[5]["delta"] = source[23:]
        events[6]["text"] = source
        events[-1]["response"]["output"][0]["content"][0]["text"] = source
        native = events[-1]["response"]["output"]
    stream = ControlledSSE(events, gate_after=len(events) - 2)
    transport = RecordingTransport([stream, ControlledSSE(reply_events(protocol))])
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    app = create_app(["--config", str(config)], transport=transport)
    async with app.run_test(size=(120, 40)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "first"
        await pilot.press("enter")
        await asyncio.wait_for(stream.blocked.wait(), timeout=2)
        await pilot.pause()
        assert "Raw heading" in screen_text(app)
        assert "## Raw heading" not in screen_text(app)
        assert "**Original**" not in screen_text(app)
        assert "    print('原文')" in screen_text(app)
        assert "Finished in" not in screen_text(app)
        stream.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [
        {"role": "user", "content": "first"}, *native,
        {"role": "user", "content": "follow-up"},
    ]
    assert transport.closed


async def test_late_thinking_and_open_tool_details_coexist_with_markdown_reflow() -> None:
    reference = "ref" * 200
    source = f"Before [short][{reference}]  \n" + "\n".join(f"Reading marker {i:03}  " for i in range(100)) + "\n\n"
    arguments = '{"path":"notes.txt"}'
    client = GatedClient([
        ToolStarted("read", "read_file", "path=notes.txt", arguments),
        ToolFinished("read", "read_file", "path=notes.txt", ToolResult("success", "# Raw output"), 0.1, arguments),
        TextDelta(source), ThinkingDelta("**Late thinking stays literal**"),
        TextDelta(f"[{reference}]: https://example.com\n"),
    ])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "Read and think"
            await pilot.press("enter")
            for index in (0, 1):
                await asyncio.wait_for(client.reached[index].wait(), timeout=2)
                client.release[index].set()
            await asyncio.wait_for(client.reached[2].wait(), timeout=2)
            await pilot.pause()
            transcript = app.query_one("#transcript", VerticalScroll)
            transcript.focus()
            await pilot.press("home")
            await pilot.wait_for_scheduled_animations()
            await pilot.click(offset=text_position(app, "Read notes.txt"))
            assert "# Raw output" in screen_text(app)
            tool_focus = app.focused
            await pilot.press("pagedown", "pagedown")
            await pilot.wait_for_scheduled_animations()
            marker = next(line.strip() for line in screen_lines(app)[2:8] if "Reading marker" in line)
            position = text_position(app, marker)
            client.release[2].set()
            await asyncio.wait_for(client.reached[3].wait(), timeout=2)
            await pilot.pause()
            assert text_position(app, marker) == position
            assert app.focused is tool_focus
            await pilot.press("tab", *"draft", "ctrl+j", *"second", "enter")
            assert app.focused is prompt
            assert prompt.text == "draft\nsecond"
            client.release[3].set()
            await asyncio.wait_for(client.reached[4].wait(), timeout=2)
            await pilot.pause()
            assert text_position(app, marker) == position
            assert app.focused is prompt
            assert client.inputs == ["Read and think"]
        finally:
            for release in client.release:
                release.set()


async def test_exit_during_markdown_reflow_waits_for_stream_cleanup() -> None:
    class ClosingClient(LLMClient):
        def __init__(self) -> None:
            self.ready = asyncio.Event()
            self.release = asyncio.Event()
            self.updating = asyncio.Event()
            self.cleaning = asyncio.Event()
            self.finish_cleanup = asyncio.Event()
            self.closed = False

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            try:
                yield TextDelta("Reading line  \n" * 100)
                self.ready.set()
                await self.release.wait()
                self.updating.set()
                yield TextDelta("\n## More content\n")
                await asyncio.Event().wait()
            finally:
                self.cleaning.set()
                await self.finish_cleanup.wait()

        async def aclose(self) -> None:
            self.closed = True

    client = ClosingClient()
    app = RepluxApp(client, model="scripted-model")

    async def session() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#prompt", TextArea).text = "Exit while reading"
            await pilot.press("enter")
            await asyncio.wait_for(client.ready.wait(), timeout=2)
            await pilot.pause()
            app.query_one("#transcript", VerticalScroll).focus()
            await pilot.press("pageup")
            await pilot.wait_for_scheduled_animations()
            client.release.set()
            await asyncio.wait_for(client.updating.wait(), timeout=2)
            app.exit()

    exiting = asyncio.create_task(session())
    try:
        await asyncio.wait_for(client.cleaning.wait(), timeout=3)
        assert not client.closed
        assert not exiting.done()
    finally:
        client.finish_cleanup.set()
        await asyncio.wait_for(exiting, timeout=3)
    assert client.closed


async def test_streaming_repeated_lines_keeps_the_same_occurrence_in_view() -> None:
    source = "\n".join("Repeated row  \n" * 4 + f"Unique landmark {i:02}  " for i in range(30))
    client = GatedClient([TextDelta(source), TextDelta("\nAdditional content  \n")])
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            app.query_one("#prompt", TextArea).text = "Repeated rows"
            await pilot.press("enter")
            await asyncio.wait_for(client.reached[0].wait(), timeout=2)
            await pilot.pause()
            app.query_one("#transcript", VerticalScroll).focus()
            await pilot.press("pageup")
            await pilot.wait_for_scheduled_animations()
            landmark = next(line.strip() for line in screen_lines(app)[2:10] if "Unique landmark" in line)
            position = text_position(app, landmark)
            client.release[0].set()
            await asyncio.wait_for(client.reached[1].wait(), timeout=2)
            await pilot.pause()
            assert text_position(app, landmark) == position
        finally:
            for release in client.release:
                release.set()


async def test_cancel_while_rendering_keeps_the_entire_returned_delta() -> None:
    class SendingClient(LLMClient):
        def __init__(self) -> None:
            self.ready = asyncio.Event()
            self.release = asyncio.Event()
            self.sending = asyncio.Event()

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            yield TextDelta("## Initial reply\n\n")
            self.ready.set()
            await self.release.wait()
            self.sending.set()
            yield TextDelta("Paragraph\n\n" * 40 + "Last returned words")
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            pass

    client = SendingClient()
    app = RepluxApp(client, model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#prompt", TextArea).text = "Cancel during rendering"
        await pilot.press("enter")
        await asyncio.wait_for(client.ready.wait(), timeout=2)
        worker = next(iter(app.workers))
        client.release.set()
        await asyncio.wait_for(client.sending.wait(), timeout=2)
        app.action_cancel_or_quit()
        with pytest.raises(WorkerCancelled):
            await asyncio.wait_for(worker.wait(), timeout=3)
        await pilot.pause()
        assert "Last returned words" in screen_text(app)
        assert "Cancelled after" in screen_text(app)
        assert "Finished in" not in screen_text(app)
