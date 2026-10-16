"""Full-screen text chat using the shared LLM interface."""

import asyncio
from contextlib import aclosing
from importlib.metadata import version
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.worker import WorkerCancelled
from textual.widgets import Collapsible, Static, TextArea

from replux.llm import IncompleteReplyError, LLMClient, LLMError, ReplyCompleted, TextDelta, ThinkingDelta


SPARK_LOGO = "\n".join((
    "     ▄█▄     ",
    "  ▄▄█████▄▄  ",
    "████  █  ████",
    "  ▀▀█████▀▀  ",
    "     ▀█▀     ",
))


class Prompt(TextArea):
    async def on_key(self, event: Key) -> None:
        if event.key in {"enter", "ctrl+j"}:
            # Keep submission and newlines ordered with typing and paste events.
            event.stop()
            event.prevent_default()
            if event.key == "enter":
                await self.run_action("app.submit_prompt")
            else:
                self.replace("\n", *self.selection, maintain_selection_offset=False)


class RepluxApp(App[None]):
    TITLE = "Replux"
    BINDINGS = [Binding("ctrl+c", "cancel_or_quit", "Cancel / Quit", priority=True)]
    CSS = """
    #composer { dock: bottom; height: auto; }
    #prompt { height: auto; min-height: 3; max-height: 8; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #transcript { height: 1fr; padding: 1 2; }
    #transcript-content { height: auto; min-height: 100%; }
    #startup { height: 5; margin-bottom: 1; }
    #startup-logo { width: 13; height: 5; margin-right: 3; color: #c6b8eb; text-wrap: nowrap; }
    #startup-info { width: 1fr; height: 3; margin-top: 1; }
    #startup-info Static { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    #startup-model, #startup-directory { color: $text-muted; }
    .user { margin-top: 1; color: $accent; }
    .reply { margin: 1 0; }
    .thinking-content { color: $text-muted; }
    .turn-status { color: $warning; }
    """

    def __init__(self, client: LLMClient, *, model: str) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._generating = False
        self._cancelling = False
        self._reply_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="composer"):
            yield Static("Ready", id="status")
            yield Prompt(placeholder="Enter to send, Ctrl+J for a new line", id="prompt")
        directory = Path.cwd()
        home = Path.home()
        display_directory = str(Path("~") / directory.relative_to(home)) if directory.is_relative_to(home) else str(directory)
        with VerticalScroll(id="transcript"):
            with Vertical(id="transcript-content"):
                with Horizontal(id="startup"):
                    yield Static(SPARK_LOGO, id="startup-logo", markup=False)
                    with Vertical(id="startup-info"):
                        yield Static(f"[bold]Replux[/bold] [dim]v{version('replux')}[/dim]", id="startup-title")
                        yield Static(self._model, id="startup-model", markup=False)
                        yield Static(display_directory, id="startup-directory", markup=False)

    def on_mount(self) -> None:
        self.query_one("#transcript", VerticalScroll).anchor()
        self.query_one("#prompt", TextArea).focus()

    def action_submit_prompt(self) -> None:
        prompt = self.query_one("#prompt", TextArea)
        user_input = prompt.text
        if self._generating or not user_input.strip():
            return
        self._generating = True
        prompt.clear()
        self.query_one("#status", Static).update("Generating...")
        self._run_reply(user_input)

    def action_cancel_or_quit(self) -> None:
        if not self._generating:
            self.exit()
        elif not self._cancelling:
            self._cancelling = True
            self.query_one("#status", Static).update("Cancelling...")
            if self._reply_task is not None:
                self._reply_task.cancel()

    def _follow_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        if transcript.is_vertical_scroll_end:
            # Scrolling past the bottom can release Textual's anchor as well.
            transcript.anchor()

    @work(name="reply", description="Stream the current reply")
    async def _run_reply(self, user_input: str) -> None:
        reply = self._reply_task = asyncio.create_task(self._show_reply(user_input))
        try:
            await asyncio.shield(reply)
        except asyncio.CancelledError:
            # Textual also cancels workers on exit. Forward cancellation once so
            # exiting during an existing cancellation cannot interrupt cleanup.
            if not reply.cancelling():
                reply.cancel()
            while not reply.done():
                try:
                    await asyncio.shield(reply)
                except asyncio.CancelledError:
                    pass
            reply.result()
            raise
        finally:
            self._reply_task = None
            self._generating = False
            self._cancelling = False
            if self.is_running:
                self.query_one("#status", Static).update("Ready")

    async def _show_reply(self, user_input: str) -> None:
        transcript = self.query_one("#transcript-content", Vertical)
        reply = Static("", classes="reply", markup=False)
        text = ""
        thinking_text = ""
        thinking: Static | None = None
        try:
            self._follow_transcript()
            await transcript.mount(Static(user_input, classes="user", markup=False), reply)
            if self._cancelling:
                raise asyncio.CancelledError
            async with aclosing(self._client.stream_reply(user_input)) as events:
                async for event in events:
                    self._follow_transcript()
                    if isinstance(event, TextDelta):
                        text += event.text
                        reply.update(text)
                    elif isinstance(event, ThinkingDelta) and event.text:
                        thinking_text += event.text
                        if thinking is None:
                            if not thinking_text.strip():
                                continue
                            thinking = Static(thinking_text, classes="thinking-content", markup=False)
                            await transcript.mount(
                                Collapsible(
                                    thinking, title="Thinking", collapsed=True,
                                    collapsed_symbol="+", expanded_symbol="-", classes="thinking",
                                ),
                                before=reply,
                            )
                        else:
                            thinking.update(thinking_text)
                    elif isinstance(event, ReplyCompleted):
                        self.query_one("#status", Static).update("Ready")
        except asyncio.CancelledError:
            if self.is_running:
                self._follow_transcript()
                await transcript.mount(Static("Cancelled", classes="turn-status", markup=False))
            raise
        except LLMError as error:
            message = str(error) if isinstance(error, IncompleteReplyError) else f"Request failed: {error}"
            self._follow_transcript()
            await transcript.mount(Static(message, classes="turn-status", markup=False))

    async def on_unmount(self) -> None:
        try:
            # Textual already cancelled these workers; let stream cleanup finish.
            for worker in self.workers:
                try:
                    await worker.wait()
                except WorkerCancelled:
                    pass
        finally:
            await self._client.aclose()
