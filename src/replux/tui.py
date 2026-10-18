"""Full-screen text chat using the shared LLM interface."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.timer import Timer
from textual.worker import WorkerCancelled
from textual.widgets import Collapsible, Static, TextArea

from replux.llm import (
    IncompleteReplyError, LLMClient, LLMError, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta,
    ToolBatchLimitError, ToolFinished, ToolStarted,
)
from replux.tools import ToolResult


SPARK_LOGO = "\n".join((
    "     ▄█▄     ",
    "  ▄▄█████▄▄  ",
    "████  █  ████",
    "  ▀▀█████▀▀  ",
    "     ▀█▀     ",
))


class TranscriptMessage(Horizontal):
    """Keep one marker beside a plain-text body, including wrapped lines."""

    def __init__(self, text: str, *, role: Literal["user", "reply"]) -> None:
        super().__init__(classes=f"message {role}-message")
        self.body = Static(text, classes=role, markup=False)
        self._marker = "›" if role == "user" else "●"

    def compose(self) -> ComposeResult:
        yield Static(self._marker, classes="message-marker", markup=False)
        yield self.body


class ThinkingView(Collapsible):
    """Toggle Thinking without scrolling the entire expanded block into view."""

    def _watch_collapsed(self, collapsed: bool) -> None:
        # Preserve the Transcript anchor and the user's current reading position.
        self._update_collapsed(collapsed)
        self.post_message(self.Collapsed(self) if collapsed else self.Expanded(self))


class ToolCallView(Vertical):
    """One Tool Call in the Transcript: running indicator, then a short preview."""

    def __init__(self, name: str, summary: str) -> None:
        super().__init__(name=name, classes="tool-call")
        self._name = name
        self._summary = summary
        self._timer: Timer | None = None
        self._bright = True
        self._finished = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="tool-heading"):
            yield Static("●", classes="tool-dot running", markup=False)
            yield Static(f"{self._name} {self._summary}", classes="tool-header", markup=False)
            yield Static("running", classes="tool-status", markup=False)
        yield Static("", classes="tool-preview", markup=False)

    def on_mount(self) -> None:
        if not self._finished:
            self._timer = self.set_interval(0.5, self._breathe)

    def _breathe(self) -> None:
        self._bright = not self._bright
        self.query_one(".tool-dot", Static).styles.text_opacity = 1.0 if self._bright else 0.3

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        self.query_one(".tool-preview", Static).display = False
        event.stop()

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        self.query_one(".tool-preview", Static).display = True
        event.stop()

    def finish(self, result: ToolResult, duration_seconds: float | None) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        dot = self.query_one(".tool-dot", Static)
        dot.remove_class("running")
        if result.status == "error":
            dot.add_class("error")
        dot.styles.text_opacity = 1.0
        self.query_one(".tool-status", Static).update(_tool_status_text(result, duration_seconds))
        lines = result.output.split("\n")
        if lines[-1:] == [""]:
            lines = lines[:-1]
        shown = lines[-5:] if self._name == "run_command" else lines[:5]
        self.query_one(".tool-preview", Static).update("\n".join(shown))
        log_path = result.details.get("log_path")
        if isinstance(log_path, str) and log_path:
            self.mount(Static(log_path, classes="tool-log", markup=False))
        if len(lines) > 5:
            self.mount(Collapsible(
                Static(result.output, classes="tool-output", markup=False),
                title=f"Show all {len(lines)} lines", collapsed=True,
                collapsed_symbol="+", expanded_symbol="-",
            ))

    def cancel(self) -> None:
        self.finish(ToolResult("cancelled", "Cancelled by the user before the tool call finished."), None)


def _tool_status_text(result: ToolResult, duration_seconds: float | None) -> str:
    label = f"error ({result.error_code})" if result.status == "error" else result.status
    return label if duration_seconds is None else f"{label} · {duration_seconds:.2f}s"


async def _show_after_cancel(
    events: AsyncGenerator[StreamEvent, None], show: Callable[[StreamEvent], Awaitable[None]],
) -> None:
    """Ask a paused reply to publish terminal tool events, then render them."""
    try:
        event = await events.athrow(asyncio.CancelledError())
    except (asyncio.CancelledError, StopAsyncIteration):
        return
    await show(event)
    while True:
        try:
            event = await anext(events)
        except (asyncio.CancelledError, StopAsyncIteration):
            return
        await show(event)


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
    #startup { height: 5; }
    #startup-logo { width: 13; height: 5; margin-right: 3; color: #c6b8eb; text-wrap: nowrap; }
    #startup-info { width: 1fr; height: 3; margin-top: 1; }
    #startup-info Static { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    #startup-model, #startup-directory { color: $text-muted; }
    .message { height: auto; margin-top: 1; color: $foreground; }
    .message-marker { width: 2; height: 1; }
    .user-message { background: $foreground 7%; }
    .user, .reply { width: 1fr; }
    .thinking {
        margin: 1 0 0 2; padding: 0; border: none;
        background: transparent;
    }
    .thinking:focus-within { background-tint: transparent; }
    .thinking > CollapsibleTitle {
        height: 1; padding: 0; color: $text-muted;
        background: transparent; text-style: none;
    }
    .thinking > CollapsibleTitle:hover { color: $text; }
    .thinking > CollapsibleTitle:focus { color: $accent; background: $boost; text-style: bold; }
    .thinking > Contents { padding: 0; }
    .thinking-content { color: $text-muted; }
    .turn-status { margin-top: 1; color: $warning; }
    .tool-call { height: auto; margin-top: 1; }
    .tool-heading { height: 1; }
    .tool-dot { width: 2; color: $text-muted; }
    .tool-dot.error { color: $error; }
    .tool-header { width: auto; margin-right: 1; }
    .tool-status { width: 1fr; color: $text-muted; text-wrap: nowrap; text-overflow: ellipsis; }
    .tool-preview, .tool-output, .tool-log { color: $text-muted; }
    .tool-log { height: auto; }
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
        user = TranscriptMessage(user_input, role="user")
        reply: TranscriptMessage | None = None
        text = ""
        thinking_text = ""
        thinking: Static | None = None
        thinking_control: Collapsible | None = None
        tools: dict[str, ToolCallView] = {}

        async def show(event: StreamEvent) -> None:
            nonlocal reply, text, thinking, thinking_text, thinking_control
            self._follow_transcript()
            if isinstance(event, TextDelta):
                text += event.text
                if reply is None:
                    if not text.strip():
                        return
                    reply = TranscriptMessage(text, role="reply")
                    await transcript.mount(reply)
                else:
                    reply.body.update(text)
            elif isinstance(event, ThinkingDelta) and event.text:
                thinking_text += event.text
                if thinking is None:
                    if not thinking_text.strip():
                        return
                    thinking = Static(thinking_text, classes="thinking-content", markup=False)
                    thinking_control = ThinkingView(
                        thinking, title="Thinking", collapsed=True,
                        collapsed_symbol="+", expanded_symbol="-", classes="thinking",
                    )
                    await transcript.mount(
                        thinking_control,
                        after=user,
                    )
                else:
                    thinking.update(thinking_text)
            elif isinstance(event, ToolStarted):
                view = ToolCallView(event.name, event.summary)
                tools[event.call_id] = view
                await transcript.mount(view)
                self.query_one("#status", Static).update(f"Running {event.name}...")
                reply = None
                text = ""
            elif isinstance(event, ToolFinished):
                finished = tools.get(event.call_id)
                if finished is None:
                    finished = ToolCallView(event.name, event.summary)
                    tools[event.call_id] = finished
                    await transcript.mount(finished)
                finished.finish(event.result, event.duration_seconds)
                if not self._cancelling:
                    self.query_one("#status", Static).update("Generating...")
                reply = None
                text = ""
            elif isinstance(event, ReplyCompleted):
                self.query_one("#status", Static).update("Ready")

        events: AsyncGenerator[StreamEvent, None] | None = None
        try:
            self._follow_transcript()
            await transcript.mount(user)
            if self._cancelling:
                raise asyncio.CancelledError
            events = self._client.stream_reply(user_input)
            while True:
                try:
                    event = await anext(events)
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                try:
                    await show(event)
                except asyncio.CancelledError:
                    # The generator is paused on the event just shown. Throw cancellation
                    # there so an unstarted call stays skipped and the UI can render it.
                    await _show_after_cancel(events, show)
                    raise
        except asyncio.CancelledError:
            for view in tools.values():
                view.cancel()
            if self.is_running:
                self._follow_transcript()
                await transcript.mount(Static("Cancelled", classes="turn-status", markup=False))
            raise
        except LLMError as error:
            if isinstance(error, (IncompleteReplyError, ToolBatchLimitError)):
                message = str(error)
            else:
                message = f"Request failed: {error}"
            self._follow_transcript()
            await transcript.mount(Static(message, classes="turn-status", markup=False))
        finally:
            try:
                if events is not None:
                    await events.aclose()
            finally:
                if thinking_control is not None:
                    thinking_control.title = "Thoughts"

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
