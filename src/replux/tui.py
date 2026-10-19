"""Full-screen text chat displaying the Agent's Turn events."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from importlib.metadata import version
import json
from math import cos, tau
from pathlib import Path
from time import monotonic

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.events import Click, Key
from textual.message import Message
from textual.timer import Timer
from textual.worker import WorkerCancelled
from textual.widgets import Collapsible, Static, TextArea

from replux.agent import AgentEvent
from replux.llm import (
    IncompleteReplyError, LLMClient, LLMError, TextDelta, ThinkingDelta,
    ToolBatchLimitError, ToolFinished, ToolStarted,
)
from replux.tools import ToolResult
from replux.reply import ReplyMessage
from replux.transcript import ThinScrollBarRender, TranscriptScroll


SPARK_LOGO = "\n".join((
    "     ▄█▄     ",
    "  ▄▄█████▄▄  ",
    "████  █  ████",
    "  ▀▀█████▀▀  ",
    "     ▀█▀     ",
))

_TOOL_STATUS_LABELS = {"success": "Success", "error": "Failed", "cancelled": "Cancelled", "skipped": "Skipped"}
_TOOL_ACTIONS = {
    "read_file": ("Read", "Reading…"), "write_file": ("Write", "Writing…"),
    "edit_file": ("Edit", "Editing…"), "find_files": ("Find", "Finding…"),
    "search_content": ("Search", "Searching…"), "run_command": ("Run", "Running…"),
}


class UserMessage(Horizontal):
    """Keep one user marker beside a literal body, including wrapped lines."""

    def __init__(self, text: str) -> None:
        super().__init__(classes="message user-message")
        self.body = Static(text, classes="user", markup=False)

    def compose(self) -> ComposeResult:
        yield Static("›", classes="message-marker", markup=False)
        yield self.body


class ThinkingView(Collapsible):
    """Toggle Thinking without scrolling the entire expanded block into view."""

    def _watch_collapsed(self, collapsed: bool) -> None:
        # Preserve the Transcript anchor and the user's current reading position.
        self._update_collapsed(collapsed)
        self.post_message(self.Collapsed(self) if collapsed else self.Expanded(self))


class ToolHeading(Horizontal, can_focus=True):
    """A whole-row detail toggle with the same mouse and keyboard behavior."""

    BINDINGS = [Binding("enter", "toggle_tool_details", "Show / hide tool details", show=False)]

    class Toggle(Message):
        pass

    def on_click(self, event: Click) -> None:
        event.stop()
        self.focus()
        self.action_toggle_tool_details()

    def action_toggle_tool_details(self) -> None:
        self.post_message(self.Toggle())


class ToolCallView(Vertical):
    """Keep a Tool Call's real arguments, bounded result, and display state together."""

    COMPONENT_CLASSES = {"tool-call--action"}

    def __init__(self, name: str, arguments: str, *, running: bool = True) -> None:
        super().__init__(name=name, classes="tool-call")
        self._tool_name = name
        self._arguments = arguments
        self._finished = False
        self._expanded = False
        self._started_at = monotonic() if running else None
        self._pulse_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with ToolHeading(classes="tool-heading"):
            yield Static("●", classes="tool-dot running" if self._started_at is not None else "tool-dot", markup=False)
            yield Static(_tool_title(self._tool_name, self._arguments), classes="tool-header", markup=False)
            yield Static("Running", classes="tool-status", markup=False)
            yield Static("▸", classes="tool-arrow", markup=False)
        yield Static("└ Running", classes="tool-preview", markup=False)
        with Vertical(classes="tool-details"):
            yield Static(f"Tool: {self._tool_name}\nStatus: Running", classes="tool-metadata", markup=False)
            yield Static("Arguments (raw JSON):", markup=False)
            yield Static(self._arguments, classes="tool-arguments", markup=False)
            yield Static("Tool Result (bounded):", markup=False)
            yield Static("", classes="tool-output", markup=False)
            yield Static("", classes="tool-result-details", markup=False)
            yield Static("", classes="tool-log", markup=False)

    def on_mount(self) -> None:
        self._update_title()
        if self._started_at is not None:
            self._started_at = monotonic()
            self._pulse_timer = self.set_interval(0.1, self._breathe)

    def _breathe(self) -> None:
        if self._started_at is not None:
            elapsed = monotonic() - self._started_at
            self.query_one(".tool-dot", Static).styles.text_opacity = 0.55 + 0.45 * (1 + cos(tau * elapsed / 3)) / 2

    def _update_title(self, result: ToolResult | None = None) -> None:
        title = _tool_title(self._tool_name, self._arguments, result)
        action_color = self.get_component_styles("tool-call--action").color.hex
        self.query_one(".tool-header", Static).update(
            Content(title).stylize(action_color, end=len(title.split(" ", 1)[0])),
        )

    def on_tool_heading_toggle(self, event: ToolHeading.Toggle) -> None:
        event.stop()
        self._expanded = not self._expanded
        self.query_one(".tool-details").display = self._expanded
        self.query_one(".tool-preview", Static).display = not self._expanded
        self.query_one(".tool-arrow", Static).update("▾" if self._expanded else "▸")

    def finish(self, result: ToolResult, duration_seconds: float | None) -> None:
        if self._finished:
            return
        self._finished = True
        self._started_at = None
        if self._pulse_timer is not None:
            self._pulse_timer.stop()
            self._pulse_timer = None
        dot = self.query_one(".tool-dot", Static)
        dot.styles.text_opacity = 1.0
        dot.remove_class("running")
        if result.status == "error":
            dot.add_class("error")
        status = _TOOL_STATUS_LABELS[result.status]
        self.set_class(result.status == "error", "failed")
        self._update_title(result)
        self.query_one(".tool-status", Static).update(status)
        self.query_one(".tool-preview", Static).update(f"└ {_tool_summary(self._tool_name, result)}")
        metadata = f"Tool: {self._tool_name}\nStatus: {status} ({result.status})"
        if result.error_code is not None:
            metadata += f"\nError code: {result.error_code}"
        if duration_seconds is not None:
            metadata += f"\nDuration: {duration_seconds:.2f}s"
        self.query_one(".tool-metadata", Static).update(metadata)
        self.query_one(".tool-output", Static).update(result.output)
        details = self.query_one(".tool-result-details", Static)
        details.update(json.dumps(result.details, ensure_ascii=False, indent=2) if result.details else "")
        details.display = bool(result.details)
        log_path = result.details.get("log_path")
        log = self.query_one(".tool-log", Static)
        log.display = isinstance(log_path, str) and bool(log_path)
        if isinstance(log_path, str) and log_path:
            log.update(log_path)

    def cancel(self) -> None:
        self.finish(ToolResult("cancelled", "Cancelled by the user before the tool call finished."), None)


def _tool_title(name: str, arguments: str, result: ToolResult | None = None, *, ongoing: bool = False) -> str:
    action = _TOOL_ACTIONS.get(name, (name, f"{name}…"))[1 if ongoing else 0]
    try:
        values = json.loads(arguments)
    except json.JSONDecodeError:
        return action
    if not isinstance(values, dict):
        return action
    key = "command" if name == "run_command" else "pattern" if name in {"find_files", "search_content"} else "path"
    target = values.get(key)
    title = f"{action} {target}" if isinstance(target, str) else action
    scope_key = "cwd" if name == "run_command" else "path"
    scope = values.get(scope_key)
    if not isinstance(scope, str) and result is not None:
        scope = result.details.get(scope_key)
    if key != "path" and isinstance(scope, str):
        title += f" in {scope}"
    return " ".join(title.split())


def _tool_summary(name: str, result: ToolResult) -> str:
    fallback = f"Failed ({result.error_code})" if result.status == "error" else _TOOL_STATUS_LABELS[result.status]
    summary = next((line.strip() for line in result.output.splitlines() if line.strip()), fallback)
    if result.status == "error" and result.error_code is not None and result.output.strip():
        reason = result.error_code.replace("_", " ").capitalize()
        if not summary.casefold().startswith(reason.casefold()):
            summary = f"{reason} · {summary}"
    details = result.details
    if result.status == "success":
        if name == "read_file":
            start, end = details.get("start_line"), details.get("end_line")
            if type(start) is int and type(end) is int:
                summary = f"Lines {start}–{end}"
        elif name == "write_file" and type(details.get("bytes")) is int:
            summary = f"Wrote {details['bytes']} bytes"
        elif name == "edit_file" and type(details.get("edits")) is int:
            noun = "operation" if details["edits"] == 1 else "operations"
            summary = f"Applied {details['edits']} replacement {noun}"
        elif name == "find_files":
            total, count = details.get("total"), details.get("count")
            if type(total) is int:
                summary = f"Found {total} files"
                if type(count) is int:
                    summary += f" · Returned {count}"
            elif type(count) is int:
                summary = f"Returned {count} files"
        elif name == "search_content" and type(details.get("count")) is int:
            noun = "line" if details["count"] == 1 else "lines"
            summary = f"Returned {details['count']} matching {noun}"
        elif name == "run_command" and type(details.get("exit_code")) is int:
            summary = f"Exited with code {details['exit_code']}"
    truncated = (
        bool(details.get("truncated_by")) or result.error_code == "output_limit_exceeded"
        or (name == "search_content" and any(line.endswith("... [line truncated]") for line in result.output.splitlines()))
    )
    return f"Truncated · {summary}" if truncated else summary


async def _show_after_cancel(
    events: AsyncGenerator[AgentEvent, None], show: Callable[[AgentEvent], Awaitable[None]],
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
    def on_mount(self) -> None:
        setattr(self.vertical_scrollbar, "renderer", ThinScrollBarRender)

    async def on_key(self, event: Key) -> None:
        if event.key in {"enter", "ctrl+j"}:
            # Keep submission and newlines ordered with typing and paste events.
            event.stop()
            event.prevent_default()
            if event.key == "enter":
                await self.run_action("app.submit_prompt")
            else:
                self.replace("\n", *self.selection, maintain_selection_offset=False)


class TurnActivity(Horizontal):
    """Show one local clock and animation for the whole Turn, including cleanup."""

    def __init__(self) -> None:
        super().__init__(id="activity")
        self._started_at: float | None = None
        self._timer: Timer | None = None
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="activity-dot", markup=False)
        yield Static("Ready", id="status", markup=False)
        yield Static("", id="activity-time", markup=False)
        yield Static("Ctrl+C to cancel", id="activity-hint", markup=False)

    def on_mount(self) -> None:
        self.stop()

    @property
    def elapsed(self) -> float:
        assert self._started_at is not None
        return monotonic() - self._started_at

    def start(self) -> None:
        self._started_at = monotonic()
        self._frame = 0
        for selector in ("#activity-dot", "#activity-time", "#activity-hint"):
            self.query_one(selector).display = True
        self.show_action("Working…")
        self._tick()
        self._timer = self.set_interval(0.1, self._tick)

    def show_action(self, action: str) -> None:
        self.query_one("#status", Static).update(action)

    def _tick(self) -> None:
        self.query_one("#activity-dot", Static).update("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._frame % 10])
        self._frame += 1
        self.query_one("#activity-time", Static).update(f"{self.elapsed:.1f}s")

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._started_at = None
        self.show_action("Ready")
        for selector in ("#activity-dot", "#activity-time", "#activity-hint"):
            self.query_one(selector).display = False


class RepluxApp(App[None]):
    TITLE = "Replux"
    BINDINGS = [Binding("ctrl+c", "cancel_or_quit", "Cancel / Quit", priority=True)]
    CSS = """
    #transcript, #prompt, ReplyHorizontal {
        scrollbar-size: 1 1;
        scrollbar-color: #626262;
        scrollbar-color-hover: #858585;
        scrollbar-color-active: #999999;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-corner-color: transparent;
    }
    #transcript:focus-within, #prompt:focus, ReplyHorizontal:focus {
        scrollbar-color: #858585;
    }
    #composer { dock: bottom; height: auto; }
    #prompt-entry { height: auto; border-top: solid $foreground 20%; padding: 0 2; }
    #prompt-marker { width: 2; height: 1; }
    #prompt { width: 1fr; height: auto; min-height: 1; max-height: 8; border: none; padding: 0; }
    #activity { height: 1; padding: 0 2; color: $text-muted; }
    #activity-dot { width: 2; color: #c6b8eb; }
    #status { width: 1fr; height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    #activity-time, #activity-hint { width: auto; margin-left: 2; height: 1; }
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
    .turn-outcome { margin: 1 0 0 2; color: $text-muted; }
    .tool-call { height: auto; margin-top: 1; }
    .tool-heading { height: 1; pointer: pointer; }
    .tool-heading:hover { background: $boost; }
    .tool-heading:focus { background: $boost; text-style: bold; }
    .tool-heading:focus .tool-arrow { color: $accent; }
    .tool-dot { width: 2; color: #d9a36f; }
    .tool-call .tool-call--action { color: #d9a36f; }
    .tool-call.failed .tool-call--action { color: $error; }
    .tool-dot.error { color: $error; }
    .tool-header { width: 1fr; margin-right: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    .tool-status { width: auto; color: $text-muted; }
    .tool-arrow { width: 2; margin-left: 1; color: $text-muted; }
    .tool-preview { height: 1; margin-left: 2; text-wrap: nowrap; text-overflow: ellipsis; }
    .tool-preview, .tool-details { color: $text-muted; }
    .tool-details { display: none; height: auto; margin-left: 2; }
    .tool-call.failed .tool-status, .tool-call.failed .tool-preview { color: $error; }
    .tool-log { height: auto; }
    """

    def __init__(self, agent: LLMClient, *, model: str) -> None:
        super().__init__()
        self._agent = agent
        self._model = model
        self._generating = False
        self._cancelling = False
        self._reply_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="composer"):
            yield TurnActivity()
            with Horizontal(id="prompt-entry"):
                yield Static("›", id="prompt-marker", markup=False)
                yield Prompt(placeholder="Enter to send, Ctrl+J for a new line", id="prompt")
        directory = Path.cwd()
        home = Path.home()
        display_directory = str(Path("~") / directory.relative_to(home)) if directory.is_relative_to(home) else str(directory)
        with TranscriptScroll(id="transcript"):
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
        self.query_one(TurnActivity).start()
        self._run_reply(user_input)

    def action_cancel_or_quit(self) -> None:
        if not self._generating:
            self.exit()
        elif not self._cancelling:
            self._cancelling = True
            self.query_one(TurnActivity).show_action("Cancelling…")
            if self._reply_task is not None:
                self._reply_task.cancel()

    def _follow_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        if transcript.is_vertical_scroll_end:
            # Scrolling past the bottom can release Textual's anchor as well.
            transcript.anchor()

    def _show_activity(self, action: str) -> None:
        if not self._cancelling:
            self.query_one(TurnActivity).show_action(action)

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
                self.query_one(TurnActivity).stop()

    async def _show_reply(self, user_input: str) -> None:
        transcript = self.query_one("#transcript-content", Vertical)
        user = UserMessage(user_input)
        reply: ReplyMessage | None = None
        text = ""
        thinking_text = ""
        thinking: Static | None = None
        thinking_control: Collapsible | None = None
        tools: dict[str, ToolCallView] = {}

        async def show(event: AgentEvent) -> None:
            if isinstance(event, TextDelta):
                # Finish displaying the returned delta before forwarding cancellation.
                rendering = asyncio.create_task(render_event(event))
                try:
                    await asyncio.shield(rendering)
                except asyncio.CancelledError:
                    await rendering
                    raise
            else:
                await render_event(event)

        async def render_event(event: AgentEvent) -> None:
            async with self.query_one(TranscriptScroll).preserve_reading_position():
                await show_event(event)

        async def show_event(event: AgentEvent) -> None:
            nonlocal reply, text, thinking, thinking_text, thinking_control
            self._follow_transcript()
            if isinstance(event, TextDelta):
                text += event.text
                if event.text.strip():
                    self._show_activity("Writing reply…")
                if reply is None:
                    if not text.strip():
                        return
                    reply = ReplyMessage()
                    await transcript.mount(reply)
                await reply.update(text)
            elif isinstance(event, ThinkingDelta) and event.text:
                thinking_text += event.text
                if event.text.strip():
                    self._show_activity("Thinking…")
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
                view = ToolCallView(event.name, event.arguments)
                tools[event.call_id] = view
                await transcript.mount(view)
                self._show_activity(_tool_title(event.name, event.arguments, ongoing=True))
                reply = None
                text = ""
            elif isinstance(event, ToolFinished):
                finished = tools.get(event.call_id)
                if finished is None:
                    finished = ToolCallView(event.name, event.arguments, running=False)
                    tools[event.call_id] = finished
                    await transcript.mount(finished)
                finished.finish(event.result, event.duration_seconds)
                self._show_activity("Working…")
                reply = None
                text = ""

        outcome: str | None = None
        try:
            self._follow_transcript()
            await transcript.mount(user)
            if self._cancelling:
                raise asyncio.CancelledError
            events = self._agent.stream_reply(user_input)
            async with aclosing(events):
                while True:
                    try:
                        event = await anext(events)
                    except StopAsyncIteration:
                        break
                    try:
                        await show(event)
                    except asyncio.CancelledError:
                        # The generator is paused on the event just shown. Throw cancellation
                        # there so an unstarted call stays skipped and the UI can render it.
                        await _show_after_cancel(events, show)
                        raise
            outcome = "Finished in"
        except asyncio.CancelledError:
            outcome = "Cancelled after"
            for view in tools.values():
                view.cancel()
            raise
        except LLMError as error:
            outcome = "Failed after"
            if isinstance(error, (IncompleteReplyError, ToolBatchLimitError)):
                message = str(error)
            else:
                message = f"Request failed: {error}"
            self._follow_transcript()
            await transcript.mount(Static(message, classes="turn-status", markup=False))
        finally:
            if thinking_control is not None:
                thinking_control.title = "Thoughts"
            if outcome is not None and self.is_running:
                elapsed = self.query_one(TurnActivity).elapsed
                self._follow_transcript()
                await transcript.mount(Static(f"{outcome} {elapsed:.1f}s", classes="turn-outcome", markup=False))

    async def on_unmount(self) -> None:
        try:
            # Textual already cancelled these workers; let stream cleanup finish.
            for worker in self.workers:
                try:
                    await worker.wait()
                except WorkerCancelled:
                    pass
        finally:
            await self._agent.aclose()
