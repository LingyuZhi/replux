"""Full-screen text chat using the shared LLM interface."""

import asyncio
from contextlib import aclosing

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.worker import WorkerCancelled
from textual.widgets import Collapsible, Header, Input, Static

from replux.llm import IncompleteReplyError, LLMClient, LLMError, ReplyCompleted, TextDelta, ThinkingDelta


class RepluxApp(App[None]):
    TITLE = "Replux"
    BINDINGS = [Binding("ctrl+c", "cancel_or_quit", "Cancel / Quit", priority=True)]
    CSS = """
    #composer { dock: bottom; height: auto; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #transcript { height: 1fr; padding: 1 2; }
    .user { margin-top: 1; color: $accent; }
    .reply { margin: 1 0; }
    .thinking-content { color: $text-muted; }
    .turn-status { color: $warning; }
    """

    def __init__(self, client: LLMClient) -> None:
        super().__init__()
        self._client = client
        self._generating = False
        self._cancelling = False
        self._reply_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="composer"):
            yield Static("Ready", id="status")
            yield Input(placeholder="Type a message and press Enter to send", id="prompt")
        yield VerticalScroll(id="transcript")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._generating or not event.value.strip():
            return
        self._generating = True
        event.input.value = ""
        self.query_one("#status", Static).update("Generating...")
        self._run_reply(event.value)

    def action_cancel_or_quit(self) -> None:
        if not self._generating:
            self.exit()
        elif not self._cancelling:
            self._cancelling = True
            self.query_one("#status", Static).update("Cancelling...")
            if self._reply_task is not None:
                self._reply_task.cancel()

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
        transcript = self.query_one("#transcript", VerticalScroll)
        reply = Static("", classes="reply", markup=False)
        text = ""
        thinking_text = ""
        thinking: Static | None = None
        try:
            await transcript.mount(Static(user_input, classes="user", markup=False), reply)
            if self._cancelling:
                raise asyncio.CancelledError
            async with aclosing(self._client.stream_reply(user_input)) as events:
                async for event in events:
                    if isinstance(event, TextDelta):
                        text += event.text
                        reply.update(text)
                        transcript.scroll_end(animate=False)
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
                        transcript.scroll_end(animate=False)
                    elif isinstance(event, ReplyCompleted):
                        self.query_one("#status", Static).update("Ready")
        except asyncio.CancelledError:
            if self.is_running:
                await transcript.mount(Static("Cancelled", classes="turn-status", markup=False))
                transcript.scroll_end(animate=False)
            raise
        except LLMError as error:
            message = str(error) if isinstance(error, IncompleteReplyError) else f"Request failed: {error}"
            await transcript.mount(Static(message, classes="turn-status", markup=False))
            transcript.scroll_end(animate=False)

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
