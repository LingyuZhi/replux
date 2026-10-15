"""Full-screen text chat using the shared LLM interface."""

from contextlib import aclosing

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.worker import WorkerCancelled
from textual.widgets import Collapsible, Header, Input, Static

from replux.llm import IncompleteReplyError, LLMClient, LLMError, ReplyCompleted, TextDelta, ThinkingDelta


class RepluxApp(App[None]):
    TITLE = "Replux"
    BINDINGS = [Binding("ctrl+c", "quit", "退出", priority=True)]
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="composer"):
            yield Static("就绪", id="status")
            yield Input(placeholder="输入问题，按 Enter 发送", id="prompt")
        yield VerticalScroll(id="transcript")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._generating or not event.value.strip():
            return
        self._generating = True
        event.input.value = ""
        self.query_one("#status", Static).update("生成中…")
        self.run_worker(self._show_reply(event.value), name="reply")

    async def _show_reply(self, user_input: str) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        reply = Static("", classes="reply", markup=False)
        await transcript.mount(Static(user_input, classes="user", markup=False), reply)
        text = ""
        thinking_text = ""
        thinking: Static | None = None
        try:
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
                                    thinking, title="思考", collapsed=True,
                                    collapsed_symbol="+", expanded_symbol="-", classes="thinking",
                                ),
                                before=reply,
                            )
                        else:
                            thinking.update(thinking_text)
                        transcript.scroll_end(animate=False)
                    elif isinstance(event, ReplyCompleted):
                        self.query_one("#status", Static).update("就绪")
        except IncompleteReplyError as error:
            await transcript.mount(Static(str(error), classes="turn-status", markup=False))
            transcript.scroll_end(animate=False)
            self.query_one("#status", Static).update("就绪")
        except LLMError as error:
            self.exit(message=str(error), return_code=1)
        self._generating = False

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
