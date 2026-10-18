"""Reply-only Markdown presentation; source text remains owned by the Turn."""

from urllib.parse import urlsplit

from markdown_it.token import Token
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.highlight import HighlightTheme, highlight
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownFence


class ReplyCodeTheme(HighlightTheme):
    STYLES = {
        ("Comment",): "#92929c italic",
        ("Keyword",): "#c6b8eb",
        ("Name",): "#d2d2dc",
        ("Name", "Function"): "#b6cbe3",
        ("Literal", "String"): "#b7c9b1",
        ("Literal", "Number"): "#d6c2a3",
        ("Operator",): "#c5c5cf",
    }


class ReplyFence(MarkdownFence):
    def __init__(self, markdown: Markdown, token: Token, code: str) -> None:
        # Textual 8.2.8 strips trailing whitespace before constructing a fence.
        super().__init__(markdown, token, token.content)

    @classmethod
    def highlight(cls, code: str, language: str, ansi: bool = False, dark: bool = False) -> Content:
        lexer = language.split()[0] if language.strip() else "text"
        # Keep token offsets in the same unexpanded source as the displayed Content.
        styled = highlight(code, language=lexer, theme=ReplyCodeTheme, tab_size=0)
        # Keep real whitespace, excluding the final line terminator from layout.
        return Content(code.removesuffix("\n"), spans=list(styled.spans))


class ReplyMarkdown(Markdown):
    BLOCKS = {**Markdown.BLOCKS, "fence": ReplyFence, "code_block": ReplyFence}
    DEFAULT_CSS = """
    ReplyMarkdown {
        padding: 0;
        MarkdownBlock {
            link-color: #c6b8eb;
            link-style: underline;
            link-color-hover: #dfd4ff;
            link-background: transparent;
            link-background-hover: transparent;
        }
        MarkdownHeader {
            content-align: left top;
            color: $foreground;
            background: transparent;
            text-style: bold;
            margin: 0 0 1 0;
            width: 1fr;
        }
        MarkdownH1 { color: #c6b8eb; }
        MarkdownH3 { color: $text-muted; }
        MarkdownH4, MarkdownH5, MarkdownH6 { color: $text-muted; text-style: bold italic; }
        MarkdownBlockQuote {
            color: $text-muted;
            border-left: solid $foreground 25%;
            background: transparent;
            padding: 0 1;
            margin: 0 0 1 0;
        }
        MarkdownFence {
            margin: 0 0 1 0;
            background: $foreground 4%;
        }
        MarkdownFence > Label { padding: 0 1; }
        MarkdownTableContent {
            keyline: thin $foreground 20%;
        }
        MarkdownTableContent > .header { color: $foreground; text-style: bold; }
        MarkdownTableCellContents {
            link-color: #c6b8eb;
            link-style: underline;
        }
        & > MarkdownBlock:last-child { margin-bottom: 0; }
    }
    """

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        event.stop()
        location = urlsplit(event.href)
        if location.scheme.lower() in {"http", "https"} and location.netloc:
            self.app.open_url(event.href)


class ReplyMessage(Horizontal):
    """Keep the role marker separate from the rendered Reply body."""

    def __init__(self) -> None:
        super().__init__(classes="message reply-message")
        self.body = ReplyMarkdown(classes="reply", open_links=False)

    def compose(self) -> ComposeResult:
        yield Static("●", classes="message-marker", markup=False)
        yield self.body

    async def update(self, text: str) -> None:
        # Parse the real source, including incomplete syntax and reference links.
        await self.body.update(text)
