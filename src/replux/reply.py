"""Reply-only Markdown presentation; source text remains owned by the Turn."""

import asyncio
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.content import Content
from textual.highlight import HighlightTheme, highlight
from textual.layout import Layout
from textual.layouts.grid import GridLayout
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownBullet, MarkdownTable, MarkdownTableContent
from textual.widgets.markdown import MarkdownBlock, MarkdownFence


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


class ReplyHorizontal(MarkdownBlock, can_focus=True):
    """Use one keyboard and overflow contract for code and tables."""

    BINDINGS = [
        Binding("left", "scroll_horizontal(-4)", "Scroll left", show=False),
        Binding("right", "scroll_horizontal(4)", "Scroll right", show=False),
    ]

    def allow_focus(self) -> bool:
        return self.show_horizontal_scrollbar

    def action_scroll_horizontal(self, amount: int) -> None:
        self.scroll_relative(x=amount, animate=False)

    async def update_from(self, block: "ReplyHorizontal") -> None:
        raise NotImplementedError


class ReplyFence(ReplyHorizontal, MarkdownFence):
    def __init__(self, markdown: Markdown, token: Token, code: str) -> None:
        # Textual 8.2.8 strips trailing whitespace before constructing a fence.
        super().__init__(markdown, token, token.content)

    @classmethod
    def highlight(cls, code: str, language: str, ansi: bool = False, dark: bool = False) -> Content:
        lexer = language.split()[0] if language.strip() else "text"
        # Keep token offsets in the same unexpanded source as the displayed Content.
        styled = highlight(code, language=lexer, theme=ReplyCodeTheme, tab_size=0)
        # Keep real whitespace, excluding the final line terminator from layout.
        content = Content(code.removesuffix("\n"), spans=list(styled.spans))
        # Expand display tabs after styling so measurement and rendering use the same cells.
        return content.expand_tabs()

    async def update_from(self, block: ReplyHorizontal) -> None:
        assert isinstance(block, ReplyFence)
        self.code = block.code
        self.lexer = block.lexer
        self._highlighted_code = block._highlighted_code
        self.set_content(self._highlighted_code)


class ReplyTableContent(MarkdownTableContent):
    """Measure complete cells instead of shrinking them to the viewport."""

    def pre_layout(self, layout: Layout) -> None:
        assert isinstance(layout, GridLayout)
        layout.shrink = False
        layout.expand = False
        layout.stretch_height = True


class ReplyTable(ReplyHorizontal, MarkdownTable):
    def _get_headers_and_rows(self) -> tuple[list[Content], list[list[Content]]]:
        headers, rows = super()._get_headers_and_rows()
        return (
            [header.expand_tabs() for header in headers],
            [[cell.expand_tabs() for cell in row] for row in rows],
        )

    def compose(self) -> ComposeResult:
        headers, rows = self._get_headers_and_rows()
        yield ReplyTableContent(headers, rows)

    async def update_from(self, block: ReplyHorizontal) -> None:
        assert isinstance(block, ReplyTable)
        headers, rows = block._get_headers_and_rows()
        content = self.query_one(ReplyTableContent)
        if (content.headers, content.rows) != (headers, rows):
            content.headers, content.rows = headers, rows
            content.styles.grid_size_columns = len(headers)
            await content.recompose()


def _block_key(widget: Widget, index: int) -> tuple[type[Widget], int]:
    return type(widget), widget.source_range[0] if isinstance(widget, MarkdownBlock) else index


async def _update_children(parent: Widget, blocks: list[Widget]) -> None:
    """Keep source-matched blocks and their layout containers mounted."""
    previous = {_block_key(widget, index): widget for index, widget in enumerate(parent.children)}
    rendered = []
    for index, block in enumerate(blocks):
        current = previous.get(_block_key(block, index))
        if current is not None:
            if isinstance(current, MarkdownBlock):
                assert isinstance(block, MarkdownBlock)
                current.source_range = block.source_range
                current._token = block._token
                current._inline_token = block._inline_token
                if isinstance(current, ReplyHorizontal):
                    assert isinstance(block, ReplyHorizontal)
                    await current.update_from(block)
                else:
                    if current._content != block._content:
                        current.set_content(block._content)
                    await _update_children(current, [*block._pending_children, *block.compose()])
            else:
                # Textual 8.2.8 list rows contain layout widgets and separate bullets.
                if isinstance(current, MarkdownBullet):
                    assert isinstance(block, MarkdownBullet)
                    current.symbol = block.symbol
                await _update_children(current, [*block._pending_children, *block.compose()])
            block = current
        rendered.append(block)
    for widget in list(parent.children):
        if widget not in rendered:
            await widget.remove()
    before: Widget | None = None
    for widget in reversed(rendered):
        if not widget.is_attached:
            await parent.mount(widget, before=before)
        before = widget


class ReplyMarkdown(Markdown):
    BLOCKS = {
        **Markdown.BLOCKS,
        "fence": ReplyFence, "code_block": ReplyFence, "table_open": ReplyTable,
    }
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
        ReplyFence {
            margin: 0 0 1 0;
            color: #d2d2d2;
            background: $foreground 4%;
        }
        ReplyHorizontal {
            overflow: auto hidden;
            scrollbar-size-horizontal: 1;
            scrollbar-size-vertical: 0;
            scrollbar-color: $foreground 30%;
            scrollbar-background: $foreground 4%;
            scrollbar-color-hover: #c6b8eb 55%;
            scrollbar-color-active: #c6b8eb 65%;
        }
        ReplyHorizontal:focus {
            background-tint: #c6b8eb 8%;
            outline-left: solid #c6b8eb 50%;
            scrollbar-color: #c6b8eb 65%;
        }
        ReplyFence > Label { padding: 0 1; text-wrap: nowrap; }
        ReplyTable { margin-bottom: 1; }
        ReplyTableContent { width: auto; }
        ReplyTableContent > .cell, ReplyTableContent > .header {
            text-wrap: nowrap;
            text-overflow: clip;
        }
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

    async def update_reply(self, text: str) -> None:
        # Reparse the complete source so late reference definitions still resolve.
        tokens = await asyncio.to_thread(MarkdownIt("gfm-like").parse, text)
        self._markdown = text
        self._table_of_contents = None
        with self.app.batch_update():
            await _update_children(self, list(self._parse_markdown(tokens)))

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
        await self.body.update_reply(text)
