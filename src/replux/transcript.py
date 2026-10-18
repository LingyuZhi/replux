"""Keep visible Transcript content in place when streamed content changes height."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher

from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widgets import Markdown, Static
from textual.widgets.markdown import MarkdownBlock


@dataclass
class ReadingAnchor:
    widget: Static
    document: Markdown | None
    source_line: int | None
    leaf_index: int
    lines: tuple[str, ...]
    row: int
    screen_y: int


def _source_location(widget: Static) -> tuple[Markdown | None, int | None]:
    source_line = None
    for node in widget.ancestors_with_self:
        if isinstance(node, MarkdownBlock) and source_line is None:
            source_line = node.source_range[0]
        if isinstance(node, Markdown):
            return node, source_line
    return None, None


def _lines(widget: Static) -> tuple[str, ...]:
    return tuple(strip.text.strip() for strip in widget.render_lines(Region(0, 0, *widget.region.size)))


def _source_leaves(document: Markdown, source_line: int | None) -> list[Static]:
    return [
        widget for widget in document.query(Static)
        if not widget.children and _source_location(widget)[1] == source_line
    ]


class TranscriptScroll(VerticalScroll):
    """Follow the bottom, or retain a source-aware visible text anchor after layout."""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self._reading_anchors: list[ReadingAnchor] = []
        self._reading_scroll_y = 0.0

    @asynccontextmanager
    async def preserve_reading_position(self) -> AsyncIterator[None]:
        if self.is_vertical_scroll_end:
            self.anchor()
        elif not self._reading_anchors:
            self._reading_scroll_y = self.scroll_y
            self._reading_anchors = self._capture_reading_position()
        try:
            yield
        finally:
            task = asyncio.current_task()
            if self._reading_anchors and self.app.is_running and task is not None and not task.cancelling():
                await self.wait_for_refresh()
                self._restore_reading_position()
            else:
                self._reading_anchors = []

    def _capture_reading_position(self) -> list[ReadingAnchor]:
        viewport = self.scrollable_content_region
        anchors = []
        for widget in self.query(Static):
            if widget.children or not widget.is_on_screen or not widget.visible:
                continue
            region = widget.region
            if not region.overlaps(viewport):
                continue
            document, source_line = _source_location(widget)
            leaf_index = _source_leaves(document, source_line).index(widget) if document is not None else 0
            lines = _lines(widget)
            for row in range(max(0, viewport.y - region.y), min(region.height, viewport.bottom - region.y)):
                if len(lines[row]) >= 3:
                    anchors.append(ReadingAnchor(widget, document, source_line, leaf_index, lines, row, region.y + row))
        anchors.sort(key=lambda anchor: (anchor.screen_y, anchor.widget.region.x))
        return anchors[:6]

    def _restore_reading_position(self) -> None:
        anchors, self._reading_anchors = self._reading_anchors, []
        # A scroll during parsing takes precedence over the captured position.
        if self.scroll_y != min(self._reading_scroll_y, self.max_scroll_y):
            return
        fallback: tuple[int, int, int] | None = None
        for anchor in anchors:
            if anchor.document is None:
                candidates = [anchor.widget] if anchor.widget.is_attached else []
            else:
                candidates = _source_leaves(anchor.document, anchor.source_line)
                # Stable cells and list items take precedence over identical text elsewhere.
                candidates = candidates[anchor.leaf_index:anchor.leaf_index + 1] + candidates[:anchor.leaf_index] + candidates[anchor.leaf_index + 1:]
            for widget in candidates:
                if not widget.display or not widget.visible:
                    continue
                lines = _lines(widget)
                # Match the line sequence, so repeated rows retain their occurrence.
                for old_row, new_row, count in SequenceMatcher(None, anchor.lines, lines, autojunk=False).get_matching_blocks():
                    if old_row <= anchor.row < old_row + count:
                        self._restore_screen_y(widget.region.y + new_row + anchor.row - old_row, anchor.screen_y)
                        return
                text_before = anchor.lines[anchor.row]
                for row, text in enumerate(lines):
                    if not text:
                        continue
                    match = SequenceMatcher(None, text_before, text, autojunk=False).find_longest_match()
                    score = match.size
                    if score >= min(12, len(text_before)) and (fallback is None or score > fallback[0]):
                        fallback = score, widget.region.y + row, anchor.screen_y
        if fallback is not None:
            self._restore_screen_y(fallback[1], fallback[2])

    def _restore_screen_y(self, current_y: int, previous_y: int) -> None:
        self.scroll_to(
            y=self.scroll_y + current_y - previous_y,
            animate=False, immediate=True, release_anchor=False,
        )
