import asyncio

import pytest
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widget import Widget
from textual.widgets import TextArea

from replux.tui import RepluxApp
from tests.test_reply_markdown import ScriptedClient
from tests.tui_support import screen_text


SCENE = (
    "\n".join(f"History line {index:03d}  " for index in range(80))
    + "\n\n```python\nvalue = '" + "code/" * 60 + "CODE_END'\n```\n\n"
    + "| Name | Value |\n| --- | --- |\n| path | " + "table/" * 60 + "TABLE_END |\n"
)


def bar_points(bar: Widget) -> list[tuple[int, int]]:
    return [(x, y) for y in range(bar.region.y, bar.region.bottom) for x in range(bar.region.x, bar.region.right)]


def bar_brightness(app: RepluxApp, bar: Widget) -> list[int]:
    brightness = []
    for x, y in bar_points(bar):
        style = app.screen.get_style_at(x, y)
        assert not style.reverse
        assert style.color is not None and style.color.triplet is not None
        red, green, blue = style.color.triplet
        assert max(red, green, blue) - min(red, green, blue) <= 2
        brightness.append(red)
    return brightness


def thumb_point(app: RepluxApp, bar: Widget) -> tuple[int, int]:
    brightness = bar_brightness(app, bar)
    points = [point for point, value in zip(bar_points(bar), brightness) if value == max(brightness)]
    return points[len(points) // 2]


async def test_all_scrollbars_use_thin_neutral_strokes_in_single_cell_gutters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = RepluxApp(ScriptedClient([SCENE]), model="scripted-model")
    async with app.run_test(size=(120, 40)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read long content"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        prompt.text = "Draft line\n" * 15
        await pilot.pause()
        transcript = app.query_one("#transcript", VerticalScroll)
        assert transcript.show_vertical_scrollbar and prompt.show_vertical_scrollbar
        horizontal = [widget for widget in app.query("*") if widget.show_horizontal_scrollbar]
        assert len(horizontal) == 2
        for widget, vertical in [(transcript, True), (prompt, True), *((widget, False) for widget in horizontal)]:
            bar = widget.vertical_scrollbar if vertical else widget.horizontal_scrollbar
            assert (bar.region.width if vertical else bar.region.height) == 1
            strips = bar.render_lines(Region(0, 0, *bar.region.size))
            assert set("".join(strip.text for strip in strips)) == ({"▏"} if vertical else {"▁"})
            brightness = bar_brightness(app, bar)
            assert min(brightness) < max(brightness) < 180
        assert prompt.text == "Draft line\n" * 15 and app.focused is prompt


@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
@pytest.mark.parametrize("kind", ["code", "table"])
async def test_horizontal_track_clicks_and_drags_reach_both_ends_with_gray_feedback(
    size: tuple[int, int], kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    value = "READ_START_" + "path/" * 65 + "READ_END"
    source = f"```\n{value}\n```" if kind == "code" else f"| Value |\n| --- |\n| {value} |"
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=size) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read wide content"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        prompt.text = "Unsent draft"
        region = next(widget for widget in app.query("*") if widget.show_horizontal_scrollbar)
        bar = region.horizontal_scrollbar
        assert "READ_START_" in screen_text(app) and "READ_END" not in screen_text(app)
        geometry = bar.region
        idle = bar_brightness(app, bar)
        await pilot.hover(offset=thumb_point(app, bar))
        hovered = bar_brightness(app, bar)
        assert max(idle) < max(hovered) < 180
        assert min(hovered) == min(idle)

        await pilot.click(offset=(bar.region.right - 1, bar.region.y))
        await pilot.wait_for_scheduled_animations()
        assert region.scroll_x > 0
        assert "READ_START_" not in screen_text(app)
        await pilot.click(offset=(bar.region.x, bar.region.y))
        await pilot.wait_for_scheduled_animations()
        assert "READ_START_" in screen_text(app)

        await pilot.mouse_down(offset=thumb_point(app, bar))
        assert max(bar_brightness(app, bar)) > max(hovered)
        await pilot.hover(offset=(bar.region.right - 1, bar.region.y))
        await pilot.mouse_up(offset=(bar.region.right - 1, bar.region.y))
        await pilot.wait_for_scheduled_animations()
        assert "READ_END" in screen_text(app)
        await pilot.mouse_down(offset=thumb_point(app, bar))
        await pilot.hover(offset=(bar.region.x, bar.region.y))
        await pilot.mouse_up(offset=(bar.region.x, bar.region.y))
        await pilot.wait_for_scheduled_animations()
        assert "READ_START_" in screen_text(app)

        await pilot.hover("#prompt")
        await pilot.press("tab", "tab")
        assert app.focused is region
        assert max(bar_brightness(app, bar)) > max(idle)
        await pilot.press("tab")
        assert app.focused is prompt and prompt.text == "Unsent draft"
        assert max(bar_brightness(app, bar)) == max(idle)
        assert bar.region == geometry and region.show_horizontal_scrollbar


@pytest.mark.parametrize("target", ["transcript", "prompt"])
async def test_vertical_track_clicks_and_drags_keep_draft_and_focus(target: str) -> None:
    source = "\n".join(f"History line {index:03d}  " for index in range(80))
    app = RepluxApp(ScriptedClient([source]), model="scripted-model")
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "Read long history"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        draft = "\n".join(f"Draft line {index:03d}" for index in range(30))
        prompt.text = draft
        await pilot.pause()
        region = app.query_one("#transcript", VerticalScroll) if target == "transcript" else prompt
        bar = region.vertical_scrollbar
        start, end = ("History line 000", "History line 079") if target == "transcript" else ("Draft line 000", "Draft line 029")
        await pilot.mouse_down(offset=thumb_point(app, bar))
        await pilot.hover(offset=(bar.region.x, bar.region.y))
        await pilot.mouse_up(offset=(bar.region.x, bar.region.y))
        await pilot.wait_for_scheduled_animations()
        assert start in screen_text(app) and end not in screen_text(app)
        await pilot.click(offset=(bar.region.x, bar.region.bottom - 1))
        await pilot.wait_for_scheduled_animations()
        assert region.scroll_y > 0 and start not in screen_text(app)
        await pilot.mouse_down(offset=thumb_point(app, bar))
        bottom = (bar.region.x, app.screen.region.bottom - 1)
        await pilot.hover(offset=bottom)
        await pilot.mouse_up(offset=bottom)
        await pilot.wait_for_scheduled_animations()
        assert end in screen_text(app)
        assert prompt.text == draft and app.focused is prompt
