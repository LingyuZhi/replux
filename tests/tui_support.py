"""Observe the terminal output through the application's rendered screen."""

from textual.content import Content

from replux.tui import RepluxApp


def screen_lines(app: RepluxApp) -> list[str]:
    return [strip.text for strip in app.screen._compositor.render_strips()]


def screen_text(app: RepluxApp) -> str:
    return "\n".join(screen_lines(app))


def text_position(app: RepluxApp, text: str) -> tuple[int, int]:
    for y, line in enumerate(screen_lines(app)):
        if text in line:
            return Content(line[:line.index(text)]).cell_length, y
    raise AssertionError(f"Text is not visible: {text!r}\n{screen_text(app)}")
