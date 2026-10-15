import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path

import pytest
import httpx2 as httpx
from textual.widgets import Input, Static

from replux.cli import create_app
from replux.llm import LLMClient, ReplyCompleted, StreamEvent, TextDelta
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport
from tests.test_config import VALID_YAML


async def test_yaml_to_sdk_to_visible_reply_before_terminal_event(tmp_path: Path) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML, encoding="utf-8")
    stream = ControlledSSE(gate_after=5)
    transport = RecordingTransport(stream)
    app = create_app(["--config", str(config)], transport=transport)

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", Input)
        assert prompt.region.bottom == 24
        assert app.query_one("#status", Static).region.bottom <= prompt.region.y
        prompt.value = "  "
        await pilot.press("enter")
        assert transport.requests == []
        assert not app.query(".reply")

        prompt.value = "你好，Replux"
        await pilot.press("enter")
        await asyncio.wait_for(stream.blocked.wait(), timeout=2)
        await pilot.pause()
        assert app.query_one(".reply", Static).content == "你好，世界\n"
        assert app.query_one("#status", Static).content == "生成中…"
        assert not stream.closed
        assert len(transport.requests) == 1

        prompt.value = "下一条草稿"
        await pilot.press("enter")
        assert len(transport.requests) == 1
        assert prompt.value == "下一条草稿"

        stream.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert stream.closed
        assert app.query_one("#status", Static).content == "就绪"
        assert app.query_one(".reply", Static).content == "你好，世界\n"
        await pilot.press("ctrl+c")

    assert transport.closed


@pytest.mark.parametrize("explicit", [False, True])
async def test_selected_yaml_controls_actual_request_until_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".replux" / "config.yaml"
    default.parent.mkdir()
    default.write_text("invalid: default-must-not-be-merged\n", encoding="utf-8")
    selected = tmp_path / "explicit.yaml" if explicit else default
    selected.write_text(
        "protocol: anthropic\n"
        "model: selected-full-model-id\n"
        "base_url: https://selected.example/custom-root\n"
        "api_key: ${LITERAL_SELECTED_KEY}\n"
        "max_output_tokens: 2048\n"
        "timeout_seconds: 7.5\n",
        encoding="utf-8",
    )
    transport = RecordingTransport(ControlledSSE())
    argv = ["--config", str(selected)] if explicit else []
    app = create_app(argv, transport=transport)
    selected.write_text("protocol: changed-after-startup\n", encoding="utf-8")

    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "one question"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert app.query_one(".reply", Static).content == "你好，世界\n"
        await pilot.press("ctrl+c")

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == "https://selected.example/custom-root/v1/messages"
    assert request.headers["x-api-key"] == "${LITERAL_SELECTED_KEY}"
    assert json.loads(request.content)["model"] == "selected-full-model-id"
    assert json.loads(request.content)["max_tokens"] == 2048
    assert request.extensions["timeout"] == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}
    assert transport.closed

    with pytest.raises(SystemExit):
        create_app(argv, transport=transport)


async def test_tui_accepts_a_scripted_llm_client_and_preserves_input() -> None:
    class ScriptedClient(LLMClient):
        def __init__(self) -> None:
            self.inputs: list[str] = []
            self.closed = False

        async def stream_reply(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
            self.inputs.append(user_input)
            yield TextDelta("[bold]这仍是正文[/bold]\n")
            yield ReplyCompleted("end_turn", None, None)

        async def aclose(self) -> None:
            self.closed = True

    client = ScriptedClient()
    app = RepluxApp(client)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "   "
        await pilot.press("enter")
        assert client.inputs == []
        prompt.value = "  [red]原样提问[/red]  "
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert client.inputs == ["  [red]原样提问[/red]  "]
        assert app.query_one(".reply", Static).content == "[bold]这仍是正文[/bold]\n"
        await pilot.press("ctrl+c")
    assert client.closed


async def test_request_error_exits_without_rendering_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML, encoding="utf-8")

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"type": "authentication_error", "message": "rejected test-secret-never-display"}})

    app = create_app(["--config", str(config)], transport=httpx.MockTransport(reject))
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "hello"
        await pilot.press("enter")
    assert app.return_code == 1
    captured = capsys.readouterr()
    assert "test-secret-never-display" not in captured.out + captured.err


async def test_exit_waits_for_in_progress_stream_cleanup(tmp_path: Path) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML, encoding="utf-8")
    stream = ControlledSSE(gate_after=0)
    stream.release_close.clear()
    transport = RecordingTransport(stream)
    app = create_app(["--config", str(config)], transport=transport)

    async def session() -> None:
        async with app.run_test() as pilot:
            app.query_one("#prompt", Input).value = "hello"
            await pilot.press("enter")
            await asyncio.wait_for(stream.blocked.wait(), timeout=2)
            app.exit()

    exiting = asyncio.create_task(session())
    try:
        await asyncio.wait_for(stream.closing.wait(), timeout=2)
        assert not stream.closed
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(exiting), timeout=0.1)
    finally:
        stream.release_close.set()
        await asyncio.wait_for(exiting, timeout=2)
    assert stream.closed
    assert transport.closed
