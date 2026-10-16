from pathlib import Path
import json
import traceback
from typing import Any

import pytest

from replux.cli import create_app
from replux.config import ConfigError, load_config
from replux.llm import ReplyCompleted, create_client
from tests.support import ControlledSSE, RecordingTransport, reply_events


VALID_YAML = (
    "protocol: anthropic\n"
    "model: custom-model-20261001\n"
    "base_url: https://gateway.example/anthropic/v2\n"
    "api_key: test-secret-never-display\n"
)


PROVIDERS_YAML = "providers:\n  - name: example\n" + "".join(
    "    " + line for line in VALID_YAML.splitlines(keepends=True)
)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("effort", ["", "auto", "low", "medium", "high", "xhigh", "max"])
async def test_active_provider_and_thinking_effort_reach_sdk_request(
    tmp_path: Path, protocol: str, effort: str,
) -> None:
    path = tmp_path / "provider.yaml"
    document = PROVIDERS_YAML.replace("protocol: anthropic", f"protocol: {protocol}")
    if effort:
        document += f"    thinking: {effort}\n"
    document += "\n  # - name: inactive\n  #   protocol: unfinished\n  #   api_key:\n"
    path.write_text(document, encoding="utf-8")
    config = load_config(path)
    transport = RecordingTransport(ControlledSSE(reply_events(protocol)))

    async with create_client(config, transport=transport) as client:
        received = [event async for event in client.stream_reply("hello")]

    assert config.name == "example"
    assert config.protocol == protocol
    assert isinstance(received[-1], ReplyCompleted)
    assert len(transport.requests) == 1
    request = transport.requests[0]
    body = json.loads(request.content)
    assert body["model"] == "custom-model-20261001"
    assert "name" not in body
    assert "thinking" not in body
    effort_field = "output_config" if protocol == "anthropic" else "reasoning"
    if effort in {"", "auto"}:
        assert effort_field not in body
    else:
        assert body[effort_field] == {"effort": effort}
    if protocol == "anthropic":
        assert body["max_tokens"] == 128000
        assert request.headers["x-api-key"] == "test-secret-never-display"
    else:
        assert "max_output_tokens" not in body
        assert body["store"] is False
        assert request.headers["authorization"] == "Bearer test-secret-never-display"
    assert request.extensions["timeout"] == {
        "connect": 5, "read": 600, "write": 600, "pool": 600,
    }


@pytest.mark.parametrize(
    "document",
    [
        "providers: null\n", "providers: []\n", "providers: {}\n", "providers: [null]\n",
        "providers: [{name: first}, {name: second}]\n",
        PROVIDERS_YAML.replace("  - name: example\n    protocol:", "  - protocol:"),
        PROVIDERS_YAML.replace("name: example", "name: ''"),
        PROVIDERS_YAML.replace("name: example", "name: 1"),
        PROVIDERS_YAML.replace("    protocol: anthropic\n", ""),
        PROVIDERS_YAML + "protocol: anthropic\n",
        PROVIDERS_YAML + "    name: duplicate-secret\n",
        PROVIDERS_YAML + "    max_output_tokens: 1024\n",
        PROVIDERS_YAML + "    timeout_seconds: 60\n",
        *[
            PROVIDERS_YAML + f"    thinking: {value}\n"
            for value in ("null", "true", "false", "42", "[]", "test-secret-never-display", "{type: adaptive}")
        ],
    ],
)
def test_invalid_provider_list_fails_before_request_without_echoing_values(
    tmp_path: Path, document: str, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")
    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(SystemExit) as caught:
        create_app(["--config", str(path)], transport=transport)
    assert caught.value.code == 2
    assert transport.requests == []
    output = capsys.readouterr()
    assert "error:" in output.err
    assert "test-secret-never-display" not in output.err
    assert "duplicate-secret" not in output.err


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        ("", None),
        ("{type: adaptive}", {"type": "adaptive"}),
        ("{type: adaptive, display: summarized}", {"type": "adaptive", "display": "summarized"}),
        ("{type: enabled, budget_tokens: 1024}", {"type": "enabled", "budget_tokens": 1024}),
        ("{type: enabled, budget_tokens: 127999, display: summarized}", {"type": "enabled", "budget_tokens": 127999, "display": "summarized"}),
        ("{type: disabled}", {"type": "disabled"}),
    ],
)
async def test_thinking_configuration_reaches_request_without_changes(
    tmp_path: Path, thinking: str, expected: dict[str, Any] | None,
) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(VALID_YAML + (f"thinking: {thinking}\n" if thinking else ""), encoding="utf-8")
    transport = RecordingTransport(ControlledSSE())

    async with create_client(load_config(path), transport=transport) as client:
        received = [event async for event in client.stream_reply("hello")]

    assert received[-1] == ReplyCompleted("end_turn", 11, 5)
    assert len(transport.requests) == 1
    request = json.loads(transport.requests[0].content)
    assert request["model"] == "custom-model-20261001"
    assert request["max_tokens"] == 128000
    if expected is None:
        assert "thinking" not in request
    else:
        assert request["thinking"] == expected


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_config_preserves_connection_values_and_sets_defaults(tmp_path: Path, protocol: str) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(
        f"protocol: {protocol}\n"
        "model: custom-model-20261001\n"
        "base_url: https://gateway.example/anthropic/v2\n"
        "api_key: ${LITERAL_KEY}\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.protocol == protocol
    assert config.model == "custom-model-20261001"
    assert config.base_url == "https://gateway.example/anthropic/v2"
    assert config.api_key == "${LITERAL_KEY}"
    assert config.max_output_tokens is None
    assert config.timeout_seconds is None
    assert "${LITERAL_KEY}" not in repr(config)


@pytest.mark.parametrize(
    "document",
    [
        "",
        "- test-secret-never-display\n",
        "api_key: [test-secret-never-display\n",
        VALID_YAML + "api_key: duplicate-secret\n",
        VALID_YAML + "test-secret-never-display: unknown\n",
        VALID_YAML + "max_output_tokens: null\n",
        VALID_YAML + "max_output_tokens: true\n",
        VALID_YAML + "max_output_tokens: 0\n",
        VALID_YAML + "max_output_tokens: 1.5\n",
        VALID_YAML + "max_output_tokens: '100'\n",
        VALID_YAML + "timeout_seconds: null\n",
        VALID_YAML + "timeout_seconds: false\n",
        VALID_YAML + "timeout_seconds: 0\n",
        VALID_YAML + "timeout_seconds: -1\n",
        VALID_YAML + "timeout_seconds: .nan\n",
        VALID_YAML + "timeout_seconds: .inf\n",
        VALID_YAML + "timeout_seconds: '10'\n",
        VALID_YAML.replace("protocol: anthropic", "protocol: unknown"),
        VALID_YAML.replace("model: custom-model-20261001\n", ""),
        VALID_YAML.replace("model: custom-model-20261001", "model: '  '"),
        VALID_YAML.replace("model: custom-model-20261001", "model: 42"),
        VALID_YAML.replace("api_key: test-secret-never-display", "api_key: null"),
        VALID_YAML.replace("api_key: test-secret-never-display", "api_key: {}"),
        *[
            VALID_YAML.replace("api_key: test-secret-never-display", f"api_key: !!{tag} test-secret-never-display")
            for tag in ("int", "float", "bool", "timestamp")
        ],
        VALID_YAML.replace("api_key: test-secret-never-display", "api_key: !!int _"),
        *[
            VALID_YAML.replace("api_key: test-secret-never-display", f'api_key: "{value}"')
            for value in (
                "test-secret-never-display\\n",
                "test-secret-never-display\\r",
                "test-secret-never-display\\0",
                "test-secret-never-display\\x7f",
                "test-secret-never-display ",
                "中文-test-secret-never-display",
            )
        ],
        VALID_YAML.replace("https://gateway.example/anthropic/v2", "file:///tmp/key"),
        VALID_YAML.replace("https://gateway.example/anthropic/v2", "https://"),
        VALID_YAML.replace("https://gateway.example/anthropic/v2", "https://host:bad"),
        VALID_YAML.replace("https://gateway.example/anthropic/v2", "https://[broken"),
        VALID_YAML + "? [test-secret-never-display]\n: value\n",
        VALID_YAML + "thinking: {type: enabled}\n",
        *[
            VALID_YAML + f"thinking: {thinking}\n"
            for thinking in (
                "null", "false", "[]", "enabled", "{}",
                "{type: null}", "{type: []}", "{type: test-secret-never-display}",
                "{type: adaptive, test-secret-never-display: unknown}",
                "{type: adaptive, type: disabled}",
                "{type: adaptive, budget_tokens: 0}",
                "{type: adaptive, budget_tokens: null}",
                "{type: disabled, budget_tokens: 1024}",
                "{type: disabled, budget_tokens: null}",
                "{type: disabled, display: summarized}",
                "{type: disabled, display: null}",
                "{type: adaptive, display: omitted}",
                "{type: adaptive, display: null}",
                "{type: enabled, budget_tokens: 1024, display: []}",
                *[
                    f"{{type: enabled, budget_tokens: {budget}}}"
                    for budget in ("null", "true", "'1024'", "1024.0", "-1", "0", "1023", "128000", "128001")
                ],
            )
        ],
        VALID_YAML + "max_output_tokens: 1024\nthinking: {type: enabled, budget_tokens: 1024}\n",
    ],
)
def test_invalid_config_fails_without_echoing_values(
    tmp_path: Path, document: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_config(path)

    displayed = "".join(traceback.format_exception(caught.value))
    assert "test-secret-never-display" not in displayed
    assert "duplicate-secret" not in displayed

    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(SystemExit) as exit_error:
        create_app(["--config", str(path)], transport=transport)
    assert exit_error.value.code == 2
    assert transport.requests == []
    captured = capsys.readouterr()
    assert "test-secret-never-display" not in captured.err
    assert "duplicate-secret" not in captured.err
    assert "error:" in captured.err
    assert "\x1b[?1049h" not in captured.out


def test_missing_config_reports_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configuration file"):
        load_config(tmp_path / "missing.yaml")
    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(SystemExit) as exit_error:
        create_app(["--config", str(tmp_path / "missing.yaml")], transport=transport)
    assert exit_error.value.code == 2
    assert transport.requests == []


@pytest.mark.parametrize("thinking", ["{type: enabled, budget_tokens: 1024}", "{type: adaptive}", "{type: disabled}", "null"])
def test_openai_thinking_fails_before_chat_or_request(
    tmp_path: Path, thinking: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "openai.yaml"
    path.write_text(
        VALID_YAML.replace("protocol: anthropic", "protocol: openai") + f"thinking: {thinking}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="thinking"):
        load_config(path)
    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(SystemExit) as caught:
        create_app(["--config", str(path)], transport=transport)
    assert caught.value.code == 2
    assert transport.requests == []
    captured = capsys.readouterr()
    assert "thinking" in captured.err
    assert "test-secret-never-display" not in captured.err
    assert "\x1b[?1049h" not in captured.out


def test_explicit_config_replaces_default_and_is_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".replux" / "config.yaml"
    default.parent.mkdir()
    default.write_text(VALID_YAML, encoding="utf-8")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        VALID_YAML.replace("custom-model-20261001", "explicit-model")
        + "max_output_tokens: 2048\ntimeout_seconds: 2.5\n",
        encoding="utf-8",
    )

    first = load_config(explicit)
    assert load_config().model == "custom-model-20261001"
    assert first.model == "explicit-model"
    assert first.max_output_tokens == 2048
    assert first.timeout_seconds == 2.5

    explicit.write_text("protocol: anthropic\n", encoding="utf-8")
    assert first.model == "explicit-model"
    with pytest.raises(ConfigError):
        load_config(explicit)
