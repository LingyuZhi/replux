from pathlib import Path
import traceback

import pytest

from replux.cli import create_app
from replux.config import ConfigError, load_config
from tests.support import ControlledSSE, RecordingTransport


VALID_YAML = (
    "protocol: anthropic\n"
    "model: custom-model-20261001\n"
    "base_url: https://gateway.example/anthropic/v2\n"
    "api_key: test-secret-never-display\n"
)


def test_config_preserves_connection_values_and_sets_defaults(tmp_path: Path) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(
        "protocol: anthropic\n"
        "model: custom-model-20261001\n"
        "base_url: https://gateway.example/anthropic/v2\n"
        "api_key: ${LITERAL_KEY}\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.protocol == "anthropic"
    assert config.model == "custom-model-20261001"
    assert config.base_url == "https://gateway.example/anthropic/v2"
    assert config.api_key == "${LITERAL_KEY}"
    assert config.max_output_tokens == 8192
    assert config.timeout_seconds == 120
    assert "${LITERAL_KEY}" not in repr(config)


@pytest.mark.parametrize(
    "document",
    [
        "",
        "- test-secret-never-display\n",
        "api_key: [test-secret-never-display\n",
        VALID_YAML + "api_key: duplicate-secret\n",
        VALID_YAML + "test-secret-never-display: unknown\n",
        VALID_YAML + "max_output_tokens: true\n",
        VALID_YAML + "max_output_tokens: 0\n",
        VALID_YAML + "max_output_tokens: 1.5\n",
        VALID_YAML + "max_output_tokens: '100'\n",
        VALID_YAML + "timeout_seconds: false\n",
        VALID_YAML + "timeout_seconds: 0\n",
        VALID_YAML + "timeout_seconds: -1\n",
        VALID_YAML + "timeout_seconds: .nan\n",
        VALID_YAML + "timeout_seconds: .inf\n",
        VALID_YAML + "timeout_seconds: '10'\n",
        VALID_YAML.replace("protocol: anthropic", "protocol: unknown"),
        VALID_YAML.replace("protocol: anthropic", "protocol: openai"),
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
    with pytest.raises(ConfigError, match="配置"):
        load_config(tmp_path / "missing.yaml")
    transport = RecordingTransport(ControlledSSE())
    with pytest.raises(SystemExit) as exit_error:
        create_app(["--config", str(tmp_path / "missing.yaml")], transport=transport)
    assert exit_error.value.code == 2
    assert transport.requests == []


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
