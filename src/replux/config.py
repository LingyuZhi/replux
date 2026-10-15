"""Read one provider configuration at startup."""

from collections.abc import Hashable
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from anthropic.types import ThinkingConfigParam
import yaml


class ConfigError(ValueError):
    """A startup error whose message contains no configuration values."""


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_scalar(self, node: yaml.ScalarNode | yaml.MappingNode) -> str:
        allowed = {"str", "int", "float", "bool", "null"}
        if node.tag.removeprefix("tag:yaml.org,2002:") not in allowed:
            raise ConfigError("Configuration contains an unsupported YAML scalar type.")
        return super().construct_scalar(node)

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Hashable, object]:
        if not isinstance(node, yaml.MappingNode):
            raise ConfigError("Configuration must be a YAML mapping.")
        result: dict[Hashable, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("Configuration field names must be strings.")
            if key in result:
                raise ConfigError("Configuration contains duplicate fields.")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class ProviderConfig:
    protocol: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    max_output_tokens: int = 8192
    timeout_seconds: float = 120
    thinking: ThinkingConfigParam | None = None


def load_config(path: Path | None = None) -> ProviderConfig:
    selected = path if path is not None else Path.home() / ".replux" / "config.yaml"
    try:
        with selected.open(encoding="utf-8") as source:
            values = yaml.load(source, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError):
        raise ConfigError("Cannot read the configuration file. Check the path, permissions, and UTF-8 encoding.") from None
    except ConfigError:
        raise
    except (yaml.YAMLError, ValueError, KeyError, IndexError):
        # PyYAML scalar constructors can raise built-in conversion errors.
        raise ConfigError("Configuration is not valid YAML.") from None

    if not isinstance(values, dict):
        raise ConfigError("Configuration must be a YAML mapping.")
    if values.get("protocol") == "openai" and "thinking" in values:
        raise ConfigError("OpenAI configuration does not support the Anthropic-only thinking field.")
    allowed = {"protocol", "model", "base_url", "api_key", "max_output_tokens", "timeout_seconds", "thinking"}
    if values.keys() - allowed:
        raise ConfigError("Configuration contains unknown fields. Supported fields: protocol, model, base_url, api_key, max_output_tokens, timeout_seconds, thinking.")

    protocol = _required_string(values, "protocol")
    if protocol not in {"anthropic", "openai"}:
        raise ConfigError("protocol must be anthropic or openai.")
    model = _required_string(values, "model")
    base_url = _required_string(values, "base_url")
    api_key = _required_string(values, "api_key")
    if api_key != api_key.strip() or any(not 32 <= ord(char) < 127 for char in api_key):
        raise ConfigError("api_key must be a valid ASCII header value, without control characters or leading or trailing whitespace.")
    try:
        url = urlsplit(base_url)
        valid_url = (
            url.scheme in {"http", "https"}
            and bool(url.hostname)
            and not any(character.isspace() for character in base_url)
            and url.username is None
            and not url.query
            and not url.fragment
        )
        url.port
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ConfigError("base_url must be a valid HTTP(S) base URL.")

    max_tokens = values.get("max_output_tokens", ProviderConfig.max_output_tokens)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ConfigError("max_output_tokens must be a positive integer.")
    timeout = values.get("timeout_seconds", ProviderConfig.timeout_seconds)
    if type(timeout) not in {int, float}:
        raise ConfigError("timeout_seconds must be a finite positive number.")
    try:
        timeout = float(timeout)
    except OverflowError:
        raise ConfigError("timeout_seconds must be a finite positive number.") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("timeout_seconds must be a finite positive number.")

    thinking = _thinking_config(values["thinking"], max_tokens) if "thinking" in values else None
    return ProviderConfig(protocol, model, base_url, api_key, max_tokens, timeout, thinking)


def _thinking_config(value: object, max_tokens: int) -> ThinkingConfigParam:
    if not isinstance(value, dict):
        raise ConfigError("thinking must be a mapping containing type.")
    if value.keys() - {"type", "budget_tokens", "display"}:
        raise ConfigError("thinking contains unknown fields. Supported fields: type, budget_tokens, display.")
    mode = value.get("type")
    if not isinstance(mode, str) or mode not in {"enabled", "adaptive", "disabled"}:
        raise ConfigError("thinking.type must be enabled, adaptive, or disabled.")
    if mode == "enabled":
        budget = value.get("budget_tokens")
        if type(budget) is not int or not 1024 <= budget < max_tokens:
            raise ConfigError("thinking.budget_tokens must be an integer of at least 1024 and less than max_output_tokens.")
    elif "budget_tokens" in value:
        raise ConfigError("thinking.budget_tokens can only be set in enabled mode.")
    if "display" in value:
        if mode == "disabled":
            raise ConfigError("thinking.display cannot be set in disabled mode.")
        if value["display"] != "summarized":
            raise ConfigError("thinking.display must be summarized.")
    return cast(ThinkingConfigParam, value)


def _required_string(values: dict[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string.")
    return value
