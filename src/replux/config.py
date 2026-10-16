"""Read one provider configuration at startup."""

from collections.abc import Hashable
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from anthropic.types import ThinkingConfigParam
import yaml


DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS = 128_000
ThinkingEffort = Literal["low", "medium", "high", "xhigh", "max"]


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
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None
    thinking: ThinkingConfigParam | None = None
    thinking_effort: ThinkingEffort | None = None
    name: str | None = None


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
    name = None
    if "providers" in values:
        if values.keys() != {"providers"}:
            raise ConfigError("Only providers is supported at the top level.")
        providers = values["providers"]
        if not isinstance(providers, list) or len(providers) != 1:
            raise ConfigError("Exactly one provider must be active. Comment out the unused providers.")
        values = providers[0]
        if not isinstance(values, dict):
            raise ConfigError("Each provider must be a YAML mapping.")
        name = _required_string(values, "name")
    allowed = {"protocol", "model", "base_url", "api_key", "max_output_tokens", "timeout_seconds", "thinking"}
    if name is not None:
        allowed = {"name", "protocol", "model", "base_url", "api_key", "thinking"}
    if values.keys() - allowed:
        raise ConfigError(f"Configuration contains unknown fields. Supported fields: {', '.join(sorted(allowed))}.")

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

    max_tokens = values.get("max_output_tokens")
    if "max_output_tokens" in values and (type(max_tokens) is not int or max_tokens <= 0):
        raise ConfigError("max_output_tokens must be a positive integer.")
    timeout = values.get("timeout_seconds")
    if "timeout_seconds" in values:
        if type(timeout) not in {int, float}:
            raise ConfigError("timeout_seconds must be a finite positive number.")
        try:
            timeout = float(cast(int | float, timeout))
        except OverflowError:
            raise ConfigError("timeout_seconds must be a finite positive number.") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigError("timeout_seconds must be a finite positive number.")

    thinking = None
    effort = None
    if "thinking" in values:
        value = values["thinking"]
        if isinstance(value, str):
            if value not in {"auto", "low", "medium", "high", "xhigh", "max"}:
                raise ConfigError("thinking must be auto, low, medium, high, xhigh, or max.")
            effort = cast(ThinkingEffort, value) if value != "auto" else None
        else:
            if name is not None:
                raise ConfigError("thinking must be auto, low, medium, high, xhigh, or max.")
            if protocol == "openai":
                raise ConfigError("OpenAI thinking must be auto, low, medium, high, xhigh, or max; Anthropic configuration mappings are not supported.")
            thinking = _thinking_config(value, max_tokens or DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS)
    return ProviderConfig(protocol, model, base_url, api_key, max_tokens, timeout, thinking, effort, name)


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
