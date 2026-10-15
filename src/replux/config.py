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
            raise ConfigError("配置包含不支持的 YAML 标量类型。")
        return super().construct_scalar(node)

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Hashable, object]:
        if not isinstance(node, yaml.MappingNode):
            raise ConfigError("配置必须是 YAML 字段映射。")
        result: dict[Hashable, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("配置字段名必须是字符串。")
            if key in result:
                raise ConfigError("配置含重复字段。")
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
        raise ConfigError("无法读取配置文件；请检查路径、权限和 UTF-8 编码。") from None
    except ConfigError:
        raise
    except (yaml.YAMLError, ValueError, KeyError, IndexError):
        # PyYAML scalar constructors can raise built-in conversion errors.
        raise ConfigError("配置不是合法的 YAML。") from None

    if not isinstance(values, dict):
        raise ConfigError("配置必须是 YAML 字段映射。")
    if values.get("protocol") == "openai" and "thinking" in values:
        raise ConfigError("OpenAI 配置不支持 Anthropic 专用的 thinking 字段。")
    allowed = {"protocol", "model", "base_url", "api_key", "max_output_tokens", "timeout_seconds", "thinking"}
    if values.keys() - allowed:
        raise ConfigError("配置含未知字段；支持 protocol、model、base_url、api_key、max_output_tokens、timeout_seconds、thinking。")

    protocol = _required_string(values, "protocol")
    if protocol not in {"anthropic", "openai"}:
        raise ConfigError("protocol 仅支持 anthropic 或 openai。")
    model = _required_string(values, "model")
    base_url = _required_string(values, "base_url")
    api_key = _required_string(values, "api_key")
    if api_key != api_key.strip() or any(not 32 <= ord(char) < 127 for char in api_key):
        raise ConfigError("api_key 必须是有效的 ASCII 请求头值，不含控制字符或首尾空白。")
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
        raise ConfigError("base_url 必须是合法的 HTTP(S) 根地址。")

    max_tokens = values.get("max_output_tokens", ProviderConfig.max_output_tokens)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ConfigError("max_output_tokens 必须是正整数。")
    timeout = values.get("timeout_seconds", ProviderConfig.timeout_seconds)
    if type(timeout) not in {int, float}:
        raise ConfigError("timeout_seconds 必须是有限正数。")
    try:
        timeout = float(timeout)
    except OverflowError:
        raise ConfigError("timeout_seconds 必须是有限正数。") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("timeout_seconds 必须是有限正数。")

    thinking = _thinking_config(values["thinking"], max_tokens) if "thinking" in values else None
    return ProviderConfig(protocol, model, base_url, api_key, max_tokens, timeout, thinking)


def _thinking_config(value: object, max_tokens: int) -> ThinkingConfigParam:
    if not isinstance(value, dict):
        raise ConfigError("thinking 必须是包含 type 的字段映射。")
    if value.keys() - {"type", "budget_tokens", "display"}:
        raise ConfigError("thinking 含未知字段；仅支持 type、budget_tokens、display。")
    mode = value.get("type")
    if not isinstance(mode, str) or mode not in {"enabled", "adaptive", "disabled"}:
        raise ConfigError("thinking.type 仅支持 enabled、adaptive 或 disabled。")
    if mode == "enabled":
        budget = value.get("budget_tokens")
        if type(budget) is not int or not 1024 <= budget < max_tokens:
            raise ConfigError("thinking.budget_tokens 必须是至少 1024 且小于 max_output_tokens 的整数。")
    elif "budget_tokens" in value:
        raise ConfigError("只有 enabled 模式可以设置 thinking.budget_tokens。")
    if "display" in value:
        if mode == "disabled":
            raise ConfigError("disabled 模式不能设置 thinking.display。")
        if value["display"] != "summarized":
            raise ConfigError("thinking.display 仅支持 summarized。")
    return cast(ThinkingConfigParam, value)


def _required_string(values: dict[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须是非空字符串。")
    return value
