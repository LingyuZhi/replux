"""Tool contract, results, registry, and the shared file-operation runner."""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import re
import threading
from typing import Any, Generic, Literal, TypeVar, get_args

from pydantic import BaseModel, ConfigDict, ValidationError


ToolCategory = Literal["read", "write", "execute"]
ToolStatus = Literal["success", "error", "cancelled", "skipped"]

FILE_OPERATION_TIMEOUT_SECONDS = 10.0
OUTPUT_LIMIT_BYTES = 50 * 1024

_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class ToolResult:
    """The model-visible outcome of one Tool Call; details must be JSON serializable."""

    status: ToolStatus
    output: str
    error_code: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in get_args(ToolStatus):
            raise ValueError(f"Unknown tool result status: {self.status!r}.")
        if (self.status == "error") != (self.error_code is not None):
            raise ValueError("Exactly the error status has an error code.")
        json.dumps(self.details)

    def to_json(self) -> str:
        payload: dict[str, object] = {"status": self.status, "output": self.output}
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.details:
            payload["details"] = self.details
        return json.dumps(payload, ensure_ascii=False)


def error_result(code: str, message: str, **details: object) -> ToolResult:
    return ToolResult("error", message, code, details)


class ToolParams(BaseModel):
    """Model arguments: strict types, no unknown fields, no numeric strings."""

    model_config = ConfigDict(strict=True, extra="forbid")


Params = TypeVar("Params", bound=BaseModel)


class Tool(ABC, Generic[Params]):
    name: str
    description: str
    params_model: type[Params]
    category: ToolCategory

    @abstractmethod
    async def execute(self, params: Params) -> ToolResult:
        """Run with validated arguments; return expected failures as error results."""


@dataclass(frozen=True)
class ToolDefinition:
    """Protocol-independent tool definition; adapters convert it to their format."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistry:
    """Tools available to the model, registered at startup and read-only afterwards."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self._definitions: list[ToolDefinition] = []

    def register(self, tool: Tool[Any]) -> None:
        if not isinstance(tool.name, str) or not _TOOL_NAME.fullmatch(tool.name):
            raise ValueError(f"Invalid tool name: {tool.name!r}.")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}.")
        if not isinstance(tool.description, str) or not tool.description.strip():
            raise ValueError(f"Tool {tool.name} needs a description.")
        if tool.category not in get_args(ToolCategory):
            raise ValueError(f"Tool {tool.name} has an unknown category: {tool.category!r}.")
        model = tool.params_model
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(f"Tool {tool.name} parameters must be a Pydantic model.")
        if model.model_config.get("strict") is not True or model.model_config.get("extra") != "forbid":
            raise ValueError(f"Tool {tool.name} parameters must be strict and forbid unknown fields.")
        schema = model.model_json_schema()
        if schema.get("type") != "object":
            raise ValueError(f"Tool {tool.name} parameters must be a JSON object.")
        self._tools[tool.name] = tool
        self._definitions.append(ToolDefinition(tool.name, tool.description, schema))

    def get(self, name: str) -> Tool[Any]:
        return self._tools[name]

    def definitions(self) -> list[ToolDefinition]:
        return list(self._definitions)

    async def execute(self, name: str, arguments: str) -> ToolResult:
        """Validate raw model arguments, then run the tool."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return error_result("unknown_tool", f"Unknown tool: {name}. Available tools: {available}.")
        try:
            params = tool.params_model.model_validate_json(arguments)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc']) or 'arguments'}: {issue['msg']}"
                for issue in error.errors(include_url=False)
            )
            return error_result("invalid_arguments", f"Invalid arguments for {name}: {problems}.")
        return await tool.execute(params)


def summarize_arguments(arguments: str, *, width: int = 100) -> str:
    """One display line for raw model arguments."""
    try:
        values = json.loads(arguments)
    except json.JSONDecodeError:
        values = None
    if isinstance(values, dict):
        summary = " ".join(
            f"{key}={value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
            for key, value in values.items()
        )
    else:
        summary = arguments
    summary = " ".join(summary.split())
    return summary if len(summary) <= width else summary[:width - 1] + "…"


class FileOperationTimeout(Exception):
    """A file operation stopped at a safe point after its time limit."""


class _Stopped(Exception):
    pass


Result = TypeVar("Result")


async def run_file_operation(
    operation: Callable[[Callable[[], None]], Result], *, timeout_seconds: float,
) -> Result:
    """Run blocking file work in a thread that stops at its checkpoints.

    Timeout and cancellation both wait until the thread has returned, so no
    caller sees the operation as finished while it can still touch files.
    """
    stop = threading.Event()

    def checkpoint() -> None:
        if stop.is_set():
            raise _Stopped

    work = asyncio.ensure_future(asyncio.to_thread(operation, checkpoint))
    cancelled = False
    try:
        await asyncio.wait({work}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        cancelled = True
    if not work.done():
        stop.set()
        while not work.done():
            try:
                await asyncio.wait({work})
            except asyncio.CancelledError:
                cancelled = True
    error = work.exception()
    if error is not None and not isinstance(error, _Stopped):
        raise error
    if cancelled:
        raise asyncio.CancelledError
    if error is not None:
        raise FileOperationTimeout
    return work.result()
