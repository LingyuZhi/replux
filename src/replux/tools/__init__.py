"""Local tools the model can call, independent of protocols and the TUI."""

from pathlib import Path

from replux.tools.base import (
    FILE_OPERATION_TIMEOUT_SECONDS, FileOperationTimeout, Tool, ToolCategory, ToolDefinition, ToolParams,
    ToolRegistry, ToolResult, ToolStatus, run_file_operation, summarize_arguments,
)
from replux.tools.read_file import ReadFileTool
from replux.tools.write_file import WriteFileTool

__all__ = [
    "FILE_OPERATION_TIMEOUT_SECONDS", "FileOperationTimeout", "Tool", "ToolCategory", "ToolDefinition",
    "ToolParams", "ToolRegistry", "ToolResult", "ToolStatus", "create_default_registry", "run_file_operation",
    "summarize_arguments",
]


def create_default_registry(root: Path) -> ToolRegistry:
    """Register the built-in tools; relative tool paths start at root."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root))
    registry.register(WriteFileTool(root))
    return registry
