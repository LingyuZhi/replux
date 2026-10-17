"""Local tools the model can call, independent of protocols and the TUI."""

from pathlib import Path

from replux.tools.base import (
    FILE_OPERATION_TIMEOUT_SECONDS, FileOperationTimeout, Tool, ToolCategory, ToolDefinition, ToolParams,
    ToolRegistry, ToolResult, ToolStatus, run_file_operation, summarize_arguments,
)
from replux.tools.edit_file import EditFileTool
from replux.tools.find_files import FindFilesTool
from replux.tools.read_file import ReadFileTool
from replux.tools.run_command import RunCommandTool
from replux.tools.search_content import SearchContentTool
from replux.tools.session import SessionDirectory
from replux.tools.write_file import WriteFileTool

__all__ = [
    "FILE_OPERATION_TIMEOUT_SECONDS", "FileOperationTimeout", "Tool", "ToolCategory", "ToolDefinition",
    "ToolParams", "ToolRegistry", "ToolResult", "ToolStatus", "create_default_registry", "run_file_operation",
    "summarize_arguments",
]


def create_default_registry(
    root: Path, *, bash: str | None = None, ripgrep: str | None = None,
) -> ToolRegistry:
    """Register the built-in tools; relative tool paths start at root."""
    session = SessionDirectory()
    registry = ToolRegistry(session)
    registry.register(ReadFileTool(root))
    registry.register(WriteFileTool(root))
    registry.register(EditFileTool(root))
    registry.register(RunCommandTool(root, session, bash=bash))
    registry.register(FindFilesTool(root, ripgrep=ripgrep))
    registry.register(SearchContentTool(root, ripgrep=ripgrep))
    return registry
