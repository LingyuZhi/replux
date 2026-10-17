"""Create or replace a UTF-8 text file."""

from collections.abc import Callable
import os
from pathlib import Path

from pydantic import Field

from replux.tools.base import (
    FILE_OPERATION_TIMEOUT_SECONDS, FileOperationTimeout, Tool, ToolParams, ToolResult, error_result,
    run_file_operation,
)
from replux.tools.commit import CommitHold, commit_file


class WriteFileParams(ToolParams):
    path: str = Field(min_length=1, description="File path. Relative paths start at the Replux startup directory.")
    content: str = Field(description="Complete new text of the file, written as UTF-8.")


class WriteFileTool(Tool[WriteFileParams]):
    name = "write_file"
    description = (
        "Create a file or replace its entire contents with the given text. "
        "Missing parent directories are created. The text is written as UTF-8 exactly as given, "
        "and is not truncated to the read output limit. Relative paths start at the Replux startup directory."
    )
    params_model = WriteFileParams
    category = "write"

    def __init__(
        self, root: Path, *, timeout_seconds: float = FILE_OPERATION_TIMEOUT_SECONDS, hold: CommitHold | None = None,
    ) -> None:
        self._root = root
        self._hold = hold
        self.timeout_seconds = timeout_seconds

    async def execute(self, params: WriteFileParams) -> ToolResult:
        target = self._root / params.path
        data = params.content.encode("utf-8")
        try:
            return await run_file_operation(
                lambda checkpoint: _write(target, data, checkpoint, self._hold),
                timeout_seconds=self.timeout_seconds,
            )
        except FileOperationTimeout:
            state, sentence = _describe(target, data)
            return error_result(
                "timeout",
                f"Writing {target} did not finish within {self.timeout_seconds:g} seconds. {sentence}",
                path=str(target), file_state=state,
            )


def _write(target: Path, data: bytes, checkpoint: Callable[[], None], hold: CommitHold | None) -> ToolResult:
    try:
        commit_file(target, data, checkpoint, hold=hold)
    except OSError as error:
        state, sentence = _describe(target, data)
        return error_result(
            _code(error, state),
            f"Cannot write {target}: {error.strerror or 'unknown error'}. {sentence}",
            path=str(target), file_state=state,
        )
    return ToolResult(
        "success", f"Wrote {len(data)} bytes to {target}.", details={"path": str(target), "bytes": len(data)},
    )


def _code(error: OSError, state: str) -> str:
    if state == "directory" or isinstance(error, IsADirectoryError):
        return "not_a_file"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, NotADirectoryError):
        return "not_a_directory"
    return "write_failed"


def _describe(path: Path, intended: bytes) -> tuple[str, str]:
    try:
        probe = Path(os.path.realpath(path)) if path.is_symlink() else path
        if probe.is_dir():
            return "directory", f"{probe} is a directory and was not replaced."
        if not probe.exists():
            return "absent", f"{probe} was not created."
        current = probe.read_bytes()
    except OSError as error:
        return "unknown", f"The state of {path} could not be read: {error.strerror or 'unknown error'}."
    if current == intended:
        return "written", f"{probe} contains the new {len(intended)} bytes."
    return "unchanged", f"{probe} still has its previous {len(current)} bytes."
