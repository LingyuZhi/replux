"""Create or replace a UTF-8 text file."""

import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic import Field

from replux.tools.base import (
    FILE_OPERATION_TIMEOUT_SECONDS, FileOperationTimeout, Tool, ToolParams, ToolResult, error_result,
    run_file_operation,
)
from replux.tools.commit import CommitHold, commit_error_code, commit_file, describe_commit


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
            return _stopped(target, data, "timeout", f"Writing {target} did not finish within {self.timeout_seconds:g} seconds.")
        except asyncio.CancelledError:
            return _stopped(target, data, "cancelled", f"Cancelled while writing {target}.")


def _write(target: Path, data: bytes, checkpoint: Callable[[], None], hold: CommitHold | None) -> ToolResult:
    try:
        commit_file(target, data, checkpoint, hold=hold)
    except OSError as error:
        state, sentence = describe_commit(target, data)
        return error_result(
            commit_error_code(error, state),
            f"Cannot write {target}: {error.strerror or 'unknown error'}. {sentence}",
            path=str(target), file_state=state,
        )
    return ToolResult(
        "success", f"Wrote {len(data)} bytes to {target}.", details={"path": str(target), "bytes": len(data)},
    )


def _stopped(target: Path, data: bytes, status: str, lead: str) -> ToolResult:
    state, sentence = describe_commit(target, data)
    if status == "cancelled":
        return ToolResult("cancelled", f"{lead} {sentence}", details={"path": str(target), "file_state": state})
    return error_result("timeout", f"{lead} {sentence}", path=str(target), file_state=state)


