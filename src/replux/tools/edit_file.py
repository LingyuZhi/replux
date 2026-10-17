"""Apply ordered unique replacements to one UTF-8 file, then write it once."""

import asyncio
from collections.abc import Callable
from pathlib import Path

from pydantic import Field

from replux.tools.base import (
    FILE_OPERATION_TIMEOUT_SECONDS, FileOperationTimeout, Tool, ToolParams, ToolResult, error_result,
    run_file_operation,
)
from replux.tools.commit import CommitHold, commit_error_code, commit_file, describe_commit


class FileEdit(ToolParams):
    old_text: str = Field(min_length=1, description="Exact text to find. It must occur once, and overlapping positions count.")
    new_text: str = Field(description="Text that replaces the match. An empty string deletes it.")


class EditFileParams(ToolParams):
    path: str = Field(min_length=1, description="File path. Relative paths start at the Replux startup directory.")
    edits: list[FileEdit] = Field(
        min_length=1,
        description="Replacements applied in order. Each one sees the text from earlier replacements.",
    )


class EditFileTool(Tool[EditFileParams]):
    name = "edit_file"
    description = (
        "Replace unique text in one existing UTF-8 file. "
        "edits is a non-empty list of old_text and new_text. old_text must match exactly once, "
        "and overlapping positions count. A later edit sees text produced by earlier edits. "
        "An empty new_text deletes the match. Every edit is checked before the file is written once. "
        "Existing newlines and a UTF-8 BOM are kept. Relative paths start at the Replux startup directory."
    )
    params_model = EditFileParams
    category = "write"

    def __init__(
        self, root: Path, *, timeout_seconds: float = FILE_OPERATION_TIMEOUT_SECONDS, hold: CommitHold | None = None,
    ) -> None:
        self._root = root
        self._hold = hold
        self.timeout_seconds = timeout_seconds

    async def execute(self, params: EditFileParams) -> ToolResult:
        target = self._root / params.path
        try:
            return await run_file_operation(
                lambda checkpoint: _edit(target, params.edits, checkpoint, self._hold),
                timeout_seconds=self.timeout_seconds,
            )
        except FileOperationTimeout:
            return _stopped(
                target, "timeout", f"Editing {target} did not finish within {self.timeout_seconds:g} seconds.",
            )
        except asyncio.CancelledError:
            return _stopped(target, "cancelled", f"Cancelled while editing {target}.")


def _edit(
    path: Path, edits: list[FileEdit], checkpoint: Callable[[], None], hold: CommitHold | None,
) -> ToolResult:
    checkpoint()
    loaded = _load(path)
    if isinstance(loaded, ToolResult):
        return loaded
    original, text = loaded
    for index, edit in enumerate(edits, start=1):
        count = _match_count(text, edit.old_text)
        if count != 1:
            code = "no_match" if count == 0 else "not_unique"
            return error_result(code, _match_message(path, index, count), path=str(path), edit=index, matches=count)
        found = text.find(edit.old_text)
        text = text[:found] + edit.new_text + text[found + len(edit.old_text):]
    # The file is already in memory. A caller can change the on-disk bytes before the check below.
    if hold is not None and hold.stage == "before":
        hold.entered.set()
        hold.release.wait()
    checkpoint()
    if _bytes_changed(path, original):
        return _file_changed(path)
    data = text.encode("utf-8")
    commit_hold = hold if hold is not None and hold.stage == "during" else None
    try:
        commit_file(path, data, checkpoint, hold=commit_hold)
    except OSError as error:
        state, sentence = describe_commit(path, data)
        return error_result(
            commit_error_code(error, state),
            f"Cannot edit {path}: {error.strerror or 'unknown error'}. {sentence}",
            path=str(path), file_state=state,
        )
    noun = "edit" if len(edits) == 1 else "edits"
    return ToolResult(
        "success", f"Applied {len(edits)} {noun} to {path}.",
        details={"path": str(path), "bytes": len(data), "edits": len(edits)},
    )


def _load(path: Path) -> tuple[bytes, str] | ToolResult:
    try:
        if path.is_dir():
            return error_result("not_a_file", f"{path} is a directory and was not changed.", path=str(path))
        data = path.read_bytes()
    except FileNotFoundError:
        return error_result("file_not_found", f"File not found: {path}.", path=str(path))
    except PermissionError:
        return error_result("permission_denied", f"Permission denied: {path}.", path=str(path))
    except NotADirectoryError:
        return error_result("not_a_directory", f"{path} is not a file path that can be edited.", path=str(path))
    except IsADirectoryError:
        return error_result("not_a_file", f"{path} is a directory and was not changed.", path=str(path))
    except OSError as error:
        return error_result(
            "read_failed", f"Cannot read {path}: {error.strerror or 'unknown error'}.", path=str(path),
        )
    if b"\0" in data:
        return error_result(
            "binary_file", f"{path} looks like a binary file; only UTF-8 text is supported.", path=str(path),
        )
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError:
        return error_result(
            "unsupported_encoding", f"{path} is not valid UTF-8 text; other encodings are not supported.",
            path=str(path),
        )


def _stopped(path: Path, status: str, lead: str) -> ToolResult:
    state, sentence = describe_commit(path)
    if status == "cancelled":
        return ToolResult("cancelled", f"{lead} {sentence}", details={"path": str(path), "file_state": state})
    return error_result("timeout", f"{lead} {sentence}", path=str(path), file_state=state)


def _bytes_changed(path: Path, original: bytes) -> bool:
    try:
        if path.is_dir() or not path.is_file():
            return True
        return path.read_bytes() != original
    except OSError:
        return True


def _file_changed(path: Path) -> ToolResult:
    state, sentence = describe_commit(path)
    return error_result(
        "file_changed",
        f"{path} changed after it was read and was not overwritten. {sentence}",
        path=str(path), file_state=state,
    )


def _match_count(text: str, needle: str) -> int:
    """Count matches, including ones that overlap."""
    count = 0
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return count
        count += 1
        start = found + 1


def _match_message(path: Path, index: int, count: int) -> str:
    if count == 0:
        return f"Edit {index} did not find old_text. {path} was not changed."
    return (
        f"Edit {index} found old_text {count} times, counting overlapping positions. "
        f"It must match exactly once. {path} was not changed."
    )
