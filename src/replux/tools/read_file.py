"""Read a UTF-8 text file as numbered lines."""

from collections.abc import Callable
import os
from pathlib import Path
import shlex
import stat
from typing import BinaryIO

from pydantic import Field

from replux.tools.base import (
    FILE_OPERATION_TIMEOUT_SECONDS, OUTPUT_LIMIT_BYTES, FileOperationTimeout, Tool, ToolParams, ToolResult,
    error_result, run_file_operation,
)


MAX_READ_LINES = 2000
_CHUNK_BYTES = 64 * 1024


class ReadFileParams(ToolParams):
    path: str = Field(min_length=1, description="File path; relative paths start at the Replux startup directory.")
    offset: int = Field(default=1, ge=1, description="First line to read, starting at 1.")
    limit: int = Field(default=MAX_READ_LINES, ge=1, le=MAX_READ_LINES, description="Maximum number of lines to read.")


class ReadFileTool(Tool[ReadFileParams]):
    name = "read_file"
    description = (
        "Read a UTF-8 text file. Each output line is '<line number>\\t<content>', numbered from 1. "
        "Use offset and limit to read part of a long file. Output stops at 2000 lines or 50 KiB, "
        "whichever comes first, and then gives the offset to continue from. "
        "Binary files and other encodings are not supported."
    )
    params_model = ReadFileParams
    category = "read"

    def __init__(self, root: Path, *, timeout_seconds: float = FILE_OPERATION_TIMEOUT_SECONDS) -> None:
        self._root = root
        self.timeout_seconds = timeout_seconds

    async def execute(self, params: ReadFileParams) -> ToolResult:
        target = self._root / params.path
        try:
            return await run_file_operation(
                lambda checkpoint: _read(target, params.offset, params.limit, checkpoint),
                timeout_seconds=self.timeout_seconds,
            )
        except FileOperationTimeout:
            return error_result(
                "timeout", f"Reading {target} did not finish within {self.timeout_seconds:g} seconds.", path=str(target),
            )


def _read(target: Path, offset: int, limit: int, checkpoint: Callable[[], None]) -> ToolResult:
    try:
        if not stat.S_ISREG(os.stat(target).st_mode):
            return error_result("not_a_file", f"{target} is not a regular file.", path=str(target))
        with open(target, "rb") as file:
            return _select(file, target, offset, limit, checkpoint)
    except (FileNotFoundError, NotADirectoryError):
        return error_result("file_not_found", f"File not found: {target}.", path=str(target))
    except PermissionError:
        return error_result("permission_denied", f"Permission denied: {target}.", path=str(target))
    except OSError as error:
        return error_result("read_failed", f"Cannot read {target}: {error.strerror or 'unknown error'}.", path=str(target))


def _select(file: BinaryIO, target: Path, offset: int, limit: int, checkpoint: Callable[[], None]) -> ToolResult:
    shown: list[str] = []
    used = 0
    number = 0
    truncated_by: str | None = None
    while (line := _next_line(file, checkpoint)) is not None:
        number += 1
        content, size, binary = line
        if binary:
            return error_result(
                "binary_file", f"{target} looks like a binary file; only UTF-8 text is supported.", path=str(target),
            )
        if number < offset:
            continue
        if len(shown) == limit:
            truncated_by = "lines"
            break
        fits = size <= OUTPUT_LIMIT_BYTES
        if fits:
            try:
                formatted = f"{number}\t{content.decode('utf-8')}"
            except UnicodeDecodeError:
                return error_result(
                    "unsupported_encoding", f"{target} is not valid UTF-8 text; other encodings are not supported.",
                    path=str(target),
                )
            cost = len(formatted.encode()) + (1 if shown else 0)
            fits = used + cost <= OUTPUT_LIMIT_BYTES
        if not fits:
            if not shown:
                command = f"sed -n '{number}p' {shlex.quote(str(target))} | head -c {OUTPUT_LIMIT_BYTES}"
                return ToolResult(
                    "success",
                    f"Line {number} is {size} bytes and does not fit in the 50 KiB read limit. "
                    f"Read part of it with a command, for example: {command}",
                    details={"path": str(target), "line": number, "line_bytes": size, "truncated_by": "bytes"},
                )
            truncated_by = "bytes"
            break
        shown.append(formatted)
        used += cost
    if not shown:
        if number == 0:
            return ToolResult("success", "The file is empty.", details={"path": str(target)})
        return error_result(
            "offset_out_of_range", f"offset {offset} is beyond the end of {target} ({number} lines).",
            path=str(target), total_lines=number,
        )
    end = offset + len(shown) - 1
    details: dict[str, object] = {"path": str(target), "start_line": offset, "end_line": end}
    output = "\n".join(shown)
    if truncated_by is not None:
        details |= {"next_offset": end + 1, "truncated_by": truncated_by}
        limit_note = " (50 KiB limit)" if truncated_by == "bytes" else ""
        output += f"\n\n[Showing lines {offset}-{end}{limit_note}. Use offset={end + 1} to continue.]"
    return ToolResult("success", output, details=details)


def _next_line(file: BinaryIO, checkpoint: Callable[[], None]) -> tuple[bytes, int, bool] | None:
    """Return the line without its newline (kept only up to the output limit), its full size, and whether it has NUL."""
    checkpoint()
    request = OUTPUT_LIMIT_BYTES + 1
    content = piece = file.readline(request)
    if not content:
        return None
    size = len(content)
    binary = b"\0" in content
    while not piece.endswith(b"\n") and len(piece) == request and not binary:
        checkpoint()
        request = _CHUNK_BYTES
        piece = file.readline(request)
        size += len(piece)
        binary = b"\0" in piece
    if piece.endswith(b"\n"):
        size -= 1
    return content.removesuffix(b"\n"), size, binary
