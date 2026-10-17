"""Find files by a glob, using ripgrep."""

import os
from pathlib import Path
import shutil
import time

from pydantic import Field

from replux.tools.base import OUTPUT_LIMIT_BYTES, Tool, ToolParams, ToolResult, error_result
from replux.tools.process import PROCESS_TERMINATION_GRACE_SECONDS, RIPGREP_INSTALL_HINT, run_process_group


FIND_RESULT_LIMIT = 1000
FIND_TIMEOUT_SECONDS = 10.0


class FindFilesParams(ToolParams):
    pattern: str = Field(
        min_length=1, description="Glob matched against each path relative to the search directory.",
    )
    path: str = Field(
        default=".", min_length=1,
        description="Directory to search. Relative paths start at the Replux startup directory. The default is that directory.",
    )
    limit: int = Field(
        default=FIND_RESULT_LIMIT, ge=1,
        description="Maximum number of results to return. The default is 1000. There is no upper bound; output still stops at 50 KiB.",
    )
    include_hidden: bool = Field(
        default=False, description="Also return hidden files and files inside hidden directories.",
    )
    include_ignored: bool = Field(
        default=False, description="Also return files ignored by .gitignore, .ignore, or .rgignore.",
    )


class FindFilesTool(Tool[FindFilesParams]):
    name = "find_files"
    description = (
        "Find files whose paths relative to a directory match a glob. "
        "path must be a directory and defaults to the Replux startup directory; relative paths start there. "
        "Results are file paths relative to that directory, in lexicographic order. "
        "The search uses ripgrep. It respects .gitignore, .ignore, and .rgignore, skips hidden files, "
        "and does not follow directory symlinks. "
        "include_hidden and include_ignored widen those two scopes separately. A glob does not override ignore rules. "
        "Names of binary files are included. No matches is a successful empty result. "
        "The default limit is 1000 results, and the output also stops at 50 KiB."
    )
    params_model = FindFilesParams
    category = "read"

    def __init__(
        self,
        root: Path,
        *,
        ripgrep: str | None = None,
        timeout_seconds: float = FIND_TIMEOUT_SECONDS,
        byte_limit: int = OUTPUT_LIMIT_BYTES,
        grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
    ) -> None:
        self._root = root
        self._ripgrep = shutil.which("rg") if ripgrep is None else ripgrep
        self.timeout_seconds = timeout_seconds
        self.byte_limit = byte_limit
        self.grace_seconds = grace_seconds

    async def execute(self, params: FindFilesParams) -> ToolResult:
        directory = _search_directory(self._root, params.path)
        if not directory.exists():
            return error_result("directory_not_found", f"Directory not found: {directory}.", path=str(directory))
        if not directory.is_dir():
            return error_result("not_a_directory", f"{directory} is not a directory.", path=str(directory))
        if not self._ripgrep:
            return error_result("ripgrep_not_found", RIPGREP_INSTALL_HINT, path=str(directory))
        listed = await _list_matching_files(
            self._ripgrep, directory, params.pattern,
            include_hidden=params.include_hidden, include_ignored=params.include_ignored,
            timeout_seconds=self.timeout_seconds, grace_seconds=self.grace_seconds,
        )
        if isinstance(listed, ToolResult):
            return listed
        return _format_paths(listed, directory=directory, limit=params.limit, byte_limit=self.byte_limit)


def _search_directory(root: Path, path: str) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = root / selected
    if selected.name == ".":
        return selected.parent
    return selected


class _Capture:
    def __init__(self) -> None:
        self.data = bytearray()

    def add(self, chunk: bytes) -> bool:
        self.data.extend(chunk)
        return True


async def _list_matching_files(
    ripgrep: str,
    directory: Path,
    pattern: str,
    *,
    include_hidden: bool,
    include_ignored: bool,
    timeout_seconds: float,
    grace_seconds: float,
) -> list[str] | ToolResult:
    """Return paths relative to directory.

    ripgrep's --glob overrides ignore rules and can surface hidden files, so the
    glob pass is intersected with a pass that applies only the requested scope.
    """
    deadline = time.monotonic() + timeout_seconds
    allowed = await _run_files(
        ripgrep, directory, pattern=None,
        include_hidden=include_hidden, include_ignored=include_ignored,
        timeout_seconds=timeout_seconds, deadline=deadline, grace_seconds=grace_seconds,
    )
    if isinstance(allowed, ToolResult):
        return allowed
    if not allowed:
        return []
    matched = await _run_files(
        ripgrep, directory, pattern=pattern,
        include_hidden=include_hidden, include_ignored=include_ignored,
        timeout_seconds=timeout_seconds, deadline=deadline, grace_seconds=grace_seconds,
    )
    if isinstance(matched, ToolResult):
        return matched
    return sorted(set(allowed) & set(matched))


async def _run_files(
    ripgrep: str,
    directory: Path,
    *,
    pattern: str | None,
    include_hidden: bool,
    include_ignored: bool,
    timeout_seconds: float,
    deadline: float,
    grace_seconds: float,
) -> list[str] | ToolResult:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return error_result(
            "timeout",
            f"Finding files in {directory} did not finish within {timeout_seconds:g} seconds. "
            "The search process was stopped.",
            path=str(directory),
        )
    capture = _Capture()
    try:
        outcome = await run_process_group(
            _argv(
                ripgrep, pattern=pattern, include_hidden=include_hidden, include_ignored=include_ignored,
            ),
            cwd=directory,
            on_output=capture.add,
            timeout_seconds=remaining,
            grace_seconds=grace_seconds,
        )
    except FileNotFoundError as error:
        if error.filename == ripgrep:
            return error_result("ripgrep_not_found", RIPGREP_INSTALL_HINT, path=str(directory))
        return error_result("directory_not_found", f"Directory not found: {directory}.", path=str(directory))
    except NotADirectoryError:
        return error_result("not_a_directory", f"{directory} is not a directory.", path=str(directory))
    except PermissionError:
        return error_result("permission_denied", f"Permission denied: {directory}.", path=str(directory))
    except OSError as error:
        return error_result(
            "search_failed", f"Cannot search {directory}: {error.strerror or 'unknown error'}.", path=str(directory),
        )
    if outcome.cancelled:
        return ToolResult(
            "cancelled",
            f"Finding files in {directory} was cancelled and its process group was stopped.",
            details={"path": str(directory)},
        )
    if outcome.timed_out:
        return error_result(
            "timeout",
            f"Finding files in {directory} did not finish within {timeout_seconds:g} seconds. "
            "The search process was stopped.",
            path=str(directory),
        )
    if outcome.return_code not in {0, 1}:
        message = _error_text(bytes(capture.data)) or f"ripgrep exited with status {outcome.return_code}."
        code = _failure_code(message)
        return error_result(code, f"Cannot search {directory}: {message}", path=str(directory))
    return _paths(bytes(capture.data))


def _argv(ripgrep: str, *, pattern: str | None, include_hidden: bool, include_ignored: bool) -> list[str]:
    argv = [
        ripgrep, "--files", "--color=never", "--no-config", "--no-require-git", "--no-ignore-global", "--null",
    ]
    if include_hidden:
        argv.append("--hidden")
    if include_ignored:
        argv.append("--no-ignore")
    if pattern is not None:
        argv.append(f"--glob={pattern}")
    return argv


def _paths(data: bytes) -> list[str]:
    return [os.fsdecode(part) for part in data.split(b"\0") if part]


def _error_text(data: bytes) -> str:
    text = os.fsdecode(data).replace("\0", "\n").strip()
    if len(text) <= 2000:
        return text
    return text[-2000:]


def _failure_code(message: str) -> str:
    if "Permission denied" in message:
        return "permission_denied"
    if "error parsing glob" in message:
        return "invalid_pattern"
    return "search_failed"


def _format_paths(paths: list[str], *, directory: Path, limit: int, byte_limit: int) -> ToolResult:
    shown: list[str] = []
    used = 0
    reason: str | None = None
    for path in paths:
        if len(shown) >= limit:
            reason = "count"
            break
        encoded = path.encode("utf-8")
        extra = len(encoded) if not shown else len(encoded) + 1
        if used + extra > byte_limit:
            reason = "bytes"
            break
        shown.append(path)
        used += extra
    details: dict[str, object] = {"path": str(directory), "count": len(shown), "total": len(paths)}
    if not paths:
        return ToolResult("success", f"No files matched in {directory}.", details=details)
    note = _note(shown, total=len(paths), directory=directory, limit=limit, byte_limit=byte_limit, reason=reason)
    if reason is not None:
        details["truncated_by"] = reason
    body = "\n".join(shown)
    output = f"{body}\n\n{note}" if body else note
    return ToolResult("success", output, details=details)


def _note(
    shown: list[str], *, total: int, directory: Path, limit: int, byte_limit: int, reason: str | None,
) -> str:
    if reason == "count":
        return (
            f"Showing the first {len(shown)} of {total} files in {directory} (result limit {limit}). "
            "Raise limit or narrow the pattern to see more."
        )
    if reason == "bytes":
        return (
            f"Showing the first {len(shown)} of {total} files in {directory} ({_byte_text(byte_limit)} limit). "
            "Narrow the pattern to see more. Raising limit does not remove the byte limit."
        )
    noun = "file" if total == 1 else "files"
    return f"{total} {noun} in {directory}."


def _byte_text(size: int) -> str:
    if size % 1024 == 0:
        return f"{size // 1024} KiB"
    return f"{size} bytes"
