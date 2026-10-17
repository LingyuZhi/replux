"""Search file contents with ripgrep."""

import os
from pathlib import Path
import shutil

from pydantic import Field

from replux.tools.base import OUTPUT_LIMIT_BYTES, Tool, ToolParams, ToolResult, error_result
from replux.tools.process import PROCESS_TERMINATION_GRACE_SECONDS, RIPGREP_INSTALL_HINT, run_process_group


SEARCH_RESULT_LIMIT = 100
SEARCH_TIMEOUT_SECONDS = 10.0
LINE_CHAR_LIMIT = 500
_LINE_MARKER = "... [line truncated]"
_LINE_BYTE_CAP = LINE_CHAR_LIMIT * 4 + 64


class SearchContentParams(ToolParams):
    pattern: str = Field(min_length=1, description="Pattern to search for. This is a regular expression unless literal is true.")
    path: str = Field(
        default=".", min_length=1,
        description="Directory or file to search. Relative paths start at the Replux startup directory. The default is that directory.",
    )
    literal: bool = Field(
        default=False, description="Match the pattern as text instead of as a regular expression. The default is a case-sensitive regular expression.",
    )
    include_hidden: bool = Field(
        default=False, description="Also search hidden files and files inside hidden directories.",
    )
    include_ignored: bool = Field(
        default=False, description="Also search files ignored by .gitignore, .ignore, or .rgignore.",
    )
    limit: int = Field(
        default=SEARCH_RESULT_LIMIT, ge=1,
        description="Maximum number of matching lines to return. The default is 100. There is no upper bound; output still stops at 50 KiB.",
    )


class SearchContentTool(Tool[SearchContentParams]):
    name = "search_content"
    description = (
        "Search the contents of a directory or one explicit file. "
        "path defaults to the Replux startup directory; relative paths start there. "
        "The pattern is a case-sensitive regular expression unless literal is true. "
        "The search uses ripgrep. It respects .gitignore, .ignore, and .rgignore, skips hidden files, "
        "does not follow directory symlinks, and skips binary content. "
        "include_hidden and include_ignored widen those two scopes separately. "
        "An explicit file is searched even when it is hidden or ignored. "
        "Each matching line is one result, '<path>:<line>:<content>', numbered from 1, with no context lines. "
        "Several matches on the same line count once. "
        "No matches is a successful empty result. "
        "The default limit is 100 matching lines. Output also stops at 50 KiB. "
        "Each line of text is cut to 500 Unicode characters; the path and line number stay intact. "
        "Raising limit can return more short matches. It does not remove the 50 KiB limit, and there is no 2000-line limit."
    )
    params_model = SearchContentParams
    category = "read"

    def __init__(
        self,
        root: Path,
        *,
        ripgrep: str | None = None,
        timeout_seconds: float = SEARCH_TIMEOUT_SECONDS,
        byte_limit: int = OUTPUT_LIMIT_BYTES,
        grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
    ) -> None:
        self._root = root
        self._ripgrep = shutil.which("rg") if ripgrep is None else ripgrep
        self.timeout_seconds = timeout_seconds
        self.byte_limit = byte_limit
        self.grace_seconds = grace_seconds

    async def execute(self, params: SearchContentParams) -> ToolResult:
        target = _search_path(self._root, params.path)
        if not target.exists():
            return error_result("path_not_found", f"Path not found: {target}.", path=str(target))
        if not target.is_dir() and not target.is_file():
            return error_result("not_a_file", f"{target} is not a file or directory.", path=str(target))
        if not self._ripgrep:
            return error_result("ripgrep_not_found", RIPGREP_INSTALL_HINT, path=str(target))
        if target.is_dir():
            cwd, selected = target, "."
        else:
            cwd, selected = target.parent, target.name
        capture = _Capture(params.limit, self.byte_limit)
        try:
            outcome = await run_process_group(
                _argv(
                    self._ripgrep, params.pattern, selected,
                    literal=params.literal, include_hidden=params.include_hidden, include_ignored=params.include_ignored,
                ),
                cwd=cwd,
                on_output=capture.add,
                timeout_seconds=self.timeout_seconds,
                grace_seconds=self.grace_seconds,
            )
        except FileNotFoundError as error:
            if error.filename == self._ripgrep:
                return error_result("ripgrep_not_found", RIPGREP_INSTALL_HINT, path=str(target))
            return error_result("path_not_found", f"Path not found: {target}.", path=str(target))
        except NotADirectoryError:
            return error_result("not_a_file", f"{target} is not a file or directory.", path=str(target))
        except PermissionError:
            return error_result("permission_denied", f"Permission denied: {target}.", path=str(target))
        except OSError as error:
            return error_result(
                "search_failed", f"Cannot search {target}: {error.strerror or 'unknown error'}.", path=str(target),
            )
        if outcome.cancelled:
            return ToolResult(
                "cancelled",
                f"Searching {target} was cancelled and its process group was stopped.",
                details={"path": str(target)},
            )
        if outcome.timed_out:
            return error_result(
                "timeout",
                f"Searching {target} did not finish within {self.timeout_seconds:g} seconds. "
                "The search process was stopped.",
                path=str(target),
            )
        # A limit only shortens a finished search. A read or pattern error still wins,
        # including when the matching lines already reached the limit.
        if outcome.return_code not in {0, 1}:
            message = capture.error_text() or f"ripgrep exited with status {outcome.return_code}."
            return error_result(_failure_code(message), f"Cannot search {target}: {message}", path=str(target))
        if capture.encoding_error is not None:
            return error_result(
                "unsupported_encoding",
                f"{capture.encoding_error} is not valid UTF-8 text; other encodings are not supported.",
                path=str(target),
            )
        if capture.unparsed:
            return error_result(
                "search_failed",
                f"Cannot search {target}: ripgrep returned a result that could not be read.",
                path=str(target),
            )
        return _format_matches(
            capture.shown, target, limit=params.limit, byte_limit=self.byte_limit, reason=capture.reason,
        )


def _search_path(root: Path, path: str) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        selected = root / selected
    if selected.name == ".":
        return selected.parent
    return selected


class _Capture:
    """Parse ripgrep's NUL records and stop once another match would not be returned."""

    def __init__(self, limit: int, byte_limit: int) -> None:
        self.shown: list[str] = []
        self.reason: str | None = None
        self.encoding_error: str | None = None
        self.unparsed = False
        self._limit = limit
        self._byte_limit = byte_limit
        self._used = 0
        self._count_reached = False
        self._mode = "path"
        self._path = bytearray()
        self._line = bytearray()
        self._error_prefix = bytearray()
        self._error_suffix = bytearray()

    def add(self, chunk: bytes) -> bool:
        """Keep the pipe draining after the result limit so ripgrep can exit on its own."""
        self._remember(chunk)
        for byte in chunk:
            self._feed(byte)
        return True

    def error_text(self) -> str:
        data = bytes(self._error_prefix)
        suffix = bytes(self._error_suffix)
        if suffix and suffix not in data:
            data = data + b"\n" + suffix
        return _error_text(data)

    def _remember(self, chunk: bytes) -> None:
        if len(self._error_prefix) < 8000:
            self._error_prefix.extend(chunk[:8000 - len(self._error_prefix)])
        self._error_suffix.extend(chunk)
        if len(self._error_suffix) > 8000:
            del self._error_suffix[:len(self._error_suffix) - 8000]

    def _feed(self, byte: int) -> None:
        if self.reason is not None or self.encoding_error is not None or self.unparsed:
            return
        if self._mode == "skip":
            if byte == 10:
                self._finish_line()
            return
        if self._mode == "path":
            if byte == 0:
                if self._count_reached:
                    self.reason = "count"
                    return
                self._mode = "line"
                self._line.clear()
                return
            if byte == 10:
                self._path.clear()
                return
            self._path.append(byte)
            return
        if byte == 10:
            self._finish_line()
            return
        self._line.append(byte)
        if len(self._line) > _LINE_BYTE_CAP:
            self._mode = "skip"

    def _finish_line(self) -> None:
        self._mode = "path"
        if self._line.endswith(b"\n"):
            del self._line[-1]
        raw_line = bytes(self._line)
        parsed = _split_match(raw_line)
        path = _clean_path(bytes(self._path))
        self._path.clear()
        self._line.clear()
        if parsed is None:
            if path and raw_line:
                self.unparsed = True
            return
        number, content_bytes = parsed
        if content_bytes.endswith(b"\r"):
            content_bytes = content_bytes[:-1]
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            self.encoding_error = path or str(path)
            return
        self._offer(path, number, content)

    def _offer(self, path: str, number: int, content: str) -> None:
        if not path:
            self.unparsed = True
            return
        line = _format_line(path, number, content)
        encoded = line.encode("utf-8")
        extra = len(encoded) if not self.shown else len(encoded) + 1
        if self._used + extra > self._byte_limit:
            self.reason = "bytes"
            return
        self.shown.append(line)
        self._used += extra
        if len(self.shown) >= self._limit:
            self._count_reached = True


def _argv(
    ripgrep: str, pattern: str, selected: str, *, literal: bool, include_hidden: bool, include_ignored: bool,
) -> list[str]:
    argv = [
        ripgrep, "--line-number", "--no-heading", "--with-filename", "--color=never", "--case-sensitive",
        "--no-config", "--no-require-git", "--no-ignore-global", "--null",
    ]
    if include_hidden:
        argv.append("--hidden")
    if include_ignored:
        argv.append("--no-ignore")
    if literal:
        argv.append("--fixed-strings")
    argv.extend(("--", pattern, selected))
    return argv


def _clean_path(raw: bytes) -> str:
    text = os.fsdecode(raw)
    if text.startswith("./"):
        return text[2:]
    return text


def _split_match(raw: bytes) -> tuple[int, bytes] | None:
    colon = raw.find(b":")
    if colon <= 0:
        return None
    try:
        number = int(raw[:colon])
    except ValueError:
        return None
    if number < 1:
        return None
    return number, raw[colon + 1:]


def _format_line(path: str, number: int, content: str) -> str:
    if len(content) > LINE_CHAR_LIMIT:
        content = content[:LINE_CHAR_LIMIT] + _LINE_MARKER
    return f"{path}:{number}:{content}"


def _format_matches(
    shown: list[str], target: Path, *, limit: int, byte_limit: int, reason: str | None,
) -> ToolResult:
    details: dict[str, object] = {"path": str(target), "count": len(shown)}
    if reason is not None:
        details["truncated_by"] = reason
    if not shown and reason is None:
        return ToolResult("success", f"No matches in {target}.", details=details)
    note = _note(shown, target=target, limit=limit, byte_limit=byte_limit, reason=reason)
    body = "\n".join(shown)
    output = f"{body}\n\n{note}" if body else note
    return ToolResult("success", output, details=details)


def _note(shown: list[str], *, target: Path, limit: int, byte_limit: int, reason: str | None) -> str:
    if reason == "count":
        return (
            f"Showing the first {len(shown)} matches in {target} (result limit {limit}). "
            "Raise limit or narrow the pattern to see more."
        )
    if reason == "bytes":
        return (
            f"Showing the first {len(shown)} matches in {target} ({_byte_text(byte_limit)} limit). "
            "Narrow the pattern to see more. Raising limit does not remove the byte limit."
        )
    noun = "match" if len(shown) == 1 else "matches"
    return f"{len(shown)} {noun} in {target}."


def _byte_text(size: int) -> str:
    if size % 1024 == 0:
        return f"{size // 1024} KiB"
    return f"{size} bytes"


def _failure_code(message: str) -> str:
    if "Permission denied" in message:
        return "permission_denied"
    if "regex parse error" in message:
        return "invalid_pattern"
    return "search_failed"


def _error_text(data: bytes) -> str:
    text = os.fsdecode(data).replace("\0", "\n").strip()
    if len(text) <= 2000:
        return text
    return text[-2000:]
