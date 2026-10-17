"""Run one short Bash command and keep its output bounded."""

from pathlib import Path
import shutil
from typing import BinaryIO

from pydantic import Field

from replux.tools.base import OUTPUT_LIMIT_BYTES, Tool, ToolParams, ToolResult, ToolStatus, error_result
from replux.tools.process import PROCESS_TERMINATION_GRACE_SECONDS, ProcessOutcome, run_process_group
from replux.tools.session import SessionDirectory


COMMAND_TIMEOUT_SECONDS = 120
COMMAND_TIMEOUT_LIMIT_SECONDS = 600
COMMAND_OUTPUT_LINE_LIMIT = 2000
COMMAND_OUTPUT_BYTE_LIMIT = OUTPUT_LIMIT_BYTES
COMMAND_LOG_BYTE_LIMIT = 100 * 1024 * 1024


class RunCommandParams(ToolParams):
    command: str = Field(
        min_length=1,
        description="Bash command text. It runs once in a new non-login, non-interactive Bash process.",
    )
    cwd: str | None = Field(
        default=None, min_length=1,
        description="Working directory for this command only. Relative paths start at the Replux startup directory.",
    )
    timeout_seconds: int = Field(
        default=COMMAND_TIMEOUT_SECONDS, ge=1, le=COMMAND_TIMEOUT_LIMIT_SECONDS,
        description="Positive time limit in seconds, at most 600. The default is 120.",
    )


class RunCommandTool(Tool[RunCommandParams]):
    name = "run_command"
    description = (
        "Run a short Bash command. Each call starts a new non-login, non-interactive Bash process "
        "with stdin closed, inherits the current environment, and combines stdout and stderr. "
        "cd and export in the command affect only that process. "
        "cwd defaults to the Replux startup directory; a relative cwd starts there. "
        "timeout_seconds defaults to 120 and cannot exceed 600. "
        "The result keeps the last 2000 lines or 50 KiB, whichever is smaller, and gives a log path "
        "when that shown output is truncated. Each command log stops at 100 MiB; "
        "reaching that limit stops the command."
    )
    params_model = RunCommandParams
    category = "execute"

    def __init__(
        self,
        root: Path,
        session: SessionDirectory,
        *,
        bash: str | None = None,
        grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
        line_limit: int = COMMAND_OUTPUT_LINE_LIMIT,
        byte_limit: int = COMMAND_OUTPUT_BYTE_LIMIT,
        log_limit: int = COMMAND_LOG_BYTE_LIMIT,
    ) -> None:
        self._root = root
        self._session = session
        self._bash = bash if bash is not None else shutil.which("bash")
        self.grace_seconds = grace_seconds
        self.line_limit = line_limit
        self.byte_limit = byte_limit
        self.log_limit = log_limit

    async def execute(self, params: RunCommandParams) -> ToolResult:
        if not self._bash:
            return error_result(
                "bash_not_found",
                "Bash was not found. Install Bash with your system package manager "
                "and make sure the bash command is on PATH.",
            )
        cwd = _resolve_cwd(self._root, params.cwd)
        if not cwd.exists():
            return error_result("cwd_not_found", f"Working directory does not exist: {cwd}.", cwd=str(cwd))
        if not cwd.is_dir():
            return error_result(
                "cwd_not_a_directory", f"Working directory is not a directory: {cwd}.", cwd=str(cwd),
            )
        log_path = self._session.create_log()
        with log_path.open("wb", buffering=0) as log_file:
            capture = _Capture(log_file, line_limit=self.line_limit, byte_limit=self.byte_limit, log_limit=self.log_limit)
            try:
                outcome = await run_process_group(
                    [self._bash, "-c", params.command],
                    cwd=cwd,
                    on_output=capture.add,
                    timeout_seconds=float(params.timeout_seconds),
                    grace_seconds=self.grace_seconds,
                )
            except OSError as error:
                return error_result(
                    "command_failed",
                    f"Cannot run the command: {error.strerror or 'unknown error'}.",
                    cwd=str(cwd),
                )
        return _result(
            outcome, capture, log_path=log_path, cwd=cwd, timeout_seconds=params.timeout_seconds,
            line_limit=self.line_limit, byte_limit=self.byte_limit, log_limit=self.log_limit,
        )


def _resolve_cwd(root: Path, cwd: str | None) -> Path:
    if cwd is None:
        return root
    selected = Path(cwd)
    if not selected.is_absolute():
        selected = root / selected
    return selected


class _Capture:
    """Append command bytes to a log and keep only a bounded tail in memory."""

    def __init__(self, log: BinaryIO, *, line_limit: int, byte_limit: int, log_limit: int) -> None:
        self._log = log
        self._line_limit = line_limit
        self._byte_limit = byte_limit
        self._log_limit = log_limit
        self.logged = 0
        self.limited = False
        self.cut_lines = False
        self.cut_bytes = False
        self.line_fragment = False
        self._tail = bytearray()

    def add(self, chunk: bytes) -> bool:
        if self.limited:
            return False
        room = self._log_limit - self.logged
        if room <= 0:
            self.limited = True
            return False
        accepted = chunk if len(chunk) <= room else chunk[:room]
        if len(accepted) < len(chunk):
            self.limited = True
        self._write(accepted)
        self._tail.extend(accepted)
        self._trim()
        return not self.limited

    def text(self) -> str:
        data = _utf8_slice(bytes(self._tail))
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    def _write(self, data: bytes) -> None:
        self._log.write(data)
        self.logged += len(data)

    def _trim(self) -> None:
        while self._tail:
            size = len(self._tail)
            lines = _line_count(self._tail)
            if lines > self._line_limit and size <= self._byte_limit:
                newline = self._tail.find(b"\n")
                if newline < 0:
                    return
                del self._tail[:newline + 1]
                self.cut_lines = True
                # Dropping through a newline removes a mid-line fragment from the front.
                self.line_fragment = False
                continue
            if size > self._byte_limit:
                self._drop_front(size - self._byte_limit)
                self.cut_bytes = True
                continue
            return

    def _drop_front(self, count: int) -> None:
        ended_on_newline = count > 0 and self._tail[count - 1] == ord("\n")
        del self._tail[:count]
        while self._tail and (self._tail[0] & 0xC0) == 0x80:
            del self._tail[:1]
            ended_on_newline = False
        if not ended_on_newline:
            self.line_fragment = True


def _line_count(data: bytearray) -> int:
    if not data:
        return 0
    newlines = data.count(b"\n")
    if data.endswith(b"\n"):
        return newlines
    return newlines + 1


def _utf8_slice(data: bytes) -> bytes:
    start = 0
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    end = len(data)
    index = end - 1
    while index >= start and (data[index] & 0xC0) == 0x80:
        index -= 1
    if index < start:
        return b""
    lead = data[index]
    if lead < 0x80:
        expected = 1
    elif lead & 0xE0 == 0xC0:
        expected = 2
    elif lead & 0xF0 == 0xE0:
        expected = 3
    elif lead & 0xF8 == 0xF0:
        expected = 4
    else:
        return data[start:index]
    if end - index < expected:
        end = index
    return data[start:end]


def _byte_text(size: int) -> str:
    if size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)} MiB"
    if size % 1024 == 0:
        return f"{size // 1024} KiB"
    return f"{size} bytes"


def _result(
    outcome: ProcessOutcome,
    capture: _Capture,
    *,
    log_path: Path,
    cwd: Path,
    timeout_seconds: int,
    line_limit: int,
    byte_limit: int,
    log_limit: int,
) -> ToolResult:
    limited = outcome.output_limited or capture.limited
    notes: list[str] = []
    details: dict[str, object] = {"exit_code": outcome.return_code, "cwd": str(cwd)}
    status: ToolStatus
    error_code: str | None = None
    if outcome.cancelled:
        status = "cancelled"
        notes.append("Command was cancelled and its process group was stopped.")
    elif limited:
        status = "error"
        error_code = "output_limit_exceeded"
        notes.append(
            f"Command stopped because captured output reached the {_byte_text(log_limit)} log limit. "
            f"The log at {log_path} contains only the captured prefix, not the complete command output."
        )
        details["log_path"] = str(log_path)
    elif outcome.timed_out:
        status = "error"
        error_code = "timeout"
        notes.append(f"Command timed out after {timeout_seconds} seconds and its process group was stopped.")
    elif outcome.return_code != 0:
        status = "error"
        error_code = "command_failed"
        notes.append(f"Command failed with exit code {outcome.return_code}.")
    else:
        status = "success"
        notes.append(f"Command finished with exit code {outcome.return_code}.")
    if status != "success":
        notes.append("Changes the command already made were kept.")
        details["side_effects"] = "kept"
    if capture.cut_lines or capture.cut_bytes:
        kept_parts: list[str] = []
        if capture.cut_lines:
            kept_parts.append(f"the last {line_limit} lines")
        if capture.cut_bytes:
            kept_parts.append(f"the last {_byte_text(byte_limit)}")
        kept = " and ".join(kept_parts)
        fragment = " The first line shown is only the tail of a longer line." if capture.line_fragment else ""
        if capture.cut_lines and capture.cut_bytes and not capture.line_fragment:
            note = (
                f"Shown output is the last {line_limit} lines. "
                f"The output also exceeded {_byte_text(byte_limit)}."
            )
        elif limited:
            note = f"Shown output is {kept} of the captured prefix.{fragment}"
        else:
            note = f"Shown output is {kept}.{fragment}"
        if limited:
            note += f" That prefix is in {log_path}."
        else:
            note += f" The full captured output is in {log_path}."
        notes.append(note)
        details["log_path"] = str(log_path)
        if capture.cut_lines and capture.cut_bytes:
            details["truncated_by"] = "lines_and_bytes"
        elif capture.cut_lines:
            details["truncated_by"] = "lines"
        else:
            details["truncated_by"] = "bytes"
        if capture.line_fragment:
            details["line_fragment"] = True
    shown = capture.text()
    output = " ".join(notes) if not shown else " ".join(notes) + "\n\n" + shown
    if error_code is None:
        return ToolResult(status, output, details=details)
    return error_result(error_code, output, **details)
