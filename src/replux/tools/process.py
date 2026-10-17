"""Run one process group and stop it without leaving children behind.

Search tools reuse this lifecycle. The direct child is not the whole group:
a command can exit while a child that ignored the first signal is still running.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import TypeVar


PROCESS_TERMINATION_GRACE_SECONDS = 2.0
_OUTPUT_CHUNK_BYTES = 16 * 1024
_POLL_SECONDS = 0.05
_KILL_WAIT_SECONDS = 1.0

_T = TypeVar("_T")


class BashNotFound(Exception):
    """Startup cannot continue because the bash executable is not on PATH."""


class RipgrepNotFound(Exception):
    """Startup cannot continue because the ripgrep executable is not on PATH."""


RIPGREP_INSTALL_HINT = (
    "Ripgrep was not found. Install ripgrep with your system package manager "
    "and make sure the rg command is on PATH before starting Replux."
)


def require_bash() -> str:
    """Return the bash executable, or raise BashNotFound with an install hint."""
    found = shutil.which("bash")
    if found is None:
        raise BashNotFound(
            "Bash was not found. Install Bash with your system package manager "
            "and make sure the bash command is on PATH before starting Replux."
        )
    return found


def require_ripgrep() -> str:
    """Return the ripgrep executable, or raise RipgrepNotFound with an install hint."""
    found = shutil.which("rg")
    if found is None:
        raise RipgrepNotFound(RIPGREP_INSTALL_HINT)
    return found


@dataclass(frozen=True)
class ProcessOutcome:
    """A process group that has been reaped. return_code is the direct child's wait status."""

    return_code: int
    timed_out: bool
    cancelled: bool
    output_limited: bool


async def run_process_group(
    argv: Sequence[str],
    *,
    cwd: Path,
    on_output: Callable[[bytes], bool],
    timeout_seconds: float,
    grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
) -> ProcessOutcome:
    """Run argv in its own session. stdin is closed. stdout and stderr are merged.

    on_output receives each chunk and returns False when capture must stop.
    Timeout, cancellation, and a False return all finish the process group before
    this coroutine returns. Cancellation is reported on the outcome; it is not re-raised.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=os.fspath(cwd),
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=_close_stdin,
    )
    pid = process.pid
    assert pid is not None
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = pid
    if pgid != pid:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Command process {pid} stayed in process group {pgid}.")
    reader = asyncio.create_task(_read_output(process, on_output))
    cancel_requested = False
    timed_out = False
    deadline = time.monotonic() + timeout_seconds
    try:
        try:
            await asyncio.wait({reader}, timeout=timeout_seconds)
        except asyncio.CancelledError:
            cancel_requested = True
        if not cancel_requested and not reader.done():
            timed_out = True
    finally:
        # A command that already closed its output keeps that result. Otherwise stop the group.
        # Closing stdout does not extend the time limit. _shield keeps cleanup running if
        # the caller is cancelled again.
        natural = _natural_eof(reader)
        parent = asyncio.current_task()
        already_cancelling = cancel_requested or (parent is not None and parent.cancelling() > 0)
        forced_timeout, saw_cancel = await _shield(_finish_process(
            process, pgid, reader, grace_seconds,
            kill=(not natural) or already_cancelling,
            remaining=deadline - time.monotonic(),
            parent=parent,
        ))
        if forced_timeout:
            timed_out = True
        if saw_cancel:
            cancel_requested = True
    assert process.returncode is not None
    if not reader.done():
        raise RuntimeError(f"Command process group {pgid} exited but its output was still open.")
    error = reader.exception()
    if error is not None:
        raise error
    # A command that already exited on its own keeps that status. A signal death while the
    # caller is cancelling is the cancellation, even if the exception arrived during cleanup.
    finished_on_its_own = natural and process.returncode >= 0
    task = asyncio.current_task()
    caller_cancelling = task is not None and task.cancelling() > 0
    cancelled = (cancel_requested or caller_cancelling) and not finished_on_its_own
    return ProcessOutcome(
        process.returncode,
        timed_out=timed_out and not cancelled,
        cancelled=cancelled,
        output_limited=reader.result() == "limit",
    )


def _close_stdin() -> None:
    os.close(0)


async def _read_output(process: asyncio.subprocess.Process, on_output: Callable[[bytes], bool]) -> str:
    assert process.stdout is not None
    while True:
        chunk = await process.stdout.read(_OUTPUT_CHUNK_BYTES)
        if not chunk:
            return "eof"
        if not on_output(chunk):
            return "limit"


def _natural_eof(reader: asyncio.Task[str]) -> bool:
    """True when the process closed its output, as opposed to timeout, cancel, or a capture limit."""
    return reader.done() and reader.exception() is None and reader.result() == "eof"


async def _finish_process(
    process: asyncio.subprocess.Process,
    pgid: int,
    reader: asyncio.Task[str],
    grace_seconds: float,
    *,
    kill: bool,
    remaining: float,
    parent: asyncio.Task[object] | None,
) -> tuple[bool, bool]:
    """Stop the group when required.

    The first flag means a closed stdout outlived the time limit.
    The second flag means the caller cancelled during that wait.
    """
    forced_timeout = False
    saw_cancel = False
    if kill or not reader.done():
        await _terminate_group(process, pgid, grace_seconds)
    else:
        # stdout already closed. Reap the shell's own status, then stop any child it left behind.
        # Poll the caller's cancel flag: this task is shielded, so its own awaits are not cancelled.
        if process.returncode is None:
            waiting = asyncio.create_task(process.wait())
            reason = await _wait_for_exit(waiting, remaining, parent)
            if reason == "timeout":
                forced_timeout = True
            elif reason == "cancelled":
                saw_cancel = True
            if reason != "done":
                await _terminate_group(process, pgid, grace_seconds)
            if not waiting.done():
                await waiting
            waiting.result()
        if not forced_timeout and not saw_cancel and _group_alive(pgid):
            await _terminate_group(process, pgid, grace_seconds)
    while not reader.done():
        await asyncio.wait({reader})
    if process.returncode is None:
        await process.wait()
    if _group_alive(pgid):
        raise RuntimeError(f"Process group {pgid} is still running after cleanup.")
    return forced_timeout, saw_cancel


async def _wait_for_exit(
    waiting: asyncio.Task[int], timeout: float, parent: asyncio.Task[object] | None,
) -> str:
    deadline = time.monotonic() + max(0.0, timeout)
    while not waiting.done():
        if parent is not None and parent.cancelling() > 0:
            return "cancelled"
        pause = deadline - time.monotonic()
        if pause <= 0:
            return "timeout"
        await asyncio.sleep(min(_POLL_SECONDS, pause))
    waiting.result()
    return "done"


async def _terminate_group(process: asyncio.subprocess.Process, pgid: int, grace_seconds: float) -> None:
    """SIGTERM the group, wait up to grace_seconds, then SIGKILL anyone still running."""
    waiting = asyncio.create_task(process.wait()) if process.returncode is None else None
    try:
        _signal_group(pgid, signal.SIGTERM)
        await _await_clear(process, pgid, waiting, grace_seconds)
        if process.returncode is None or _group_alive(pgid):
            _signal_group(pgid, signal.SIGKILL)
            await _await_clear(process, pgid, waiting, _KILL_WAIT_SECONDS)
        if process.returncode is None or _group_alive(pgid):
            raise RuntimeError(f"Process group {pgid} did not exit after SIGKILL.")
    finally:
        if waiting is not None and not waiting.done():
            _signal_group(pgid, signal.SIGKILL)
            await waiting


async def _await_clear(
    process: asyncio.subprocess.Process,
    pgid: int,
    waiting: asyncio.Task[int] | None,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if waiting is not None and waiting.done():
            waiting.result()
        if process.returncode is not None and not _group_alive(pgid):
            return
        pause = min(_POLL_SECONDS, deadline - time.monotonic())
        if pause <= 0:
            break
        if waiting is not None and not waiting.done():
            await asyncio.wait({waiting}, timeout=pause)
        else:
            await asyncio.sleep(pause)
    if waiting is not None and waiting.done():
        waiting.result()


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can return EPERM for a group that has already exited.
        return _group_listed(pgid)
    return True


def _group_listed(pgid: int) -> bool:
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pgid=,stat="], check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "ps failed"
        raise RuntimeError(f"Cannot list process group {pgid}: {detail}.")
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == str(pgid) and not parts[1].startswith("Z"):
            return True
    return False


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        if not _group_listed(pgid):
            return
        raise


async def _shield(call: Awaitable[_T]) -> _T:
    """Finish call even when the surrounding task is cancelled again."""
    task = asyncio.ensure_future(call)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        return task.result()
