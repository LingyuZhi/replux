"""Replace one file by writing a temporary file in its directory, then renaming it."""

from collections.abc import Callable
import errno
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Literal


class CommitHold:
    """Pause one file commit so a caller can cancel or inspect that point.

    ``before`` runs before any directory or temporary file is created.
    ``during`` runs after the temporary file is complete and before it replaces the target.
    """

    def __init__(self, stage: Literal["before", "during"]) -> None:
        if stage not in ("before", "during"):
            raise ValueError(f"Unknown commit stage: {stage!r}.")
        self.stage: Literal["before", "during"] = stage
        self.entered = threading.Event()
        self.release = threading.Event()


def commit_file(
    path: Path, data: bytes, checkpoint: Callable[[], None], *, hold: CommitHold | None = None,
) -> None:
    """Replace ``path`` with ``data``.

    A symlink stays in place and the bytes go to its target. Missing parent
    directories are created only after ``checkpoint``. An existing regular file
    keeps its permission bits. Once the temporary file is being written, this
    function finishes the replacement before returning. A failed replacement
    deletes the temporary file.
    """
    if hold is not None and hold.stage == "before":
        hold.entered.set()
        hold.release.wait()
    checkpoint()
    destination = target_file(path)
    if destination.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(destination))
    _ensure_parent(destination)
    fd, temp = _open_temp(destination.parent)
    replaced = False
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        mode = _existing_mode(destination)
        if mode is not None:
            os.chmod(temp, mode)
        if hold is not None and hold.stage == "during":
            hold.entered.set()
            hold.release.wait()
        os.replace(temp, destination)
        replaced = True
    finally:
        if not replaced:
            temp.unlink(missing_ok=True)


def target_file(path: Path) -> Path:
    """Return the file a write replaces. A symlink resolves to its target.

    A symlink loop has no target file. Raising here leaves the link in place.
    """
    if not path.is_symlink():
        return path
    resolved = Path(os.path.realpath(path))
    if resolved.is_symlink():
        raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), str(path))
    return resolved


def commit_error_code(error: OSError, state: str) -> str:
    """Map a failed replacement to a stable tool error code."""
    if state == "directory" or isinstance(error, IsADirectoryError):
        return "not_a_file"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, NotADirectoryError):
        return "not_a_directory"
    return "write_failed"


def describe_commit(path: Path, intended: bytes | None = None) -> tuple[str, str]:
    """Return the target's state after a commit stops or fails.

    ``written`` is used only when ``intended`` is present and those bytes are
    already on disk. The sentence reports that known state.
    """
    try:
        probe = target_file(path)
        if probe.is_dir():
            return "directory", f"{probe} is a directory and was not replaced."
        if not probe.exists():
            return "absent", f"{probe} was not created."
        current = probe.read_bytes()
    except OSError as error:
        if path.is_symlink():
            return "symlink", f"{path} is still a symlink and was not replaced."
        return "unknown", f"The state of {path} could not be read: {error.strerror or 'unknown error'}."
    if intended is not None and current == intended:
        return "written", f"{probe} contains the new {len(intended)} bytes."
    return "not_replaced", f"{probe} was not replaced. It contains {len(current)} bytes."


def _ensure_parent(destination: Path) -> None:
    parent = destination.parent
    cursor = parent
    while not cursor.exists() and cursor.parent != cursor:
        cursor = cursor.parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(cursor))
    parent.mkdir(parents=True, exist_ok=True)


def _existing_mode(destination: Path) -> int | None:
    try:
        info = os.stat(destination)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return stat.S_IMODE(info.st_mode)


def _open_temp(directory: Path) -> tuple[int, Path]:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    for _ in range(100):
        path = directory / f".replux-{secrets.token_hex(8)}.tmp"
        try:
            return os.open(path, flags, 0o666), path
        except FileExistsError:
            continue
    raise OSError(f"Could not create a temporary file in {directory}.")
