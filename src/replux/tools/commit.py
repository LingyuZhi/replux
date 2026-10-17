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
    destination = Path(os.path.realpath(path)) if path.is_symlink() else path
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
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass


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
