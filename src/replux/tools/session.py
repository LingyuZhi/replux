"""Files owned by one Replux session."""

import os
from pathlib import Path
import shutil
import tempfile


class SessionDirectory:
    """A private temporary directory for one session.

    The directory is created on first use. cleanup removes only that directory.
    """

    def __init__(self) -> None:
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = Path(tempfile.mkdtemp(prefix="replux-"))
        return self._path

    def create_log(self) -> Path:
        descriptor, name = tempfile.mkstemp(prefix="command-", suffix=".log", dir=self.path)
        os.close(descriptor)
        return Path(name)

    def cleanup(self) -> None:
        path = self._path
        if path is None:
            return
        shutil.rmtree(path)
        self._path = None
