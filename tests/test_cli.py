import fcntl
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import pytest

from tests.test_config import VALID_YAML


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("entry", ["command", "module"])
def test_public_entry_opens_full_screen_and_restores_terminal(
    tmp_path: Path, entry: str, protocol: str
) -> None:
    config = tmp_path / "provider.yaml"
    config.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    command = (
        [str(Path(sys.executable).with_name("replux"))]
        if entry == "command"
        else [sys.executable, "-m", "replux"]
    )
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    original = termios.tcgetattr(slave)
    process = subprocess.Popen(
        [*command, "--config", str(config)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
    )
    output = bytearray()

    def read_until(marker: bytes) -> None:
        deadline = time.monotonic() + 10
        while marker not in output:
            remaining = deadline - time.monotonic()
            assert remaining > 0, output.decode(errors="replace")
            ready, _, _ = select.select([master], [], [], remaining)
            assert ready, output.decode(errors="replace")
            output.extend(os.read(master, 65536))

    try:
        read_until("输入问题".encode())
        assert b"\x1b[?1049h" in output
        assert b"Replux" in output
        os.write(master, b"\x03")
        read_until(b"\x1b[?1049l")
        assert process.wait(timeout=10) == 0
        assert termios.tcgetattr(slave) == original
        assert b"test-secret-never-display" not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        os.close(slave)
