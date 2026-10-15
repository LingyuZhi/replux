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
@pytest.mark.parametrize("entry", ["command", "module", "cancel", "defect", "early-exit"])
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
    injected = entry in {"cancel", "defect", "early-exit"}
    if injected:
        command = [sys.executable, "-c", """
import os
import replux.cli as cli
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, reply_events

protocol = os.environ["REPLUX_TEST_PROTOCOL"]
scenario = os.environ["REPLUX_TEST_SCENARIO"]
if scenario == "early-exit":
    class ExitingApp(RepluxApp):
        def on_input_submitted(self, event):
            super().on_input_submitted(event)
            self.exit()
    cli.RepluxApp = ExitingApp
gate = 2 if protocol == "anthropic" else 4
events = reply_events(protocol)
stream = ControlledSSE(
    events if scenario == "cancel" else events[:gate + 1],
    gate_after=gate if scenario == "cancel" else None,
    read_error=RuntimeError("injected defect") if scenario == "defect" else None,
)
transport = RecordingTransport(stream)
app = cli.create_app(transport=transport)
app.run()
print(f"REPLUX_RESOURCES:{stream.closed}:{transport.closed}:{len(transport.requests)}", flush=True)
raise SystemExit(app.return_code)
"""]
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    original = termios.tcgetattr(slave)
    process = subprocess.Popen(
        [*command, "--config", str(config)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "TERM": "xterm-256color", "REPLUX_TEST_PROTOCOL": protocol, "REPLUX_TEST_SCENARIO": entry},
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
        read_until("Type a message".encode())
        assert b"\x1b[?1049h" in output
        assert b"Replux" in output
        if injected:
            os.write(master, b"hello\r")
            if entry == "cancel":
                read_until("你好".encode())
                os.write(master, b"\x03")
                read_until("Cancelled".encode())
                assert process.poll() is None
                os.write(master, b"\x03")
            read_until(b"REPLUX_RESOURCES:False:True:0" if entry == "early-exit" else b"REPLUX_RESOURCES:True:True:1")
        else:
            os.write(master, b"\x03")
        read_until(b"\x1b[?1049l")
        assert process.wait(timeout=10) == (1 if entry == "defect" else 0)
        while select.select([master], [], [], 0)[0]:
            output.extend(os.read(master, 65536))
        assert b"RuntimeWarning" not in output
        assert b"Task exception was never retrieved" not in output
        if entry == "defect":
            assert b"injected defect" in output
        assert termios.tcgetattr(slave) == original
        assert b"test-secret-never-display" not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        os.close(slave)
