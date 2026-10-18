"""Bash commands, process-group cleanup, and session logs."""

import asyncio
from collections.abc import Iterator
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import tracemalloc
from typing import Any

import pytest
import httpx2 as httpx
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled

from replux.cli import create_app
from replux.llm import (
    IncompleteReplyError, LLMError, ReplyCompleted, ToolFinished, ToolStarted, create_client,
)
from replux.tools import Tool, ToolParams, ToolRegistry, ToolResult, create_default_registry
from replux.tools.process import PROCESS_TERMINATION_GRACE_SECONDS
from replux.tools.run_command import (
    COMMAND_LOG_BYTE_LIMIT, COMMAND_OUTPUT_BYTE_LIMIT, COMMAND_OUTPUT_LINE_LIMIT, COMMAND_TIMEOUT_LIMIT_SECONDS,
    COMMAND_TIMEOUT_SECONDS, RunCommandTool,
)
from replux.tools.session import SessionDirectory
from replux.tools.write_file import WriteFileTool
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events
from tests.test_config import VALID_YAML
from tests.test_tool_batch import batch_events, open_client, result_pairs, text_gate
from tests.test_tool_ui import statuses


_RESISTANT_CHILD = """
import os
import signal
import time

marker = open(os.environ["MARKER"], "a", buffering=1)

def on_term(signum, frame):
    marker.write("TERM %.6f\\n" % time.monotonic())
    marker.flush()

signal.signal(signal.SIGTERM, on_term)
while True:
    marker.write("tick %.6f\\n" % time.monotonic())
    marker.flush()
    time.sleep(0.05)
"""


class Commands:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions: list[SessionDirectory] = []

    def registry(
        self,
        *,
        grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
        line_limit: int = COMMAND_OUTPUT_LINE_LIMIT,
        byte_limit: int = COMMAND_OUTPUT_BYTE_LIMIT,
        log_limit: int = COMMAND_LOG_BYTE_LIMIT,
    ) -> ToolRegistry:
        session = SessionDirectory()
        self.sessions.append(session)
        registry = ToolRegistry(session)
        registry.register(WriteFileTool(self.root))
        registry.register(RunCommandTool(
            self.root, session,
            grace_seconds=grace_seconds, line_limit=line_limit, byte_limit=byte_limit, log_limit=log_limit,
        ))
        return registry

    def close(self) -> None:
        for session in self.sessions:
            session.cleanup()


@pytest.fixture
def commands(tmp_path: Path) -> Iterator[Commands]:
    box = Commands(tmp_path)
    yield box
    box.close()


async def run(registry: ToolRegistry, **arguments: Any) -> ToolResult:
    return await asyncio.wait_for(registry.execute("run_command", json.dumps(arguments)), timeout=8)


def command_call(call_id: str, command: str, **extra: object) -> tuple[str, str, tuple[str, ...]]:
    payload = json.dumps({"command": command, **extra}, ensure_ascii=False)
    marker = '{"command": "'
    assert payload.startswith(marker)
    return (call_id, "run_command", (marker, payload[len(marker):]))


def last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line]
    return lines[-1]


def process_state(pid: int) -> str:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    completed = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        return "dead"
    if completed.stdout.strip().startswith("Z"):
        return "zombie"
    return "running"


def resistant_command(directory: Path) -> tuple[str, Path, Path, Path]:
    script = directory / "resistant.py"
    script.write_text(_RESISTANT_CHILD, encoding="utf-8")
    leader = directory / "leader.pid"
    child = directory / "child.pid"
    marker = directory / "marker.txt"
    command = " ".join((
        f"echo $$ > {shlex.quote(str(leader))};",
        f"export MARKER={shlex.quote(str(marker))};",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} &",
        f"echo $! > {shlex.quote(str(child))};",
        "sleep 30",
    ))
    return command, leader, child, marker


async def wait_for_tick(marker: Path) -> None:
    for _ in range(100):
        if marker.exists() and "tick " in marker.read_text(encoding="utf-8"):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"{marker} never recorded a tick")


async def wait_for_text(path: Path) -> str:
    for _ in range(100):
        if path.exists() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8")
        await asyncio.sleep(0.02)
    raise AssertionError(f"{path} was not written")


def _provider_file(tmp_path: Path, protocol: str) -> Path:
    path = tmp_path / "provider.yaml"
    path.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    return path


def test_run_command_schema_is_execute_and_keeps_the_production_limits() -> None:
    registry = create_default_registry(Path("/startup"))
    tool = registry.get("run_command")
    definition = next(item for item in registry.definitions() if item.name == "run_command")
    assert isinstance(tool, RunCommandTool)
    assert tool.category == "execute"
    assert tool.grace_seconds == PROCESS_TERMINATION_GRACE_SECONDS == 2
    assert tool.line_limit == COMMAND_OUTPUT_LINE_LIMIT == 2000
    assert tool.byte_limit == COMMAND_OUTPUT_BYTE_LIMIT == 50 * 1024
    assert tool.log_limit == COMMAND_LOG_BYTE_LIMIT == 100 * 1024 * 1024
    assert definition.input_schema == tool.params_model.model_json_schema()
    schema = definition.input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["command"]
    assert schema["properties"]["command"]["minLength"] == 1
    assert {"minLength": 1, "type": "string"} in schema["properties"]["cwd"]["anyOf"]
    assert {"type": "null"} in schema["properties"]["cwd"]["anyOf"]
    assert schema["properties"]["timeout_seconds"]["default"] == COMMAND_TIMEOUT_SECONDS == 120
    assert schema["properties"]["timeout_seconds"]["minimum"] == 1
    assert schema["properties"]["timeout_seconds"]["maximum"] == COMMAND_TIMEOUT_LIMIT_SECONDS == 600
    assert definition.description.isascii()
    assert "non-interactive" in definition.description and "600" in definition.description
    assert "100 MiB" in definition.description
    assert registry.session is not None
    registry.session.cleanup()


@pytest.mark.parametrize("arguments", [
    {"command": ""},
    {"command": "printf hi", "cwd": ""},
    {"command": "printf hi", "timeout_seconds": 0},
    {"command": "printf hi", "timeout_seconds": -1},
    {"command": "printf hi", "timeout_seconds": 601},
    {"command": "printf hi", "timeout_seconds": 1.5},
    {"command": "printf hi", "timeout_seconds": "10"},
    {"command": "printf hi", "timeout_seconds": True},
    {"command": 1},
    {"cwd": "sub"},
    {"command": "printf hi", "extra": True},
    {},
])
async def test_invalid_command_arguments_do_not_run(
    commands: Commands, tmp_path: Path, arguments: dict[str, Any],
) -> None:
    target = tmp_path / "side.txt"
    registry = commands.registry()
    result = await registry.execute("run_command", json.dumps(arguments))
    assert (result.status, result.error_code) == ("error", "invalid_arguments")
    assert result.output.isascii() and result.output
    assert not target.exists()


async def test_the_maximum_timeout_is_accepted_and_a_larger_one_is_not_clamped(commands: Commands) -> None:
    registry = commands.registry()
    allowed = await run(registry, command="printf ok", timeout_seconds=600)
    assert allowed.status == "success"
    assert allowed.details["exit_code"] == 0
    assert "ok" in allowed.output
    rejected = await registry.execute("run_command", json.dumps({"command": "printf too-big", "timeout_seconds": 601}))
    assert rejected.error_code == "invalid_arguments"
    assert "too-big" not in rejected.output


async def test_a_fresh_bash_merges_output_closes_stdin_and_returns_the_exit_code(
    commands: Commands, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPLUX_PARENT_MARK", "from-parent")
    registry = commands.registry()
    first = await run(registry, command=(
        "printf 'from-stdout\\n'; printf 'from-stderr\\n' >&2; printf '%s\\n' $$; "
        "if echo probe >&0 2>/dev/null; then printf 'OPEN\\n'; else printf 'CLOSED\\n'; fi; "
        "if shopt -q login_shell; then printf 'LOGIN\\n'; else printf 'NOT-LOGIN\\n'; fi; "
        "case $- in *i*) printf 'INTERACTIVE\\n';; *) printf 'NOT-INTERACTIVE\\n';; esac; "
        "printf '%s\\n' \"$REPLUX_PARENT_MARK\""
    ))
    second = await run(registry, command="printf '%s\\n' $$")
    assert first.status == "success"
    assert first.error_code is None
    assert first.details["exit_code"] == 0
    output = first.output
    stdout_at = output.index("from-stdout")
    assert output.index("from-stderr") > stdout_at
    assert "CLOSED" in output
    assert "NOT-LOGIN" in output
    assert "NOT-INTERACTIVE" in output
    assert "from-parent" in output
    assert "OPEN" not in output
    flags = output.replace("NOT-LOGIN", "").replace("NOT-INTERACTIVE", "")
    assert "LOGIN" not in flags
    assert "INTERACTIVE" not in flags
    first_pid = int(output.split("from-stderr", 1)[1].split()[0])
    second_pid = int(last_line(second.output))
    assert first_pid != second_pid
    assert process_state(first_pid) == "dead"
    assert process_state(second_pid) == "dead"


async def test_cwd_and_exports_stay_inside_one_command(
    commands: Commands, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "other").mkdir()
    registry = commands.registry()
    monkeypatch.chdir(tmp_path / "sub")
    default_cwd = await run(registry, command="pwd")
    assert Path(last_line(default_cwd.output)) == tmp_path
    explicit = await run(registry, command="printf from-cwd > note.txt && pwd", cwd="sub")
    assert (tmp_path / "sub" / "note.txt").read_text(encoding="utf-8") == "from-cwd"
    assert Path(last_line(explicit.output)) == tmp_path / "sub"
    await run(registry, command="cd other && printf from-shell > inside.txt && export REPLUX_SENTINEL=from-command")
    assert (tmp_path / "other" / "inside.txt").read_text(encoding="utf-8") == "from-shell"
    followed = await run(registry, command='printf "pwd=%s env=%s\\n" "$(pwd)" "${REPLUX_SENTINEL-unset}"')
    followed_line = last_line(followed.output)
    assert followed_line == f"pwd={tmp_path} env=unset"
    written = await registry.execute("write_file", json.dumps({"path": "local.txt", "content": "from-tool"}))
    assert written.status == "success"
    assert (tmp_path / "local.txt").read_text(encoding="utf-8") == "from-tool"
    assert not (tmp_path / "other" / "local.txt").exists()
    assert not (tmp_path / "sub" / "local.txt").exists()


async def test_a_missing_or_non_directory_cwd_does_not_run(commands: Commands, tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    registry = commands.registry()
    missing = await run(registry, command="printf ran > side.txt", cwd="missing")
    assert missing.error_code == "cwd_not_found"
    assert not (tmp_path / "side.txt").exists()
    not_dir = await run(registry, command="printf ran > side.txt", cwd="file.txt")
    assert not_dir.error_code == "cwd_not_a_directory"
    assert not (tmp_path / "side.txt").exists()


async def test_empty_success_stays_success_and_nonzero_returns_the_output(commands: Commands, tmp_path: Path) -> None:
    registry = commands.registry()
    quiet = await run(registry, command=":")
    assert quiet.status == "success"
    assert quiet.error_code is None
    assert quiet.details["exit_code"] == 0
    assert quiet.output == "Command finished with exit code 0."
    target = tmp_path / "partial.txt"
    failed = await run(registry, command=f"printf partial > {shlex.quote(str(target))}; printf 'failed line\\n' >&2; exit 4")
    assert failed.status == "error"
    assert failed.error_code == "command_failed"
    assert failed.details["exit_code"] == 4
    assert failed.details["side_effects"] == "kept"
    assert "failed line" in failed.output
    assert "Changes the command already made were kept." in failed.output
    assert target.read_text(encoding="utf-8") == "partial"


async def test_cancel_stops_a_command_that_already_closed_stdout(commands: Commands, tmp_path: Path) -> None:
    leader = tmp_path / "leader.pid"
    registry = commands.registry(grace_seconds=0.4)
    tool = registry.get("run_command")
    assert isinstance(tool, RunCommandTool)
    running = asyncio.create_task(tool.execute(tool.params_model(
        command=f"echo $$ > {shlex.quote(str(leader))}; exec >/dev/null; sleep 30",
        timeout_seconds=30,
    )))
    await wait_for_text(leader)
    await asyncio.sleep(0.2)
    running.cancel()
    started = time.monotonic()
    result = await asyncio.wait_for(running, timeout=3)
    assert result.status == "cancelled"
    assert process_state(int(leader.read_text())) == "dead"
    assert time.monotonic() - started < 2


async def test_a_byte_cut_then_a_line_cut_does_not_keep_a_stale_fragment(
    commands: Commands, tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"abcdefghi\nL2\nL3\n")
    registry = commands.registry(byte_limit=10, line_limit=2)
    result = await run(registry, command=f"cat {shlex.quote(str(sample))}")
    assert result.details["truncated_by"] == "lines_and_bytes"
    assert "line_fragment" not in result.details
    assert "tail of a longer line" not in result.output
    assert "the last 2 lines" in result.output
    assert "also exceeded 10 bytes" in result.output
    assert result.output.endswith("L2\nL3\n")


async def test_closing_stdout_does_not_outrun_the_timeout(commands: Commands, tmp_path: Path) -> None:
    leader = tmp_path / "leader.pid"
    registry = commands.registry(grace_seconds=5)
    started = time.monotonic()
    result = await run(
        registry,
        command=f"echo $$ > {shlex.quote(str(leader))}; exec >/dev/null; sleep 30",
        timeout_seconds=1,
    )
    assert result.error_code == "timeout"
    assert process_state(int(leader.read_text())) == "dead"
    assert time.monotonic() - started < 2.5


async def test_timeout_kills_a_process_that_exits_on_the_first_signal(commands: Commands, tmp_path: Path) -> None:
    leader = tmp_path / "leader.pid"
    registry = commands.registry(grace_seconds=5)
    started = time.monotonic()
    result = await run(registry, command=f"echo $$ > {shlex.quote(str(leader))}; sleep 30", timeout_seconds=1)
    elapsed = time.monotonic() - started
    pid = int(leader.read_text())
    assert result.error_code == "timeout"
    assert result.details["exit_code"] == -signal.SIGTERM
    assert result.details["side_effects"] == "kept"
    assert "process group was stopped" in result.output
    assert process_state(pid) == "dead"
    assert elapsed < 2.5


async def test_timeout_kills_a_child_that_survives_the_leader(commands: Commands, tmp_path: Path) -> None:
    command, leader, child, marker = resistant_command(tmp_path)
    registry = commands.registry(grace_seconds=0.6)
    started = time.monotonic()
    result = await run(registry, command=command, timeout_seconds=1)
    elapsed = time.monotonic() - started
    leader_pid = int((await wait_for_text(leader)).strip())
    child_pid = int(child.read_text().strip())
    term_at, last_tick = _term_gap(marker)
    assert result.error_code == "timeout"
    assert result.details["exit_code"] == -signal.SIGTERM
    assert process_state(leader_pid) == "dead"
    assert process_state(child_pid) == "dead"
    assert last_tick - term_at >= 0.4
    assert 1.3 <= elapsed < 3


async def test_cancel_does_not_finish_until_the_surviving_child_is_reaped(commands: Commands, tmp_path: Path) -> None:
    command, leader, child, marker = resistant_command(tmp_path)
    registry = commands.registry(grace_seconds=0.6)
    tool = registry.get("run_command")
    assert isinstance(tool, RunCommandTool)
    running = asyncio.create_task(tool.execute(tool.params_model(command=command, timeout_seconds=30)))
    await wait_for_tick(marker)
    leader_pid = int(leader.read_text().strip())
    child_pid = int(child.read_text().strip())
    running.cancel()
    await asyncio.sleep(0.12)
    assert not running.done()
    assert process_state(child_pid) == "running"
    assert "TERM" in marker.read_text(encoding="utf-8")
    result = await asyncio.wait_for(running, timeout=3)
    assert result.status == "cancelled"
    assert result.error_code is None
    assert result.details["exit_code"] == -signal.SIGTERM
    assert result.details["side_effects"] == "kept"
    assert "process group was stopped" in result.output
    assert process_state(leader_pid) == "dead"
    assert process_state(child_pid) == "dead"


def _term_gap(marker: Path) -> tuple[float, float]:
    lines = marker.read_text(encoding="utf-8").splitlines()
    term_index = next(index for index, line in enumerate(lines) if line.startswith("TERM "))
    term_at = float(lines[term_index].split()[1])
    ticks = [float(line.split()[1]) for line in lines[term_index + 1:] if line.startswith("tick ")]
    assert ticks
    return term_at, ticks[-1]


async def test_line_and_byte_limits_keep_the_tail_and_a_full_log(commands: Commands, tmp_path: Path) -> None:
    script = tmp_path / "lines.py"
    script.write_text(
        "import sys\nfor number in range(1, 2002):\n    sys.stdout.write(f'L{number}\\n')\n",
        encoding="utf-8",
    )
    registry = commands.registry()
    lines = await run(registry, command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    full_lines = "".join(f"L{number}\n" for number in range(1, 2002))
    log_path = Path(str(lines.details["log_path"]))
    assert lines.status == "success"
    assert lines.details["truncated_by"] == "lines"
    assert log_path.read_text(encoding="utf-8") == full_lines
    assert "the last 2000 lines" in lines.output
    assert "The full captured output is in " in lines.output
    assert "\nL1\n" not in lines.output
    assert lines.output.rstrip().endswith("L2001")
    assert log_path.is_relative_to(Path(tempfile.gettempdir()))

    wide = tmp_path / "wide.py"
    wide.write_text(
        "import sys\nline = 'a' * 100 + '\\n'\nfor _ in range(600):\n    sys.stdout.write(line)\n",
        encoding="utf-8",
    )
    wide_result = await run(registry, command=f"{shlex.quote(sys.executable)} {shlex.quote(str(wide))}")
    full_wide = (b"a" * 100 + b"\n") * 600
    wide_log = Path(str(wide_result.details["log_path"]))
    assert wide_result.status == "success"
    assert wide_result.details["truncated_by"] == "bytes"
    assert "2000 lines" not in wide_result.output
    assert "the last 50 KiB" in wide_result.output
    assert wide_log.read_bytes() == full_wide
    assert wide_result.output.encode().endswith(full_wide[-50 * 1024:])


async def test_a_long_utf8_line_is_cut_on_a_character_boundary(commands: Commands) -> None:
    # "你" is 3 bytes. Eight characters are 24 bytes. A 10-byte tail starts inside a character,
    # so the shown text is the last three characters.
    text = "你" * 8
    registry = commands.registry(byte_limit=10, line_limit=50)
    result = await run(registry, command=f"printf %s {shlex.quote(text)}")
    assert result.details["truncated_by"] == "bytes"
    assert result.details["line_fragment"] is True
    assert "tail of a longer line" in result.output
    assert result.output.endswith("你" * 3)
    assert "\ufffd" not in result.output
    assert Path(str(result.details["log_path"])).read_text(encoding="utf-8") == text

    aligned = commands.registry(byte_limit=9, line_limit=50)
    exact = await run(aligned, command=f"printf %s {shlex.quote(text)}")
    assert exact.output.endswith("你" * 3)
    assert exact.details["line_fragment"] is True
    assert "\ufffd" not in exact.output


async def test_the_production_byte_limit_keeps_a_utf8_tail(commands: Commands) -> None:
    # 17067 characters are 51201 bytes. The 50 KiB tail drops a split character, leaving 17066.
    count = (50 * 1024) // 3 + 1
    assert len(("你" * count).encode()) == 50 * 1024 + 1
    script_text = "你" * count
    registry = commands.registry()
    result = await run(registry, command=f"printf %s {shlex.quote(script_text)}")
    assert result.details["truncated_by"] == "bytes"
    assert result.details["line_fragment"] is True
    assert result.output.endswith("你" * 17066)
    assert "\ufffd" not in result.output
    assert Path(str(result.details["log_path"])).read_bytes() == script_text.encode()


async def test_large_output_stays_out_of_memory(commands: Commands, tmp_path: Path) -> None:
    script = tmp_path / "big.py"
    script.write_text(
        "import sys\nremaining = 2_000_000\nchunk = b'x' * 8192\n"
        "while remaining:\n    piece = chunk[:remaining]\n    sys.stdout.buffer.write(piece)\n    remaining -= len(piece)\n",
        encoding="utf-8",
    )
    registry = commands.registry(byte_limit=4096, log_limit=3_000_000)
    tracemalloc.start()
    try:
        result = await run(registry, command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    log_path = Path(str(result.details["log_path"]))
    assert log_path.stat().st_size == 2_000_000
    assert len(result.output.encode()) < 20_000
    assert peak < 1_000_000


async def test_a_log_limit_stops_the_command_and_keeps_only_the_prefix(
    commands: Commands, tmp_path: Path,
) -> None:
    script = tmp_path / "prefix.py"
    script.write_text(
        "import os, sys, time\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "sys.stdout.buffer.write(b'abcdefghij' * 500)\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    pid_path = tmp_path / "writer.pid"
    registry = commands.registry(log_limit=1000, byte_limit=100)
    result = await run(
        registry,
        command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {shlex.quote(str(pid_path))}",
    )
    log_path = Path(str(result.details["log_path"]))
    assert result.error_code == "output_limit_exceeded"
    assert result.details["side_effects"] == "kept"
    assert log_path.read_bytes() == b"abcdefghij" * 100
    assert "not the complete command output" in result.output
    assert "full captured output" not in result.output
    assert process_state(int(pid_path.read_text())) == "dead"


async def test_session_logs_remain_readable_and_disappear_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = Path(tempfile.mkdtemp(prefix="replux-other-"))
    kept = other / "keep.txt"
    kept.write_text("stay", encoding="utf-8")
    script = tmp_path / "lines.py"
    script.write_text("import sys\nfor number in range(1, 2002):\n    sys.stdout.write(f'L{number}\\n')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events("anthropic", (command_call(
            "call_run", f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
        ),))),
        ControlledSSE(reply_events()),
    ])
    log_path: Path | None = None
    try:
        async with create_client(provider_config(), transport=transport) as client:
            monkeypatch.chdir(tmp_path.parent)
            received = [event async for event in client.stream_reply("run")]
            finished = next(event for event in received if isinstance(event, ToolFinished))
            log_path = Path(str(finished.result.details["log_path"]))
            assert log_path.read_text(encoding="utf-8") == "".join(f"L{number}\n" for number in range(1, 2002))
            assert log_path.parent.exists()
        assert log_path is not None
        assert not log_path.exists()
        assert not log_path.parent.exists()
        assert kept.read_text(encoding="utf-8") == "stay"
    finally:
        if other.exists():
            for child in other.iterdir():
                child.unlink()
            other.rmdir()


async def test_an_internal_error_still_removes_only_this_session(tmp_path: Path) -> None:
    other = Path(tempfile.mkdtemp(prefix="replux-other-"))
    kept = other / "keep.txt"
    kept.write_text("stay", encoding="utf-8")

    class BoomParams(ToolParams):
        command: str

    class Boom(Tool[BoomParams]):
        name = "boom"
        description = "Raise after the command has produced a log."
        params_model = BoomParams
        category = "execute"

        async def execute(self, params: BoomParams) -> ToolResult:
            raise RuntimeError("injected defect")

    session = SessionDirectory()
    registry = ToolRegistry(session)
    registry.register(RunCommandTool(tmp_path, session, line_limit=2))
    registry.register(Boom())
    calls = (
        command_call("call_run", "printf 'L1\\nL2\\nL3\\n'"),
        ("call_boom", "boom", ('{"command": "', 'x"}')),
    )
    transport = RecordingTransport([ControlledSSE(batch_events("anthropic", calls))])
    log_path: Path | None = None
    try:
        async with open_client("anthropic", transport, registry) as client:
            received: list[object] = []
            with pytest.raises(RuntimeError, match="injected defect"):
                async for event in client.stream_reply("run"):
                    received.append(event)
            finished = next(event for event in received if isinstance(event, ToolFinished))
            assert isinstance(finished, ToolFinished)
            log_path = Path(str(finished.result.details["log_path"]))
            assert log_path.read_text(encoding="utf-8") == "L1\nL2\nL3\n"
            assert log_path.exists()
        assert log_path is not None
        assert not log_path.parent.exists()
        assert kept.read_text(encoding="utf-8") == "stay"
    finally:
        session.cleanup()
        if other.exists():
            for child in other.iterdir():
                child.unlink()
            other.rmdir()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_both_protocols_run_a_real_command_and_replay_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    target = tmp_path / "side.txt"
    command = f"printf 'from-stdout\\n'; printf 'from-stderr\\n' >&2; printf kept > {shlex.quote(str(target))}"
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (command_call("call_run", command),), decoy=True)),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("run it")]
        _ = [event async for event in client.stream_reply("follow-up")]
    assert target.read_text(encoding="utf-8") == "kept"
    finished = next(event for event in received if isinstance(event, ToolFinished))
    assert finished.result.status == "success"
    assert finished.result.details["exit_code"] == 0
    assert "from-stdout" in finished.result.output
    assert "from-stderr" in finished.result.output
    assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["call_run"]
    summary, follow_up = transport.requests[1], transport.requests[2]
    assert result_pairs(protocol, summary)[0][1]["details"]["exit_code"] == 0
    assert result_pairs(protocol, follow_up)[0][0] == "call_run"
    summary_body = json.loads(summary.content)
    assert summary_body["tool_choice"] == ({"type": "none"} if protocol == "anthropic" else "none")
    assert "run_command" in {tool["name"] for tool in json.loads(transport.requests[0].content)["tools"]}


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_a_nonzero_command_stops_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    made = tmp_path / "partial.txt"
    skipped = tmp_path / "later.txt"
    calls = (
        command_call("call_bad", f"printf partial > {shlex.quote(str(made))}; printf 'failed line\\n' >&2; exit 4"),
        command_call("call_later", f"printf no > {shlex.quote(str(skipped))}"),
    )
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("run two")]
    assert made.read_text(encoding="utf-8") == "partial"
    assert not skipped.exists()
    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [(event.call_id, event.result.status) for event in finished] == [
        ("call_bad", "error"), ("call_later", "skipped"),
    ]
    assert finished[0].result.error_code == "command_failed"
    assert finished[0].result.details["exit_code"] == 4
    assert "failed line" in finished[0].result.output
    assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["call_bad"]
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == ["error", "skipped"]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancelling_a_command_keeps_its_side_effect_and_skips_the_rest(
    tmp_path: Path, protocol: str,
) -> None:
    target = tmp_path / "kept.txt"
    later = tmp_path / "later.txt"
    leader = tmp_path / "leader.pid"
    command = (
        f"echo $$ > {shlex.quote(str(leader))}; "
        f"printf kept > {shlex.quote(str(target))}; "
        "sleep 30"
    )
    calls = (command_call("call_run", command), command_call("call_later", f"printf no > {shlex.quote(str(later))}"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        reply = client.stream_reply("run them")
        while not isinstance(await anext(reply), ToolStarted):
            pass
        pending = asyncio.create_task(anext(reply))
        await wait_for_text(target)
        pending.cancel()
        finished = await asyncio.wait_for(pending, timeout=3)
        skipped = await asyncio.wait_for(anext(reply), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await anext(reply)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert isinstance(finished, ToolFinished) and isinstance(skipped, ToolFinished)
    assert finished.result.status == "cancelled"
    assert skipped.result.status == "skipped"
    assert target.read_text(encoding="utf-8") == "kept"
    assert not later.exists()
    assert process_state(int(leader.read_text())) == "dead"
    assert "process group was stopped" in finished.result.output
    assert finished.result.details["exit_code"] == -signal.SIGTERM
    statuses_seen = [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])]
    assert statuses_seen == ["cancelled", "skipped"]
    assert result_pairs(protocol, transport.requests[1])[0][1]["details"]["exit_code"] == -signal.SIGTERM
    assert "你好" not in transport.requests[1].content.decode()
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["incomplete", "cancel"])
async def test_a_failed_summary_keeps_the_command_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    target = tmp_path / "kept.txt"
    calls = (command_call("call_run", f"printf kept > {shlex.quote(str(target))}"),)
    monkeypatch.chdir(tmp_path)
    requests: list[httpx.Request] = []
    held: list[ControlledSSE] = []
    summary_started = asyncio.Event()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            events, gate = batch_events(protocol, calls), None
        elif len(requests) == 2 and ending == "incomplete":
            events, gate = reply_events(protocol)[:-1], None
        elif len(requests) == 2 and ending == "cancel":
            events, gate = reply_events(protocol), text_gate(protocol)
        else:
            events, gate = reply_events(protocol), None
        stream = ControlledSSE(events, gate_after=gate)
        if len(requests) == 2 and ending == "cancel":
            stream.release_close.clear()
            held.append(stream)
        if len(requests) == 2:
            summary_started.set()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with create_client(provider_config(protocol), transport=httpx.MockTransport(respond)) as client:
        monkeypatch.chdir(tmp_path.parent)
        received: list[object] = []

        async def consume() -> None:
            async for event in client.stream_reply("run it"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            if ending == "cancel":
                await asyncio.wait_for(summary_started.wait(), timeout=2)
                await asyncio.wait_for(held[0].blocked.wait(), timeout=2)
                consuming.cancel()
                await asyncio.wait_for(held[0].closing.wait(), timeout=2)
            else:
                with pytest.raises(LLMError):
                    await asyncio.wait_for(consuming, timeout=2)
        finally:
            if held:
                held[0].release_close.set()
            if ending == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(consuming, timeout=2)
        assert not any(isinstance(event, ReplyCompleted) for event in received)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert target.read_text(encoding="utf-8") == "kept"
    pairs = result_pairs(protocol, requests[-1])
    assert [(call_id, payload["status"], payload["details"]["exit_code"]) for call_id, payload in pairs] == [
        ("call_run", "success", 0),
    ]
    assert "你好" not in json.dumps(json.loads(requests[-1].content), ensure_ascii=False)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["eof", "cleanup"])
async def test_an_unfinished_command_response_has_no_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    target = tmp_path / "side.txt"
    calls = (command_call("call_run", f"printf ran > {shlex.quote(str(target))}"),)
    events = batch_events(protocol, calls)
    if ending == "eof" and protocol == "anthropic":
        del events[-2:]
    elif ending == "eof":
        events.pop()
    stream = ControlledSSE(events, close_error=httpx.ReadError("offline cleanup failure") if ending == "cleanup" else None)
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([stream, ControlledSSE(reply_events(protocol))])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received: list[object] = []
        if ending == "eof":
            with pytest.raises(IncompleteReplyError):
                async for event in client.stream_reply("run"):
                    received.append(event)
        else:
            with pytest.raises(LLMError) as caught:
                async for event in client.stream_reply("run"):
                    received.append(event)
            assert not isinstance(caught.value, IncompleteReplyError)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert not target.exists()
    assert not any(isinstance(event, (ToolStarted, ToolFinished)) for event in received)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_summarizes_the_command_and_expand_does_not_read_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    command = "printf '%s\\n' " + " ".join(f"L{number}" for number in range(1, 13))
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    session = SessionDirectory()
    registry = ToolRegistry(session)
    registry.register(RunCommandTool(tmp_path, session, line_limit=8))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (command_call("call_run", command),))),
        summary,
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            app.query_one("#prompt", TextArea).text = "run"
            await pilot.press("enter")
            await asyncio.wait_for(summary.blocked.wait(), timeout=2)
            await pilot.pause()
            view = app.query_one(".tool-call")
            preview = str(view.query_one(".tool-preview", Static).content)
            assert preview == "└ Truncated · Exited with code 0"
            log_path = Path(str(view.query_one(".tool-log", Static).content))
            assert log_path.read_text(encoding="utf-8").startswith("L1\n")
            assert app.query_one("#status", Static).content == "Working…"
            log_path.write_text("SECRET LOG", encoding="utf-8")
            output = view.query_one(".tool-details")
            assert not output.display
            await pilot.click(view.query_one(".tool-heading"))
            assert output.display
            expanded = str(output.query_one(".tool-output", Static).content)
            assert "SECRET LOG" not in expanded
            assert "L12" in expanded
            assert "L5" in expanded
            assert len(transport.requests) == 2
            summary.release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    finally:
        session.cleanup()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_stays_cancelling_until_the_process_group_is_reaped(
    tmp_path: Path, protocol: str,
) -> None:
    command, leader, child, marker = resistant_command(tmp_path)
    later = tmp_path / "later.txt"
    session = SessionDirectory()
    registry = ToolRegistry(session)
    registry.register(RunCommandTool(tmp_path, session, grace_seconds=0.8))
    calls = (command_call("call_run", command), command_call("call_later", f"printf no > {shlex.quote(str(later))}"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    try:
        async with app.run_test(size=(80, 32)) as pilot:
            prompt = app.query_one("#prompt", TextArea)
            prompt.text = "run them"
            await pilot.press("enter")
            await wait_for_tick(marker)
            await pilot.pause()
            assert app.query_one("#status", Static).content == f"Run {command}…"
            assert process_state(int(child.read_text())) == "running"
            worker = next(iter(app.workers))
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.query_one("#status", Static).content == "Cancelling…"
            assert not app.query(".turn-outcome")
            assert app.query_one(".tool-dot", Static).has_class("running")
            assert process_state(int(child.read_text())) == "running"
            prompt.text = "草稿"
            await pilot.press("enter")
            assert prompt.text == "草稿"
            assert len(transport.requests) == 1
            with pytest.raises(WorkerCancelled):
                await asyncio.wait_for(worker.wait(), timeout=3)
            await pilot.pause()
            assert process_state(int(leader.read_text())) == "dead"
            assert process_state(int(child.read_text())) == "dead"
            assert not later.exists()
            assert str(app.query_one(".turn-outcome", Static).content).startswith("Cancelled after ")
            assert app.query_one("#status", Static).content == "Ready"
            shown = [status.split(" · ", 1)[0] for status in statuses(app)]
            preview = str(app.query_one(".tool-preview", Static).content)
            assert shown == ["Cancelled", "Skipped"], preview
            prompt.text = "follow-up"
            await pilot.press("enter")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == [
            "cancelled", "skipped",
        ]
        assert "你好" not in transport.requests[1].content.decode()
    finally:
        session.cleanup()


def test_startup_without_bash_stops_before_any_tool_or_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(SystemExit) as caught:
        create_app(["--config", str(_provider_file(tmp_path, "anthropic"))])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "Bash was not found" in error
    assert "Install Bash" in error
    assert "PATH" in error


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_startup_with_bash_keeps_the_provider_config(
    tmp_path: Path, protocol: str, capsys: pytest.CaptureFixture[str],
) -> None:
    transport = RecordingTransport(ControlledSSE(reply_events(protocol)))
    app = create_app(["--config", str(_provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert "custom-model-20261001" in str(app.query_one("#startup-model", Static).content)
        assert app.query_one("#status", Static).content == "Ready"
    assert "Bash was not found" not in capsys.readouterr().err
    assert transport.requests == []
