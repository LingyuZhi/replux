"""Find files with ripgrep: relative paths, scope, and limits."""

import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pytest
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled

from replux.cli import create_app
from replux.llm import ToolFinished, ToolStarted, create_client
from replux.tools import ToolRegistry, ToolResult, create_default_registry
from replux.tools.base import OUTPUT_LIMIT_BYTES
from replux.tools.find_files import FIND_RESULT_LIMIT, FIND_TIMEOUT_SECONDS, FindFilesTool
from replux.tools.process import PROCESS_TERMINATION_GRACE_SECONDS
from replux.tools.write_file import WriteFileTool
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events
from tests.test_config import VALID_YAML
from tests.test_run_command import _term_gap, process_state, wait_for_tick
from tests.test_tool_batch import batch_events, open_client, result_pairs
from tests.test_tool_ui import statuses
from tests.test_write_file import write_call


async def find(registry: ToolRegistry, **arguments: Any) -> ToolResult:
    return await registry.execute("find_files", json.dumps(arguments))


async def test_find_files_lists_matching_files_in_lexicographic_order(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "B.txt").write_text("B", encoding="utf-8")
    (package / "a+.txt").write_text("+", encoding="utf-8")
    (package / "a").mkdir()
    (package / "a" / "b.txt").write_text("ab", encoding="utf-8")
    (package / "notes.md").write_text("notes", encoding="utf-8")
    (package / "folder.txt").mkdir()
    (package / "folder.txt" / "inner.txt").write_text("inner", encoding="utf-8")
    result = await find(create_default_registry(tmp_path), pattern="*.txt", path="pkg")
    assert result.status == "success"
    assert result.error_code is None
    assert result.output == (
        "B.txt\n"
        "a+.txt\n"
        "a/b.txt\n"
        "folder.txt/inner.txt\n"
        "\n"
        f"4 files in {package}."
    )
    assert result.details == {"path": str(package), "count": 4, "total": 4}


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_project(root: Path) -> None:
    _write(root / "keep.txt")
    _write(root / "src" / "a.py")
    _write(root / "src" / "nested" / "c.py")
    _write(root / "src" / "secret.log")
    _write(root / "public.log")
    _write(root / "secret.log")
    _write(root / "ignored.txt")
    _write(root / "dir" / "skip.txt")
    _write(root / "dotignored.txt")
    _write(root / "rgignored.txt")
    _write(root / ".hidden.txt")
    _write(root / ".hidden_dir" / "inside.txt")
    _write(root / "sub" / ".secret.txt")
    _write(root / "data.bin", "placeholder")
    (root / "data.bin").write_bytes(b"\0\1\2")
    (root / ".gitignore").write_text("ignored.txt\ndir/skip.txt\n*.log\n!public.log\n", encoding="utf-8")
    (root / ".ignore").write_text("dotignored.txt\n", encoding="utf-8")
    (root / ".rgignore").write_text("rgignored.txt\n", encoding="utf-8")
    real = root / "real"
    real.mkdir()
    _write(real / "a.txt")
    os.symlink(real, root / "linkdir")
    os.symlink(root / "keep.txt", root / "filelink.txt")


def _paths(output: str) -> list[str]:
    note_at = output.find("\n\n")
    body = output if note_at < 0 else output[:note_at]
    return [line for line in body.split("\n") if line]


async def test_default_scope_skips_ignored_hidden_and_directory_symlinks(tmp_path: Path) -> None:
    write_project(tmp_path)
    result = await find(create_default_registry(tmp_path), pattern="*")
    assert result.status == "success"
    assert _paths(result.output) == [
        "data.bin", "keep.txt", "public.log", "real/a.txt", "src/a.py", "src/nested/c.py",
    ]
    assert "secret.log" not in result.output
    assert "filelink.txt" not in result.output
    assert "linkdir/" not in result.output


async def test_a_matching_glob_does_not_bypass_ignore_or_hidden_files(tmp_path: Path) -> None:
    write_project(tmp_path)
    logs = await find(create_default_registry(tmp_path), pattern="*.log")
    hidden = await find(create_default_registry(tmp_path), pattern=".hidden.txt")
    nested = await find(create_default_registry(tmp_path), pattern="nested/*.py", path="src")
    assert logs.status == hidden.status == nested.status == "success"
    assert _paths(logs.output) == ["public.log"]
    assert hidden.output == f"No files matched in {tmp_path}."
    assert _paths(nested.output) == ["nested/c.py"]
    assert nested.details["path"] == str(tmp_path / "src")


async def test_scope_switches_widen_hidden_and_ignored_files_separately(tmp_path: Path) -> None:
    write_project(tmp_path)
    hidden = await find(create_default_registry(tmp_path), pattern="*", include_hidden=True)
    ignored = await find(create_default_registry(tmp_path), pattern="*", include_ignored=True)
    both = await find(
        create_default_registry(tmp_path), pattern="*", include_hidden=True, include_ignored=True,
    )
    hidden_paths = _paths(hidden.output)
    ignored_paths = _paths(ignored.output)
    both_paths = _paths(both.output)
    assert ".hidden.txt" in hidden_paths and ".hidden_dir/inside.txt" in hidden_paths
    assert "sub/.secret.txt" in hidden_paths
    assert "secret.log" not in hidden_paths and "ignored.txt" not in hidden_paths
    assert "dotignored.txt" not in hidden_paths and "rgignored.txt" not in hidden_paths
    assert "ignored.txt" in ignored_paths and "secret.log" in ignored_paths and "src/secret.log" in ignored_paths
    assert "dir/skip.txt" in ignored_paths and "dotignored.txt" in ignored_paths and "rgignored.txt" in ignored_paths
    assert ".hidden.txt" not in ignored_paths and "sub/.secret.txt" not in ignored_paths
    assert ".hidden.txt" in both_paths and "secret.log" in both_paths and "rgignored.txt" in both_paths
    assert hidden_paths == sorted(hidden_paths)
    assert ignored_paths == sorted(ignored_paths)
    linked = await find(create_default_registry(tmp_path), pattern="*", path="linkdir")
    assert _paths(linked.output) == ["a.txt"]
    assert linked.details["path"] == str(tmp_path / "linkdir")


async def test_paths_are_directories_resolved_from_the_startup_directory(tmp_path: Path) -> None:
    startup = tmp_path / "startup"
    elsewhere = tmp_path / "elsewhere"
    startup.mkdir()
    _write(startup / "pkg" / "a.py")
    _write(elsewhere / "b.py")
    _write(startup / "pkg" / "note.txt")
    registry = create_default_registry(startup)
    relative = await find(registry, pattern="*.py", path="pkg")
    absolute = await find(registry, pattern="*.py", path=str(elsewhere))
    default = await find(registry, pattern="*.py")
    missing = await find(registry, pattern="*.py", path="missing")
    file_path = await find(registry, pattern="*.py", path="pkg/a.py")
    empty = await find(registry, pattern="*.nope", path="pkg")
    assert _paths(relative.output) == ["a.py"]
    assert relative.details["path"] == str(startup / "pkg")
    assert _paths(absolute.output) == ["b.py"]
    assert absolute.details["path"] == str(elsewhere)
    assert _paths(default.output) == ["pkg/a.py"]
    assert default.details["path"] == str(startup)
    assert (missing.status, missing.error_code) == ("error", "directory_not_found")
    assert str(startup / "missing") in missing.output
    assert (file_path.status, file_path.error_code) == ("error", "not_a_directory")
    assert empty.status == "success" and empty.error_code is None
    assert empty.output == f"No files matched in {startup / 'pkg'}."
    assert empty.details == {"path": str(startup / "pkg"), "count": 0, "total": 0}


def test_find_files_schema_is_read_and_keeps_the_production_limits() -> None:
    registry = create_default_registry(Path("/startup"))
    tool = registry.get("find_files")
    definition = next(item for item in registry.definitions() if item.name == "find_files")
    assert isinstance(tool, FindFilesTool)
    assert tool.category == "read"
    assert tool.timeout_seconds == FIND_TIMEOUT_SECONDS == 10
    assert tool.byte_limit == OUTPUT_LIMIT_BYTES == 50 * 1024
    assert tool.grace_seconds == PROCESS_TERMINATION_GRACE_SECONDS == 2
    assert definition.input_schema == tool.params_model.model_json_schema()
    schema = definition.input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["pattern"]
    assert schema["properties"]["pattern"]["minLength"] == 1
    assert schema["properties"]["path"]["default"] == "."
    assert schema["properties"]["path"]["minLength"] == 1
    limit = schema["properties"]["limit"]
    assert (limit["type"], limit["default"], limit["minimum"]) == ("integer", FIND_RESULT_LIMIT, 1)
    assert FIND_RESULT_LIMIT == 1000
    assert "maximum" not in limit
    assert schema["properties"]["include_hidden"]["default"] is False
    assert schema["properties"]["include_ignored"]["default"] is False
    assert definition.description.isascii()
    assert "ripgrep" in definition.description
    assert "1000" in definition.description and "50 KiB" in definition.description
    assert ".gitignore" in definition.description and ".rgignore" in definition.description


@pytest.mark.parametrize("arguments", [
    {},
    {"pattern": ""},
    {"pattern": "*.txt", "path": ""},
    {"pattern": "*.txt", "limit": 0},
    {"pattern": "*.txt", "limit": -1},
    {"pattern": "*.txt", "limit": 1.5},
    {"pattern": "*.txt", "limit": "10"},
    {"pattern": "*.txt", "include_hidden": "true"},
    {"pattern": "*.txt", "include_hidden": 1},
    {"pattern": "*.txt", "include_ignored": "false"},
    {"pattern": 1},
    {"pattern": "*.txt", "extra": True},
])
async def test_invalid_find_arguments_do_not_search(tmp_path: Path, arguments: dict[str, Any]) -> None:
    _write(tmp_path / "keep.txt")
    result = await create_default_registry(tmp_path).execute("find_files", json.dumps(arguments))
    assert (result.status, result.error_code) == ("error", "invalid_arguments")
    assert result.output.isascii() and result.output
    assert "keep.txt" not in result.output


async def test_unreadable_directories_and_bad_globs_are_errors(tmp_path: Path) -> None:
    visible = tmp_path / "visible.txt"
    visible.write_text("seen", encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "secret.txt").write_text("hidden", encoding="utf-8")
    os.chmod(blocked, 0)
    try:
        denied = await find(create_default_registry(tmp_path), pattern="*.txt")
    finally:
        os.chmod(blocked, 0o755)
    assert denied.status == "error"
    assert denied.error_code == "permission_denied"
    assert "Permission denied" in denied.output
    assert denied.details["path"] == str(tmp_path)
    invalid = await find(create_default_registry(tmp_path), pattern="[")
    assert invalid.status == "error"
    assert invalid.error_code == "invalid_pattern"
    assert "error parsing glob" in invalid.output


async def test_an_invalid_glob_is_an_error_even_when_nothing_is_in_scope(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "secret.log").write_text("x", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    empty_pattern = await find(registry, pattern="[", path="empty")
    ignored_only = await find(registry, pattern="[")
    assert empty_pattern.status == ignored_only.status == "error"
    assert empty_pattern.error_code == ignored_only.error_code == "invalid_pattern"
    assert "error parsing glob" in empty_pattern.output
    assert "error parsing glob" in ignored_only.output
    matched = await find(registry, pattern="*.txt", path="empty")
    assert matched.status == "success"
    assert matched.output == f"No files matched in {empty}."


async def test_a_missing_ripgrep_binary_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "keep.txt")
    registry = ToolRegistry()
    registry.register(FindFilesTool(tmp_path, ripgrep=""))
    missing = await find(registry, pattern="*.txt")
    assert missing.status == "error"
    assert missing.error_code == "ripgrep_not_found"
    assert "Ripgrep was not found" in missing.output
    assert "Install ripgrep" in missing.output and "PATH" in missing.output
    absent = ToolRegistry()
    absent.register(FindFilesTool(tmp_path, ripgrep=str(tmp_path / "missing-rg")))
    gone = await find(absent, pattern="*.txt")
    assert gone.error_code == "ripgrep_not_found"
    assert not (tmp_path / "missing-rg").exists()


def _numbered(directory: Path, count: int) -> None:
    directory.mkdir()
    for number in range(count):
        (directory / f"{number:04d}.txt").write_text("x", encoding="utf-8")


async def test_the_default_count_limit_keeps_the_lexicographic_head(tmp_path: Path) -> None:
    folder = tmp_path / "p"
    _numbered(folder, 1001)
    result = await find(create_default_registry(tmp_path), pattern="*.txt", path="p")
    paths = _paths(result.output)
    assert result.status == "success"
    assert result.details["truncated_by"] == "count"
    assert result.details["count"] == 1000
    assert result.details["total"] == 1001
    assert paths[0] == "0000.txt"
    assert paths[-1] == "0999.txt"
    assert "1000.txt" not in paths
    assert f"Showing the first 1000 of 1001 files in {folder} (result limit 1000)." in result.output
    assert "2000" not in result.output
    assert "50 KiB" not in result.output


async def test_a_higher_limit_can_pass_2000_files_when_the_bytes_still_fit(tmp_path: Path) -> None:
    folder = tmp_path / "n"
    _numbered(folder, 2500)
    result = await find(create_default_registry(tmp_path), pattern="*.txt", path="n", limit=2500)
    assert result.status == "success"
    assert result.error_code is None
    assert "truncated_by" not in result.details
    assert result.details["count"] == result.details["total"] == 2500
    assert result.output.startswith("0000.txt\n")
    assert result.output.endswith(f"\n\n2500 files in {folder}.")
    assert "2000.txt" in _paths(result.output)
    assert "2000 lines" not in result.output and "line limit" not in result.output


async def test_the_byte_limit_keeps_whole_paths_and_survives_a_higher_count_limit(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "你.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(FindFilesTool(tmp_path, byte_limit=11))
    counted = await find(registry, pattern="*.txt", limit=100)
    assert counted.status == "success"
    assert counted.details["truncated_by"] == "bytes"
    assert _paths(counted.output) == ["a.txt", "b.txt"]
    assert "c.txt" not in counted.output and "你.txt" not in counted.output
    assert "\ufffd" not in counted.output
    assert f"Showing the first 2 of 4 files in {tmp_path} (11 bytes limit)." in counted.output
    assert "Raising limit does not remove the byte limit." in counted.output
    assert "2000" not in counted.output
    tight = ToolRegistry()
    tight.register(FindFilesTool(tmp_path, byte_limit=4))
    none = await find(tight, pattern="*.txt")
    assert none.details["truncated_by"] == "bytes"
    assert none.details["count"] == 0
    assert "a.txt" not in none.output
    assert "\ufffd" not in none.output
    raised = ToolRegistry()
    raised.register(FindFilesTool(tmp_path, byte_limit=11))
    again = await find(raised, pattern="*.txt", limit=3)
    assert again.details["truncated_by"] == "bytes"
    assert _paths(again.output) == ["a.txt", "b.txt"]


async def test_the_production_byte_limit_truncates_before_the_count_limit(tmp_path: Path) -> None:
    folder = tmp_path / "w"
    folder.mkdir()
    for number in range(700):
        (folder / f"{'n' * 80}{number:03d}.txt").write_text("x", encoding="utf-8")
    default = await find(create_default_registry(tmp_path), pattern="*.txt", path="w")
    raised = await find(create_default_registry(tmp_path), pattern="*.txt", path="w", limit=5000)
    assert default.status == raised.status == "success"
    assert default.details["truncated_by"] == raised.details["truncated_by"] == "bytes"
    assert default.details["count"] == raised.details["count"]
    assert isinstance(default.details["count"], int) and default.details["count"] < 700
    assert _paths(default.output) == _paths(raised.output)
    assert "50 KiB limit" in default.output
    assert "2000" not in default.output
    assert len("\n".join(_paths(default.output)).encode()) <= 50 * 1024


_SLOW_CHILD = """
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


def install_slow_ripgrep(directory: Path) -> tuple[Path, Path, Path, Path]:
    child_script = directory / "resistant.py"
    child_script.write_text(_SLOW_CHILD, encoding="utf-8")
    leader = directory / "leader.pid"
    child = directory / "child.pid"
    marker = directory / "marker.txt"
    ripgrep = directory / "rg"
    ripgrep.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import time\n"
        "open(os.environ['LEADER'], 'w').write(str(os.getpid()))\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, {str(child_script)!r}])\n"
        "open(os.environ['CHILD'], 'w').write(str(pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    ripgrep.chmod(0o755)
    return ripgrep, leader, child, marker


def arm_slow_ripgrep(monkeypatch: pytest.MonkeyPatch, leader: Path, child: Path, marker: Path) -> None:
    monkeypatch.setenv("LEADER", str(leader))
    monkeypatch.setenv("CHILD", str(child))
    monkeypatch.setenv("MARKER", str(marker))


def find_call(call_id: str, pattern: str, **extra: object) -> tuple[str, str, tuple[str, ...]]:
    payload = json.dumps({"pattern": pattern, **extra}, ensure_ascii=False)
    marker = '{"pattern": "'
    assert payload.startswith(marker)
    return (call_id, "find_files", (marker, payload[len(marker):]))


def provider_file(directory: Path, protocol: str) -> Path:
    path = directory / "provider.yaml"
    path.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    return path


async def test_timeout_stops_the_search_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ripgrep, leader, child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, child, marker)
    registry = ToolRegistry()
    registry.register(FindFilesTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=1, grace_seconds=0.6))
    started = time.monotonic()
    result = await find(registry, pattern="*")
    elapsed = time.monotonic() - started
    leader_pid = int(leader.read_text())
    child_pid = int(child.read_text())
    term_at, last_tick = _term_gap(marker)
    assert result.status == "error", result.output
    assert result.error_code == "timeout", result.output
    assert "The search process was stopped." in result.output
    assert "within 1 seconds" in result.output
    assert process_state(leader_pid) == "dead"
    assert process_state(child_pid) == "dead"
    assert last_tick - term_at >= 0.4
    assert 1.3 <= elapsed < 3


async def test_cancel_waits_until_the_search_process_group_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ripgrep, leader, child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, child, marker)
    registry = ToolRegistry()
    tool = FindFilesTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=30, grace_seconds=0.6)
    registry.register(tool)
    running = asyncio.create_task(find(registry, pattern="*"))
    await wait_for_tick(marker)
    leader_pid = int(leader.read_text())
    child_pid = int(child.read_text())
    running.cancel()
    await asyncio.sleep(0.12)
    assert not running.done()
    assert process_state(child_pid) == "running"
    assert "TERM" in marker.read_text(encoding="utf-8")
    result = await asyncio.wait_for(running, timeout=3)
    assert result.status == "cancelled"
    assert result.error_code is None
    assert "process group was stopped" in result.output
    assert process_state(leader_pid) == "dead"
    assert process_state(child_pid) == "dead"


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_both_protocols_find_real_files_and_replay_the_sorted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "b.txt").write_text("b", encoding="utf-8")
    (package / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "notes.md").write_text("notes", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (find_call("call_find", "*.txt", path="pkg"),), decoy=True)),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("find the text files")]
        _ = [event async for event in client.stream_reply("follow-up")]
    finished = next(event for event in received if isinstance(event, ToolFinished))
    assert finished.result.status == "success"
    assert _paths(finished.result.output) == ["a.txt", "b.txt"]
    assert "notes.md" not in finished.result.output
    assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["call_find"]
    summary, follow_up = transport.requests[1], transport.requests[2]
    replayed = result_pairs(protocol, summary)[0][1]
    assert replayed["status"] == "success"
    assert _paths(str(replayed["output"])) == ["a.txt", "b.txt"]
    assert result_pairs(protocol, follow_up)[0][0] == "call_find"
    summary_body = json.loads(summary.content)
    assert summary_body["tool_choice"] == ({"type": "none"} if protocol == "anthropic" else "none")
    tools = json.loads(transport.requests[0].content)["tools"]
    definition = next(tool for tool in tools if tool["name"] == "find_files")
    schema = definition["input_schema"] if protocol == "anthropic" else definition["parameters"]
    assert schema["properties"]["include_hidden"]["default"] is False
    assert schema["properties"]["include_ignored"]["default"] is False
    assert schema["properties"]["limit"]["default"] == 1000
    assert "permission" not in schema["properties"]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancelling_a_search_skips_the_rest_of_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    ripgrep, leader, _child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, tmp_path / "child.pid", marker)
    later = tmp_path / "later.txt"
    registry = ToolRegistry()
    registry.register(FindFilesTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=30, grace_seconds=0.6))
    registry.register(WriteFileTool(tmp_path))
    calls = (find_call("call_find", "*.txt"), write_call("call_later", "later.txt", "no"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("find then write")
        while not isinstance(await anext(reply), ToolStarted):
            pass
        pending = asyncio.create_task(anext(reply))
        await wait_for_tick(marker)
        pending.cancel()
        finished = await asyncio.wait_for(pending, timeout=3)
        skipped = await asyncio.wait_for(anext(reply), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await anext(reply)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert isinstance(finished, ToolFinished) and isinstance(skipped, ToolFinished)
    assert finished.result.status == "cancelled"
    assert "process group was stopped" in finished.result.output
    assert skipped.result.status == "skipped"
    assert not later.exists()
    assert process_state(int(leader.read_text())) == "dead"
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == [
        "cancelled", "skipped",
    ]
    assert "你好" not in transport.requests[1].content.decode()
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_summarizes_counts_and_expand_matches_the_model_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt", ".hidden.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / ".secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "secret.log").write_text("log", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".secret.txt\nsecret.log\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (find_call("call_find", "*.txt", include_hidden=True),))),
        summary,
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(100, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "find text files"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        view = app.query_one(".tool-call")
        assert view.query_one(".tool-header", Static).content == f"Find *.txt in {tmp_path}"
        assert view.query_one(".tool-status", Static).content == "Success"
        preview = str(view.query_one(".tool-preview", Static).content)
        assert preview == "└ Found 7 files · Returned 7"
        assert app.query_one("#status", Static).content == "Generating..."
        assert "Confirm" not in str(app.query_one("#status", Static).content)
        output = view.query_one(".tool-details")
        await pilot.click(view.query_one(".tool-heading"))
        expanded = str(output.query_one(".tool-output", Static).content)
        assert ".secret.txt" not in expanded
        assert "secret.log" not in expanded
        assert len(transport.requests) == 2
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
    replayed = result_pairs(protocol, transport.requests[1])[0][1]
    assert expanded == replayed["output"]
    assert replayed["status"] == "success"
    assert _paths(expanded)[0] == ".hidden.txt"
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_stays_cancelling_until_the_search_process_group_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    ripgrep, leader, child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, child, marker)
    later = tmp_path / "later.txt"
    registry = ToolRegistry()
    registry.register(FindFilesTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=30, grace_seconds=0.8))
    registry.register(WriteFileTool(tmp_path))
    calls = (find_call("call_find", "*.txt"), write_call("call_later", "later.txt", "no"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "find them"
        await pilot.press("enter")
        await wait_for_tick(marker)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Running find_files..."
        assert process_state(int(child.read_text())) == "running"
        worker = next(iter(app.workers))
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Cancelling..."
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
        assert app.query_one("#status", Static).content == "Ready"
        shown = [status.split(" · ", 1)[0] for status in statuses(app)]
        assert shown == ["Cancelled", "Skipped"]
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == [
        "cancelled", "skipped",
    ]
    assert "你好" not in transport.requests[1].content.decode()


def test_startup_without_ripgrep_stops_before_any_tool_or_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bash = bin_dir / "bash"
    bash.write_text("#!/bin/sh\n", encoding="utf-8")
    bash.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(SystemExit) as caught:
        create_app(["--config", str(provider_file(tmp_path, "anthropic"))])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "Ripgrep was not found" in error
    assert "Install ripgrep" in error
    assert "PATH" in error
    assert "Bash was not found" not in error
