"""Search file contents with ripgrep: matches, scope, and limits."""

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any

import pytest
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled

from replux.cli import create_app
from replux.llm import ToolFinished, ToolStarted, create_client
from replux.tools import ToolRegistry, ToolResult, create_default_registry
from replux.tools.base import OUTPUT_LIMIT_BYTES
from replux.tools.process import PROCESS_TERMINATION_GRACE_SECONDS
from replux.tools.search_content import (
    LINE_CHAR_LIMIT, SEARCH_RESULT_LIMIT, SEARCH_TIMEOUT_SECONDS, SearchContentTool,
)
from replux.tools.write_file import WriteFileTool
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events
from tests.test_find_files import arm_slow_ripgrep, install_slow_ripgrep, provider_file
from tests.test_run_command import _term_gap, process_state, wait_for_tick
from tests.test_tool_batch import batch_events, open_client, result_pairs
from tests.test_tool_ui import statuses
from tests.test_write_file import write_call


async def search(registry: ToolRegistry, **arguments: Any) -> ToolResult:
    return await registry.execute("search_content", json.dumps(arguments))


async def test_search_content_returns_case_sensitive_regex_matches(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "notes.txt").write_text("alpha\nALPHA\nalpha beta\n", encoding="utf-8")
    result = await search(create_default_registry(tmp_path), pattern="alpha", path="pkg")
    assert result.status == "success"
    assert result.error_code is None
    assert result.output == (
        "notes.txt:1:alpha\n"
        "notes.txt:3:alpha beta\n"
        "\n"
        f"2 matches in {package}."
    )
    assert result.details == {"path": str(package), "count": 2}


async def test_no_match_is_a_successful_empty_result(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "notes.txt").write_text("alpha\n", encoding="utf-8")
    result = await search(create_default_registry(tmp_path), pattern="missing", path="pkg")
    assert result.status == "success"
    assert result.error_code is None
    assert result.output == f"No matches in {package}."
    assert result.details == {"path": str(package), "count": 0}
    assert "exited with status" not in result.output


async def test_literal_matches_text_and_regex_uses_the_pattern(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("a.b\naxb\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    regex = await search(registry, pattern="a.b")
    literal = await search(registry, pattern="a.b", literal=True)
    escaped = await search(registry, pattern=r"a\.b", literal=True)
    assert regex.status == literal.status == escaped.status == "success"
    assert regex.output == (
        "notes.txt:1:a.b\n"
        "notes.txt:2:axb\n"
        "\n"
        f"2 matches in {tmp_path}."
    )
    assert literal.output == (
        "notes.txt:1:a.b\n"
        "\n"
        f"1 match in {tmp_path}."
    )
    assert escaped.output == f"No matches in {tmp_path}."
    assert escaped.error_code is None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_project(root: Path) -> None:
    _write(root / "keep.txt", "keep-hit\n")
    _write(root / "src" / "a.py", "code-hit\n")
    _write(root / "src" / "secret.log", "logged-secret-hit\n")
    _write(root / "public.log", "public-hit\n")
    _write(root / "secret.log", "logged-secret-hit\n")
    _write(root / "ignored.txt", "ignored-hit\n")
    _write(root / "dotignored.txt", "dot-hit\n")
    _write(root / "rgignored.txt", "rg-hit\n")
    _write(root / ".hidden.txt", "hidden-hit\n")
    _write(root / ".hidden_dir" / "inside.txt", "inside-hit\n")
    _write(root / "sub" / ".secret.txt", "nested-dot-hit\n")
    (root / "data.bin").write_bytes(b"before\0binary-hit\n")
    _write(root / "real" / "a.txt", "linked-hit\n")
    outside = root.parent / "outside"
    outside.mkdir()
    _write(outside / "x.txt", "outside-hit\n")
    os.symlink(outside, root / "dirlink")
    os.symlink(root / "real", root / "linkdir")
    os.symlink(root / "keep.txt", root / "filelink.txt")
    (root / ".gitignore").write_text("ignored.txt\n*.log\n!public.log\n", encoding="utf-8")
    (root / ".ignore").write_text("dotignored.txt\n", encoding="utf-8")
    (root / ".rgignore").write_text("rgignored.txt\n", encoding="utf-8")


def _lines(output: str) -> list[str]:
    note_at = output.find("\n\n")
    body = output if note_at < 0 else output[:note_at]
    return [line for line in body.split("\n") if line]


async def test_default_scope_skips_ignored_hidden_binary_and_directory_symlinks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_project(project)
    result = await search(create_default_registry(tmp_path), pattern="hit", path="project")
    assert result.status == "success"
    assert sorted(_lines(result.output)) == [
        "keep.txt:1:keep-hit",
        "public.log:1:public-hit",
        "real/a.txt:1:linked-hit",
        "src/a.py:1:code-hit",
    ]
    assert "secret-hit" not in result.output
    assert "hidden-hit" not in result.output
    assert "ignored-hit" not in result.output
    assert "binary-hit" not in result.output
    assert "outside-hit" not in result.output
    assert "dirlink/" not in result.output
    assert "linkdir/" not in result.output
    assert "filelink.txt" not in result.output


async def test_scope_switches_widen_hidden_and_ignored_content_separately(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_project(project)
    registry = create_default_registry(tmp_path)
    hidden = await search(registry, pattern="hit", path="project", include_hidden=True)
    ignored = await search(registry, pattern="hit", path="project", include_ignored=True)
    hidden_lines = _lines(hidden.output)
    ignored_lines = _lines(ignored.output)
    assert "hidden-hit" in hidden.output and ".hidden.txt:1:hidden-hit" in hidden_lines
    assert ".hidden_dir/inside.txt:1:inside-hit" in hidden_lines
    assert "sub/.secret.txt:1:nested-dot-hit" in hidden_lines
    assert "logged-secret-hit" not in hidden.output and "ignored.txt" not in hidden.output
    assert "dotignored.txt" not in hidden.output and "rgignored.txt" not in hidden.output
    assert "outside-hit" not in hidden.output
    assert "ignored.txt:1:ignored-hit" in ignored_lines
    assert "secret.log:1:logged-secret-hit" in ignored_lines
    assert "src/secret.log:1:logged-secret-hit" in ignored_lines
    assert "dotignored.txt:1:dot-hit" in ignored_lines
    assert "rgignored.txt:1:rg-hit" in ignored_lines
    assert ".hidden.txt" not in ignored.output and "sub/.secret.txt" not in ignored.output
    linked = await search(registry, pattern="hit", path="project/linkdir")
    assert _lines(linked.output) == ["a.txt:1:linked-hit"]
    assert linked.details["path"] == str(project / "linkdir")


async def test_an_explicit_file_is_searched_and_binary_content_is_skipped(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write_project(project)
    registry = create_default_registry(tmp_path)
    ignored = await search(registry, pattern="hit", path="project/ignored.txt")
    hidden = await search(registry, pattern="hit", path="project/.hidden.txt")
    binary = await search(registry, pattern="hit", path="project/data.bin")
    linked = await search(registry, pattern="hit", path="project/filelink.txt")
    assert ignored.status == hidden.status == binary.status == linked.status == "success"
    assert ignored.output == (
        "ignored.txt:1:ignored-hit\n"
        "\n"
        f"1 match in {project / 'ignored.txt'}."
    )
    assert hidden.output == (
        ".hidden.txt:1:hidden-hit\n"
        "\n"
        f"1 match in {project / '.hidden.txt'}."
    )
    assert binary.output == f"No matches in {project / 'data.bin'}."
    assert binary.error_code is None
    assert binary.details["count"] == 0
    assert linked.output == (
        "filelink.txt:1:keep-hit\n"
        "\n"
        f"1 match in {project / 'filelink.txt'}."
    )


async def test_each_matching_line_is_one_result_without_neighboring_lines(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("before\naa aa\nafter\n", encoding="utf-8")
    result = await search(create_default_registry(tmp_path), pattern="aa")
    assert result.status == "success"
    assert result.details["count"] == 1
    assert result.output == (
        "notes.txt:2:aa aa\n"
        "\n"
        f"1 match in {tmp_path}."
    )


async def test_invalid_regex_and_read_failures_are_explicit_errors(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("alpha\n", encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "secret.txt").write_text("alpha\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    invalid = await search(registry, pattern="[")
    missing = await search(registry, pattern="alpha", path="missing.txt")
    os.chmod(blocked, 0)
    try:
        denied = await search(registry, pattern="alpha")
    finally:
        os.chmod(blocked, 0o755)
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "secret.log").write_text("alpha\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    ignored_only = await search(registry, pattern="[")
    empty_pattern = await search(registry, pattern="[", path="empty")
    assert invalid.status == ignored_only.status == empty_pattern.status == "error"
    assert invalid.error_code == ignored_only.error_code == empty_pattern.error_code == "invalid_pattern"
    assert "regex parse error" in invalid.output
    assert "alpha" not in invalid.output
    assert missing.status == "error"
    assert missing.error_code == "path_not_found"
    assert f"Path not found: {tmp_path / 'missing.txt'}" in missing.output
    assert denied.status == "error"
    assert denied.error_code == "permission_denied"
    assert "Permission denied" in denied.output
    assert "notes.txt:1:alpha" not in denied.output
    lines = "".join(f"m{number}\n" for number in range(101))
    (tmp_path / "many.txt").write_text(lines, encoding="utf-8")
    os.chmod(blocked, 0)
    try:
        truncated = await search(registry, pattern="m", limit=1)
    finally:
        os.chmod(blocked, 0o755)
    assert truncated.status == "error"
    assert truncated.error_code == "permission_denied"
    assert "truncated_by" not in truncated.details
    assert "Showing the first" not in truncated.output


async def test_a_match_that_is_not_utf8_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_bytes(b"caf\xe9\n")
    result = await search(create_default_registry(tmp_path), pattern="caf")
    assert result.status == "error"
    assert result.error_code == "unsupported_encoding"
    assert "not valid UTF-8" in result.output
    assert "\ufffd" not in result.output


@pytest.mark.parametrize("arguments", [
    {},
    {"pattern": ""},
    {"pattern": "alpha", "path": ""},
    {"pattern": "alpha", "literal": "true"},
    {"pattern": "alpha", "literal": 1},
    {"pattern": "alpha", "include_hidden": "false"},
    {"pattern": "alpha", "include_ignored": 0},
    {"pattern": "alpha", "limit": 0},
    {"pattern": "alpha", "limit": -1},
    {"pattern": "alpha", "limit": 1.5},
    {"pattern": "alpha", "limit": "10"},
    {"pattern": 1},
    {"pattern": "alpha", "extra": True},
])
async def test_invalid_search_arguments_do_not_search(tmp_path: Path, arguments: dict[str, Any]) -> None:
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")
    result = await create_default_registry(tmp_path).execute("search_content", json.dumps(arguments))
    assert (result.status, result.error_code) == ("error", "invalid_arguments")
    assert result.output.isascii() and result.output
    assert "alpha" not in result.output


async def test_the_default_count_limit_keeps_the_first_100_matches(tmp_path: Path) -> None:
    lines = [f"m{number:04d}" for number in range(101)]
    (tmp_path / "notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = await search(create_default_registry(tmp_path), pattern=r"m\d+")
    shown = _lines(result.output)
    assert result.status == "success"
    assert result.details["truncated_by"] == "count"
    assert result.details["count"] == 100
    assert shown[0] == "notes.txt:1:m0000"
    assert shown[-1] == "notes.txt:100:m0099"
    assert "m0100" not in shown
    assert f"Showing the first 100 matches in {tmp_path} (result limit 100)." in result.output
    assert "50 KiB" not in result.output
    assert "2000" not in result.output


async def test_a_line_is_cut_to_500_unicode_characters_without_splitting_utf8(tmp_path: Path) -> None:
    long_line = "你" * 600
    (tmp_path / "notes.txt").write_text(f"{long_line}\nshort\n", encoding="utf-8")
    (tmp_path / "exact.txt").write_text(("你" * 500) + "\n", encoding="utf-8")
    (tmp_path / "bytes.txt").write_text(("你" * 167) + "\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    long = await search(registry, pattern="你", path="notes.txt")
    exact = await search(registry, pattern="你", path="exact.txt")
    by_bytes = await search(registry, pattern="你", path="bytes.txt")
    assert long.status == exact.status == by_bytes.status == "success"
    assert _lines(long.output) == [f"notes.txt:1:{'你' * 500}... [line truncated]"]
    assert "\ufffd" not in long.output
    assert "short" not in long.output
    assert long.details["count"] == 1
    assert "truncated_by" not in long.details
    assert _lines(exact.output) == [f"exact.txt:1:{'你' * 500}"]
    assert "[line truncated]" not in exact.output
    assert _lines(by_bytes.output) == [f"bytes.txt:1:{'你' * 167}"]
    assert "[line truncated]" not in by_bytes.output
    assert len(("你" * 167).encode()) == 501


async def test_the_production_byte_limit_keeps_whole_matches(tmp_path: Path) -> None:
    body = "a" * 500
    (tmp_path / "notes.txt").write_text("\n".join(body for _ in range(100)) + "\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    default = await search(registry, pattern="a+")
    raised = await search(registry, pattern="a+", limit=5000)
    assert default.status == raised.status == "success"
    assert default.details["truncated_by"] == raised.details["truncated_by"] == "bytes"
    assert default.details["count"] == raised.details["count"]
    assert isinstance(default.details["count"], int) and default.details["count"] < 100
    assert _lines(default.output) == _lines(raised.output)
    assert "50 KiB limit" in default.output
    assert "Raising limit does not remove the byte limit." in default.output
    assert "2000" not in default.output
    shown = _lines(default.output)
    assert shown[0] == f"notes.txt:1:{body}"
    assert shown[-1].startswith("notes.txt:")
    assert shown[-1].endswith(body)
    assert "\ufffd" not in default.output
    assert len("\n".join(shown).encode()) <= 50 * 1024


async def test_a_higher_limit_can_pass_2000_short_matches(tmp_path: Path) -> None:
    lines = [f"m{number}" for number in range(2500)]
    (tmp_path / "t.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = await search(create_default_registry(tmp_path), pattern=r"m\d+", limit=2500)
    shown = _lines(result.output)
    assert result.status == "success"
    assert result.error_code is None
    assert "truncated_by" not in result.details
    assert result.details["count"] == 2500
    assert shown[0] == "t.txt:1:m0"
    assert shown[1999] == "t.txt:2000:m1999"
    assert shown[-1] == "t.txt:2500:m2499"
    assert "2000 lines" not in result.output and "line limit" not in result.output


async def test_the_byte_limit_drops_a_whole_match_and_keeps_utf8(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hi\n你你\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(SearchContentTool(tmp_path, byte_limit=20))
    counted = await search(registry, pattern=".")
    assert counted.status == "success"
    assert counted.details["truncated_by"] == "bytes"
    assert _lines(counted.output) == ["notes.txt:1:hi"]
    assert "你" not in counted.output
    assert "\ufffd" not in counted.output
    assert "20 bytes limit" in counted.output
    assert "Raising limit does not remove the byte limit." in counted.output
    tight = ToolRegistry()
    tight.register(SearchContentTool(tmp_path, byte_limit=4))
    none = await search(tight, pattern=".")
    assert none.details["truncated_by"] == "bytes"
    assert none.details["count"] == 0
    assert "hi" not in none.output
    assert "你" not in none.output
    assert "\ufffd" not in none.output


async def test_a_colon_in_the_file_name_stays_with_the_path(tmp_path: Path) -> None:
    (tmp_path / "a:b.txt").write_text("alpha:beta\n", encoding="utf-8")
    result = await search(create_default_registry(tmp_path), pattern="alpha")
    assert result.status == "success"
    assert _lines(result.output) == ["a:b.txt:1:alpha:beta"]


async def test_a_crlf_match_omits_the_line_break(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_bytes(b"alpha\r\nbeta\r\n")
    result = await search(create_default_registry(tmp_path), pattern="alpha")
    assert result.status == "success"
    assert _lines(result.output) == ["notes.txt:1:alpha"]


def test_search_content_schema_is_read_and_keeps_the_production_limits() -> None:
    registry = create_default_registry(Path("/startup"))
    tool = registry.get("search_content")
    assert isinstance(tool, SearchContentTool)
    definition = next(item for item in registry.definitions() if item.name == "search_content")
    assert [item.name for item in registry.definitions()] == [
        "read_file", "write_file", "edit_file", "run_command", "find_files", "search_content",
    ]
    names = ("read_file", "write_file", "edit_file", "run_command", "find_files", "search_content")
    assert [registry.get(name).category for name in names] == ["read", "write", "write", "execute", "read", "read"]
    assert tool.category == "read"
    assert tool.timeout_seconds == SEARCH_TIMEOUT_SECONDS == 10
    assert tool.byte_limit == OUTPUT_LIMIT_BYTES == 50 * 1024
    assert tool.grace_seconds == PROCESS_TERMINATION_GRACE_SECONDS == 2
    assert SEARCH_RESULT_LIMIT == 100
    assert LINE_CHAR_LIMIT == 500
    assert definition.input_schema == tool.params_model.model_json_schema()
    schema = definition.input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["pattern"]
    assert schema["properties"]["pattern"]["minLength"] == 1
    assert schema["properties"]["path"]["default"] == "."
    assert schema["properties"]["literal"]["default"] is False
    limit = schema["properties"]["limit"]
    assert (limit["type"], limit["default"], limit["minimum"]) == ("integer", 100, 1)
    assert "maximum" not in limit
    assert schema["properties"]["include_hidden"]["default"] is False
    assert schema["properties"]["include_ignored"]["default"] is False
    assert definition.description.isascii()
    assert "ripgrep" in definition.description
    assert "case-sensitive" in definition.description and "literal" in definition.description
    assert ".gitignore" in definition.description and ".rgignore" in definition.description
    assert "binary" in definition.description and "explicit file" in definition.description
    assert "100" in definition.description and "50 KiB" in definition.description and "500" in definition.description
    assert "2000-line limit" in definition.description


async def test_paths_resolve_from_the_startup_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = tmp_path / "startup"
    elsewhere = tmp_path / "elsewhere"
    _write(startup / "pkg" / "notes.txt", "inside-hit\n")
    _write(elsewhere / "note.txt", "outside-hit\n")
    monkeypatch.chdir(tmp_path)
    registry = create_default_registry(startup)
    relative = await search(registry, pattern="hit", path="pkg")
    explicit = await search(registry, pattern="hit", path="pkg/notes.txt")
    absolute = await search(registry, pattern="hit", path=str(elsewhere / "note.txt"))
    omitted = await search(registry, pattern="hit")
    assert _lines(relative.output) == ["notes.txt:1:inside-hit"]
    assert relative.details["path"] == str(startup / "pkg")
    assert explicit.output == (
        "notes.txt:1:inside-hit\n"
        "\n"
        f"1 match in {startup / 'pkg' / 'notes.txt'}."
    )
    assert _lines(absolute.output) == ["note.txt:1:outside-hit"]
    assert absolute.details["path"] == str(elsewhere / "note.txt")
    assert _lines(omitted.output) == ["pkg/notes.txt:1:inside-hit"]
    assert "outside-hit" not in omitted.output


async def test_a_missing_ripgrep_binary_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(SearchContentTool(tmp_path, ripgrep=""))
    missing = await search(registry, pattern="alpha")
    assert missing.status == "error"
    assert missing.error_code == "ripgrep_not_found"
    assert "Ripgrep was not found" in missing.output
    assert "Install ripgrep" in missing.output and "PATH" in missing.output
    absent = ToolRegistry()
    absent.register(SearchContentTool(tmp_path, ripgrep=str(tmp_path / "missing-rg")))
    gone = await search(absent, pattern="alpha")
    assert gone.error_code == "ripgrep_not_found"


async def test_timeout_stops_the_search_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ripgrep, leader, child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, child, marker)
    registry = ToolRegistry()
    registry.register(SearchContentTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=1, grace_seconds=0.6))
    started = time.monotonic()
    result = await search(registry, pattern="alpha")
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
    registry.register(SearchContentTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=30, grace_seconds=0.6))
    running = asyncio.create_task(search(registry, pattern="alpha"))
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


def search_call(call_id: str, pattern: str, **extra: object) -> tuple[str, str, tuple[str, ...]]:
    payload = json.dumps({"pattern": pattern, **extra}, ensure_ascii=False)
    marker = '{"pattern": "'
    assert payload.startswith(marker)
    return (call_id, "search_content", (marker, payload[len(marker):]))


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_both_protocols_search_a_directory_and_an_explicit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "notes.txt").write_text("bat\nBAT\nb.t\n", encoding="utf-8")
    (tmp_path / "only.txt").write_text("a.b\naxb\n", encoding="utf-8")
    calls = (
        search_call("call_dir", "b.t", path="pkg"),
        search_call("call_file", "a.b", path="only.txt", literal=True),
    )
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls, decoy=True)),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("find those strings")]
        _ = [event async for event in client.stream_reply("follow-up")]
    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["call_dir", "call_file"]
    assert _lines(finished[0].result.output) == ["notes.txt:1:bat", "notes.txt:3:b.t"]
    assert "BAT" not in finished[0].result.output
    assert _lines(finished[1].result.output) == ["only.txt:1:a.b"]
    assert "axb" not in finished[1].result.output
    summary, follow_up = transport.requests[1], transport.requests[2]
    replayed = result_pairs(protocol, summary)
    assert [call_id for call_id, _payload in replayed] == ["call_dir", "call_file"]
    assert _lines(str(replayed[0][1]["output"])) == ["notes.txt:1:bat", "notes.txt:3:b.t"]
    assert _lines(str(replayed[1][1]["output"])) == ["only.txt:1:a.b"]
    assert result_pairs(protocol, follow_up)[0][0] == "call_dir"
    summary_body = json.loads(summary.content)
    assert summary_body["tool_choice"] == ({"type": "none"} if protocol == "anthropic" else "none")
    tools = json.loads(transport.requests[0].content)["tools"]
    definition = next(tool for tool in tools if tool["name"] == "search_content")
    schema = definition["input_schema"] if protocol == "anthropic" else definition["parameters"]
    assert schema["properties"]["literal"]["default"] is False
    assert schema["properties"]["limit"]["default"] == 100
    assert "permission" not in schema["properties"]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancelling_a_search_skips_the_rest_of_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    ripgrep, leader, _child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, tmp_path / "child.pid", marker)
    later = tmp_path / "later.txt"
    registry = ToolRegistry()
    registry.register(SearchContentTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=30, grace_seconds=0.6))
    registry.register(WriteFileTool(tmp_path))
    calls = (search_call("call_find", "alpha"), write_call("call_later", "later.txt", "no"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("search then write")
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
async def test_the_ui_summarizes_matching_lines_and_expand_matches_the_model_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "notes.txt").write_text("".join(f"hit {number}\n" for number in range(1, 8)), encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("hidden hit\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (search_call("call_find", "hit"),))),
        summary,
    ])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(100, 40)) as pilot:
        app.query_one("#prompt", TextArea).text = "search for hit"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        view = app.query_one(".tool-call")
        assert view.query_one(".tool-header", Static).content == f"Search hit in {tmp_path}"
        assert view.query_one(".tool-status", Static).content == "Success"
        preview = str(view.query_one(".tool-preview", Static).content)
        assert preview == "└ Returned 7 matching lines"
        assert "hidden hit" not in preview
        assert app.query_one("#status", Static).content == "Generating..."
        output = view.query_one(".tool-details")
        await pilot.click(view.query_one(".tool-heading"))
        expanded = str(output.query_one(".tool-output", Static).content)
        assert len(transport.requests) == 2
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
    replayed = result_pairs(protocol, transport.requests[1])[0][1]
    assert expanded == replayed["output"]
    assert replayed["status"] == "success"
    assert _lines(expanded)[0] == "notes.txt:1:hit 1"
    assert _lines(expanded)[6] == "notes.txt:7:hit 7"
    assert "hidden hit" not in expanded
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_stays_cancelling_until_the_search_process_group_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    ripgrep, leader, child, marker = install_slow_ripgrep(tmp_path)
    arm_slow_ripgrep(monkeypatch, leader, child, marker)
    later = tmp_path / "later.txt"
    registry = ToolRegistry()
    registry.register(SearchContentTool(tmp_path, ripgrep=str(ripgrep), timeout_seconds=30, grace_seconds=0.8))
    registry.register(WriteFileTool(tmp_path))
    calls = (search_call("call_find", "alpha"), write_call("call_later", "later.txt", "no"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "search them"
        await pilot.press("enter")
        await wait_for_tick(marker)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Running search_content..."
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
