import asyncio
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal

import pytest
import httpx2 as httpx
from textual.widgets import Static, TextArea
from textual.worker import WorkerCancelled

from replux.cli import create_app
from replux.llm import (
    IncompleteReplyError, LLMError, ReplyCompleted, TextDelta, ToolFinished, ToolStarted, create_client,
)
from replux.tools import FILE_OPERATION_TIMEOUT_SECONDS, ToolRegistry, ToolResult, create_default_registry
from replux.tools.commit import CommitHold
from replux.tools.edit_file import EditFileTool
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events
from tests.test_tool_batch import batch_events, open_client, result_pairs, text_gate
from tests.test_tool_ui import provider_file, statuses
from tests.test_write_file import write_call


async def edit(registry: ToolRegistry, **arguments: Any) -> ToolResult:
    return await registry.execute("edit_file", json.dumps(arguments))


def edit_call(call_id: str, path: str, edits: list[dict[str, str]]) -> tuple[str, str, tuple[str, ...]]:
    payload = json.dumps({"path": path, "edits": edits}, ensure_ascii=False)
    head, tail = payload.split(', "edits"', 1)
    return (call_id, "edit_file", (head, ', "edits"' + tail))


async def test_later_edits_match_text_produced_by_earlier_edits(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": "hello", "new_text": "hello world"},
        {"old_text": "world", "new_text": "there"},
    ])
    assert result.status == "success"
    assert target.read_bytes() == b"hello there"
    assert result.details["bytes"] == len(b"hello there")
    assert b"hello there".decode() not in result.output


def test_edit_file_schema_is_strict_and_classified_as_write() -> None:
    registry = create_default_registry(Path("/startup"))
    tool = registry.get("edit_file")
    definition = next(item for item in registry.definitions() if item.name == "edit_file")
    assert isinstance(tool, EditFileTool)
    assert tool.category == "write"
    assert tool.timeout_seconds == FILE_OPERATION_TIMEOUT_SECONDS == 10
    assert definition.input_schema == tool.params_model.model_json_schema()
    schema = definition.input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["path", "edits"]
    assert schema["properties"]["path"]["minLength"] == 1
    edits = schema["properties"]["edits"]
    assert edits["type"] == "array"
    assert edits["minItems"] == 1
    item = edits["items"]
    if "$ref" in item:
        item = schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
    assert item["additionalProperties"] is False
    assert item["required"] == ["old_text", "new_text"]
    assert item["properties"]["old_text"]["minLength"] == 1
    assert item["properties"]["new_text"]["type"] == "string"
    assert "minLength" not in item["properties"]["new_text"]
    assert "old_string" not in json.dumps(schema)
    assert definition.description.isascii()
    assert "overlapping" in definition.description and "BOM" in definition.description


@pytest.mark.parametrize("arguments", [
    {"path": "a.txt", "edits": [{"old_text": "a", "new_text": "b"}], "extra": True},
    {"path": "a.txt", "old_string": "a", "new_string": "b"},
    {"path": "a.txt", "edits": []},
    {"path": "a.txt", "edits": [{"old_text": "", "new_text": "b"}]},
    {"path": "a.txt", "edits": [{"old_text": "a", "new_text": "b", "extra": True}]},
    {"path": "a.txt", "edits": [{"old_text": 1, "new_text": "b"}]},
    {"path": "a.txt", "edits": [{"old_text": "a", "new_text": None}]},
    {"path": "a.txt", "edits": "a"},
    {"path": "", "edits": [{"old_text": "a", "new_text": "b"}]},
    {"edits": [{"old_text": "a", "new_text": "b"}]},
    {"path": "a.txt"},
    {},
])
async def test_invalid_edit_arguments_do_not_change_the_file(
    tmp_path: Path, arguments: dict[str, Any],
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"keep")
    result = await create_default_registry(tmp_path).execute("edit_file", json.dumps(arguments))
    assert (result.status, result.error_code) == ("error", "invalid_arguments")
    assert result.output.isascii() and result.output
    assert target.read_bytes() == b"keep"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_an_empty_new_text_deletes_the_one_match(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"aXb")
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": "X", "new_text": ""},
    ])
    assert result.status == "success"
    assert target.read_bytes() == b"ab"
    assert result.details["edits"] == 1


async def test_zero_matches_leave_the_file_bytes_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": "missing", "new_text": "x"},
    ])
    assert (result.status, result.error_code) == ("error", "no_match")
    assert result.details["edit"] == 1
    assert result.details["matches"] == 0
    assert "was not changed" in result.output
    assert target.read_bytes() == b"hello"
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize(("content", "old_text", "matches"), [
    (b"a a", "a", 2),
    (b"aaa", "aa", 2),
])
async def test_repeated_and_overlapping_matches_are_rejected(
    tmp_path: Path, content: bytes, old_text: str, matches: int,
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(content)
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": old_text, "new_text": "b"},
    ])
    assert (result.status, result.error_code) == ("error", "not_unique")
    assert result.details["matches"] == matches
    assert "overlapping" in result.output
    assert target.read_bytes() == content
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_later_mismatch_does_not_write_the_earlier_edit(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": "hello", "new_text": "hello world"},
        {"old_text": "missing", "new_text": "x"},
    ])
    assert (result.status, result.error_code, result.details["edit"]) == ("error", "no_match", 2)
    assert target.read_bytes() == b"hello"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_successful_edits_appear_in_one_commit(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "hello", "new_text": "hello world"},
        {"old_text": "world", "new_text": "there"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    temps = list(tmp_path.glob(".replux-*"))
    assert [item.read_bytes() for item in temps] == [b"hello there"]
    assert target.read_bytes() == b"hello"
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert result.status == "success"
    assert target.read_bytes() == b"hello there"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_an_edit_keeps_the_bom_newlines_mode_and_symlink(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "file.txt"
    original = b"\xef\xbb\xbfhello\r\nworld\r\n"
    target.write_bytes(original)
    target.chmod(0o640)
    link = tmp_path / "link.txt"
    link.symlink_to(Path("real") / "file.txt")
    link_before = os.lstat(link)
    result = await edit(create_default_registry(tmp_path), path="link.txt", edits=[
        {"old_text": "world", "new_text": "there"},
    ])
    assert result.status == "success"
    assert target.read_bytes() == b"\xef\xbb\xbfhello\r\nthere\r\n"
    assert os.lstat(link).st_ino == link_before.st_ino
    assert os.readlink(link) == "real/file.txt"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(real_dir.glob(".replux-*")) == []

    strict = await edit(create_default_registry(tmp_path), path="link.txt", edits=[
        {"old_text": "hello\nthere", "new_text": "x"},
    ])
    assert (strict.status, strict.error_code) == ("error", "no_match")
    assert target.read_bytes() == b"\xef\xbb\xbfhello\r\nthere\r\n"


@pytest.mark.parametrize(("content", "code"), [
    (b"a\0b", "binary_file"),
    (b"caf\xe9", "unsupported_encoding"),
])
async def test_binary_and_non_utf8_files_are_left_unchanged(
    tmp_path: Path, content: bytes, code: str,
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(content)
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": "a", "new_text": "b"},
    ])
    assert (result.status, result.error_code) == ("error", code)
    assert result.output.isascii() and result.output
    assert target.read_bytes() == content
    assert list(tmp_path.glob(".replux-*")) == []


async def test_an_external_change_after_the_read_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "hello", "new_text": "there"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    target.write_bytes(b"external")
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code) == ("error", "file_changed")
    assert "was not overwritten" in result.output
    assert target.read_bytes() == b"external"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_missing_file_or_directory_is_not_created(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)
    missing = await edit(registry, path="missing.txt", edits=[{"old_text": "a", "new_text": "b"}])
    directory = tmp_path / "dir"
    directory.mkdir()
    (directory / "keep").write_bytes(b"stay")
    blocked = await edit(registry, path="dir", edits=[{"old_text": "a", "new_text": "b"}])
    assert (missing.status, missing.error_code) == ("error", "file_not_found")
    assert (blocked.status, blocked.error_code) == ("error", "not_a_file")
    assert not (tmp_path / "missing.txt").exists()
    assert (directory / "keep").read_bytes() == b"stay"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_permission_denied_reports_the_unchanged_file_and_removes_the_temp(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can write without permission bits")
    parent = tmp_path / "locked"
    parent.mkdir()
    target = parent / "a.txt"
    target.write_bytes(b"old")
    parent.chmod(0o555)
    try:
        result = await edit(create_default_registry(tmp_path), path="locked/a.txt", edits=[
            {"old_text": "old", "new_text": "new"},
        ])
    finally:
        parent.chmod(0o755)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "permission_denied", "not_replaced")
    assert "was not replaced. It contains 3 bytes." in result.output
    assert target.read_bytes() == b"old"
    assert list(parent.glob(".replux-*")) == []


async def test_a_long_edit_keeps_every_byte_and_does_not_return_the_body(tmp_path: Path) -> None:
    body = "é" * (50 * 1024)
    target = tmp_path / "a.txt"
    target.write_text("HEAD" + body, encoding="utf-8")
    result = await edit(create_default_registry(tmp_path), path="a.txt", edits=[
        {"old_text": "HEAD", "new_text": "TAIL"},
    ])
    assert result.status == "success"
    assert target.read_text(encoding="utf-8") == "TAIL" + body
    assert body not in result.output
    assert result.details["bytes"] == len(("TAIL" + body).encode())


async def test_relative_paths_stay_at_the_startup_directory_and_absolute_paths_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "a.txt").write_bytes(b"old")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"old")
    monkeypatch.chdir(startup)
    registry = create_default_registry(Path.cwd())
    monkeypatch.chdir(elsewhere)
    relative = await edit(registry, path="a.txt", edits=[{"old_text": "old", "new_text": "new"}])
    absolute = await edit(registry, path=str(outside), edits=[{"old_text": "old", "new_text": "abs"}])
    assert (startup / "a.txt").read_bytes() == b"new"
    assert not (elsewhere / "a.txt").exists()
    assert relative.details["path"] == str(startup / "a.txt")
    assert outside.read_bytes() == b"abs"
    assert absolute.details["path"] == str(outside)


async def test_timeout_before_commit_leaves_the_previous_bytes_and_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, timeout_seconds=0.05, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "old", "new_text": "new"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(running), timeout=0.2)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".replux-*")) == []
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "timeout", "not_replaced")
    assert "0.05 seconds" in result.output
    assert "was not replaced. It contains 3 bytes." in result.output
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_cancel_before_commit_leaves_the_file_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "old", "new_text": "new"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    running.cancel()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
    assert target.read_bytes() == b"old"
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert result.status == "cancelled"
    assert result.details["file_state"] == "not_replaced"
    assert "was not replaced" in result.output
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_cancel_during_commit_finishes_the_edit(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "old", "new_text": "new"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    running.cancel()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
    assert target.read_bytes() == b"old"
    assert len(list(tmp_path.glob(".replux-*"))) == 1
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert result.status == "success"
    assert result.details["bytes"] == 3
    assert target.read_bytes() == b"new"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_file_removed_after_the_read_is_not_recreated(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "hello", "new_text": "there"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    target.unlink()
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "file_changed", "absent")
    assert "was not recreated" in result.output
    assert "was not created" not in result.output
    assert not target.exists()
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_permission_change_after_the_read_is_not_reported_as_file_changed(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can read without permission bits")
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "hello", "new_text": "there"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    target.chmod(0o000)
    try:
        hold.release.set()
        result = await asyncio.wait_for(running, timeout=2)
    finally:
        target.chmod(0o644)
    assert (result.status, result.error_code) == ("error", "permission_denied")
    assert "file_changed" not in result.output
    assert target.read_bytes() == b"hello"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_an_external_change_is_kept_when_it_matches_the_edited_bytes(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"hello")
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(edit(registry, path="a.txt", edits=[
        {"old_text": "hello", "new_text": "there"},
    ]))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    target.write_bytes(b"there")
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code) == ("error", "file_changed")
    assert target.read_bytes() == b"there"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_directory_symlink_or_loop_is_left_in_place(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "keep").write_bytes(b"stay")
    link = tmp_path / "link"
    link.symlink_to("real", target_is_directory=True)
    directory = await edit(create_default_registry(tmp_path), path="link", edits=[
        {"old_text": "stay", "new_text": "changed"},
    ])
    assert (directory.status, directory.error_code) == ("error", "not_a_file")
    assert os.readlink(link) == "real"
    assert (real / "keep").read_bytes() == b"stay"

    left = tmp_path / "a"
    right = tmp_path / "b"
    left.symlink_to("b")
    right.symlink_to("a")
    loop = await edit(create_default_registry(tmp_path), path="a", edits=[
        {"old_text": "a", "new_text": "b"},
    ])
    assert (loop.status, loop.error_code) == ("error", "read_failed")
    assert os.readlink(left) == "b"
    assert os.readlink(right) == "a"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_broken_symlink_is_not_replaced_with_a_new_file(tmp_path: Path) -> None:
    link = tmp_path / "link.txt"
    link.symlink_to("missing.txt")
    result = await edit(create_default_registry(tmp_path), path="link.txt", edits=[
        {"old_text": "a", "new_text": "b"},
    ])
    assert (result.status, result.error_code) == ("error", "file_not_found")
    assert os.readlink(link) == "missing.txt"
    assert not (tmp_path / "missing.txt").exists()
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_both_protocols_edit_a_real_file_and_replay_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "notes.txt").write_bytes(b"alpha")
    calls = (edit_call("call_edit", "notes.txt", [
        {"old_text": "alpha", "new_text": "alpha beta"},
        {"old_text": "beta", "new_text": "gamma"},
    ]),)
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls, decoy=True)),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("edit the file")]
        _ = [event async for event in client.stream_reply("follow-up")]

    assert (tmp_path / "notes.txt").read_bytes() == b"alpha gamma"
    assert not (tmp_path / "wrong.txt").exists()
    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [(event.call_id, event.result.status, event.result.details["bytes"], event.result.details["edits"]) for event in finished] == [
        ("call_edit", "success", len(b"alpha gamma"), 2),
    ]
    assert "alpha gamma" not in finished[0].result.output
    summary, follow_up = transport.requests[1], transport.requests[2]
    assert [(call_id, payload["status"], payload["details"]["bytes"]) for call_id, payload in result_pairs(protocol, summary)] == [
        ("call_edit", "success", len(b"alpha gamma")),
    ]
    assert [call_id for call_id, _ in result_pairs(protocol, follow_up)] == ["call_edit"]
    summary_body = json.loads(summary.content)
    assert summary_body["tool_choice"] == ({"type": "none"} if protocol == "anthropic" else "none")
    assert "edit_file" in {tool["name"] for tool in json.loads(transport.requests[0].content)["tools"]}
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_an_edit_batch_keeps_the_finished_file_and_skips_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"old")
    (tmp_path / "blocked").mkdir()
    (tmp_path / "later.txt").write_bytes(b"keep")
    calls = (
        edit_call("call_ok", "a.txt", [{"old_text": "old", "new_text": "done"}]),
        write_call("call_bad", "blocked", "nope"),
        edit_call("call_later", "later.txt", [{"old_text": "keep", "new_text": "changed"}]),
    )
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("edit then write")]
    assert (tmp_path / "a.txt").read_bytes() == b"done"
    assert (tmp_path / "later.txt").read_bytes() == b"keep"
    assert not (tmp_path / "blocked" / "nope").exists()
    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [(event.call_id, event.result.status) for event in finished] == [
        ("call_ok", "success"), ("call_bad", "error"), ("call_later", "skipped"),
    ]
    assert finished[1].result.error_code == "not_a_file"
    assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["call_ok", "call_bad"]
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == [
        "success", "error", "skipped",
    ]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["eof", "cleanup"])
async def test_an_unfinished_edit_response_leaves_the_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"old")
    calls = (edit_call("call_ok", "a.txt", [{"old_text": "old", "new_text": "new"}]),)
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
                async for event in client.stream_reply("edit"):
                    received.append(event)
        else:
            with pytest.raises(LLMError) as caught:
                async for event in client.stream_reply("edit"):
                    received.append(event)
            assert not isinstance(caught.value, IncompleteReplyError)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert (tmp_path / "a.txt").read_bytes() == b"old"
    assert not any(isinstance(event, (ToolStarted, ToolFinished)) for event in received)
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["incomplete", "cancel"])
async def test_a_failed_summary_keeps_the_edit_and_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"old")
    calls = (edit_call("call_ok", "a.txt", [{"old_text": "old", "new_text": "kept"}]),)
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
            async for event in client.stream_reply("edit it"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            if ending == "cancel":
                await asyncio.wait_for(summary_started.wait(), timeout=2)
                await asyncio.wait_for(held[0].blocked.wait(), timeout=2)
                assert any(isinstance(event, TextDelta) and event.text == "你好" for event in received)
                consuming.cancel()
                await asyncio.wait_for(held[0].closing.wait(), timeout=2)
                with pytest.raises(RuntimeError, match="in progress"):
                    await anext(client.stream_reply("too soon"))
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

    assert (tmp_path / "a.txt").read_bytes() == b"kept"
    pairs = result_pairs(protocol, requests[-1])
    assert [(call_id, payload["status"], payload["details"]["bytes"]) for call_id, payload in pairs] == [
        ("call_ok", "success", 4),
    ]
    assert "你好" not in json.dumps(json.loads(requests[-1].content), ensure_ascii=False)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("stage", ["before", "during"])
async def test_cancelling_a_turn_keeps_only_an_edit_that_already_started(
    tmp_path: Path, protocol: str, stage: Literal["before", "during"],
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    (tmp_path / "later.txt").write_bytes(b"keep")
    hold = CommitHold(stage)
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    calls = (
        edit_call("call_edit", "a.txt", [{"old_text": "old", "new_text": "new"}]),
        edit_call("call_later", "later.txt", [{"old_text": "keep", "new_text": "changed"}]),
    )
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("edit them")
        while not isinstance(await anext(reply), ToolStarted):
            pass
        pending = asyncio.create_task(anext(reply))
        assert await asyncio.to_thread(hold.entered.wait, 2)
        pending.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.1)
        assert target.read_bytes() == b"old"
        assert (tmp_path / "later.txt").read_bytes() == b"keep"
        hold.release.set()
        finished = await asyncio.wait_for(pending, timeout=2)
        skipped = await asyncio.wait_for(anext(reply), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await anext(reply)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert isinstance(finished, ToolFinished) and isinstance(skipped, ToolFinished)
    assert skipped.result.status == "skipped"
    assert (tmp_path / "later.txt").read_bytes() == b"keep"
    assert list(tmp_path.glob(".replux-*")) == []
    if stage == "before":
        assert target.read_bytes() == b"old"
        assert finished.result.status == "cancelled"
        assert finished.result.details["file_state"] == "not_replaced"
        assert "was not replaced" in finished.result.output
    else:
        assert target.read_bytes() == b"new"
        assert finished.result.status == "success"
        assert finished.result.details["bytes"] == 3
    statuses_seen = [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])]
    assert statuses_seen == [finished.result.status, "skipped"]
    assert "你好" not in transport.requests[1].content.decode()
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_shows_a_real_edit_and_stays_busy_until_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "notes.txt").write_bytes(b"hello")
    monkeypatch.chdir(tmp_path)
    calls = (edit_call("call_edit", "pkg/notes.txt", [{"old_text": "hello", "new_text": "there"}]),)
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([ControlledSSE(batch_events(protocol, calls)), summary])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 32)) as pilot:
        app.query_one("#prompt", TextArea).text = "edit notes"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        assert (tmp_path / "pkg" / "notes.txt").read_bytes() == b"there"
        view = app.query_one(".tool-call")
        assert view.query_one(".tool-header", Static).content == "Edit pkg/notes.txt"
        assert "pkg/notes.txt" in str(view.query_one(".tool-header", Static).content)
        assert str(view.query_one(".tool-status", Static).content).startswith("Success")
        assert "Applied" in str(view.query_one(".tool-preview", Static).content)
        assert app.query_one("#status", Static).content == "Generating..."
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
    assert (tmp_path / "pkg" / "notes.txt").read_bytes() == b"there"
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_shows_a_failed_edit_batch_without_changing_the_skipped_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"old")
    (tmp_path / "later.txt").write_bytes(b"keep")
    monkeypatch.chdir(tmp_path)
    calls = (
        edit_call("call_ok", "a.txt", [{"old_text": "old", "new_text": "done"}]),
        edit_call("call_bad", "missing.txt", [{"old_text": "a", "new_text": "b"}]),
        edit_call("call_later", "later.txt", [{"old_text": "keep", "new_text": "changed"}]),
    )
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([ControlledSSE(batch_events(protocol, calls)), summary])
    app = create_app(["--config", str(provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 36)) as pilot:
        app.query_one("#prompt", TextArea).text = "edit three"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        assert statuses(app) == ["Success", "Failed", "Skipped"]
        failed = list(app.query(".tool-call"))[1]
        await pilot.click(failed.query_one(".tool-heading"))
        assert "Error code: file_not_found" in str(failed.query_one(".tool-metadata", Static).content)
        assert app.query_one("#status", Static).content == "Generating..."
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
    assert (tmp_path / "a.txt").read_bytes() == b"done"
    assert (tmp_path / "later.txt").read_bytes() == b"keep"
    assert not (tmp_path / "missing.txt").exists()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_stays_cancelling_until_an_in_progress_edit_finishes(
    tmp_path: Path, protocol: str,
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    (tmp_path / "later.txt").write_bytes(b"keep")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(EditFileTool(tmp_path, hold=hold))
    calls = (
        edit_call("call_edit", "a.txt", [{"old_text": "old", "new_text": "new"}]),
        edit_call("call_later", "later.txt", [{"old_text": "keep", "new_text": "changed"}]),
    )
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "edit them"
        await pilot.press("enter")
        assert await asyncio.to_thread(hold.entered.wait, 2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Running edit_file..."
        assert target.read_bytes() == b"old"
        worker = next(iter(app.workers))
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Cancelling..."
        assert target.read_bytes() == b"old"
        assert len(list(tmp_path.glob(".replux-*"))) == 1
        prompt.text = "草稿"
        await pilot.press("enter")
        assert prompt.text == "草稿"
        assert len(transport.requests) == 1
        hold.release.set()
        with pytest.raises(WorkerCancelled):
            await asyncio.wait_for(worker.wait(), timeout=2)
        await pilot.pause()
        assert target.read_bytes() == b"new"
        assert (tmp_path / "later.txt").read_bytes() == b"keep"
        assert list(tmp_path.glob(".replux-*")) == []
        assert app.query_one("#status", Static).content == "Ready"
        assert statuses(app) == ["Success", "Skipped"]
        prompt.text = "follow-up"
        await pilot.press("enter")
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == ["success", "skipped"]
    assert "你好" not in transport.requests[1].content.decode()
