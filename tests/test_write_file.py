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
from replux.tools import (
    FILE_OPERATION_TIMEOUT_SECONDS, ToolRegistry, ToolResult, create_default_registry,
)
from replux.tools.commit import CommitHold
from replux.tools.write_file import WriteFileTool
from replux.tui import RepluxApp
from tests.support import ControlledSSE, RecordingTransport, provider_config, reply_events
from tests.test_config import VALID_YAML
from tests.test_tool_batch import batch_events, open_client, result_pairs, text_gate
from tests.test_tool_ui import statuses


async def write(registry: ToolRegistry, **arguments: Any) -> ToolResult:
    return await registry.execute("write_file", json.dumps(arguments))


def write_call(call_id: str, path: str, content: str) -> tuple[str, str, tuple[str, ...]]:
    payload = json.dumps({"path": path, "content": content}, ensure_ascii=False)
    head, tail = payload.split(', "content"', 1)
    return (call_id, "write_file", (head, ', "content"' + tail))


async def test_write_file_creates_overwrites_and_keeps_the_given_bytes(tmp_path: Path) -> None:
    content = "第一行\r\n\ufeff"
    registry = create_default_registry(tmp_path)
    created = await write(registry, path="pkg/notes.txt", content=content)
    target = tmp_path / "pkg" / "notes.txt"
    assert created.status == "success"
    assert target.read_bytes() == content.encode()
    assert created.details["bytes"] == len(content.encode())

    long = "é" * (50 * 1024)
    overwritten = await write(registry, path="pkg/notes.txt", content=long)
    assert overwritten.status == "success"
    assert target.read_bytes() == long.encode()
    assert long not in overwritten.output
    assert overwritten.details["bytes"] == len(long.encode())

    emptied = await write(registry, path="pkg/notes.txt", content="")
    assert emptied.status == "success"
    assert target.read_bytes() == b""


async def test_commit_replaces_a_temp_file_and_keeps_the_symlink_and_mode(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "file.txt"
    target.write_bytes(b"old")
    target.chmod(0o640)
    link = tmp_path / "link.txt"
    link.symlink_to(Path("real") / "file.txt")
    link_before = os.lstat(link)
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(write(registry, path="link.txt", content="new"))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    temps = list(real_dir.glob(".replux-*"))
    assert [item.read_bytes() for item in temps] == [b"new"]
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".replux-*")) == []
    assert os.readlink(link) == "real/file.txt"
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert result.status == "success"
    assert target.read_bytes() == b"new"
    assert list(real_dir.glob(".replux-*")) == []
    assert os.lstat(link).st_ino == link_before.st_ino
    assert os.readlink(link) == "real/file.txt"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_write_file_schema_is_strict_and_classified_as_write() -> None:
    registry = create_default_registry(Path("/startup"))
    tool = registry.get("write_file")
    definition = next(item for item in registry.definitions() if item.name == "write_file")
    assert isinstance(tool, WriteFileTool)
    assert tool.category == "write"
    assert tool.timeout_seconds == FILE_OPERATION_TIMEOUT_SECONDS == 10
    assert definition.input_schema == tool.params_model.model_json_schema()
    schema = definition.input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["path", "content"]
    assert schema["properties"]["path"]["minLength"] == 1
    assert schema["properties"]["content"]["type"] == "string"
    assert "default" not in schema["properties"]["content"]
    assert "minLength" not in schema["properties"]["content"]
    assert definition.description.isascii()
    assert "parent" in definition.description and "UTF-8" in definition.description


@pytest.mark.parametrize("arguments", [
    {"path": "a.txt", "content": "x", "extra": True},
    {"path": 7, "content": "x"},
    {"path": "a.txt", "content": 7},
    {"path": "a.txt", "content": True},
    {"path": "a.txt", "content": None},
    {"path": "", "content": "x"},
    {"content": "x"},
    {"path": "a.txt"},
    {},
])
async def test_invalid_write_arguments_do_not_change_the_file(
    tmp_path: Path, arguments: dict[str, Any],
) -> None:
    target = tmp_path / "a.txt"
    target.write_text("keep", encoding="utf-8")
    result = await create_default_registry(tmp_path).execute("write_file", json.dumps(arguments))
    assert (result.status, result.error_code) == ("error", "invalid_arguments")
    assert result.output.isascii() and result.output
    assert target.read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_arguments_that_are_not_a_json_object_do_not_write(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)
    for arguments in ['{"path": "a.txt"', '["a.txt"]']:
        result = await registry.execute("write_file", arguments)
        assert (result.status, result.error_code) == ("error", "invalid_arguments")
    assert list(tmp_path.iterdir()) == []


async def test_relative_paths_stay_at_the_startup_directory_and_absolute_paths_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup = tmp_path / "startup"
    startup.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    outside = tmp_path / "outside" / "a.txt"
    monkeypatch.chdir(startup)
    registry = create_default_registry(Path.cwd())
    monkeypatch.chdir(elsewhere)
    created = await write(registry, path="pkg/a.txt", content="123")
    assert (startup / "pkg" / "a.txt").read_bytes() == b"123"
    assert not (elsewhere / "pkg").exists()
    assert created.details["path"] == str(startup / "pkg" / "a.txt")
    absolute = await write(registry, path=str(outside), content="abs")
    assert outside.read_bytes() == b"abs"
    assert absolute.details["path"] == str(outside)


async def test_a_new_file_uses_the_process_umask(tmp_path: Path) -> None:
    previous = os.umask(0o027)
    try:
        result = await write(create_default_registry(tmp_path), path="a.txt", content="hi")
    finally:
        os.umask(previous)
    assert result.status == "success"
    assert stat.S_IMODE((tmp_path / "a.txt").stat().st_mode) == 0o640


async def test_a_broken_symlink_keeps_the_link_and_creates_its_target(tmp_path: Path) -> None:
    link = tmp_path / "link.txt"
    link.symlink_to("nested/missing.txt")
    result = await write(create_default_registry(tmp_path), path="link.txt", content="hi")
    assert result.status == "success"
    assert os.readlink(link) == "nested/missing.txt"
    assert (tmp_path / "nested" / "missing.txt").read_bytes() == b"hi"
    assert list(tmp_path.glob(".replux-*")) == []
    assert list((tmp_path / "nested").glob(".replux-*")) == []


async def test_writing_a_directory_or_through_a_directory_symlink_changes_nothing(tmp_path: Path) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()
    (directory / "keep").write_text("stay", encoding="utf-8")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to("real", target_is_directory=True)
    registry = create_default_registry(tmp_path)
    direct = await write(registry, path="dir", content="x")
    linked = await write(registry, path="link", content="x")
    assert (direct.status, direct.error_code, direct.details["file_state"]) == ("error", "not_a_file", "directory")
    assert "was not replaced" in direct.output
    assert (linked.status, linked.error_code) == ("error", "not_a_file")
    assert link.is_symlink() and os.readlink(link) == "real"
    assert (directory / "keep").read_text(encoding="utf-8") == "stay"
    assert list(real.iterdir()) == []
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_parent_that_is_a_file_is_left_unchanged(tmp_path: Path) -> None:
    parent = tmp_path / "file"
    parent.write_text("keep", encoding="utf-8")
    result = await write(create_default_registry(tmp_path), path="file/child.txt", content="x")
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "not_a_directory", "absent")
    assert "was not created" in result.output
    assert parent.read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_permission_denied_reports_the_unchanged_file_and_removes_the_temp(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can write without permission bits")
    parent = tmp_path / "locked"
    parent.mkdir()
    target = parent / "a.txt"
    target.write_text("old", encoding="utf-8")
    parent.chmod(0o555)
    try:
        result = await write(create_default_registry(tmp_path), path="locked/a.txt", content="new")
    finally:
        parent.chmod(0o755)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "permission_denied", "not_replaced")
    assert "was not replaced. It contains 3 bytes." in result.output
    assert "previous" not in result.output
    assert target.read_text(encoding="utf-8") == "old"
    assert list(parent.glob(".replux-*")) == []


async def test_a_failed_replace_reports_the_directory_and_removes_the_temp(tmp_path: Path) -> None:
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(write(registry, path="a.txt", content="new"))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    blocked = tmp_path / "a.txt"
    blocked.mkdir()
    (blocked / "keep").write_text("stay", encoding="utf-8")
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "not_a_file", "directory")
    assert "was not replaced" in result.output
    assert (blocked / "keep").read_text(encoding="utf-8") == "stay"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_failed_replace_reports_when_the_new_bytes_are_already_there(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(write(registry, path="a.txt", content="new"))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    temps = list(tmp_path.glob(".replux-*"))
    assert len(temps) == 1
    target.write_bytes(b"new")
    temps[0].unlink()
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "write_failed", "written")
    assert "contains the new 3 bytes" in result.output
    assert target.read_bytes() == b"new"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_timeout_before_commit_leaves_the_previous_bytes_and_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    hold = CommitHold("before")
    tool = WriteFileTool(tmp_path, timeout_seconds=0.05, hold=hold)
    registry = ToolRegistry()
    registry.register(tool)
    running = asyncio.create_task(write(registry, path="a.txt", content="new"))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(running), timeout=0.2)
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".replux-*")) == []
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "timeout", "not_replaced")
    assert "0.05 seconds" in result.output
    assert "was not replaced. It contains 3 bytes." in result.output
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_cancel_before_commit_creates_nothing(tmp_path: Path) -> None:
    hold = CommitHold("before")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(write(registry, path="nested/a.txt", content="new"))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    running.cancel()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
    assert not (tmp_path / "nested").exists()
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert result.status == "cancelled"
    assert result.details["file_state"] == "absent"
    assert "was not created" in result.output
    assert not (tmp_path / "nested").exists()
    assert list(tmp_path.glob(".replux-*")) == []


async def test_cancel_during_commit_finishes_the_replacement(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(write(registry, path="a.txt", content="new"))
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


async def test_a_failed_replace_reports_the_bytes_now_on_disk_not_the_old_ones(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    running = asyncio.create_task(write(registry, path="a.txt", content="new"))
    assert await asyncio.to_thread(hold.entered.wait, 2)
    target.write_bytes(b"zzzzzzz")
    temps = list(tmp_path.glob(".replux-*"))
    assert len(temps) == 1
    temps[0].unlink()
    hold.release.set()
    result = await asyncio.wait_for(running, timeout=2)
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "write_failed", "not_replaced")
    assert "contains 7 bytes" in result.output
    assert "previous" not in result.output
    assert target.read_bytes() == b"zzzzzzz"
    assert list(tmp_path.glob(".replux-*")) == []


async def test_a_symlink_loop_is_left_in_place(tmp_path: Path) -> None:
    link = tmp_path / "a"
    other = tmp_path / "b"
    link.symlink_to("b")
    other.symlink_to("a")
    result = await write(create_default_registry(tmp_path), path="a", content="x")
    assert os.readlink(link) == "b"
    assert os.readlink(other) == "a"
    assert (result.status, result.error_code, result.details["file_state"]) == ("error", "write_failed", "symlink")
    assert "still a symlink" in result.output
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_both_protocols_write_real_files_and_replay_the_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "old.txt").write_bytes(b"previous")
    created = "第一行\r\n\ufeff"
    long = "é" * (50 * 1024)
    calls = (
        write_call("call_new", "pkg/new.txt", created),
        write_call("call_old", "old.txt", "replaced"),
        write_call("call_long", "long.txt", long),
    )
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls, decoy=True)),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("write the files")]
        _ = [event async for event in client.stream_reply("follow-up")]

    assert (tmp_path / "pkg" / "new.txt").read_bytes() == created.encode()
    assert (tmp_path / "old.txt").read_bytes() == b"replaced"
    assert (tmp_path / "long.txt").read_bytes() == long.encode()
    assert not (tmp_path / "wrong.txt").exists()
    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [event.result.status for event in finished] == ["success", "success", "success"]
    assert [event.result.details["bytes"] for event in finished] == [
        len(created.encode()), len(b"replaced"), len(long.encode()),
    ]
    assert long not in finished[2].result.output
    tool_events = [event for event in received if isinstance(event, (ToolStarted, ToolFinished))]
    assert [type(event) for event in tool_events] == [ToolStarted, ToolFinished, ToolStarted, ToolFinished, ToolStarted, ToolFinished]
    summary, follow_up = transport.requests[1], transport.requests[2]
    assert [item[1]["details"]["bytes"] for item in result_pairs(protocol, summary)] == [
        len(created.encode()), len(b"replaced"), len(long.encode()),
    ]
    assert [item[0] for item in result_pairs(protocol, follow_up)] == ["call_new", "call_old", "call_long"]
    summary_body = json.loads(summary.content)
    assert summary_body["tool_choice"] == ({"type": "none"} if protocol == "anthropic" else "none")
    assert "write_file" in {tool["name"] for tool in json.loads(transport.requests[0].content)["tools"]}
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_a_write_batch_keeps_the_finished_file_and_skips_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "blocked").mkdir()
    (tmp_path / "blocked" / "keep").write_text("stay", encoding="utf-8")
    (tmp_path / "later.txt").write_text("keep", encoding="utf-8")
    calls = (
        write_call("call_ok", "ok.txt", "done"),
        write_call("call_bad", "blocked", "nope"),
        write_call("call_later", "later.txt", "changed"),
    )
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("write three")]
    assert (tmp_path / "ok.txt").read_bytes() == b"done"
    assert (tmp_path / "blocked" / "keep").read_text(encoding="utf-8") == "stay"
    assert (tmp_path / "later.txt").read_text(encoding="utf-8") == "keep"
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
async def test_an_unfinished_write_response_leaves_every_target_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    (tmp_path / "ok.txt").write_text("old", encoding="utf-8")
    calls = (write_call("call_ok", "ok.txt", "new"), write_call("call_later", "pkg/later.txt", "x"))
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
                async for event in client.stream_reply("write"):
                    received.append(event)
        else:
            with pytest.raises(LLMError) as caught:
                async for event in client.stream_reply("write"):
                    received.append(event)
            assert not isinstance(caught.value, IncompleteReplyError)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "pkg").exists()
    assert not any(isinstance(event, (ToolStarted, ToolFinished)) for event in received)
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["incomplete", "cancel"])
async def test_a_failed_summary_keeps_the_written_file_and_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    calls = (write_call("call_ok", "a.txt", "kept"),)
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
            async for event in client.stream_reply("write it"):
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
async def test_cancelling_a_turn_keeps_only_a_commit_that_already_started(
    tmp_path: Path, protocol: str, stage: Literal["before", "during"],
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    (tmp_path / "later.txt").write_bytes(b"keep")
    hold = CommitHold(stage)
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    calls = (write_call("call_write", "a.txt", "new"), write_call("call_later", "later.txt", "changed"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("write them")
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


def _provider_file(tmp_path: Path, protocol: str) -> Path:
    path = tmp_path / "provider.yaml"
    path.write_text(VALID_YAML.replace("protocol: anthropic", f"protocol: {protocol}"), encoding="utf-8")
    return path


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_shows_a_real_write_and_stays_busy_until_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = (write_call("call_new", "pkg/notes.txt", "hello"),)
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([ControlledSSE(batch_events(protocol, calls)), summary])
    app = create_app(["--config", str(_provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 32)) as pilot:
        app.query_one("#prompt", TextArea).text = "write notes"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        assert (tmp_path / "pkg" / "notes.txt").read_bytes() == b"hello"
        view = app.query_one(".tool-call")
        assert view.query_one(".tool-header", Static).content == "Write pkg/notes.txt"
        assert "pkg/notes.txt" in str(view.query_one(".tool-header", Static).content)
        assert str(view.query_one(".tool-status", Static).content).startswith("Success")
        assert "Wrote" in str(view.query_one(".tool-preview", Static).content)
        assert app.query_one("#status", Static).content == "Generating..."
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
    assert (tmp_path / "pkg" / "notes.txt").read_bytes() == b"hello"
    assert list(tmp_path.glob(".replux-*")) == []


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_shows_a_failed_write_batch_without_changing_the_skipped_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "blocked").mkdir()
    (tmp_path / "later.txt").write_text("keep", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = (
        write_call("call_ok", "ok.txt", "done"),
        write_call("call_bad", "blocked", "nope"),
        write_call("call_later", "later.txt", "changed"),
    )
    summary = ControlledSSE(reply_events(protocol), gate_after=0)
    transport = RecordingTransport([ControlledSSE(batch_events(protocol, calls)), summary])
    app = create_app(["--config", str(_provider_file(tmp_path, protocol))], transport=transport)
    async with app.run_test(size=(80, 36)) as pilot:
        app.query_one("#prompt", TextArea).text = "write three"
        await pilot.press("enter")
        await asyncio.wait_for(summary.blocked.wait(), timeout=2)
        await pilot.pause()
        assert statuses(app) == ["Success", "Failed", "Skipped"]
        failed = list(app.query(".tool-call"))[1]
        await pilot.click(failed.query_one(".tool-heading"))
        assert "Error code: not_a_file" in str(failed.query_one(".tool-metadata", Static).content)
        assert app.query_one("#status", Static).content == "Generating..."
        summary.release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Ready"
    assert (tmp_path / "ok.txt").read_bytes() == b"done"
    assert (tmp_path / "later.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "blocked" / "nope").exists()


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_ui_stays_cancelling_until_an_in_progress_commit_finishes(
    tmp_path: Path, protocol: str,
) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    (tmp_path / "later.txt").write_bytes(b"keep")
    hold = CommitHold("during")
    registry = ToolRegistry()
    registry.register(WriteFileTool(tmp_path, hold=hold))
    calls = (write_call("call_write", "a.txt", "new"), write_call("call_later", "later.txt", "changed"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    app = RepluxApp(open_client(protocol, transport, registry), model="scripted-model")
    async with app.run_test(size=(80, 32)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "write them"
        await pilot.press("enter")
        assert await asyncio.to_thread(hold.entered.wait, 2)
        await pilot.pause()
        assert app.query_one("#status", Static).content == "Running write_file..."
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
