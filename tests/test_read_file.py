import asyncio
import json
import os
from pathlib import Path
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict
import pytest

from replux.tools import (
    FILE_OPERATION_TIMEOUT_SECONDS, FileOperationTimeout, Tool, ToolParams, ToolRegistry, ToolResult,
    create_default_registry, run_file_operation,
)
from replux.tools.read_file import ReadFileTool


async def read(registry: ToolRegistry, **arguments: Any) -> ToolResult:
    return await registry.execute("read_file", json.dumps(arguments))


def test_read_file_schema_comes_from_its_strict_parameter_model() -> None:
    registry = create_default_registry(Path("/startup"))
    definition = next(item for item in registry.definitions() if item.name == "read_file")
    tool = registry.get("read_file")
    assert definition.name == "read_file"
    assert definition.input_schema == tool.params_model.model_json_schema()
    schema = definition.input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["path"]
    assert schema["properties"]["path"]["minLength"] == 1
    offset, limit = schema["properties"]["offset"], schema["properties"]["limit"]
    assert (offset["type"], offset["default"], offset["minimum"]) == ("integer", 1, 1)
    assert (limit["type"], limit["default"], limit["minimum"], limit["maximum"]) == ("integer", 2000, 1, 2000)
    assert tool.category == "read"
    assert definition.description.isascii() and "offset" in definition.description
    assert isinstance(tool, ReadFileTool)
    assert tool.timeout_seconds == FILE_OPERATION_TIMEOUT_SECONDS == 10


@pytest.mark.parametrize("arguments", [
    {"path": "a.txt", "extra": True},
    {"path": 7},
    {"path": "a.txt", "offset": "2"},
    {"path": "a.txt", "limit": 2.0},
    {"path": "a.txt", "limit": True},
    {"path": ""},
    {"path": "a.txt", "offset": 0},
    {"path": "a.txt", "limit": 0},
    {"path": "a.txt", "limit": 2001},
    {},
])
async def test_invalid_model_arguments_become_an_error_result_without_reading(
    tmp_path: Path, arguments: dict[str, Any],
) -> None:
    (tmp_path / "a.txt").write_text("content\n", encoding="utf-8")
    result = await create_default_registry(tmp_path).execute("read_file", json.dumps(arguments))
    assert result.status == "error"
    assert result.error_code == "invalid_arguments"
    assert "content" not in result.output
    assert result.output.isascii()


async def test_arguments_that_are_not_a_json_object_are_invalid(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)
    for arguments in ['{"path": "a.txt"', '["a.txt"]']:
        result = await registry.execute("read_file", arguments)
        assert (result.status, result.error_code) == ("error", "invalid_arguments")


async def test_unknown_tool_is_an_error_result(tmp_path: Path) -> None:
    result = await create_default_registry(tmp_path).execute("delete_everything", "{}")
    assert (result.status, result.error_code) == ("error", "unknown_tool")
    assert "read_file" in result.output


class EchoParams(ToolParams):
    text: str


class EchoTool(Tool[EchoParams]):
    name = "echo"
    description = "Echo the text."
    params_model = EchoParams
    category = "read"

    async def execute(self, params: EchoParams) -> ToolResult:
        return ToolResult("success", params.text)


class LooseParams(BaseModel):
    text: str


class ForgivingParams(BaseModel):
    model_config = ConfigDict(strict=True)
    text: str


def test_registry_finds_and_lists_registered_tools() -> None:
    registry = ToolRegistry()
    echo = EchoTool()
    registry.register(echo)
    assert registry.get("echo") is echo
    assert [definition.name for definition in registry.definitions()] == ["echo"]
    with pytest.raises(KeyError):
        registry.get("missing")


@pytest.mark.parametrize("change", [
    {"name": "bad name"}, {"name": ""}, {"name": "x" * 65}, {"description": " "},
    {"category": "admin"}, {"params_model": LooseParams}, {"params_model": ForgivingParams},
    {"params_model": dict},
])
def test_registry_rejects_invalid_definitions_immediately(change: dict[str, Any]) -> None:
    tool = EchoTool()
    for attribute, value in change.items():
        setattr(tool, attribute, value)
    registry = ToolRegistry()
    with pytest.raises((TypeError, ValueError)):
        registry.register(tool)
    assert registry.definitions() == []


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError, match="echo"):
        registry.register(EchoTool())


async def test_internal_errors_in_a_tool_are_not_converted_into_results() -> None:
    class BrokenTool(EchoTool):
        async def execute(self, params: EchoParams) -> ToolResult:
            raise TypeError("injected defect")

    registry = ToolRegistry()
    registry.register(BrokenTool())
    with pytest.raises(TypeError, match="injected defect"):
        await registry.execute("echo", '{"text": "hi"}')


@pytest.mark.parametrize("status", ["success", "error", "cancelled", "skipped"])
def test_tool_result_serializes_its_status_code_and_details(status: Any) -> None:
    code = "file_not_found" if status == "error" else None
    result = ToolResult(status, "Output ✓", code, {"next_offset": 3, "range": [1, 2]})
    assert json.loads(result.to_json()) == {
        "status": status, "output": "Output ✓", "details": {"next_offset": 3, "range": [1, 2]},
        **({"error_code": code} if code else {}),
    }


@pytest.mark.parametrize("arguments", [
    ("done", "x", None, {}), ("error", "x", None, {}), ("success", "x", "file_not_found", {}),
    ("success", "x", None, {"path": Path("a")}),
])
def test_tool_result_rejects_invalid_construction(arguments: tuple[Any, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ToolResult(*arguments)


async def test_relative_paths_use_the_startup_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = tmp_path / "startup"
    (startup / "pkg").mkdir(parents=True)
    (startup / "pkg" / "a.txt").write_text("first\nsecond\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "pkg").mkdir(parents=True)
    (elsewhere / "pkg" / "a.txt").write_text("wrong\n", encoding="utf-8")
    monkeypatch.chdir(startup)
    registry = create_default_registry(Path.cwd())
    monkeypatch.chdir(elsewhere)
    result = await read(registry, path="pkg/a.txt")
    assert result == ToolResult("success", "1\tfirst\n2\tsecond", details={
        "path": str(startup / "pkg" / "a.txt"), "start_line": 1, "end_line": 2,
    })


async def test_absolute_paths_and_explicit_symlinks_read_the_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("目标\n", encoding="utf-8")
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "link.txt").symlink_to(outside)
    registry = create_default_registry(startup)
    assert (await read(registry, path=str(outside))).output == "1\t目标"
    via_link = await read(registry, path="link.txt")
    assert via_link.output == "1\t目标"
    assert (startup / "link.txt").is_symlink()


async def test_offset_and_limit_select_actual_lines_with_a_continuation(tmp_path: Path) -> None:
    lines = [f"line {number}" for number in range(1, 11)]
    (tmp_path / "a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    result = await read(registry, path="a.txt", offset=3, limit=4)
    assert result.status == "success"
    assert result.output == "3\tline 3\n4\tline 4\n5\tline 5\n6\tline 6\n\n[Showing lines 3-6. Use offset=7 to continue.]"
    assert result.details == {
        "path": str(tmp_path / "a.txt"), "start_line": 3, "end_line": 6, "next_offset": 7, "truncated_by": "lines",
    }
    last = await read(registry, path="a.txt", offset=7, limit=4)
    assert last.output == "7\tline 7\n8\tline 8\n9\tline 9\n10\tline 10"
    assert "next_offset" not in last.details


async def test_lines_keep_carriage_returns_and_final_lines_without_newline(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"one\r\n\ntwo")
    result = await read(create_default_registry(tmp_path), path="a.txt")
    assert result.output == "1\tone\r\n2\t\n3\ttwo"


async def test_empty_file_and_offset_past_the_end(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    empty = await read(registry, path="empty.txt")
    assert (empty.status, empty.output) == ("success", "The file is empty.")
    beyond = await read(registry, path="a.txt", offset=3)
    assert (beyond.status, beyond.error_code) == ("error", "offset_out_of_range")
    assert "2 lines" in beyond.output


@pytest.mark.parametrize(("setup", "code"), [
    ("missing", "file_not_found"),
    ("under_file", "file_not_found"),
    ("directory", "not_a_file"),
    ("latin1", "unsupported_encoding"),
    ("binary", "binary_file"),
    ("permission", "permission_denied"),
])
async def test_expected_read_failures_are_explicit_error_results(tmp_path: Path, setup: str, code: str) -> None:
    target = tmp_path / "target"
    if setup == "directory":
        target.mkdir()
    elif setup == "latin1":
        target.write_bytes("café\n".encode("latin-1"))
    elif setup == "binary":
        target.write_bytes(b"PK\x03\x04\x00\x00text\n")
    elif setup == "permission":
        if os.geteuid() == 0:
            pytest.skip("root can read files without permission bits")
        target.write_text("secret\n", encoding="utf-8")
        target.chmod(0)
    elif setup == "under_file":
        (tmp_path / "file").write_text("x", encoding="utf-8")
        target = tmp_path / "file" / "child"
    result = await read(create_default_registry(tmp_path), path=str(target.relative_to(tmp_path)))
    assert (result.status, result.error_code) == ("error", code)
    assert result.output.isascii() and result.output
    assert "secret" not in result.output


async def test_default_line_limit_keeps_the_head_and_reports_the_next_offset(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("".join(f"{n}\n" for n in range(1, 2501)), encoding="utf-8")
    result = await read(create_default_registry(tmp_path), path="long.txt")
    body, notice = result.output.split("\n\n")
    assert body.split("\n") == [f"{n}\t{n}" for n in range(1, 2001)]
    assert notice == "[Showing lines 1-2000. Use offset=2001 to continue.]"
    assert result.details["next_offset"] == 2001
    assert result.details["truncated_by"] == "lines"


async def test_byte_limit_keeps_whole_utf8_lines_and_reports_the_next_offset(tmp_path: Path) -> None:
    lines = ["汉字" * 50 + "é" for _ in range(400)]
    (tmp_path / "wide.txt").write_text("\n".join(lines), encoding="utf-8")
    result = await read(create_default_registry(tmp_path), path="wide.txt", offset=2)
    body, notice = result.output.split("\n\n")
    shown = body.split("\n")
    end = result.details["end_line"]
    assert isinstance(end, int)
    assert shown == [f"{n}\t{lines[n - 1]}" for n in range(2, end + 1)]
    assert len(body.encode()) <= 50 * 1024
    assert len((body + f"\n{end + 1}\t{lines[end]}").encode()) > 50 * 1024
    assert result.details == {
        "path": str(tmp_path / "wide.txt"), "start_line": 2, "end_line": end,
        "next_offset": end + 1, "truncated_by": "bytes",
    }
    assert notice == f"[Showing lines 2-{end} (50 KiB limit). Use offset={end + 1} to continue.]"


async def test_oversized_first_line_is_explained_without_a_forged_offset(tmp_path: Path) -> None:
    (tmp_path / "min.js").write_text("x" * (60 * 1024) + "\nnext\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    result = await read(registry, path="min.js")
    assert result.status == "success"
    assert result.details == {"path": str(tmp_path / "min.js"), "line": 1, "line_bytes": 60 * 1024, "truncated_by": "bytes"}
    assert "Line 1 is 61440 bytes" in result.output
    assert "head -c" in result.output and "min.js" in result.output
    assert "xxx" not in result.output
    assert (await read(registry, path="min.js", offset=2)).output == "2\tnext"


async def test_file_operation_timeout_waits_until_the_worker_stops() -> None:
    entered, release, passed = threading.Event(), threading.Event(), threading.Event()

    def operation(checkpoint: Any) -> str:
        entered.set()
        release.wait()
        checkpoint()
        passed.set()
        return "finished"

    running = asyncio.create_task(run_file_operation(operation, timeout_seconds=0.01))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
    finally:
        release.set()
    with pytest.raises(FileOperationTimeout):
        await asyncio.wait_for(running, timeout=2)
    assert not passed.is_set()


async def test_cancelled_file_operation_waits_until_the_worker_stops() -> None:
    entered, release, passed = threading.Event(), threading.Event(), threading.Event()

    def operation(checkpoint: Any) -> str:
        entered.set()
        release.wait()
        checkpoint()
        passed.set()
        return "finished"

    running = asyncio.create_task(run_file_operation(operation, timeout_seconds=10))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        running.cancel()
        running.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=2)
    assert not passed.is_set()


async def test_a_finished_file_operation_keeps_its_result_when_cancellation_arrived() -> None:
    entered, release = threading.Event(), threading.Event()

    def operation(checkpoint: Any) -> str:
        entered.set()
        release.wait()
        return "finished"

    running = asyncio.create_task(run_file_operation(operation, timeout_seconds=10))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        running.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
    finally:
        release.set()
    assert await asyncio.wait_for(running, timeout=2) == "finished"
