import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from replux.llm import (
    AnthropicAdapter, Conversation, IncompleteReplyError, ReplyCompleted, StreamEvent, TextDelta, ThinkingDelta,
    ToolBatchLimitError, ToolFinished, ToolStarted, create_client,
)
from replux.tools import Tool, ToolParams, ToolRegistry, ToolResult, create_default_registry
from tests.support import (
    ControlledSSE, RecordingTransport, native_reply, native_tool_use, provider_config, reply_events, tool_use_events,
)


def anthropic_tools(root: Path) -> list[dict[str, Any]]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in create_default_registry(root).definitions()
    ]


def tool_result_message(call_id: str, result: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": call_id, "content": result, "is_error": is_error,
    }]}


def messages(request: Any) -> list[dict[str, Any]]:
    """Request messages with tool_result content decoded from its JSON text."""
    decoded = json.loads(request.content)["messages"]
    for message in decoded:
        if isinstance(message["content"], list):
            for block in message["content"]:
                if block["type"] == "tool_result":
                    block["content"] = json.loads(block["content"])
    return list(decoded)


async def test_anthropic_reads_a_real_file_and_summarizes_with_tools_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("第一行\nsecond\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    streams = [ControlledSSE(tool_use_events()), ControlledSSE(reply_events()), ControlledSSE(reply_events())]
    transport = RecordingTransport(streams)
    async with create_client(provider_config(), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        received = [event async for event in client.stream_reply("读一下 notes.txt")]
        _ = [event async for event in client.stream_reply("follow-up")]

    result = ToolResult("success", "1\t第一行\n2\tsecond", details={
        "path": str(tmp_path / "notes.txt"), "start_line": 1, "end_line": 2,
    })
    finished = received[3]
    assert isinstance(finished, ToolFinished) and finished.duration_seconds is not None
    assert received == [
        ThinkingDelta("先看文件"), TextDelta("我先读文件。"),
        ToolStarted("toolu_read", "read_file", "path=notes.txt"),
        ToolFinished("toolu_read", "read_file", "path=notes.txt", result, finished.duration_seconds),
        TextDelta("你好"), TextDelta("，世界\n"), ReplyCompleted("end_turn", 11, 5),
    ]
    first, summary, follow_up = (json.loads(request.content) for request in transport.requests)
    assert first["tools"] == anthropic_tools(tmp_path)
    assert "tool_choice" not in first
    assert first["messages"] == [{"role": "user", "content": "读一下 notes.txt"}]
    assert summary["tools"] == first["tools"]
    assert summary["tool_choice"] == {"type": "none"}
    committed = [
        {"role": "user", "content": "读一下 notes.txt"},
        native_tool_use({"path": "notes.txt"}),
        tool_result_message("toolu_read", json.loads(result.to_json())),
    ]
    assert messages(transport.requests[1]) == committed
    assert messages(transport.requests[2]) == [*committed, *native_reply("anthropic"), {"role": "user", "content": "follow-up"}]
    assert "tool_choice" not in follow_up
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("tool_call", ["missing_file", "unknown_tool", "invalid_arguments"])
async def test_tool_errors_are_returned_to_the_model_and_the_turn_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_call: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    name, arguments, code = {
        "missing_file": ("read_file", '{"path": "absent.txt"}', "file_not_found"),
        "unknown_tool": ("delete_files", '{"path": "absent.txt"}', "unknown_tool"),
        "invalid_arguments": ("read_file", '{"path": "absent.txt", "offset": "2"}', "invalid_arguments"),
    }[tool_call]
    transport = RecordingTransport([ControlledSSE(tool_use_events((arguments,), name=name)), ControlledSSE(reply_events())])
    async with create_client(provider_config(), transport=transport) as client:
        received = [event async for event in client.stream_reply("read it")]
    [finished] = [event for event in received if isinstance(event, ToolFinished)]
    assert (finished.result.status, finished.result.error_code) == ("error", code)
    assert received[-1] == ReplyCompleted("end_turn", 11, 5)
    [block] = messages(transport.requests[1])[-1]["content"]
    assert block["is_error"] is True
    assert block["content"]["status"] == "error"
    assert block["content"]["error_code"] == code


async def test_tool_runs_only_after_the_response_completes_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("while streaming\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    events = tool_use_events()
    arguments_done = next(index for index, event in enumerate(events) if event == {"type": "content_block_stop", "index": 2})
    pending = ControlledSSE(events, gate_after=arguments_done)
    pending.release_close.clear()
    transport = RecordingTransport([pending, ControlledSSE(reply_events())])
    received: list[StreamEvent] = []
    async with create_client(provider_config(), transport=transport) as client:
        async def consume() -> None:
            async for event in client.stream_reply("read it"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(pending.blocked.wait(), timeout=2)
            assert received == [ThinkingDelta("先看文件"), TextDelta("我先读文件。")]
            notes.write_text("while closing\n", encoding="utf-8")
            pending.release.set()
            await asyncio.wait_for(pending.closing.wait(), timeout=2)
            assert received == [ThinkingDelta("先看文件"), TextDelta("我先读文件。")]
            notes.write_text("after cleanup\n", encoding="utf-8")
        finally:
            pending.release.set()
            pending.release_close.set()
        await asyncio.wait_for(consuming, timeout=2)
    [finished] = [event for event in received if isinstance(event, ToolFinished)]
    assert finished.result.output == "1\tafter cleanup"
    assert len(transport.requests) == 2


@pytest.mark.parametrize("ending", ["eof", "output_limit", "invalid_json"])
async def test_unfinished_tool_response_executes_nothing_and_stays_out_of_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str,
) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    events = tool_use_events(('{"path": "notes.txt"',) if ending == "invalid_json" else ('{"path": ', '"notes.txt"}'))
    if ending == "eof":
        events = events[:-2]
    elif ending == "output_limit":
        events[-2]["delta"]["stop_reason"] = "max_tokens"
    unfinished = ControlledSSE(events)
    transport = RecordingTransport([unfinished, ControlledSSE(reply_events())])
    received: list[StreamEvent] = []
    async with create_client(provider_config(), transport=transport) as client:
        with pytest.raises(IncompleteReplyError):
            async for event in client.stream_reply("read it"):
                received.append(event)
        assert received == [ThinkingDelta("先看文件"), TextDelta("我先读文件。")]
        assert len(transport.requests) == 1
        _ = [event async for event in client.stream_reply("follow-up")]
    assert unfinished.closed
    assert messages(transport.requests[1]) == [{"role": "user", "content": "follow-up"}]


@pytest.mark.parametrize("call_id", ["", "duplicate"])
async def test_unpairable_tool_calls_execute_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call_id: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    events = tool_use_events(call_id=call_id)
    if call_id == "duplicate":
        start = next(index for index, event in enumerate(events) if event.get("content_block", {}).get("type") == "tool_use")
        copies = [{**event, "index": 3} for event in events[start:-2]]
        events[-2:-2] = copies
    transport = RecordingTransport([ControlledSSE(events), ControlledSSE(reply_events())])
    async with create_client(provider_config(), transport=transport) as client:
        received: list[StreamEvent] = []
        with pytest.raises(IncompleteReplyError, match="IDs"):
            async for event in client.stream_reply("read it"):
                received.append(event)
    assert not any(isinstance(event, (ToolStarted, ToolFinished)) for event in received)
    assert len(transport.requests) == 1


class WaitParams(ToolParams):
    path: str


class WaitTool(Tool[WaitParams]):
    name = "read_file"
    description = "Wait until released."
    params_model = WaitParams
    category = "read"

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, params: WaitParams) -> ToolResult:
        self.entered.set()
        await self.release.wait()
        return ToolResult("success", "released")


@pytest.mark.parametrize("ending", ["cancel_during_tool", "close_after_start"])
async def test_ended_tool_batch_commits_its_real_results_without_a_summary(ending: str) -> None:
    tool = WaitTool()
    registry = ToolRegistry()
    registry.register(tool)
    transport = RecordingTransport([ControlledSSE(tool_use_events()), ControlledSSE(reply_events())])
    async with Conversation(AnthropicAdapter(provider_config(), transport=transport), registry) as client:
        reply = client.stream_reply("read it")
        while not isinstance(await anext(reply), ToolStarted):
            pass
        if ending == "cancel_during_tool":
            pending = asyncio.create_task(anext(reply))
            await asyncio.wait_for(tool.entered.wait(), timeout=2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            await reply.aclose()
        assert not tool.release.is_set()
        assert len(transport.requests) == 1
        _ = [event async for event in client.stream_reply("follow-up")]
    status, output = {
        "cancel_during_tool": ("cancelled", "Cancelled by the user before the tool call finished."),
        "close_after_start": ("skipped", "Not run because the user cancelled the Turn."),
    }[ending]
    assert messages(transport.requests[1]) == [
        {"role": "user", "content": "read it"},
        native_tool_use({"path": "notes.txt"}),
        tool_result_message("toolu_read", {"status": status, "output": output}, is_error=True),
        {"role": "user", "content": "follow-up"},
    ]


async def test_tool_request_in_the_summary_stops_the_turn_and_keeps_earlier_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    again = tool_use_events(call_id="toolu_again")
    transport = RecordingTransport([ControlledSSE(tool_use_events()), ControlledSSE(again), ControlledSSE(reply_events())])
    async with create_client(provider_config(), transport=transport) as client:
        received: list[StreamEvent] = []
        with pytest.raises(ToolBatchLimitError):
            async for event in client.stream_reply("read it"):
                received.append(event)
        assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["toolu_read"]
        assert not any(isinstance(event, ReplyCompleted) for event in received)
        _ = [event async for event in client.stream_reply("follow-up")]
    history = messages(transport.requests[2])
    assert history[:2] == [{"role": "user", "content": "read it"}, native_tool_use({"path": "notes.txt"})]
    assert history[2]["content"][0]["tool_use_id"] == "toolu_read"
    assert history[3:] == [{"role": "user", "content": "follow-up"}]
