"""One Tool Batch: order, stop on failure, cancellation, and summary recovery."""

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import threading
from typing import Any

import pytest
import httpx2 as httpx

from replux.llm import (
    AnthropicAdapter, Conversation, IncompleteReplyError, LLMClient, LLMError, ModelResponse, OpenAIAdapter,
    ProtocolAdapter, ReplyCompleted, StreamEvent, TextDelta, ToolBatchLimitError, ToolFinished, ToolStarted,
    create_client,
)
from replux.tools import Tool, ToolParams, ToolRegistry, ToolResult, create_default_registry, run_file_operation
from replux.tools.read_file import ReadFileParams, ReadFileTool
from tests.support import (
    ControlledSSE, RecordingTransport, function_call_events, provider_config, reply_events, tool_batch_events,
)
from tests.test_tool_turn import messages, responses_input


class RecordingReadFile(ReadFileTool):
    """Real reads, plus the paths that actually entered execute."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.paths: list[str] = []

    async def execute(self, params: ReadFileParams) -> ToolResult:
        self.paths.append(params.path)
        return await super().execute(params)


class GateParams(ToolParams):
    path: str


class GatedRead(Tool[GateParams]):
    """Blocks in a file-operation thread until released, then stops at its checkpoint."""

    name = "read_file"
    description = "Read after a checkpoint."
    params_model = GateParams
    category = "read"

    def __init__(self, side_effect: Path) -> None:
        self.side_effect = side_effect
        self.runs = 0
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.stopped = asyncio.Event()

    async def execute(self, params: GateParams) -> ToolResult:
        self.runs += 1
        self.entered.set()
        try:
            return await run_file_operation(self._work, timeout_seconds=5)
        finally:
            self.stopped.set()

    def _work(self, checkpoint: Callable[[], None]) -> ToolResult:
        self.release.wait()
        checkpoint()
        self.side_effect.write_text("ran", encoding="utf-8")
        return ToolResult("success", "ran")


class BoomParams(ToolParams):
    path: str


class BoomRead(Tool[BoomParams]):
    """Raises a programming error only after its worker has returned."""

    name = "read_file"
    description = "Fail after a checkpoint."
    params_model = BoomParams
    category = "read"

    def __init__(self) -> None:
        self.runs = 0
        self.entered = asyncio.Event()
        self.release = threading.Event()

    async def execute(self, params: BoomParams) -> ToolResult:
        self.runs += 1
        self.entered.set()

        def work(checkpoint: Callable[[], None]) -> ToolResult:
            self.release.wait()
            checkpoint()
            raise TypeError("injected defect")

        return await run_file_operation(work, timeout_seconds=5)


ToolRequest = tuple[str, str, tuple[str, ...]]


def path_pieces(name: str) -> tuple[str, ...]:
    return ('{"path": "', name + '"}')


def read_call(call_id: str, name: str) -> ToolRequest:
    return (call_id, "read_file", path_pieces(name))


def batch_events(
    protocol: str, calls: tuple[ToolRequest, ...], *, decoy: bool = False,
) -> list[dict[str, Any]]:
    if protocol == "anthropic":
        return tool_batch_events(calls, decoy_input={"path": "wrong.txt"} if decoy else None)
    paired = tuple(
        (f"fc_{index}", call_id, name, pieces) for index, (call_id, name, pieces) in enumerate(calls)
    )
    stored = tuple('{"path": "wrong.txt"}' for _ in calls) if decoy else None
    return function_call_events(paired, completed_arguments=stored)


def open_client(protocol: str, transport: httpx.AsyncBaseTransport, registry: ToolRegistry) -> LLMClient:
    config = provider_config(protocol)
    adapter: ProtocolAdapter[Any]
    if protocol == "anthropic":
        adapter = AnthropicAdapter(config, transport=transport)
    else:
        adapter = OpenAIAdapter(config, transport=transport)
    return Conversation[Any](adapter, registry)


def recording_client(
    protocol: str, transport: httpx.AsyncBaseTransport, root: Path,
) -> tuple[LLMClient, RecordingReadFile]:
    tool = RecordingReadFile(root)
    registry = ToolRegistry()
    registry.register(tool)
    return open_client(protocol, transport, registry), tool


def result_pairs(protocol: str, request: httpx.Request) -> list[tuple[str, dict[str, Any]]]:
    if protocol == "anthropic":
        pairs: list[tuple[str, dict[str, Any]]] = []
        for message in messages(request):
            content = message["content"]
            if isinstance(content, list):
                for block in content:
                    if block["type"] == "tool_result":
                        pairs.append((str(block["tool_use_id"]), block["content"]))
        return pairs
    return [
        (str(item["call_id"]), item["output"])
        for item in responses_input(request)
        if item.get("type") == "function_call_output"
    ]


def text_gate(protocol: str) -> int:
    events = reply_events(protocol)
    if protocol == "anthropic":
        return next(index for index, event in enumerate(events) if event.get("delta", {}).get("text") == "你好")
    return next(index for index, event in enumerate(events) if event.get("delta") == "你好")


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_one_batch_reads_files_in_order_and_keeps_argument_pieces_apart(
    tmp_path: Path, protocol: str,
) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    calls = (read_call("call_a", "a.txt"), read_call("call_b", "b.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls, decoy=True)),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(reply_events(protocol)),
    ])
    client, tool = recording_client(protocol, transport, tmp_path)
    async with client:
        received = [event async for event in client.stream_reply("read both")]
        _ = [event async for event in client.stream_reply("follow-up")]

    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [event.call_id for event in finished] == ["call_a", "call_b"]
    assert [event.result.output for event in finished] == ["1\talpha", "1\tbeta"]
    positions = [index for index, event in enumerate(received) if isinstance(event, (ToolStarted, ToolFinished))]
    assert [type(received[index]) for index in positions] == [ToolStarted, ToolFinished, ToolStarted, ToolFinished]
    assert tool.paths == ["a.txt", "b.txt"]
    assert isinstance(received[-1], ReplyCompleted)
    pairs = result_pairs(protocol, transport.requests[1])
    assert [call_id for call_id, _ in pairs] == ["call_a", "call_b"]
    assert [payload["output"] for _, payload in pairs] == ["1\talpha", "1\tbeta"]
    assert "wrong.txt" not in json.dumps(pairs)
    if protocol == "openai":
        replayed = [item for item in responses_input(transport.requests[1]) if item.get("type") == "function_call"]
        assert [(item["id"], item["call_id"]) for item in replayed] == [("fc_0", "call_a"), ("fc_1", "call_b")]
    else:
        blocks = [block for block in messages(transport.requests[1])[1]["content"] if block["type"] == "tool_use"]
        assert [(block["id"], block["input"]) for block in blocks] == [
            ("call_a", {"path": "a.txt"}), ("call_b", {"path": "b.txt"}),
        ]
        assert messages(transport.requests[1])[1]["content"][0]["signature"] == "tool-signature+/="
    summary = json.loads(transport.requests[1].content)
    follow_up = json.loads(transport.requests[2].content)
    if protocol == "anthropic":
        assert summary["tool_choice"] == {"type": "none"}
        assert "tool_choice" not in follow_up
    else:
        assert summary["tool_choice"] == "none"
        assert "tool_choice" not in follow_up


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_first_failure_skips_the_rest_without_running_them(tmp_path: Path, protocol: str) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    calls = (read_call("call_ok", "ok.txt"), read_call("call_missing", "missing.txt"), read_call("call_later", "later.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    client, tool = recording_client(protocol, transport, tmp_path)
    async with client:
        received = [event async for event in client.stream_reply("read three")]

    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [event.call_id for event in finished] == ["call_ok", "call_missing", "call_later"]
    assert [event.result.status for event in finished] == ["success", "error", "skipped"]
    assert finished[0].result.output == "1\talpha"
    assert finished[1].result.error_code == "file_not_found"
    assert finished[1].duration_seconds is not None
    assert finished[2].result.output == "Not run because an earlier tool call in this batch did not succeed."
    assert finished[2].duration_seconds is None
    assert not any(isinstance(event, ToolStarted) and event.call_id == "call_later" for event in received)
    assert tool.paths == ["ok.txt", "missing.txt"]
    assert isinstance(received[-1], ReplyCompleted)
    pairs = result_pairs(protocol, transport.requests[1])
    assert [(call_id, payload["status"]) for call_id, payload in pairs] == [
        ("call_ok", "success"), ("call_missing", "error"), ("call_later", "skipped"),
    ]
    assert "should-not" not in json.dumps(pairs)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("kind", ["unknown_tool", "schema"])
async def test_replayable_call_errors_stop_the_batch(tmp_path: Path, protocol: str, kind: str) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    calls: tuple[ToolRequest, ...]
    if kind == "unknown_tool":
        calls = (
            ("call_bad", "delete_files", path_pieces("ok.txt")),
            read_call("call_later", "later.txt"),
        )
    else:
        calls = (
            read_call("call_ok", "ok.txt"),
            ("call_bad", "read_file", ('{"path": "ok.txt", "offset": "', '2"}')),
            read_call("call_later", "later.txt"),
        )
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    client, tool = recording_client(protocol, transport, tmp_path)
    async with client:
        received = [event async for event in client.stream_reply("read")]

    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert finished[-1].call_id == "call_later"
    assert finished[-1].result.status == "skipped"
    assert "should-not" not in finished[-1].result.output
    bad = next(event for event in finished if event.call_id == "call_bad")
    assert bad.result.status == "error"
    assert bad.result.error_code == ("unknown_tool" if kind == "unknown_tool" else "invalid_arguments")
    assert tool.paths == ([] if kind == "unknown_tool" else ["ok.txt"])
    assert isinstance(received[-1], ReplyCompleted)
    assert "later.txt" not in tool.paths


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_replayable_invalid_json_stops_the_batch_without_a_forged_call(
    tmp_path: Path, protocol: str,
) -> None:
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    # OpenAI stores arguments as text, so invalid JSON is still a replayable call.
    # Anthropic tool_use.input has to be an object, so the same text cannot be replayed.
    calls: tuple[ToolRequest, ...] = (("call_bad", "read_file", ('{"path":',)), read_call("call_later", "later.txt"))
    if protocol == "anthropic":
        calls = (read_call("call_ok", "later.txt"), ("call_bad", "read_file", ('{"path":',)))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    client, tool = recording_client(protocol, transport, tmp_path)
    received: list[StreamEvent] = []
    async with client:
        if protocol == "anthropic":
            with pytest.raises(IncompleteReplyError):
                async for event in client.stream_reply("read"):
                    received.append(event)
            assert tool.paths == []
            assert not any(isinstance(event, (ToolStarted, ToolFinished)) for event in received)
            _ = [event async for event in client.stream_reply("follow-up")]
            assert messages(transport.requests[1]) == [{"role": "user", "content": "follow-up"}]
            return
        received = [event async for event in client.stream_reply("read")]
    finished = [event for event in received if isinstance(event, ToolFinished)]
    assert [(event.call_id, event.result.status, event.result.error_code) for event in finished] == [
        ("call_bad", "error", "invalid_arguments"),
        ("call_later", "skipped", None),
    ]
    assert tool.paths == []
    replayed = [item for item in responses_input(transport.requests[1]) if item.get("type") == "function_call"]
    assert replayed[0]["arguments"] == '{"path":'
    assert replayed[0]["call_id"] == "call_bad"
    assert "should-not" not in json.dumps(result_pairs(protocol, transport.requests[1]))


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("kind", ["missing", "duplicate"])
async def test_unpairable_ids_execute_nothing_even_when_an_earlier_call_is_valid(
    tmp_path: Path, protocol: str, kind: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    bad_id = "" if kind == "missing" else "call_ok"
    calls = (read_call("call_ok", "ok.txt"), read_call(bad_id, "later.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    client, tool = recording_client(protocol, transport, tmp_path)
    received: list[StreamEvent] = []
    async with client:
        with pytest.raises(IncompleteReplyError, match="IDs"):
            async for event in client.stream_reply("read"):
                received.append(event)
        assert tool.paths == []
        assert not any(isinstance(event, (ToolStarted, ToolFinished, ReplyCompleted)) for event in received)
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["eof", "cleanup"])
async def test_a_broken_response_executes_nothing_after_arguments_are_complete(
    tmp_path: Path, protocol: str, ending: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    events = batch_events(protocol, (read_call("call_ok", "ok.txt"),))
    if ending == "eof" and protocol == "anthropic":
        del events[-2:]
    elif ending == "eof":
        events.pop()
    stream = ControlledSSE(events, close_error=httpx.ReadError("offline cleanup failure") if ending == "cleanup" else None)
    transport = RecordingTransport([stream, ControlledSSE(reply_events(protocol))])
    client, tool = recording_client(protocol, transport, tmp_path)
    received: list[StreamEvent] = []
    async with client:
        if ending == "eof":
            with pytest.raises(IncompleteReplyError):
                async for event in client.stream_reply("read"):
                    received.append(event)
        else:
            with pytest.raises(LLMError) as caught:
                async for event in client.stream_reply("read"):
                    received.append(event)
            assert not isinstance(caught.value, IncompleteReplyError)
        assert tool.paths == []
        assert not any(isinstance(event, (ToolStarted, ToolFinished, ReplyCompleted)) for event in received)
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]
    assert stream.closed


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancel_waits_for_cleanup_then_commits_cancelled_and_skipped_results(
    tmp_path: Path, protocol: str,
) -> None:
    side_effect = tmp_path / "ran.txt"
    tool = GatedRead(side_effect)
    registry = ToolRegistry()
    registry.register(tool)
    calls = (read_call("call_run", "ok.txt"), read_call("call_next", "later.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("read them")
        while not isinstance(await anext(reply), ToolStarted):
            pass
        pending = asyncio.create_task(anext(reply))
        await asyncio.wait_for(tool.entered.wait(), timeout=2)
        pending.cancel()
        pending.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.1)
        assert not side_effect.exists()
        assert not tool.stopped.is_set()
        tool.release.set()
        cancelled = await asyncio.wait_for(pending, timeout=2)
        skipped = await asyncio.wait_for(anext(reply), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await anext(reply)
        assert tool.stopped.is_set()
        assert len(transport.requests) == 1
        _ = [event async for event in client.stream_reply("follow-up")]

    assert tool.runs == 1
    assert not side_effect.exists()
    assert isinstance(cancelled, ToolFinished)
    assert cancelled.call_id == "call_run"
    assert cancelled.result.status == "cancelled"
    assert cancelled.result.output == "Cancelled by the user before the tool call finished."
    assert cancelled.duration_seconds is not None
    assert isinstance(skipped, ToolFinished)
    assert skipped.call_id == "call_next"
    assert skipped.result.status == "skipped"
    assert skipped.duration_seconds is None
    pairs = result_pairs(protocol, transport.requests[1])
    assert [(call_id, payload["status"]) for call_id, payload in pairs] == [
        ("call_run", "cancelled"), ("call_next", "skipped"),
    ]
    body = json.loads(transport.requests[1].content)
    assert "你好" not in json.dumps(body, ensure_ascii=False)
    choice = "tool_choice"
    assert choice not in body
    assert len(transport.requests) == 2


class FinishingRead(GatedRead):
    """Returns after release without another checkpoint, so the finished result is real."""

    def _work(self, checkpoint: Callable[[], None]) -> ToolResult:
        self.release.wait()
        self.side_effect.write_text("ran", encoding="utf-8")
        return ToolResult("success", "ran")


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_cancel_keeps_a_finished_tool_result_and_skips_the_rest(tmp_path: Path, protocol: str) -> None:
    side_effect = tmp_path / "ran.txt"
    tool = FinishingRead(side_effect)
    registry = ToolRegistry()
    registry.register(tool)
    calls = (read_call("call_run", "ok.txt"), read_call("call_next", "later.txt"))
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)), ControlledSSE(reply_events(protocol)),
    ])
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("read them")
        while not isinstance(await anext(reply), ToolStarted):
            pass
        pending = asyncio.create_task(anext(reply))
        await asyncio.wait_for(tool.entered.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.1)
        tool.release.set()
        finished = await asyncio.wait_for(pending, timeout=2)
        skipped = await asyncio.wait_for(anext(reply), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await anext(reply)
        _ = [event async for event in client.stream_reply("follow-up")]
    assert side_effect.read_text(encoding="utf-8") == "ran"
    assert isinstance(finished, ToolFinished) and finished.result.status == "success"
    assert finished.result.output == "ran"
    assert isinstance(skipped, ToolFinished) and skipped.result.status == "skipped"
    assert [payload["status"] for _, payload in result_pairs(protocol, transport.requests[1])] == ["success", "skipped"]
    assert len(transport.requests) == 2


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("ending", ["incomplete", "cancel", "http_error"])
async def test_a_failed_summary_keeps_tool_results_and_drops_the_summary_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, ending: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    requests: list[httpx.Request] = []
    streams: list[ControlledSSE] = []
    held: list[ControlledSSE] = []
    summary_started = asyncio.Event()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 2 and ending == "http_error":
            summary_started.set()
            return httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}})
        if len(requests) == 1:
            events = batch_events(protocol, (read_call("call_ok", "ok.txt"),))
            gate = None
        elif len(requests) == 2 and ending == "incomplete":
            events = reply_events(protocol)[:-1]
            gate = None
        elif len(requests) == 2 and ending == "cancel":
            events = reply_events(protocol)
            gate = text_gate(protocol)
        else:
            events = reply_events(protocol)
            gate = None
        stream = ControlledSSE(events, gate_after=gate)
        if len(requests) == 2 and ending == "cancel":
            stream.release_close.clear()
            held.append(stream)
        if len(requests) == 2:
            summary_started.set()
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with create_client(provider_config(protocol), transport=httpx.MockTransport(respond)) as client:
        monkeypatch.chdir(tmp_path.parent)
        received: list[StreamEvent] = []

        async def consume() -> None:
            async for event in client.stream_reply("read it"):
                received.append(event)

        consuming = asyncio.create_task(consume())
        try:
            if ending == "cancel":
                await asyncio.wait_for(summary_started.wait(), timeout=2)
                await asyncio.wait_for(held[0].blocked.wait(), timeout=2)
                assert any(isinstance(event, TextDelta) and event.text == "你好" for event in received)
                consuming.cancel()
                await asyncio.wait_for(held[0].closing.wait(), timeout=2)
                assert not held[0].closed
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
        finished = [event for event in received if isinstance(event, ToolFinished)]
        assert [event.result.status for event in finished] == ["success"]
        assert finished[0].result.output == "1\talpha"
        assert not any(isinstance(event, ReplyCompleted) for event in received)
        _ = [event async for event in client.stream_reply("follow-up")]

    field = "messages" if protocol == "anthropic" else "input"
    history = json.loads(requests[-1].content)[field]
    assert history[0] == {"role": "user", "content": "read it"}
    assert history[-1] == {"role": "user", "content": "follow-up"}
    pairs = result_pairs(protocol, requests[-1])
    assert pairs == [("call_ok", json.loads(ToolResult(
        "success", "1\talpha", details={"path": str(tmp_path / "ok.txt"), "start_line": 1, "end_line": 1},
    ).to_json()))]
    assert "你好" not in json.dumps(history, ensure_ascii=False)
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_a_valid_tool_call_in_the_summary_stops_without_a_second_batch(
    tmp_path: Path, protocol: str,
) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "later.txt").write_text("should-not\n", encoding="utf-8")
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (read_call("call_ok", "ok.txt"),))),
        ControlledSSE(batch_events(protocol, (read_call("call_again", "later.txt"),))),
        ControlledSSE(reply_events(protocol)),
    ])
    client, tool = recording_client(protocol, transport, tmp_path)
    received: list[StreamEvent] = []
    async with client:
        with pytest.raises(ToolBatchLimitError, match="tool batch"):
            async for event in client.stream_reply("read"):
                received.append(event)
        assert [event.call_id for event in received if isinstance(event, ToolStarted)] == ["call_ok"]
        assert not any(isinstance(event, ReplyCompleted) for event in received)
        assert tool.paths == ["ok.txt"]
        _ = [event async for event in client.stream_reply("follow-up")]
    body = json.dumps(json.loads(transport.requests[2].content), ensure_ascii=False)
    assert "call_again" not in body
    assert "should-not" not in body
    assert "call_ok" in body
    assert len(transport.requests) == 3


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_adapter_parses_another_tool_response_without_a_batch_limit_or_summary(
    tmp_path: Path, protocol: str,
) -> None:
    calls = (read_call("call_ok", "ok.txt"),)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, calls)),
        ControlledSSE(batch_events(protocol, calls)),
    ])
    config = provider_config(protocol)
    adapter: ProtocolAdapter[Any]
    if protocol == "anthropic":
        adapter = AnthropicAdapter(config, transport=transport)
    else:
        adapter = OpenAIAdapter(config, transport=transport)
    tools = create_default_registry(tmp_path).definitions()
    try:
        allowed = [event async for event in adapter.stream_response([], tools, tools_allowed=True)]
        disabled = [event async for event in adapter.stream_response([], tools, tools_allowed=False)]
    finally:
        await adapter.aclose()
    assert isinstance(allowed[-1], ModelResponse) and [call.id for call in allowed[-1].tool_calls] == ["call_ok"]
    assert isinstance(disabled[-1], ModelResponse) and [call.id for call in disabled[-1].tool_calls] == ["call_ok"]
    assert len(transport.requests) == 2
    second = json.loads(transport.requests[1].content)
    assert second["tool_choice"] == ({"type": "none"} if protocol == "anthropic" else "none")


async def test_the_registry_runs_the_same_tool_twice(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("alpha\n", encoding="utf-8")
    registry = create_default_registry(tmp_path)
    first = await registry.execute("read_file", '{"path": "ok.txt"}')
    second = await registry.execute("read_file", '{"path": "ok.txt"}')
    assert first.status == second.status == "success"
    assert first.output == second.output == "1\talpha"


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_the_next_turn_can_run_its_own_tool_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    transport = RecordingTransport([
        ControlledSSE(batch_events(protocol, (read_call("call_a", "a.txt"),))),
        ControlledSSE(reply_events(protocol)),
        ControlledSSE(batch_events(protocol, (read_call("call_b", "b.txt"),))),
        ControlledSSE(reply_events(protocol)),
    ])
    async with create_client(provider_config(protocol), transport=transport) as client:
        monkeypatch.chdir(tmp_path.parent)
        first = [event async for event in client.stream_reply("first")]
        second = [event async for event in client.stream_reply("second")]
    assert [event.result.output for event in first if isinstance(event, ToolFinished)] == ["1\talpha"]
    assert [event.result.output for event in second if isinstance(event, ToolFinished)] == ["1\tbeta"]
    assert isinstance(first[-1], ReplyCompleted) and isinstance(second[-1], ReplyCompleted)
    assert len(transport.requests) == 4


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_an_internal_tool_error_surfaces_after_cleanup_and_is_not_a_result(
    tmp_path: Path, protocol: str,
) -> None:
    tool = BoomRead()
    registry = ToolRegistry()
    registry.register(tool)
    calls = (read_call("call_boom", "ok.txt"), read_call("call_later", "later.txt"))
    stream = ControlledSSE(batch_events(protocol, calls))
    transport = RecordingTransport([stream, ControlledSSE(reply_events(protocol))])
    received: list[StreamEvent] = []
    async with open_client(protocol, transport, registry) as client:
        reply = client.stream_reply("read")

        async def consume() -> None:
            async for event in reply:
                received.append(event)

        consuming = asyncio.create_task(consume())
        await asyncio.wait_for(tool.entered.wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(consuming), timeout=0.1)
        tool.release.set()
        with pytest.raises(TypeError, match="injected defect"):
            await asyncio.wait_for(consuming, timeout=2)
        assert tool.runs == 1
        assert not any(isinstance(event, ToolFinished) for event in received)
        _ = [event async for event in client.stream_reply("follow-up")]
    field = "messages" if protocol == "anthropic" else "input"
    assert json.loads(transport.requests[1].content)[field] == [{"role": "user", "content": "follow-up"}]
    assert stream.closed
    assert len(transport.requests) == 2
