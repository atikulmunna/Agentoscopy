import asyncio

import pytest

from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.sandbox.base import SandboxError
from agentoscopy.sandbox.recording import RecordingSandbox
from agentoscopy.testing.fake_sandbox import FakeSandbox, exit_with


@pytest.fixture
def recorder(tmp_path):
    recorder = TrajectoryRecorder(tmp_path / "attempt-1.jsonl", tmp_path / "artifacts", "t", 1)
    yield recorder
    recorder.close()


def payloads(recorder):
    return [(event["type"], event["payload"]) for event in read_events(recorder.path)]


def test_exec_records_call_and_result(recorder):
    sandbox = RecordingSandbox(FakeSandbox(handler=exit_with(3, b"out", b"err")), recorder)

    result = asyncio.run(sandbox.exec("make test", timeout_s=9))

    assert result.exit_code == 3
    (call_type, call), (result_type, outcome) = payloads(recorder)
    assert call_type == "tool_call"
    assert call == {"tool": "exec", "args": {"cmd": "make test", "timeout_s": 9}}
    assert result_type == "tool_result"
    assert outcome["exit_code"] == 3
    assert (outcome["stdout"], outcome["stderr"]) == ("out", "err")
    assert outcome["error"] is None


def test_read_failure_is_recorded_and_reraised(recorder):
    sandbox = RecordingSandbox(FakeSandbox(), recorder)

    with pytest.raises(SandboxError):
        asyncio.run(sandbox.read_file("missing.txt"))

    (_, call), (_, outcome) = payloads(recorder)
    assert call == {"tool": "read_file", "args": {"path": "missing.txt"}}
    assert outcome["exit_code"] is None
    assert "no such file" in outcome["error"]


def test_write_records_path_size_and_content(recorder):
    inner = FakeSandbox()
    sandbox = RecordingSandbox(inner, recorder)

    asyncio.run(sandbox.write_file("app.py", b"print(1)\n"))

    assert inner.files == {"app.py": b"print(1)\n"}
    (_, call), (_, outcome) = payloads(recorder)
    assert call["args"]["path"] == "app.py"
    assert call["args"]["bytes"] == 9
    assert call["args"]["content"] == "print(1)\n"
    assert outcome["exit_code"] == 0
