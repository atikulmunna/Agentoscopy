import json

import pytest

from agentoscopy.recorder.trajectory import INLINE_OUTPUT_LIMIT, TrajectoryRecorder, read_events


@pytest.fixture
def recorder(tmp_path):
    recorder = TrajectoryRecorder(tmp_path / "t" / "attempt-1.jsonl", tmp_path / "a", "trial-1", 1)
    yield recorder
    recorder.close()


def test_events_are_sequenced_and_flushed_immediately(recorder):
    recorder.write("trial_start", {"task_id": "demo"})
    recorder.write("agent_message", {"text": "hi"})

    # Read while the file is still open: every event must already be on disk.
    events = read_events(recorder.path)
    assert [event["seq"] for event in events] == [0, 1]
    assert [event["type"] for event in events] == ["trial_start", "agent_message"]
    assert events[0]["trial_id"] == "trial-1"
    assert events[0]["attempt"] == 1
    assert events[1]["payload"] == {"text": "hi"}
    assert events[0]["ts"].endswith("Z")


def test_adapter_recorder_only_allows_agent_events(recorder):
    adapter_recorder = recorder.for_adapter()
    adapter_recorder.event("agent_message", text="working")
    adapter_recorder.event("error", source="agent", message="oops")

    for forbidden in ("tool_call", "tool_result", "model_response", "trial_end"):
        with pytest.raises(ValueError, match="adapters may only write"):
            adapter_recorder.event(forbidden, tool="exec")

    assert [event["type"] for event in read_events(recorder.path)] == ["agent_message", "error"]


def test_short_blobs_stay_inline(recorder):
    fields = recorder.inline_blobs(stdout=b"ok", stderr=b"")

    assert fields == {"stdout": "ok", "stderr": "", "output_ref": None}


def test_long_blobs_are_truncated_and_stored_in_full(recorder):
    long_output = b"x" * (INLINE_OUTPUT_LIMIT + 10)
    fields = recorder.inline_blobs(stdout=long_output, stderr=b"err")

    assert fields["stdout"] == "x" * INLINE_OUTPUT_LIMIT + "\n[truncated]"
    stored = json.loads((recorder.artifacts_dir / fields["output_ref"]).read_text())
    assert stored == {"stdout": long_output.decode(), "stderr": "err"}
