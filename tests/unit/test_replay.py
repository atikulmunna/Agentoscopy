"""Replay (WF-11): rerunning a trial against the model calls it recorded."""

import asyncio
import json

import anthropic
import pytest
import yaml
from conftest import BASE_TASK, running_gateway

from agentoscopy.cli.main import main
from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.replay import RecordedCall, load_recording, request_hash
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import Budget, load_task
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend
from agentoscopy.worker.trial import attempt_paths

FIX = {"write": "app.py", "content": "fixed"}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture
def project(tmp_path, write_task, tasks_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "agentoscopy.cli.run.DockerBackend", lambda: FakeBackend(grader=passes_when_fixed)
    )
    home = tmp_path / "home"
    store = Store(home / "agentoscopy.db")
    write_task({**BASE_TASK, "id": "task-a"})
    store.save_task_version(load_task(tasks_dir, "task-a"), "fake-image", [], [])
    yield store, home, tasks_dir, tmp_path
    store.close()


def cli(project, *args):
    _, home, tasks_dir, _ = project
    return main([*args, "--home", str(home), "--tasks-dir", str(tasks_dir)])


def recorded_trial(project, script):
    """Run one trial of a scripted agent; returns its trial row."""
    store, _, _, root = project
    agent = root / "agent.yaml"
    agent.write_text(
        yaml.safe_dump(
            {
                "name": "scripted",
                "adapter": "python",
                "entrypoint": "agentoscopy.testing.scripted_agent:ScriptedAgent",
                "params": {"script": script},
            }
        )
    )
    assert cli(project, "run", "--task", "task-a", "--agent", str(agent), "--trials", "1") == 0
    (trial,) = store.trial_rows(store.list_runs()[0]["run_id"])
    return trial


def source_paths(project, trial):
    return attempt_paths(project[1], trial["run_id"], trial["trial_id"], trial["attempt"])


def test_a_replay_reproduces_the_trial_without_new_spend(project, capsys):
    store = project[0]
    source = recorded_trial(project, [{"model_calls": 2}, FIX, {"model_calls": 1, "stream": True}])
    capsys.readouterr()

    assert cli(project, "replay", source["trial_id"]) == 0

    out = capsys.readouterr().out
    assert "replaying 3 recorded model calls" in out
    assert "reproduced: same outcome from the same model calls" in out
    replay_run = store.list_runs()[0]
    (replayed,) = store.trial_rows(replay_run["run_id"])
    assert (replay_run["mode"], replay_run["labels"]) == (
        "replay",
        {"replay_of": source["trial_id"]},
    )
    assert replayed["source_trial_id"] == source["trial_id"]
    assert (replayed["outcome"], replayed["steps"]) == ("pass", 3)
    assert replayed["cost_usd"] == 0 and source["cost_usd"] > 0

    def answers(run_id, trial_id, attempt):
        trajectory, _ = attempt_paths(project[1], run_id, trial_id, attempt)
        events = read_events(trajectory)
        return [e["payload"] for e in events if e["type"] == "model_response"]

    original = answers(source["run_id"], source["trial_id"], source["attempt"])
    again = answers(replay_run["run_id"], replayed["trial_id"], 1)
    assert [a["content"] for a in again] == [a["content"] for a in original]
    assert all(a["replayed"] and a["cost_usd"] == 0 for a in again)


def test_a_changed_request_stops_the_replay(project, capsys):
    store = project[0]
    source = recorded_trial(project, [{"model_calls": 2}, FIX])
    _, artifacts = source_paths(project, source)
    request = json.loads((artifacts / "requests/2.json").read_text())
    request["messages"][0]["content"] = "something the agent never sent"
    (artifacts / "requests/2.json").write_text(json.dumps(request))

    assert cli(project, "replay", source["trial_id"]) == 1

    out = capsys.readouterr().out
    assert "FAIL REPLAY_DIVERGED at step 2: the request differs from the one recorded" in out
    (replayed,) = store.trial_rows(store.list_runs()[0]["run_id"])
    assert (replayed["outcome"], replayed["error_code"]) == ("infra_error", "REPLAY_DIVERGED")


def test_a_call_the_source_never_made_stops_the_replay(project, capsys):
    source = recorded_trial(project, [{"model_calls": 2}, FIX])
    trajectory, _ = source_paths(project, source)
    events = [e for e in read_events(trajectory) if e.get("payload", {}).get("call_index") != 2]
    trajectory.write_text("".join(json.dumps(e) + "\n" for e in events))

    assert cli(project, "replay", source["trial_id"]) == 1
    assert "at step 2: the source trial made no model call at this step" in capsys.readouterr().out


def test_trials_without_a_recording_cannot_be_replayed(project, capsys):
    source = recorded_trial(project, [FIX])  # no model calls

    assert cli(project, "replay", source["trial_id"]) == 2
    assert cli(project, "replay", "no-such-trial") == 2
    err = capsys.readouterr().err
    assert "NO_RECORDING" in err and "NOT_FOUND: no trial no-such-trial" in err


def test_a_replay_run_is_not_resumed(project, capsys):
    store = project[0]
    source = recorded_trial(project, [{"model_calls": 1}, FIX])
    cli(project, "replay", source["trial_id"])
    replay_run = store.list_runs()[0]["run_id"]
    store.set_run_status(replay_run, "running")  # as if its process had crashed
    capsys.readouterr()

    assert cli(project, "run", "--resume", replay_run) == 2
    assert "REPLAY_NOT_RESUMABLE" in capsys.readouterr().err


def test_a_recorded_error_is_answered_again(tmp_path):
    body = {"model": "mock", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]}
    error = json.dumps(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "too long"}}
    )
    recording = [RecordedCall(request_hash(body), 400, error)]

    async def scenario():
        async with running_gateway() as gateway:
            recorder = TrajectoryRecorder(tmp_path / "t.jsonl", tmp_path / "a", "t", 1)
            session = gateway.open_session("t", Budget(), recorder, "s", replay=recording)
            client = anthropic.AsyncAnthropic(
                base_url=session.endpoint.base_url, api_key=session.endpoint.token
            )
            try:
                await client.messages.create(**body)
            except anthropic.BadRequestError as exc:
                return exc.status_code, exc.body
            finally:
                await client.close()
                recorder.close()

    status, answer = asyncio.run(scenario())

    assert status == 400 and answer["error"]["message"] == "too long"


def test_load_recording_keeps_the_agents_own_max_tokens(project):
    source = recorded_trial(project, [{"model_calls": 1, "max_tokens": 777}, FIX])

    (call,) = load_recording(*source_paths(project, source))

    _, artifacts = source_paths(project, source)
    sent = json.loads((artifacts / "requests/1.json").read_text())
    assert call.request_hash == request_hash({**sent, "max_tokens": 777})
    assert call.status == 200 and call.response["role"] == "assistant"
