"""Fork (WF-11, FR-RPL-03): replay a trial up to a step, then continue live."""

import asyncio
import json

import anthropic
import pytest
import yaml
from conftest import BASE_TASK, running_gateway

from agentoscopy.cli.main import main
from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.replay import Playback, RecordedCall, missing_tools, with_injection
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
    store = Store(tmp_path / "home" / "agentoscopy.db")
    write_task({**BASE_TASK, "id": "task-a"})
    store.save_task_version(load_task(tasks_dir, "task-a"), "fake-image", [], [])
    yield store, tmp_path
    store.close()


def config(project, name, script):
    path = project[1] / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "adapter": "python",
                "entrypoint": "agentoscopy.testing.scripted_agent:ScriptedAgent",
                "params": {"script": script},
            }
        )
    )
    return path


def cli(project, *args):
    _, root = project
    return main([*args, "--home", str(root / "home"), "--tasks-dir", str(root / "tasks")])


def source_trial(project, script):
    store = project[0]
    agent = config(project, "original", script)
    assert cli(project, "run", "--task", "task-a", "--agent", str(agent), "--trials", "1") == 0
    (trial,) = store.trial_rows(store.list_runs()[0]["run_id"])
    return trial


def newest_trial(project):
    store = project[0]
    run = store.list_runs()[0]
    (trial,) = store.trial_rows(run["run_id"])
    return run, trial


def responses(project, run, trial):
    trajectory, _ = attempt_paths(project[1] / "home", run["run_id"], trial["trial_id"], 1)
    return [e["payload"] for e in read_events(trajectory) if e["type"] == "model_response"]


def test_a_fork_replays_up_to_its_step_and_then_goes_live(project, capsys):
    source = source_trial(project, [{"model_calls": 3}, FIX])
    capsys.readouterr()

    assert cli(project, "fork", source["trial_id"], "--at-step", "2") == 0

    out = capsys.readouterr().out
    assert "replaying 1 recorded model calls, then live" in out
    assert f"#/side/{source['trial_id']}/" in out
    run, fork = newest_trial(project)
    assert run["mode"] == "fork"
    assert run["labels"] == {"fork_of": source["trial_id"], "fork_at": "2"}
    assert fork["source_trial_id"] == source["trial_id"]
    answers = responses(project, run, fork)
    assert [bool(a.get("replayed")) for a in answers] == [True, False, False]
    assert answers[0]["cost_usd"] == 0 and answers[1]["cost_usd"] > 0


def test_a_fork_can_continue_with_another_config(project, capsys):
    source = source_trial(project, [{"model_calls": 2}, FIX])
    idler = config(project, "idler", [{"model_calls": 2}])  # never writes the fix

    assert cli(project, "fork", source["trial_id"], "--at-step", "2", "--agent", str(idler)) == 0

    run, fork = newest_trial(project)
    assert (source["outcome"], fork["outcome"]) == ("pass", "fail")
    assert "fork" in capsys.readouterr().out


def test_an_injected_message_joins_the_request_at_the_fork_step(project):
    source = source_trial(project, [{"model_calls": 2}, FIX])

    assert (
        cli(project, "fork", source["trial_id"], "--at-step", "2", "--inject", "Try harder.") == 0
    )

    run, fork = newest_trial(project)
    _, artifacts = attempt_paths(project[1] / "home", run["run_id"], fork["trial_id"], 1)
    sent = json.loads((artifacts / "requests/2.json").read_text())
    assert sent["messages"][-1]["content"][-1] == {"type": "text", "text": "Try harder."}
    first = json.loads((artifacts / "requests/1.json").read_text())
    assert "Try harder." not in json.dumps(first)
    assert run["labels"]["injected"] == "yes"


def test_the_fork_step_must_be_within_the_recording(project, capsys):
    source = source_trial(project, [{"model_calls": 2}, FIX])

    assert cli(project, "fork", source["trial_id"], "--at-step", "4") == 2
    assert "FORK_STEP_OUT_OF_RANGE: trial" in capsys.readouterr().err
    assert (
        cli(project, "fork", source["trial_id"], "--at-step", "3") == 0
    )  # all replayed, then live
    with pytest.raises(SystemExit):
        cli(project, "fork", source["trial_id"], "--at-step", "0")


def test_a_fork_run_is_not_resumed(project, capsys):
    store = project[0]
    source = source_trial(project, [{"model_calls": 2}, FIX])
    cli(project, "fork", source["trial_id"], "--at-step", "2")
    run, _ = newest_trial(project)
    store.set_run_status(run["run_id"], "running")
    capsys.readouterr()

    assert cli(project, "run", "--resume", run["run_id"]) == 2
    assert "a fork run is not resumed" in capsys.readouterr().err


def test_a_recorded_tool_call_the_fork_cannot_make_stops_it(tmp_path):
    message = {
        "id": "m", "type": "message", "role": "assistant", "model": "mock",
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}],
    }  # fmt: skip
    playback = Playback((RecordedCall("any-hash", 200, message),), fork_at=2)
    request = {
        "model": "mock", "max_tokens": 32, "messages": [{"role": "user", "content": "go"}],
        "tools": [{"name": "write_file", "input_schema": {"type": "object"}}],
    }  # fmt: skip

    async def scenario():
        async with running_gateway() as gateway:
            recorder = TrajectoryRecorder(tmp_path / "t.jsonl", tmp_path / "a", "t", 1)
            session = gateway.open_session("t", Budget(), recorder, "s", playback=playback)
            client = anthropic.AsyncAnthropic(
                base_url=session.endpoint.base_url, api_key=session.endpoint.token
            )
            try:
                await client.messages.create(**request)
            except anthropic.BadRequestError as exc:
                return exc.body, session.replay_divergence
            finally:
                await client.close()
                recorder.close()

    body, divergence = asyncio.run(scenario())

    assert body["error"]["type"] == "replay_diverged_error"
    assert divergence == "step 1: the recorded answer calls bash, which this config lacks"


def test_missing_tools_and_injection_helpers():
    answer = {"content": [{"type": "text"}, {"type": "tool_use", "name": "bash"}]}
    assert missing_tools(answer, {"tools": [{"name": "bash"}]}) == []
    assert missing_tools(answer, {}) == ["bash"]

    text = {"type": "text", "text": "hint"}
    as_string = {"messages": [{"role": "user", "content": "do it"}]}
    assert with_injection(as_string, "hint")["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "do it"}, text]}
    ]
    after_assistant = {"messages": [{"role": "assistant", "content": "ok"}]}
    assert with_injection(after_assistant, "hint")["messages"][-1] == {
        "role": "user",
        "content": [text],
    }
    assert as_string["messages"][0]["content"] == "do it"  # the original request is left alone
