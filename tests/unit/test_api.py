"""REST API over real runs produced by the scheduler with a fake sandbox and the mock model."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from conftest import BASE_TASK, running_gateway

from agentoscopy.api.server import create_app
from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import pin_tasks, plan_trials
from agentoscopy.scheduler.runner import RunExecutor
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.storage.store import RunSettings, Store
from agentoscopy.testing.fake_sandbox import FakeBackend
from agentoscopy.testing.scripted_agent import ScriptedAgent

TOKEN = "test-token"
TASK_IDS = ["task-0", "task-1", "task-2", "task-3"]


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


def execute(store, home, pinned, name, script, labels):
    config = AgentConfig(
        name=name,
        adapter="python",
        entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent",
        params={"script": script},
    )
    settings = RunSettings(
        suite_id="api-suite", suite_version=1, trials_per_task=4, budget_usd=None, seed=1,
        concurrency=4, harness_version="test", labels=labels,
    )  # fmt: skip
    run_id = store.create_run(config, settings, plan_trials(pinned, 4, seed=1))

    async def scenario():
        async with running_gateway() as gateway:
            await RunExecutor(
                store, run_id, pinned, config, FakeBackend(grader=passes_when_fixed), gateway,
                home, concurrency=4, seed=1, adapter_factory=lambda _: ScriptedAgent(),
            ).execute()  # fmt: skip

    asyncio.run(scenario())
    store.save_summary(run_id, run_summary(store, run_id, refresh=True))
    return run_id


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    tasks_dir, home = root / "tasks", root / "home"
    store = Store(home / "agentoscopy.db")
    for index, task_id in enumerate(TASK_IDS):
        (tasks_dir / task_id).mkdir(parents=True)
        spec = {**BASE_TASK, "id": task_id, "category": "coding" if index < 2 else "docs"}
        (tasks_dir / task_id / "task.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
        store.save_task_version(load_task(tasks_dir, task_id), "fake-image", [], [])
    pinned = pin_tasks(store, tasks_dir, TASK_IDS)
    fix = [{"model_calls": 1}, {"write": "app.py", "content": "fixed"}]
    baseline = execute(store, home, pinned, "fixer", fix, {"branch": "main"})
    candidate = execute(store, home, pinned, "idler", [{"model_calls": 1}], {"branch": "feat"})
    yield SimpleNamespace(store=store, home=home, baseline=baseline, candidate=candidate)
    store.close()


def call(world, path, *, token=TOKEN, headers=None, read=None):
    async def scenario():
        app = create_app(world.store, world.home, TOKEN)
        async with TestClient(TestServer(app)) as client:
            sent = {"Authorization": f"Bearer {token}"} if token else {}
            response = await client.get(path, headers={**sent, **(headers or {})})
            if read:
                return response.status, await read(response)
            body = await response.text()
            try:
                return response.status, json.loads(body)
            except ValueError:
                return response.status, body

    return asyncio.run(scenario())


def test_ui_shell_is_public_but_the_api_needs_the_token(world):
    assert call(world, "/", token=None)[0] == 200
    status, script = call(world, "/ui/app.js", token=None)
    assert status == 200 and "textContent" in script

    for token in (None, "wrong"):
        status, body = call(world, "/runs", token=token)
        assert status == 401
        assert body["error"]["code"] == "UNAUTHORIZED"


def test_requests_for_foreign_hosts_are_refused(world):
    status, body = call(world, "/runs", headers={"Host": "attacker.example"})

    assert status == 403
    assert body["error"]["code"] == "HOST_NOT_ALLOWED"


def test_runs_are_listed_with_headline_numbers_and_filters(world):
    _, body = call(world, "/runs")
    rates = {run["run_id"]: run["macro_pass_rate"] for run in body["runs"]}

    assert rates == {world.baseline: 1.0, world.candidate: 0.0}
    assert all(run["finished_trials"] == run["total_trials"] == 16 for run in body["runs"])
    _, idlers = call(world, "/runs?config=idler")
    _, main = call(world, "/runs?label=branch=main")
    assert [run["run_id"] for run in idlers["runs"]] == [world.candidate]
    assert [run["run_id"] for run in main["runs"]] == [world.baseline]


def test_run_summary_trials_and_missing_runs(world):
    _, summary = call(world, f"/runs/{world.baseline}/summary")
    _, trials = call(world, f"/runs/{world.baseline}/trials")
    status, missing = call(world, "/runs/no-such-run")

    categories = {item["value"]: item["mean"] for item in summary["slices"]["category"]}
    assert categories == {"coding": 1.0, "docs": 1.0}
    assert len(trials["trials"]) == 16
    assert (status, missing["error"]["code"]) == (404, "NOT_FOUND")


def test_trial_detail_paginated_trajectory_and_diff(world):
    _, trials = call(world, f"/runs/{world.baseline}/trials")
    trial_id = trials["trials"][0]["trial_id"]

    _, trial = call(world, f"/trials/{trial_id}")
    _, page = call(world, f"/trials/{trial_id}/trajectory?offset=1&limit=2")
    _, diff = call(world, f"/trials/{trial_id}/diff")
    status, bad = call(world, f"/trials/{trial_id}/trajectory?offset=-1")

    assert trial["grades"][0]["grader_name"] == "tests" and trial["grades"][0]["passed"] is True
    assert [attempt["attempt"] for attempt in trial["attempts"]] == [1]
    assert page["total"] > 3 and len(page["events"]) == 2
    assert page["events"][0]["seq"] == 1
    assert diff["diff"] == "A /workspace/app.py"
    assert (status, bad["error"]["code"]) == (400, "BAD_REQUEST")


def test_compare_endpoint(world):
    _, result = call(world, f"/compare?base={world.baseline}&cand={world.candidate}")
    status, missing = call(world, f"/compare?base={world.baseline}")
    not_found, _ = call(world, f"/compare?base={world.baseline}&cand=nope")

    assert result["verdict"] == "REGRESSION"
    assert result["classes"] == {"regressed": 4}  # 4 of 4 to 0 of 4: p = 1/70
    assert result["config_diff"][0]["field"] == "name"
    assert (status, missing["error"]["code"]) == (400, "BAD_REQUEST")
    assert not_found == 404


def test_events_stream_reports_run_progress(world):
    async def first_event(response):
        lines = []
        while len(lines) < 2:
            lines.append((await response.content.readline()).decode().strip())
        return lines

    status, lines = call(world, f"/events?token={TOKEN}", token=None, read=first_event)

    assert status == 200
    assert lines[0] == "event: runs"
    snapshot = json.loads(lines[1].removeprefix("data: "))
    assert {run["run_id"] for run in snapshot} == {world.baseline, world.candidate}


def test_artifacts_outside_the_gym_home_are_refused(world, tmp_path):
    outside = tmp_path / "elsewhere.jsonl"
    outside.write_text("{}\n")
    _, trials = call(world, f"/runs/{world.candidate}/trials")
    trial_id = trials["trials"][0]["trial_id"]
    with sqlite3.connect(world.home / "agentoscopy.db") as conn:
        conn.execute(
            "UPDATE trial_attempts SET trajectory_uri = ? WHERE trial_id = ?",
            (str(outside), trial_id),
        )

    status, body = call(world, f"/trials/{trial_id}/trajectory")

    assert status == 404
    assert "outside the Agentoscopy home" in body["error"]["message"]
