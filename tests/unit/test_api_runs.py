"""Starting and cancelling runs over HTTP, and the Prometheus metrics endpoint."""

import asyncio
import re
from types import SimpleNamespace

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from conftest import BASE_TASK

from agentoscopy.api.runs import RunService
from agentoscopy.api.server import create_app
from agentoscopy.launch import Directories
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import load_task
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FIX = {"write": "app.py", "content": "fixed"}
AGENTS = {"fixer": [{"model_calls": 1}, FIX], "sleeper": [{"sleep": 30}, FIX]}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture
def world(tmp_path, write_task, tasks_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    home, configs = tmp_path / "home", tmp_path / "configs"
    store = Store(home / "agentoscopy.db")
    write_task({**BASE_TASK, "id": "task-a"})
    store.save_task_version(load_task(tasks_dir, "task-a"), "fake-image", [], [])
    write_task({**BASE_TASK, "id": "not-validated"})
    configs.mkdir()
    for name, script in AGENTS.items():
        config = {
            "name": name,
            "adapter": "python",
            "entrypoint": "agentoscopy.testing.scripted_agent:ScriptedAgent",
            "params": {"script": script},
        }
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(config))
    yield SimpleNamespace(store=store, home=home, tasks_dir=tasks_dir, configs=configs)
    store.close()


def serve(world, scenario):
    """Run `scenario(client)` against an app that executes runs with fake sandboxes."""
    service = RunService(
        Directories(world.tasks_dir, world.tasks_dir, world.home),
        world.configs,
        FakeBackend(grader=passes_when_fixed),
    )

    async def main():
        app = create_app(world.store, world.home, TOKEN, runs=service)
        async with TestClient(TestServer(app)) as client:
            return await scenario(client)

    return asyncio.run(main())


async def start(client, **body):
    response = await client.post(
        "/runs", json={"agent": "fixer", "tasks": ["task-a"], **body}, headers=AUTH
    )
    return response.status, await response.json()


async def wait_for(client, run_id, statuses, timeout_s=30):
    for _ in range(int(timeout_s / 0.05)):
        run = await (await client.get(f"/runs/{run_id}", headers=AUTH)).json()
        if run["status"] in statuses:
            return run
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {statuses}")


def test_a_run_started_over_http_executes_in_the_server(world):
    async def scenario(client):
        status, created = await start(client, trials=3, labels={"source": "api"}, seed=4)
        run = await wait_for(client, created["run_id"], {"completed"})
        trials = await (await client.get(f"/runs/{created['run_id']}/trials", headers=AUTH)).json()
        return status, created, run, trials["trials"]

    status, created, run, trials = serve(world, scenario)

    assert (status, created["status"]) == (201, "pending")
    assert (run["labels"], run["seed"], run["total_trials"]) == ({"source": "api"}, 4, 3)
    assert {trial["outcome"] for trial in trials} == {"pass"}
    assert world.store.get_summary(created["run_id"])["counts"]["pass"] == 3


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"agent": None}, 400, "BAD_REQUEST"),
        ({"agent": "../fixer"}, 400, "BAD_REQUEST"),
        ({"agent": "fixer.yaml"}, 400, "BAD_REQUEST"),
        ({"agent": "nope"}, 422, "UNKNOWN_AGENT"),
        ({"suite": "smoke"}, 400, "BAD_REQUEST"),  # a suite and tasks
        ({"tasks": None}, 400, "BAD_REQUEST"),  # neither
        ({"tasks": ["../task-a"]}, 400, "BAD_REQUEST"),
        ({"trials": 0}, 400, "BAD_REQUEST"),
        ({"trials": "3"}, 400, "BAD_REQUEST"),
        ({"trials": True}, 400, "BAD_REQUEST"),
        ({"concurrency": 1000}, 400, "BAD_REQUEST"),
        ({"budget_usd": -1}, 400, "BAD_REQUEST"),
        ({"seed": 1.5}, 400, "BAD_REQUEST"),
        ({"review_rate": 2}, 400, "BAD_REQUEST"),
        ({"labels": {"k": 1}}, 400, "BAD_REQUEST"),
        ({"entrypoint": "os:system"}, 400, "BAD_REQUEST"),
        ({"tasks": ["not-validated"]}, 422, "UNVALIDATED_TASKS"),
        ({"budget_usd": 0.01}, 422, "BUDGET_BELOW_TRIAL_CAP"),
    ],
)
def test_invalid_run_requests_are_refused_before_anything_runs(world, body, status, code):
    async def scenario(client):
        return await start(client, **body)

    got, error = serve(world, scenario)

    assert (got, error["error"]["code"]) == (status, code)
    assert world.store.list_runs() == []


def test_run_requests_need_the_token(world):
    async def scenario(client):
        response = await client.post("/runs", json={"agent": "fixer", "tasks": ["task-a"]})
        return response.status

    assert serve(world, scenario) == 401


def test_a_running_run_is_cancelled_over_http(world):
    async def scenario(client):
        _, created = await start(client, agent="sleeper", trials=4, concurrency=2)
        run_id = created["run_id"]
        while world.store.count_in_flight(run_id) < 2:
            await asyncio.sleep(0.05)
        response = await client.delete(f"/runs/{run_id}", headers=AUTH)
        accepted = response.status, await response.json()
        run = await wait_for(client, run_id, {"cancelled"})
        again = await client.delete(f"/runs/{run_id}", headers=AUTH)
        return accepted, run, again.status

    (status, body), run, again = serve(world, scenario)

    assert (status, body["status"]) == (202, "cancelling")
    assert run["status"] == "cancelled" and run["finished_trials"] == 4
    assert again == 200  # cancelling a cancelled run changes nothing
    outcomes = {row["outcome"] for row in world.store.trial_rows(run["run_id"])}
    assert outcomes == {"cancelled"}


def test_finished_or_unknown_runs_cannot_be_cancelled(world):
    async def scenario(client):
        _, created = await start(client)
        await wait_for(client, created["run_id"], {"completed"})
        finished = await client.delete(f"/runs/{created['run_id']}", headers=AUTH)
        unknown = await client.delete("/runs/no-such-run", headers=AUTH)
        return (finished.status, await finished.json()), unknown.status

    (status, body), unknown = serve(world, scenario)

    assert (status, body["error"]["code"]) == (409, "RUN_FINISHED")
    assert unknown == 404


def test_server_shutdown_cancels_the_runs_it_was_executing(world):
    async def scenario(client):
        _, created = await start(client, agent="sleeper", trials=2)
        while world.store.count_in_flight(created["run_id"]) < 1:
            await asyncio.sleep(0.05)
        return created["run_id"]

    run_id = serve(world, scenario)

    assert world.store.run_status(run_id) == "cancelled"


def metric(text, line_start):
    match = re.search(rf"^{re.escape(line_start)} (\S+)$", text, re.MULTILINE)
    assert match, f"no {line_start} in metrics"
    return float(match.group(1))


def test_metrics_cover_queue_trials_errors_latency_and_spend(world):
    async def scenario(client):
        _, created = await start(client, trials=3)
        await wait_for(client, created["run_id"], {"completed"})
        response = await client.get("/metrics", headers=AUTH)
        unauthorized = await client.get("/metrics")
        return (
            response.status,
            response.headers["Content-Type"],
            await response.text(),
            unauthorized.status,
        )

    status, content_type, text, unauthorized = serve(world, scenario)

    assert (status, unauthorized) == (200, 401)
    assert content_type.startswith("text/plain; version=0.0.4")
    assert metric(text, "agentoscopy_queue_depth") == 0
    assert metric(text, 'agentoscopy_active_trials{state="RUNNING"}') == 0
    assert metric(text, 'agentoscopy_trials_total{outcome="pass"}') == 3
    assert metric(text, 'agentoscopy_attempts_total{outcome="pass"}') == 3
    assert metric(text, 'agentoscopy_runs{status="completed"}') == 1
    assert metric(text, 'agentoscopy_trial_duration_seconds_bucket{le="+Inf"}') == 3
    assert metric(text, "agentoscopy_trial_duration_seconds_count") == 3
    assert metric(text, "agentoscopy_gateway_latency_seconds_count") == 3  # one call per trial
    assert metric(text, 'agentoscopy_spend_usd_total{kind="agent"}') > 0
    assert "# TYPE agentoscopy_infra_errors_total counter" in text
    buckets = [float(v) for v in re.findall(r'_bucket\{le="[0-9.]+"\} (\S+)', text)]
    assert buckets == sorted(buckets)  # cumulative
