"""The review queue and calibration endpoints, over a run graded by the mock judge."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from conftest import BASE_TASK, running_gateway

from agentoscopy.api.server import create_app
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import pin_tasks, plan_trials
from agentoscopy.scheduler.runner import RunExecutor
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.stats.calibration import MIN_REVIEWS
from agentoscopy.storage.store import RunSettings, Store
from agentoscopy.testing.fake_sandbox import FakeBackend
from agentoscopy.testing.scripted_agent import ScriptedAgent

TOKEN = "test-token"
TRIALS = MIN_REVIEWS + 2
JUDGE = {
    "name": "quality",
    "type": "llm_judge",
    "rubric": "Pass if the fix is minimal.",
    "required": False,  # so every trial passes, whatever the mock judge says
}
TASK = {**BASE_TASK, "id": "judged-task", "graders": [*BASE_TASK["graders"], JUDGE]}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A finished run of TRIALS trials, each graded by the mock judge and queued for review."""
    root = tmp_path_factory.mktemp("reviews")
    tasks_dir, home = root / "tasks", root / "home"
    (tasks_dir / TASK["id"]).mkdir(parents=True)
    (tasks_dir / TASK["id"] / "task.yaml").write_text(yaml.safe_dump(TASK), encoding="utf-8")
    store = Store(home / "agentoscopy.db")
    store.save_task_version(load_task(tasks_dir, TASK["id"]), "fake-image", [], [])
    pinned = pin_tasks(store, tasks_dir, [TASK["id"]])
    config = AgentConfig(
        name="fixer",
        adapter="python",
        entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent",
        params={"script": [{"write": "app.py", "content": "fixed"}], "final_message": "Fixed."},
    )
    settings = RunSettings(
        suite_id=None, suite_version=None, trials_per_task=TRIALS, budget_usd=None, seed=1,
        concurrency=4, harness_version="test", judge_model="mock",
    )  # fmt: skip
    run_id = store.create_run(config, settings, plan_trials(pinned, TRIALS, seed=1))

    async def scenario():
        async with running_gateway() as gateway:
            await RunExecutor(
                store, run_id, pinned, config, FakeBackend(grader=passes_when_fixed), gateway,
                home, concurrency=4, seed=1, adapter_factory=lambda _: ScriptedAgent(),
                judge_model="mock", review_rate=1.0,
            ).execute()  # fmt: skip

    asyncio.run(scenario())
    store.close()
    return SimpleNamespace(home=home, run_id=run_id)


@pytest.fixture
def world(template, tmp_path):
    """The template run with a database of its own, so each test can submit reviews."""
    database = tmp_path / "agentoscopy.db"
    with sqlite3.connect(template.home / "agentoscopy.db") as source:
        with sqlite3.connect(database) as copy:
            source.backup(copy)
    copy.close()
    source.close()
    store = Store(database)
    yield SimpleNamespace(store=store, home=template.home, run_id=template.run_id)
    store.close()


def call(world, method, path, body=None, *, token=TOKEN, raw=None):
    async def scenario():
        async with TestClient(TestServer(create_app(world.store, world.home, TOKEN))) as client:
            headers = {"Authorization": f"Bearer {token}"}
            data = raw if raw is not None else None if body is None else json.dumps(body)
            response = await client.request(method, path, headers=headers, data=data)
            return response.status, await response.json()

    return asyncio.run(scenario())


def random_items(world):
    _, queue = call(world, "GET", "/review/queue?limit=100")
    return [item for item in queue["items"] if item["sample_source"] == "random"]


def judge_passed(world, item):
    return world.store.grade_for(item["trial_id"], item["attempt"], item["grader_name"])["passed"]


def test_queue_items_show_the_work_but_not_the_verdict(world):
    status, queue = call(world, "GET", "/review/queue?limit=5")

    assert status == 200
    assert queue["pending"] > TRIALS  # a random sample per trial, plus unsure verdicts
    sources = [item["sample_source"] for item in queue["items"]]
    assert sources == sorted(sources, key=["low_confidence", "random", "manual"].index)
    item = queue["items"][0]
    assert item["grader_name"] == "quality" and item["rubric"] == JUDGE["rubric"]
    assert item["instructions"] == BASE_TASK["instructions"]
    assert item["final_message"] == "Fixed."
    assert item["changed_files"] == ["A /workspace/app.py"]
    assert any("app.py" in line for line in item["actions"])
    hidden = {"passed", "score", "judge", "outcome", "rationale", "confidence"}
    assert not hidden & set(item)


def test_submitting_reveals_the_verdict_once(world):
    (item, *_) = random_items(world)

    status, revealed = call(world, "POST", f"/review/{item['review_id']}", {"passed": True})
    again, error = call(world, "POST", f"/review/{item['review_id']}", {"passed": False})

    assert status == 200
    assert revealed["judge"]["passed"] is judge_passed(world, item)
    assert revealed["judge"]["metadata"]["judge_model"] == "mock"
    assert (revealed["outcome"], revealed["original_outcome"]) == ("pass", None)
    assert (again, error["error"]["code"]) == (409, "ALREADY_REVIEWED")


def test_an_override_changes_the_trial_outcome(world):
    (item, *_) = random_items(world)

    _, revealed = call(
        world, "POST", f"/review/{item['review_id']}", {"passed": False, "override": True}
    )
    _, trial = call(world, "GET", f"/trials/{item['trial_id']}")

    assert (revealed["outcome"], revealed["original_outcome"]) == ("fail", "pass")
    assert (trial["outcome"], trial["original_outcome"]) == ("fail", "pass")


@pytest.mark.parametrize(
    ("body", "raw"),
    [
        ({}, None),
        ({"passed": "yes"}, None),
        ({"passed": True, "score": 1.5}, None),
        ({"passed": True, "score": True}, None),
        ({"passed": True, "override": 1}, None),
        ({"passed": True, "note": "x" * 2001}, None),
        ({"passed": True, "reviewer": 7}, None),
        (None, "not json"),
        (None, "[1, 2]"),
    ],
)
def test_malformed_reviews_are_rejected(world, body, raw):
    (item, *_) = random_items(world)

    status, error = call(world, "POST", f"/review/{item['review_id']}", body, raw=raw)

    assert (status, error["error"]["code"]) == (400, "BAD_REQUEST")
    assert world.store.get_review(item["review_id"])["reviewed_at"] is None


def test_unknown_reviews_and_missing_tokens_are_refused(world):
    status, error = call(world, "POST", "/review/no-such-review", {"passed": True})
    unauthorized, _ = call(world, "POST", "/review/queue", {"trial_id": "x"}, token="wrong")

    assert (status, error["error"]["code"]) == (404, "NOT_FOUND")
    assert unauthorized == 401


def test_trials_can_be_queued_by_hand(world):
    trial_id = world.store.trial_rows(world.run_id)[0]["trial_id"]

    audit, created = call(world, "POST", "/review/queue", {"trial_id": trial_id})
    judged, _ = call(
        world, "POST", "/review/queue", {"trial_id": trial_id, "grader_name": "quality"}
    )
    duplicate, dup = call(world, "POST", "/review/queue", {"trial_id": trial_id})

    assert (audit, judged) == (201, 201)
    assert world.store.get_review(created["review_id"])["sample_source"] == "manual"
    assert (duplicate, dup["error"]["code"]) == (409, "ALREADY_QUEUED")


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({}, 400, "BAD_REQUEST"),
        ({"trial_id": "no-such-trial"}, 404, "NOT_FOUND"),
        ({"trial_id": "TRIAL", "grader_name": "tests"}, 422, "NOT_A_JUDGE_GRADER"),
        ({"trial_id": "TRIAL", "grader_name": "nope"}, 422, "NOT_A_JUDGE_GRADER"),
    ],
)
def test_bad_manual_queue_requests_are_refused(world, body, status, code):
    trial_id = world.store.trial_rows(world.run_id)[0]["trial_id"]
    body = {key: trial_id if value == "TRIAL" else value for key, value in body.items()}

    got, error = call(world, "POST", "/review/queue", body)

    assert (got, error["error"]["code"]) == (status, code)


def test_calibration_needs_enough_random_reviews(world):
    items = random_items(world)
    for item in items[: MIN_REVIEWS - 1]:
        call(world, "POST", f"/review/{item['review_id']}", {"passed": judge_passed(world, item)})

    _, report = call(world, "GET", "/calibration")

    (judge,) = report["judges"]
    assert (report["min_reviews"], report["kappa_threshold"]) == (MIN_REVIEWS, 0.6)
    assert (judge["status"], judge["random_reviews"]) == ("insufficient_data", MIN_REVIEWS - 1)


def test_kappa_over_thirty_random_reviews_calibrates_the_judge(world):
    """The M3 exit criterion: kappa over at least 30 randomly sampled human reviews."""
    items = random_items(world)
    assert len(items) >= MIN_REVIEWS
    for item in items:
        call(world, "POST", f"/review/{item['review_id']}", {"passed": judge_passed(world, item)})

    _, report = call(world, "GET", "/calibration")
    _, summary = call(world, "GET", f"/runs/{world.run_id}/summary")

    (judge,) = report["judges"]
    assert judge["random_reviews"] == len(items)
    assert (judge["status"], judge["kappa"], judge["agreement"]) == ("calibrated", 1.0, 1.0)
    assert summary["uncalibrated_judges"] == [] and summary["judge_cost_usd"] > 0


def test_an_uncalibrated_judge_is_flagged_on_the_run(world):
    for item in random_items(world):
        disagree = {"passed": not judge_passed(world, item)}
        call(world, "POST", f"/review/{item['review_id']}", disagree)

    _, report = call(world, "GET", "/calibration")
    _, summary = call(world, "GET", f"/runs/{world.run_id}/summary")

    (judge,) = report["judges"]
    assert judge["status"] == "uncalibrated" and judge["kappa"] < 0
    assert len(judge["disagreements"]) == judge["all_reviews"]
    assert summary["uncalibrated_judges"] == [f"{TASK['id']}/quality"]
