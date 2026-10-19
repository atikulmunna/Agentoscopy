"""Tagging failed trials by hand (FR-FA-01)."""

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentoscopy.api.server import create_app
from agentoscopy.api.tags import TAXONOMY, normalize
from agentoscopy.gateway.session import Usage
from agentoscopy.reporting import run_summary
from agentoscopy.spec import AgentConfig
from agentoscopy.stats.report import render
from agentoscopy.storage.store import NewTrial, RunSettings, Store
from agentoscopy.worker.trial import AttemptResult

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CONFIG = AgentConfig(name="agent", adapter="python", entrypoint="pkg.mod:Agent")
SETTINGS = RunSettings(
    suite_id=None, suite_version=None, trials_per_task=1, budget_usd=None, seed=1,
    concurrency=1, harness_version="test",
)  # fmt: skip


@pytest.fixture
def world(tmp_path):
    """A finished run with one failed trial and one passed trial."""
    store = Store(tmp_path / "agentoscopy.db")
    run_id = store.create_run(
        CONFIG, SETTINGS, [NewTrial(f"task-{i}", 1, 0, i, 1.0) for i in range(2)]
    )
    ids = []
    for outcome in ("fail", "pass"):
        trial = store.claim_next(run_id, "w").trial
        result = AttemptResult(
            outcome=outcome, termination="agent_done", score=0.0, grades=[], error_code=None,
            error=None, cleanup_errors=[], usage=Usage(), duration_s=1.0,
            trajectory_path=Path("t.jsonl"), final_state_path=None,
        )  # fmt: skip
        store.finish_attempt(trial, "w", result)
        ids.append(trial.trial_id)
    store.set_run_status(run_id, "completed", finished=True)
    yield store, tmp_path, run_id, *ids
    store.close()


def call(world, method, path, body=None):
    store, home = world[0], world[1]

    async def scenario():
        async with TestClient(TestServer(create_app(store, home, TOKEN))) as client:
            response = await client.request(method, path, json=body, headers=AUTH)
            return response.status, await response.json()

    return asyncio.run(scenario())


def test_failed_trials_take_taxonomy_and_free_form_tags(world):
    store, _, _, failed, _ = world

    first, _ = call(
        world, "POST", f"/trials/{failed}/tags", {"tag": "wrong_fix", "note": "off by two"}
    )
    status, body = call(world, "POST", f"/trials/{failed}/tags", {"tag": "Flaky Network"})

    assert (first, status) == (201, 201)
    assert body["failure_tags"] == [
        {"tag": "flaky_network", "source": "human", "note": None},
        {"tag": "wrong_fix", "source": "human", "note": "off by two"},
    ]


def test_tagging_again_updates_the_note(world):
    failed = world[3]
    call(world, "POST", f"/trials/{failed}/tags", {"tag": "gave_up", "note": "first"})

    _, body = call(world, "POST", f"/trials/{failed}/tags", {"tag": "gave_up", "note": "second"})

    assert body["failure_tags"] == [{"tag": "gave_up", "source": "human", "note": "second"}]


def test_reviewers_remove_their_own_tags_but_not_the_harnesses(world):
    store, _, _, failed, _ = world
    store.add_failure_tag(failed, "reward_hacking", "auto", "tamper_check vetoed the trial")
    call(world, "POST", f"/trials/{failed}/tags", {"tag": "looped"})

    removed, body = call(world, "DELETE", f"/trials/{failed}/tags/looped")
    kept, error = call(world, "DELETE", f"/trials/{failed}/tags/reward_hacking")
    missing, _ = call(world, "DELETE", f"/trials/{failed}/tags/looped")

    assert removed == 200 and [t["tag"] for t in body["failure_tags"]] == ["reward_hacking"]
    assert (kept, error["error"]["code"]) == (409, "AUTOMATIC_TAG")
    assert missing == 404


def test_a_reviewers_tag_never_rewrites_an_automatic_one(world):
    store, _, _, failed, _ = world
    store.add_failure_tag(failed, "reward_hacking", "auto", "vetoed")

    call(world, "POST", f"/trials/{failed}/tags", {"tag": "reward_hacking", "note": "agreed"})

    assert store.failure_tag(failed, "reward_hacking")["source"] == "auto"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({}, 400, "BAD_REQUEST"),
        ({"tag": "x"}, 400, "BAD_REQUEST"),
        ({"tag": "9lives"}, 400, "BAD_REQUEST"),
        ({"tag": "a" * 41}, 400, "BAD_REQUEST"),
        ({"tag": "<script>"}, 400, "BAD_REQUEST"),
        ({"tag": "ok_tag", "note": "n" * 501}, 400, "BAD_REQUEST"),
    ],
)
def test_bad_tags_are_refused(world, body, status, code):
    got, error = call(world, "POST", f"/trials/{world[3]}/tags", body)

    assert (got, error["error"]["code"]) == (status, code)


def test_only_failed_trials_are_tagged(world):
    passed = world[4]

    status, error = call(world, "POST", f"/trials/{passed}/tags", {"tag": "wrong_fix"})
    unknown, _ = call(world, "POST", "/trials/no-such-trial/tags", {"tag": "wrong_fix"})

    assert (status, error["error"]["code"]) == (422, "NOT_FAILED")
    assert unknown == 404


def test_the_taxonomy_is_served(world):
    _, body = call(world, "GET", "/failure-tags")

    assert [item["tag"] for item in body["taxonomy"]] == list(TAXONOMY)
    assert all(normalize(tag) == tag for tag in TAXONOMY)


def test_run_summaries_count_tags(world):
    store, _, run_id, failed, _ = world
    call(world, "POST", f"/trials/{failed}/tags", {"tag": "wrong_fix"})
    store.add_failure_tag(failed, "reward_hacking", "auto")

    summary = run_summary(store, run_id)

    assert summary["failure_tags"] == {"reward_hacking": 1, "wrong_fix": 1}
    assert "failure tags: reward_hacking 1, wrong_fix 1" in render(
        summary, store.get_run(run_id), "table"
    )
