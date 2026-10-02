import sqlite3
from pathlib import Path

import pytest

from agentoscopy.gateway.session import Usage
from agentoscopy.graders.base import GradeResult
from agentoscopy.spec import AgentConfig
from agentoscopy.stats.calibration import calibrate, cohen_kappa
from agentoscopy.storage.store import NewTrial, RunSettings, Store, StoreError
from agentoscopy.worker.trial import AttemptResult

CONFIG = AgentConfig(name="agent", adapter="python", entrypoint="pkg.mod:Agent")
SETTINGS = RunSettings(
    suite_id=None, suite_version=None, trials_per_task=1, budget_usd=None, seed=1,
    concurrency=1, harness_version="test", judge_model="judge-1",
)  # fmt: skip


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "agentoscopy.db")
    yield store
    store.close()


def judged(passed, confidence=0.9, model="judge-1"):
    metadata = {"confidence": confidence, "judge_model": model}
    return GradeResult("quality", 1.0 if passed else 0.0, passed, "why", metadata, "judge")


def command_grade(passed=True):
    return GradeResult("tests", 1.0 if passed else 0.0, passed, "ran", {}, "deterministic")


def finished(store, *grade_lists, outcome="pass"):
    """A run whose trials finished with the given grades; returns the trial ids."""
    trials = [NewTrial(f"task-{i}", 1, 0, i, 1.0) for i in range(len(grade_lists))]
    run_id = store.create_run(CONFIG, SETTINGS, trials)
    ids = []
    for grades in grade_lists:
        trial = store.claim_next(run_id, "w").trial
        result = AttemptResult(
            outcome=outcome, termination="agent_done", score=1.0, grades=list(grades),
            error_code=None, error=None, cleanup_errors=[], usage=Usage(), duration_s=1.0,
            trajectory_path=Path("t.jsonl"), final_state_path=None,
        )  # fmt: skip
        store.finish_attempt(trial, "w", result)
        ids.append(trial.trial_id)
    return run_id, ids


def review(store, review_id, passed, override=False):
    return store.submit_review(
        review_id, reviewer="me", passed=passed, score=None, note=None, override=override
    )


def test_an_item_is_queued_once_per_reason(store):
    _, (trial,) = finished(store, [command_grade(), judged(True)])

    first = store.queue_review(trial, 1, "quality", "random")
    assert first is not None
    assert store.queue_review(trial, 1, "quality", "random") is None
    assert store.queue_review(trial, 1, "quality", "manual") is not None
    assert store.pending_reviews() == 2


def test_the_queue_puts_unsure_judges_first_then_random_samples(store):
    _, ids = finished(store, *[[judged(True)] for _ in range(3)])
    store.queue_review(ids[0], 1, "", "manual")
    store.queue_review(ids[1], 1, "quality", "random")
    store.queue_review(ids[2], 1, "quality", "low_confidence")

    queue = store.review_queue(limit=10)

    assert [item["sample_source"] for item in queue] == ["low_confidence", "random", "manual"]
    assert queue[0]["task_id"] == "task-2" and queue[0]["task_version"] == 1
    assert len(store.review_queue(limit=1)) == 1


def test_a_review_is_recorded_once_and_leaves_the_queue(store):
    _, (trial,) = finished(store, [judged(True)])
    review_id = store.queue_review(trial, 1, "quality", "random")

    assert review(store, review_id, passed=False) is True
    assert review(store, review_id, passed=True) is False  # already reviewed
    assert store.pending_reviews() == 0
    stored = store.get_review(review_id)
    assert (stored["passed"], stored["reviewer"], stored["override"]) == (0, "me", 0)
    assert store.get_trial(trial)["outcome"] == "pass"  # no override, no change


def test_an_override_replaces_the_outcome_and_keeps_the_original(store):
    run_id, (trial,) = finished(store, [command_grade()], outcome="pass")
    store.save_summary(run_id, {"stale": True})
    first = store.queue_review(trial, 1, "", "random")
    second = store.queue_review(trial, 1, "", "manual")

    assert review(store, first, passed=False, override=True)
    assert store.get_summary(run_id) is None  # recomputed on the next read
    row = store.get_trial(trial)
    assert (row["outcome"], row["original_outcome"]) == ("fail", "pass")

    review(store, second, passed=True, override=True)
    row = store.get_trial(trial)
    assert (row["outcome"], row["original_outcome"]) == ("pass", "pass")  # first original kept


def test_calibration_rows_pair_human_and_judge_labels(store):
    _, ids = finished(store, [command_grade(), judged(True)], [command_grade(), judged(False)])
    for trial in ids:
        store.queue_review(trial, 1, "quality", "random")
        store.queue_review(trial, 1, "", "random")  # trial audits are not judge calibration
    for item in store.review_queue(limit=10):
        review(store, item["review_id"], passed=True)
    store.queue_review(ids[0], 1, "quality", "manual")  # pending: not counted

    rows = store.calibration_rows()

    assert [(row["human_passed"], row["judge_passed"]) for row in rows] == [
        (True, True),
        (True, False),
    ]
    assert {row["judge_model"] for row in rows} == {"judge-1"}
    assert {row["grader_name"] for row in rows} == {"quality"}


def test_a_database_from_another_schema_version_is_refused(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE runs (run_id TEXT)")
        conn.execute("PRAGMA user_version = 1")
    conn.close()

    with pytest.raises(StoreError, match="schema version 1"):
        Store(path)


def test_a_fresh_database_gets_the_current_schema_version(tmp_path):
    Store(tmp_path / "new.db").close()
    Store(tmp_path / "new.db").close()  # reopening the same version is fine


# Calibration statistics ------------------------------------------------------------------


def pairs(both_pass, human_only, judge_only, both_fail):
    return (
        [(True, True)] * both_pass
        + [(True, False)] * human_only
        + [(False, True)] * judge_only
        + [(False, False)] * both_fail
    )


def test_kappa_corrects_agreement_for_chance():
    # p_o = 0.7; p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.5; kappa = (0.7 - 0.5) / 0.5
    assert cohen_kappa(pairs(20, 5, 10, 15)) == pytest.approx(0.4)
    assert cohen_kappa(pairs(10, 0, 0, 10)) == pytest.approx(1.0)
    assert cohen_kappa(pairs(0, 10, 10, 0)) == pytest.approx(-1.0)


def test_kappa_is_undefined_without_variation():
    assert cohen_kappa([]) is None
    assert cohen_kappa(pairs(30, 0, 0, 0)) is None


def rows(labels, source="random", model="judge-1", grader="quality"):
    return [
        {
            "review_id": f"r{i}",
            "trial_id": f"t{i}",
            "task_id": "task",
            "grader_name": grader,
            "judge_model": model,
            "sample_source": source,
            "human_passed": human,
            "judge_passed": judge,
        }
        for i, (human, judge) in enumerate(labels)
    ]


def test_a_judge_is_calibrated_with_enough_random_reviews_and_high_kappa():
    (report,) = calibrate(rows(pairs(14, 1, 1, 14)))

    assert report["status"] == "calibrated"
    assert report["random_reviews"] == 30
    assert report["agreement"] == pytest.approx(28 / 30)
    assert report["confusion"] == {
        "human_pass_judge_pass": 14,
        "human_pass_judge_fail": 1,
        "human_fail_judge_pass": 1,
        "human_fail_judge_fail": 14,
    }
    assert len(report["disagreements"]) == 2


def test_low_kappa_over_enough_reviews_is_uncalibrated():
    (report,) = calibrate(rows(pairs(20, 5, 10, 15)))

    assert (report["status"], report["kappa"]) == ("uncalibrated", pytest.approx(0.4))


def test_too_few_random_reviews_is_insufficient_data():
    (report,) = calibrate(rows(pairs(14, 0, 0, 15)))

    assert (report["status"], report["random_reviews"]) == ("insufficient_data", 29)


def test_a_judge_that_always_agrees_on_one_label_is_undefined():
    (report,) = calibrate(rows(pairs(30, 0, 0, 0)))

    assert (report["status"], report["kappa"]) == ("undefined", None)


def test_enriched_reviews_show_as_disagreements_but_stay_out_of_the_statistics():
    hard_cases = rows(pairs(0, 5, 5, 0), source="low_confidence")
    for row in hard_cases:
        row["review_id"] = "hard-" + row["review_id"]

    (report,) = calibrate(rows(pairs(15, 0, 0, 15)) + hard_cases)

    assert (report["status"], report["kappa"]) == ("calibrated", pytest.approx(1.0))
    assert (report["random_reviews"], report["all_reviews"]) == (30, 40)
    assert {row["sample_source"] for row in report["disagreements"]} == {"low_confidence"}


def test_each_judge_model_is_calibrated_separately():
    reports = calibrate(rows(pairs(1, 0, 0, 1), model="a") + rows(pairs(1, 0, 0, 1), model="b"))

    assert [report["judge_model"] for report in reports] == ["a", "b"]
