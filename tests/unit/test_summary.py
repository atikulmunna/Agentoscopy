import pytest

from agentoscopy.stats.report import render
from agentoscopy.stats.summary import bootstrap_ci, pass_at_k, pass_hat_k, summarize


@pytest.mark.parametrize(
    ("n", "c", "k", "at_k", "hat_k"),
    [
        (3, 1, 3, 1.0, 0.0),
        (5, 2, 2, 0.7, 0.1),  # 1 - C(3,2)/C(5,2) and C(2,2)/C(5,2)
        (3, 3, 3, 1.0, 1.0),
        (3, 0, 1, 0.0, 0.0),
    ],
)
def test_pass_at_k_and_pass_hat_k(n, c, k, at_k, hat_k):
    assert pass_at_k(n, c, k) == pytest.approx(at_k)
    assert pass_hat_k(n, c, k) == pytest.approx(hat_k)


def test_pass_at_k_needs_enough_trials():
    assert pass_at_k(2, 1, 3) is None
    assert pass_hat_k(2, 1, 3) is None


def test_bootstrap_is_seeded_and_bounded():
    values = [0.0, 0.5, 1.0, 1.0, 0.25]

    low, high = bootstrap_ci(values, seed=7, resamples=2000)

    assert (low, high) == bootstrap_ci(values, seed=7, resamples=2000)
    assert 0.0 <= low <= sum(values) / len(values) <= high <= 1.0
    assert bootstrap_ci([0.5, 0.5], seed=1, resamples=100) == (0.5, 0.5)


def rows(task_id, *outcomes, cost=0.1):
    return [
        {
            "task_id": task_id,
            "trial_index": i,
            "outcome": o,
            "cost_usd": cost,
            "steps": 2,
            "duration_s": 1.0,
        }
        for i, o in enumerate(outcomes)
    ]


def test_summary_scores_only_pass_and_fail():
    trials = (
        rows("always", "pass", "pass", "pass")
        + rows("flaky", "pass", "fail", "infra_error")
        + rows("broken", "infra_error", "skipped", "cancelled")
    )

    summary = summarize(trials, trials_per_task=3, seed=1, spent_usd=0.9, resamples=500)

    by_task = {task["task_id"]: task for task in summary["tasks"]}
    assert by_task["always"]["pass_at_k"] == 1.0
    assert by_task["flaky"]["n"] == 2 and by_task["flaky"]["pass_at_k"] is None  # n < k
    assert summary["macro_pass_rate"] == pytest.approx(0.75)  # mean of 1.0 and 0.5
    assert summary["micro_pass_rate"] == pytest.approx(4 / 5)
    assert summary["flaky_tasks"] == ["flaky"]
    assert summary["unscored_tasks"] == ["broken"]
    assert summary["counts"] == {
        "fail": 1,
        "pass": 4,
        "infra_error": 2,
        "skipped": 1,
        "cancelled": 1,
    }
    assert summary["cost_per_pass_usd"] == pytest.approx(0.9 / 4)


def test_summary_of_nothing_scored_has_no_rates():
    summary = summarize(rows("a", "skipped"), trials_per_task=1, seed=1, spent_usd=0.0)

    assert summary["macro_pass_rate"] is None
    assert summary["macro_ci_95"] is None
    assert summary["cost_per_pass_usd"] is None


@pytest.mark.parametrize("output_format", ["table", "md", "json"])
def test_reports_render_in_every_format(output_format):
    summary = summarize(
        rows("task-a", "pass", "fail"), trials_per_task=2, seed=1, spent_usd=0.2, resamples=100
    )
    run = {
        "run_id": "r1",
        "suite_id": "smoke",
        "suite_version": 1,
        "config_name": "agent",
        "trials_per_task": 2,
        "seed": 1,
        "status": "completed",
        "flags": ["BUDGET_TRUNCATED"],
    }

    text = render(summary, run, output_format)

    assert "task-a" in text
    assert "BUDGET_TRUNCATED" in text
    if output_format != "json":
        assert "pass@2" in text and "50.0%" in text


def test_summary_slices_by_task_metadata():
    trials = rows("a", "pass", "pass") + rows("b", "fail", "fail") + rows("c", "pass", "fail")
    specs = {
        "a": {"category": "coding", "difficulty": "easy", "tags": ["py", "fast"]},
        "b": {"category": "coding", "tags": ["py"]},
        "c": {"category": "docs"},
    }

    summary = summarize(
        trials, trials_per_task=2, seed=1, spent_usd=0, task_specs=specs, resamples=50
    )

    assert summary["slices"]["category"] == [
        {"value": "coding", "tasks": 2, "mean": 0.5},
        {"value": "docs", "tasks": 1, "mean": 0.5},
    ]
    assert summary["slices"]["difficulty"] == [{"value": "easy", "tasks": 1, "mean": 1.0}]
    assert {item["value"]: item["mean"] for item in summary["slices"]["tag"]} == {
        "fast": 1.0,
        "py": 0.5,
    }


@pytest.mark.parametrize("output_format", ["table", "md"])
def test_reports_show_judge_spend_and_uncalibrated_judges(output_format):
    summary = summarize(rows("task-a", "pass"), trials_per_task=1, seed=1, spent_usd=0.1)
    summary |= {"judge_cost_usd": 0.0123, "uncalibrated_judges": ["task-a/quality"]}
    run = {
        "run_id": "r1", "suite_id": None, "config_name": "agent", "trials_per_task": 1,
        "seed": 1, "status": "completed", "flags": [],
    }  # fmt: skip

    text = render(summary, run, output_format)

    assert "judging $0.0123" in text
    assert "uncalibrated judges" in text and "task-a/quality" in text
