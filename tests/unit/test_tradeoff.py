"""Cost against quality across runs (agentoscopy.stats.tradeoff)."""

import pytest

from agentoscopy.stats.compare import CompareError, RunInputs
from agentoscopy.stats.tradeoff import tradeoff_runs


def run(run_id, outcomes_by_task, cost, version=1):
    """A run whose every trial costs `cost`; `outcomes_by_task` maps task id to outcomes."""
    trials = [
        {
            "trial_id": f"{run_id}-{task_id}-{index}", "task_id": task_id,
            "task_version": version, "trial_index": index, "outcome": outcome,
            "cost_usd": cost, "steps": 2, "duration_s": 1.0, "input_tokens": 1,
            "output_tokens": 1,
        }
        for task_id, outcomes in outcomes_by_task.items()
        for index, outcome in enumerate(outcomes)
    ]  # fmt: skip
    return RunInputs({"run_id": run_id, "config_name": run_id}, trials, {}, {})


def passing(share, tasks=("a", "b", "c", "d")):
    """Outcomes where the first `share` of the tasks pass both trials and the rest fail."""
    cut = round(share * len(tasks))
    return {task: ["pass", "pass"] if i < cut else ["fail", "fail"] for i, task in enumerate(tasks)}


def test_the_frontier_is_the_runs_no_other_run_beats_on_both():
    runs = [
        run("strong", passing(1.0), cost=0.010),
        run("cheap", passing(0.25), cost=0.001),
        run("dominated", passing(0.5), cost=0.008),  # "middle" passes more for less
        run("middle", passing(0.75), cost=0.005),
    ]

    result = tradeoff_runs(runs)

    points = {point["run_id"]: point for point in result["runs"]}
    order = [point["run_id"] for point in result["runs"]]
    assert order == ["cheap", "middle", "dominated", "strong"]  # cheapest first
    frontier = {run_id for run_id, point in points.items() if point["on_frontier"]}
    assert frontier == {"cheap", "middle", "strong"}
    middle = points["middle"]
    assert (middle["pass_rate"], middle["trials"], middle["passes"]) == (0.75, 8, 6)
    assert middle["cost_per_trial_usd"] == pytest.approx(0.005)
    low, high = middle["pass_rate_ci_95"]
    assert low < 0.75 < high
    assert result["shared_tasks"] == 4


def test_identical_runs_are_both_on_the_frontier():
    result = tradeoff_runs([run("one", passing(0.5), 0.002), run("two", passing(0.5), 0.002)])

    assert all(point["on_frontier"] for point in result["runs"])


def test_the_interval_stays_within_possible_pass_rates():
    nearly = {"a": ["pass", "pass"], "b": ["pass", "pass"], "c": ["pass", "fail"]}

    point = tradeoff_runs([run("x", nearly, 0.001), run("y", nearly, 0.002)])["runs"][0]

    low, high = point["pass_rate_ci_95"]
    assert 0.0 < low < point["pass_rate"] and high == 1.0  # the raw t bound is above 1


def test_only_tasks_every_run_scored_at_one_version_count():
    wide = run("wide", {"a": ["pass"], "b": ["fail"], "c": ["fail"]}, 0.001)
    narrow = run("narrow", {"a": ["fail"], "b": ["fail"]}, 0.001)
    other_version = run("drifted", {"a": ["pass"], "b": ["fail"]}, 0.001, version=2)

    result = tradeoff_runs([wide, narrow])

    assert result["shared_tasks"] == 2  # c is missing from "narrow"
    assert {point["run_id"]: point["pass_rate"] for point in result["runs"]} == {
        "wide": 0.5, "narrow": 0.0,
    }  # fmt: skip
    assert result["warnings"] and result["warnings"][0].startswith("only 2 shared tasks")
    with pytest.raises(CompareError, match="NO_SHARED_TASKS"):
        tradeoff_runs([wide, other_version])


def test_unscored_trials_do_not_count_toward_cost():
    with_errors = run("errors", {"a": ["pass", "infra_error"]}, 0.004)
    with_errors.trials[1]["cost_usd"] = 0.5  # an infra error that spent a lot

    (point, _) = tradeoff_runs([with_errors, run("plain", {"a": ["pass"]}, 0.004)])["runs"]

    assert point["cost_per_trial_usd"] == pytest.approx(0.004) and point["trials"] == 1


def test_one_run_has_nothing_to_weigh_against():
    with pytest.raises(CompareError, match="RUN_COUNT"):
        tradeoff_runs([run("alone", passing(1.0), 0.001)])
