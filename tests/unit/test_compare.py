import random

import pytest

from agentoscopy.stats.compare import (
    CompareError,
    RunInputs,
    compare_runs,
    config_diff,
    fisher_worse,
    mean_ci,
    t_975,
)
from agentoscopy.stats.report import render_ci_comment


def run_inputs(run_id, tasks, specs=None, config=None):
    """`tasks` maps task id to (outcomes, version, cost per trial)."""
    trials = []
    for task_id, (outcomes, version, cost) in tasks.items():
        for index, outcome in enumerate(outcomes):
            trials.append(
                {
                    "trial_id": f"{run_id}-{task_id}-{index}",
                    "task_id": task_id,
                    "task_version": version,
                    "trial_index": index,
                    "outcome": outcome,
                    "cost_usd": cost,
                    "steps": 2,
                    "duration_s": 1.0,
                    "input_tokens": 100,
                    "output_tokens": 10,
                }
            )
    run = {"run_id": run_id, "config_name": run_id, "config_hash": run_id}
    return RunInputs(run, trials, specs or {}, config or {})


def task(outcomes, version=1, cost=0.01):
    return (outcomes, version, cost)


PASS4, FAIL4 = ["pass"] * 4, ["fail"] * 4


@pytest.mark.parametrize(
    ("base_passes", "base_n", "cand_passes", "cand_n", "expected"),
    [
        (3, 3, 0, 3, 1 / 20),  # the best 3 vs 3 can do: exactly 0.05, never below
        (4, 4, 0, 4, 1 / 70),
        (6, 6, 2, 6, 28 / 924),
        (2, 4, 2, 4, 0.7571428571),
    ],
)
def test_fisher_one_sided(base_passes, base_n, cand_passes, cand_n, expected):
    assert fisher_worse(base_passes, base_n, cand_passes, cand_n) == pytest.approx(expected)


def test_tasks_are_classified_by_significance_not_by_size():
    baseline = run_inputs(
        "b",
        {
            "regressed": task(PASS4),
            "improved": task(FAIL4),
            "changed": task(["pass"] * 3),
            "flaky": task(["pass", "fail"] * 2),
            "stable-pass": task(PASS4),
            "stable-fail": task(FAIL4),
        },
    )
    candidate = run_inputs(
        "c",
        {
            "regressed": task(FAIL4),
            "improved": task(PASS4),
            "changed": task(["fail"] * 3),
            "flaky": task(["fail", "pass"] * 2),
            "stable-pass": task(PASS4),
            "stable-fail": task(FAIL4),
        },
    )

    result = compare_runs(baseline, candidate)

    classes = {row["task_id"]: row["class"] for row in result["tasks"]}
    assert classes == {
        "regressed": "regressed",
        "improved": "improved",
        "changed": "changed",  # 3/3 to 0/3 is a big drop, but p = 0.05 is not below 0.05
        "flaky": "flaky",
        "stable-pass": "stable_pass",
        "stable-fail": "stable_fail",
    }


def test_consistent_drop_across_tasks_is_a_regression():
    tasks = {f"t{i}": task(["pass"] * 3) for i in range(12)}
    worse = {f"t{i}": task(["fail"] * 3) for i in range(12)}

    result = compare_runs(run_inputs("b", tasks), run_inputs("c", worse))

    assert result["verdict"] == "REGRESSION"
    assert result["delta"] == pytest.approx(-1.0)
    assert result["delta_ci_95"][1] < 0
    assert result["warnings"] == []


def test_identical_runs_show_no_change():
    tasks = {f"t{i}": task(["pass", "fail", "pass"]) for i in range(12)}

    result = compare_runs(run_inputs("b", tasks), run_inputs("c", tasks))

    assert result["verdict"] == "NO_SIGNIFICANT_CHANGE"
    assert result["delta"] == 0


def test_a_regressed_critical_task_fails_the_comparison_alone():
    stable = {f"t{i}": task(PASS4) for i in range(11)}
    specs = {"key-task": {"critical": True}}
    baseline = run_inputs("b", {**stable, "key-task": task(PASS4)}, specs)
    candidate = run_inputs("c", {**stable, "key-task": task(FAIL4)}, specs)

    result = compare_runs(baseline, candidate)

    assert result["delta_ci_95"][1] >= 0  # the aggregate alone would not call it
    assert result["verdict"] == "REGRESSION"


def test_only_shared_scored_versions_are_compared():
    baseline = run_inputs(
        "b",
        {
            "shared": task(PASS4),
            "gone": task(PASS4),
            "drifted": task(PASS4, version=1),
            "broken": task(["infra_error"] * 2),
        },
    )
    candidate = run_inputs(
        "c",
        {
            "shared": task(PASS4),
            "new": task(PASS4),
            "drifted": task(PASS4, version=2),
            "broken": task(PASS4),
        },
    )

    result = compare_runs(baseline, candidate)
    with_drift = compare_runs(baseline, candidate, allow_version_drift=True)

    assert [row["task_id"] for row in result["tasks"]] == ["shared"]
    assert result["excluded"] == {
        "only_in_baseline": ["gone"],
        "only_in_candidate": ["new"],
        "version_drift": ["drifted"],
        "unscored": ["broken"],
    }
    assert [row["task_id"] for row in with_drift["tasks"]] == ["drifted", "shared"]
    assert any("drifted: comparing version 1 with 2" in w for w in with_drift["warnings"])
    assert any("only 1 shared task:" in w for w in result["warnings"])


def test_runs_without_shared_tasks_cannot_be_compared():
    with pytest.raises(CompareError, match="NO_SHARED_TASKS"):
        compare_runs(run_inputs("b", {"a": task(PASS4)}), run_inputs("c", {"b": task(PASS4)}))


def test_cost_and_slice_deltas():
    specs = {"x": {"category": "coding", "tags": ["py"]}, "y": {"category": "docs"}}
    baseline = run_inputs("b", {"x": task(PASS4, cost=0.01), "y": task(PASS4)}, specs)
    candidate = run_inputs("c", {"x": task(FAIL4, cost=0.03), "y": task(PASS4)}, specs)

    result = compare_runs(baseline, candidate)

    assert result["metric_deltas"]["cost_usd"] == pytest.approx(0.01)  # mean of +0.02 and 0
    categories = {item["value"]: item["mean"] for item in result["slices"]["category"]}
    assert categories == {"coding": -1.0, "docs": 0.0}
    assert result["slices"]["tag"] == [{"value": "py", "tasks": 1, "mean": -1.0}]


def test_config_diff_reports_changed_fields_and_text_diffs():
    before = {"name": "a", "params": {"model": "m1", "effort": "high", "system": "one\ntwo"}}
    after = {"name": "a", "params": {"model": "m2", "system": "one\nthree"}, "extra": 1}

    changes = {change["field"]: change for change in config_diff(before, after)}

    assert set(changes) == {"params.model", "params.effort", "params.system", "extra"}
    assert changes["params.effort"]["candidate"] is None
    assert "-two" in changes["params.system"]["diff"]
    assert "+three" in changes["params.system"]["diff"]


def config_lines(before, after):
    tasks = {"t": task(PASS4)}
    result = compare_runs(
        run_inputs("b", tasks, config=before), run_inputs("c", tasks, config=after)
    )
    comment = render_ci_comment(result, overridden=False, truncated=False)
    return [line for line in comment.splitlines() if line.startswith("- config ")]


def test_the_comment_shows_where_a_long_config_value_changed():
    script = "def paginate(items, page, per_page):\n    start = (page - 1) * per_page\n"
    broken = script.replace("per_page\n", "per_page + 1\n")

    [line] = config_lines({"params": {"code": script}}, {"params": {"code": broken}})

    before, after = line.removeprefix("- config params.code: ").split(" -> ")
    assert before.startswith("...") and after.startswith("...")
    assert "* per_page " in before and "* per_page + 1" in after


def test_short_config_values_are_shown_whole():
    assert config_lines({"params": {"model": "m1"}}, {"params": {"model": "m2"}}) == [
        "- config params.model: m1 -> m2"
    ]


@pytest.mark.parametrize(
    ("count", "phrase"), [(1, "over 1 shared task\n"), (2, "over 2 shared tasks\n")]
)
def test_the_comment_counts_shared_tasks_in_words(count, phrase):
    tasks = {f"t{i}": task(PASS4) for i in range(count)}
    result = compare_runs(run_inputs("b", tasks), run_inputs("c", tasks))

    assert phrase in render_ci_comment(result, overridden=False, truncated=False)


def test_a_config_compared_with_itself_rarely_gives_a_verdict():
    """Acceptance criterion 6: of 100 self-comparisons (a 20-task suite without critical
    tasks, 3 trials each, seeded noise), at most 10 may give a verdict other than
    NO_SIGNIFICANT_CHANGE. The design target is about 5%."""
    rng = random.Random(6)
    pass_rates = [rng.random() for _ in range(20)]

    def noisy_run(name):
        tasks = {
            f"task-{i}": task(["pass" if rng.random() < rate else "fail" for _ in range(3)])
            for i, rate in enumerate(pass_rates)
        }
        return run_inputs(name, tasks)

    verdicts = [compare_runs(noisy_run("a"), noisy_run("b"))["verdict"] for _ in range(100)]

    assert sum(verdict != "NO_SIGNIFICANT_CHANGE" for verdict in verdicts) <= 10


def test_mean_ci_is_a_t_interval():
    low, high = mean_ci([0.0, 1.0, 0.0, 1.0])

    # mean 0.5, sample sd 0.57735, t(3) = 3.182
    expected = (0.5 - 3.182 * 0.57735 / 2, 0.5 + 3.182 * 0.57735 / 2)
    assert (low, high) == pytest.approx(expected, rel=1e-4)
    assert mean_ci([-1.0, -1.0, -1.0]) == (-1.0, -1.0)  # no spread
    assert mean_ci([0.25]) == (0.25, 0.25)  # a single task: no spread to measure
    assert t_975(1) == 12.706 and abs(t_975(40) - 2.021) < 1e-3
