"""Builds summaries, comparisons, calibration reports, and review items from the store, for
the CLI and the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentoscopy.graders.fsdiff import changed_files
from agentoscopy.recorder.trajectory import action_lines, read_events
from agentoscopy.stats.calibration import calibrate
from agentoscopy.stats.compare import RunInputs, compare_runs
from agentoscopy.stats.summary import summarize
from agentoscopy.storage.store import Store


class NotFound(Exception):
    """A requested run, trial, or review does not exist."""


def run_summary(store: Store, run_id: str, *, refresh: bool = False) -> dict[str, Any]:
    """The stored summary, or a freshly computed one (always fresh for unfinished runs).

    Judge calibration changes as reviews come in, so its warnings are always computed fresh.
    """
    run = _run(store, run_id)
    stored = None if refresh or not run["finished_at"] else store.get_summary(run_id)
    if stored and "judge_cost_usd" in stored:
        summary = stored
    else:
        summary = summarize(
            store.trial_rows(run_id),
            trials_per_task=run["trials_per_task"],
            seed=run["seed"],
            spent_usd=store.run_spent(run_id),
            task_specs=store.task_specs_for_run(run_id),
        )
        summary["judge_cost_usd"] = store.run_judge_spent(run_id)
    return {**summary, "uncalibrated_judges": _uncalibrated_judges(store, run_id)}


def comparison(
    store: Store, baseline_id: str, candidate_id: str, *, allow_version_drift: bool = False
) -> dict[str, Any]:
    return compare_runs(
        _inputs(store, baseline_id),
        _inputs(store, candidate_id),
        allow_version_drift=allow_version_drift,
    )


def calibration(store: Store) -> list[dict[str, Any]]:
    return calibrate(store.calibration_rows())


def safe_artifact(home: Path, uri: str | None) -> Path | None:
    """A stored artifact path, or None if it is missing or outside the Agentoscopy home."""
    if not uri:
        return None
    path = Path(uri).resolve()
    return path if path.is_relative_to(home.resolve()) and path.is_file() else None


def review_item(store: Store, home: Path, review: dict[str, Any]) -> dict[str, Any]:
    """What a reviewer sees: the task, the rubric, and the agent's work. The judge's verdict
    and the trial outcome are left out until the review is submitted (FR-REV-02)."""
    trial = store.get_trial(review["trial_id"])
    if trial is None:
        raise NotFound(f"no trial {review['trial_id']}")
    spec = store.task_spec(review["task_id"], review["task_version"])
    graders = {grader["name"]: grader for grader in spec.get("graders", [])}
    attempt = next((a for a in trial["attempts"] if a["attempt"] == review["attempt"]), {})
    trajectory = safe_artifact(home, attempt.get("trajectory_uri"))
    events = read_events(trajectory) if trajectory else []
    diff = safe_artifact(home, trial["final_state_uri"])
    changes = changed_files(diff.read_text(encoding="utf-8")) if diff else []
    messages = [e["payload"].get("text") for e in events if e["type"] == "agent_message"]
    return {
        "review_id": review["review_id"],
        "trial_id": review["trial_id"],
        "run_id": review["run_id"],
        "task_id": review["task_id"],
        "attempt": review["attempt"],
        "grader_name": review["grader_name"],
        "sample_source": review["sample_source"],
        "instructions": spec.get("instructions"),
        "rubric": graders.get(review["grader_name"], {}).get("rubric"),
        "final_message": messages[-1] if messages else None,
        "changed_files": [f"{kind} {path}" for kind, path in changes],  # as the judge sees them
        "actions": action_lines(events),
    }


def _uncalibrated_judges(store: Store, run_id: str) -> list[str]:
    judged = {
        (task_id, grader["name"])
        for task_id, spec in store.task_specs_for_run(run_id).items()
        for grader in spec.get("graders", [])
        if grader.get("type") == "llm_judge"
    }
    return sorted(
        f"{report['task_id']}/{report['grader_name']}"
        for report in calibration(store)
        if report["status"] == "uncalibrated"
        and (report["task_id"], report["grader_name"]) in judged
    )


def _inputs(store: Store, run_id: str) -> RunInputs:
    run = _run(store, run_id)
    return RunInputs(
        run=run,
        trials=store.trial_rows(run_id),
        task_specs=store.task_specs_for_run(run_id),
        config=store.config_spec(run["config_hash"]),
    )


def _run(store: Store, run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise NotFound(f"no run {run_id}")
    return run
