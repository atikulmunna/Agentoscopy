"""Builds run summaries and comparisons from the store, for the CLI and the API."""

from __future__ import annotations

from typing import Any

from agentoscopy.stats.compare import RunInputs, compare_runs
from agentoscopy.stats.summary import summarize
from agentoscopy.storage.store import Store


class NotFound(Exception):
    """A requested run does not exist."""


def run_summary(store: Store, run_id: str, *, refresh: bool = False) -> dict[str, Any]:
    """The stored summary, or a freshly computed one (always fresh for unfinished runs)."""
    run = _run(store, run_id)
    stored = None if refresh or not run["finished_at"] else store.get_summary(run_id)
    if stored and "slices" in stored:
        return stored
    return summarize(
        store.trial_rows(run_id),
        trials_per_task=run["trials_per_task"],
        seed=run["seed"],
        spent_usd=store.run_spent(run_id),
        task_specs=store.task_specs_for_run(run_id),
    )


def comparison(
    store: Store, baseline_id: str, candidate_id: str, *, allow_version_drift: bool = False
) -> dict[str, Any]:
    return compare_runs(
        _inputs(store, baseline_id),
        _inputs(store, candidate_id),
        allow_version_drift=allow_version_drift,
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
