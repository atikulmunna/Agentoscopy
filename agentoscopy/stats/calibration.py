"""Judge calibration (FR-REV-04, FR-REV-05): how often each judge agrees with human reviewers.

Agreement and Cohen's kappa use only randomly sampled reviews. Low-confidence and manually added
items are enriched for hard cases and would bias kappa downward, so they appear among the
disagreements but stay out of the statistics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

KAPPA_THRESHOLD = 0.6
MIN_REVIEWS = 30


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Kappa for (human, judge) pass/fail labels; None when undefined."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(human == judge for human, judge in pairs) / n
    human_yes = sum(human for human, _ in pairs) / n
    judge_yes = sum(judge for _, judge in pairs) / n
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    if expected == 1:  # both raters used a single label throughout
        return None
    return (observed - expected) / (1 - expected)


def calibrate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One report per judge grader: (task, grader, judge model)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["grader_name"], row["judge_model"] or "")].append(row)
    return [_report(key, group) for key, group in sorted(groups.items())]


def _report(key: tuple[str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_id, grader_name, judge_model = key
    random = [row for row in rows if row["sample_source"] == "random"]
    pairs = [(row["human_passed"], row["judge_passed"]) for row in random]
    kappa = cohen_kappa(pairs)
    confusion = {
        "human_pass_judge_pass": sum(h and j for h, j in pairs),
        "human_pass_judge_fail": sum(h and not j for h, j in pairs),
        "human_fail_judge_pass": sum(not h and j for h, j in pairs),
        "human_fail_judge_fail": sum(not h and not j for h, j in pairs),
    }
    return {
        "task_id": task_id,
        "grader_name": grader_name,
        "judge_model": judge_model,
        "random_reviews": len(random),
        "all_reviews": len(rows),
        "agreement": sum(h == j for h, j in pairs) / len(pairs) if pairs else None,
        "kappa": kappa,
        "confusion": confusion,
        "status": _status(len(random), kappa),
        "disagreements": [
            {key: row[key] for key in ("review_id", "trial_id", "sample_source")}
            | {"human_passed": row["human_passed"], "judge_passed": row["judge_passed"]}
            for row in rows
            if row["human_passed"] != row["judge_passed"]
        ],
    }


def _status(random_reviews: int, kappa: float | None) -> str:
    if random_reviews < MIN_REVIEWS:
        return "insufficient_data"
    if kappa is None:
        return "undefined"
    return "uncalibrated" if kappa < KAPPA_THRESHOLD else "calibrated"
