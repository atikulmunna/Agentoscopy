"""Run summary rendering: terminal table, Markdown, or JSON (FR-AGG-06)."""

from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

FORMATS = ("table", "md", "json")
DIFF_CONTEXT_CHARS = 20  # kept before the first difference when a config change is shortened


def render(summary: dict[str, Any], run: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps({"run": run, "summary": summary}, indent=2)
    if output_format == "md":
        return _markdown(summary, run)
    return _table(summary, run)


def _headline(summary: dict[str, Any], run: dict[str, Any]) -> list[str]:
    suite = suite_name(run)
    macro = _pct(summary["macro_pass_rate"])
    ci = summary.get("macro_ci_95")
    ci_text = f" (95% CI {_pct(ci[0])} to {_pct(ci[1])})" if ci else ""
    counts = ", ".join(f"{name} {count}" for name, count in summary["counts"].items())
    cost_per_pass = summary["cost_per_pass_usd"]
    lines = [
        f"run {run['run_id']}: {suite}, agent {run.get('config_name')}, "
        f"{run['trials_per_task']} trials per task, seed {run['seed']}",
        f"status {run['status']}" + (f", flags {', '.join(run['flags'])}" if run["flags"] else ""),
        f"macro pass rate {macro}{ci_text}, micro {_pct(summary['micro_pass_rate'])}",
        f"trials: {counts}",
        f"cost ${summary['total_cost_usd']:.4f}"
        + (f", ${cost_per_pass:.4f} per pass" if cost_per_pass is not None else ""),
    ]
    if summary.get("judge_cost_usd"):
        lines.append(f"judging ${summary['judge_cost_usd']:.4f} (on top of agent cost)")
    if summary.get("uncalibrated_judges"):
        judges = ", ".join(summary["uncalibrated_judges"])
        lines.append(
            f"warning: uncalibrated judges, so pass rates that use them may be off: {judges}"
        )
    if summary["flaky_tasks"]:
        lines.append(f"flaky: {', '.join(summary['flaky_tasks'])}")
    if summary.get("failure_tags"):
        tags = ", ".join(f"{tag} {count}" for tag, count in summary["failure_tags"].items())
        lines.append(f"failure tags: {tags}")
    if summary["unscored_tasks"]:
        lines.append(f"unscored (no valid trials): {', '.join(summary['unscored_tasks'])}")
    return lines


def suite_name(run: dict[str, Any]) -> str:
    if not run.get("suite_id"):
        return "ad hoc tasks"
    if run.get("suite_version") is None:
        return f"{run['suite_id']} ({run.get('labels', {}).get('filter', 'filtered')})"
    return f"{run['suite_id']} v{run['suite_version']}"


def _header(k: int) -> tuple[str, ...]:
    return ("task", "n", "pass", "rate", f"pass@{k}", f"pass^{k}", "cost/trial", "steps")


def _task_rows(summary: dict[str, Any]) -> list[tuple[str, ...]]:
    return [
        (
            task["task_id"],
            str(task["n"]),
            str(task["passes"]),
            _pct(task["pass_rate"]),
            _pct(task["pass_at_k"]),
            _pct(task["pass_hat_k"]),
            "-" if task["mean_cost_usd"] is None else f"${task['mean_cost_usd']:.4f}",
            "-" if task["mean_steps"] is None else f"{task['mean_steps']:.1f}",
        )
        for task in summary["tasks"]
    ]


def _table(summary: dict[str, Any], run: dict[str, Any]) -> str:
    lines = _text_table(_header(summary["k"]), _task_rows(summary))
    return "\n".join([*_headline(summary, run), "", *lines])


def _markdown(summary: dict[str, Any], run: dict[str, Any]) -> str:
    table = _markdown_table(_header(summary["k"]), _task_rows(summary))
    headline = [f"- {line}" for line in _headline(summary, run)[1:]]
    title = f"# Run {run['run_id']}"
    return "\n".join([title, "", *headline, "", *table])


def _text_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in [header, *rows]
    ]


def _markdown_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{_tenths(value * 100)}%"


def _tenths(value: float) -> Decimal:
    """One decimal place, ties rounded away from zero as the web UI's toFixed rounds them, so
    both show 57 of 80 as 71.3%; format's .1f rounds ties to even and would show 71.2%."""
    return Decimal(value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


CLASS_ORDER = ("regressed", "improved", "changed", "flaky", "stable_pass", "stable_fail")


def render_comparison(comparison: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(comparison, indent=2)
    lines = _comparison_headline(comparison)
    header = ("task", "class", "baseline", "candidate", "delta", "p(worse)", "cost delta")
    rows = [_comparison_row(task) for task in _ordered(comparison["tasks"])]
    if output_format == "md":
        title = f"# {comparison['verdict']}"
        table = _markdown_table(header, rows)
        return "\n".join([title, "", *(f"- {line}" for line in lines[1:]), "", *table])
    return "\n".join([*lines, "", *_text_table(header, rows)])


def render_tradeoff(tradeoff: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(tradeoff, indent=2)
    shared = tradeoff["shared_tasks"]
    lines = [
        f"pass rate against agent cost over {shared} shared task{'' if shared == 1 else 's'}, "
        "cheapest first",
        "frontier: no other run passes at least as often for no more cost",
        *(f"warning: {warning}" for warning in tradeoff["warnings"]),
    ]
    header = ("run", "config", "pass rate", "95% CI", "cost/trial", "passes", "frontier")
    rows = [_tradeoff_row(point) for point in tradeoff["runs"]]
    if output_format == "md":
        title = "# Cost against quality"
        table = _markdown_table(header, rows)
        return "\n".join([title, "", *(f"- {line}" for line in lines), "", *table])
    return "\n".join([*lines, "", *_text_table(header, rows)])


def _tradeoff_row(point: dict[str, Any]) -> tuple[str, ...]:
    low, high = point["pass_rate_ci_95"]
    return (
        point["run_id"],
        point["config_name"] or "-",
        _pct(point["pass_rate"]),
        f"{_pct(low)} to {_pct(high)}",
        f"${point['cost_per_trial_usd']:.4f}",
        f"{point['passes']}/{point['trials']}",
        "yes" if point["on_frontier"] else "no",
    )


def render_ci_comment(comparison: dict[str, Any], *, overridden: bool, truncated: bool) -> str:
    """The pull request comment for `agentoscopy ci` (WF-10 step 4), in Markdown."""
    verdict = comparison["verdict"].replace("_", " ").capitalize()
    lines = [f"## Agentoscopy: {verdict}", ""]
    if overridden:
        lines += [
            "> The `eval-override` label is set, so this regression does not fail the check.",
            "",
        ]
    if truncated:
        lines += [
            "> The candidate run hit its budget and skipped trials, so these results are "
            "incomplete.",
            "",
        ]
    # The Markdown comparison minus its own title, which repeats the verdict.
    lines.append(render_comparison(comparison, "md").split("\n", 2)[2])
    return "\n".join(lines) + "\n"


def _comparison_headline(comparison: dict[str, Any]) -> list[str]:
    base, cand = comparison["baseline"], comparison["candidate"]
    low, high = comparison["delta_ci_95"]
    classes = ", ".join(
        f"{name.replace('_', ' ')} {comparison['classes'][name]}"
        for name in CLASS_ORDER
        if name in comparison["classes"]
    )
    excluded = ", ".join(
        f"{name.replace('_', ' ')} {len(ids)}" for name, ids in comparison["excluded"].items()
    )
    deltas = comparison["metric_deltas"]
    shared = comparison["shared_tasks"]
    lines = [
        f"{comparison['verdict']}: candidate {cand['run_id']} vs baseline {base['run_id']}",
        f"delta pass rate {_points(comparison['delta'])} "
        f"(95% CI {_points(low)} to {_points(high)}) "
        f"over {shared} shared task{'' if shared == 1 else 's'}",
        f"baseline {_pct(base['macro_pass_rate'])} ({base['config_name']}), "
        f"candidate {_pct(cand['macro_pass_rate'])} ({cand['config_name']})",
        f"per-trial deltas: cost {_signed(deltas['cost_usd'], '$', 4)}, "
        f"tokens {_signed(deltas['tokens'], '', 0)}, steps {_signed(deltas['steps'], '', 1)}, "
        f"duration {_signed(deltas['duration_s'], '', 1)}s",
        f"classes: {classes}",
        f"excluded: {excluded}",
    ]
    lines += [f"warning: {warning}" for warning in comparison["warnings"]]
    for change in comparison["config_diff"]:
        before, after = _from_first_difference(change["baseline"], change["candidate"])
        lines.append(f"config {change['field']}: {before} -> {after}")
    return lines


def _comparison_row(task: dict[str, Any]) -> tuple[str, ...]:
    base, cand = task["baseline"], task["candidate"]
    name = task["task_id"] + (" (critical)" if task["critical"] else "")
    return (
        name,
        task["class"].replace("_", " "),
        f"{base['passes']}/{base['n']}",
        f"{cand['passes']}/{cand['n']}",
        _points(task["delta"]),
        f"{task['p_worse']:.3f}",
        _signed(task["metric_deltas"]["cost_usd"], "$", 4),
    )


def _ordered(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(tasks, key=lambda task: (CLASS_ORDER.index(task["class"]), task["delta"]))


def _points(value: float) -> str:
    return f"{_tenths(value * 100):+} points"


def _signed(value: float | None, prefix: str, digits: int) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{prefix}{abs(value):.{digits}f}"


def _from_first_difference(before: Any, after: Any, limit: int = 60) -> tuple[str, str]:
    """Both values, shortened to start a little before they first differ: long values that
    share their opening would otherwise look the same."""
    texts = [_one_line(before), _one_line(after)]
    start = max(0, len(os.path.commonprefix(texts)) - DIFF_CONTEXT_CHARS)
    lead = "..." if start else ""
    before_text, after_text = (_clip(lead + text[start:], limit) for text in texts)
    return before_text, after_text


def _one_line(value: Any) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text.replace("\n", " ")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
