"""Prometheus metrics (NFR-OBS-02) in the text exposition format.

They are computed from the results database when scraped, so they cover runs executed by any
process: `agentoscopy run` in a terminal as well as runs started through the API.
"""

from __future__ import annotations

from typing import Any

from agentoscopy.storage.store import IN_FLIGHT_STATES, Store

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
DURATION_BUCKETS = (10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0)


def render_metrics(store: Store) -> str:
    data = store.metrics(DURATION_BUCKETS)
    states = data["trial_states"]
    lines: list[str] = []
    _family(
        lines,
        "queue_depth",
        "gauge",
        "Trials waiting to be dispatched.",
        [({}, states.get("QUEUED", 0))],
    )
    _family(
        lines,
        "active_trials",
        "gauge",
        "Trials holding a sandbox, by state.",
        [({"state": state}, states.get(state, 0)) for state in IN_FLIGHT_STATES],
    )
    _family(lines, "runs", "gauge", "Runs by status.", _by(data["run_statuses"], "status"))
    _family(
        lines,
        "trials_total",
        "counter",
        "Finished trials by outcome.",
        _by(data["trial_outcomes"], "outcome"),
    )
    _family(
        lines,
        "attempts_total",
        "counter",
        "Finished trial attempts by outcome; with infra_errors_total, the infra error rate.",
        _by(data["attempt_outcomes"], "outcome"),
    )
    _family(
        lines,
        "infra_errors_total",
        "counter",
        "Attempts that ended in an infra error, by error code.",
        _by(data["infra_errors"], "error_code"),
    )
    _histogram(lines, data["durations"])
    _family(
        lines,
        "gateway_latency_seconds",
        "summary",
        "Time model calls took through the gateway.",
        [],
    )
    lines.append(f"agentoscopy_gateway_latency_seconds_sum {data['model_latency_s']:.6f}")
    lines.append(f"agentoscopy_gateway_latency_seconds_count {data['model_calls']}")
    _family(
        lines,
        "spend_usd_total",
        "counter",
        "Model spend in US dollars; judge spend is kept apart from agent spend.",
        [
            ({"kind": "agent"}, data["agent_spend_usd"]),
            ({"kind": "judge"}, data["judge_spend_usd"]),
        ],
    )
    return "\n".join(lines) + "\n"


def _family(
    lines: list[str],
    name: str,
    kind: str,
    help_text: str,
    samples: list[tuple[dict[str, str], float]],
) -> None:
    full = f"agentoscopy_{name}"
    lines += [f"# HELP {full} {help_text}", f"# TYPE {full} {kind}"]
    lines += [f"{full}{_labels(labels)} {_number(value)}" for labels, value in samples]


def _histogram(lines: list[str], durations: dict[str, Any]) -> None:
    name = "agentoscopy_trial_duration_seconds"
    lines += [f"# HELP {name} Wall-clock time of finished trials.", f"# TYPE {name} histogram"]
    for le, count in zip(DURATION_BUCKETS, durations["buckets"], strict=True):
        lines.append(f'{name}_bucket{{le="{le:g}"}} {count}')
    lines.append(f'{name}_bucket{{le="+Inf"}} {durations["count"]}')
    lines.append(f"{name}_sum {durations['sum']:.3f}")
    lines.append(f"{name}_count {durations['count']}")


def _by(counts: dict[Any, int], label: str) -> list[tuple[dict[str, str], float]]:
    return [({label: str(key)}, value) for key, value in sorted(counts.items(), key=str)]


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in labels.items()) + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value: float) -> str:
    return str(value) if isinstance(value, int) else f"{value:.6f}"
