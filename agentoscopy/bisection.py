"""Find the step where a failed trial went wrong, by bisecting over fork points.

A failed trial is forked at step k with a config that can pass the task, the rescue config:
the trial's first k - 1 recorded model answers are replayed, then the rescue config takes
over. While those forks still pass, the trial's mistake has not happened yet; once they
fail, it has. A binary search over k finds the last step from which the trial can still be
rescued, so the answer recorded at that step is the one that sealed the failure.

Live model calls vary, so each probe runs several forks and counts as rescued when most of
them pass. The search assumes that a trial that cannot be rescued from one step cannot be
rescued from any later step either; every probe is reported so the evidence can be checked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Probe:
    """The forks of one trial at one step."""

    step: int
    passes: int
    trials: int
    run_id: str
    problem: str | None = None  # set when a fork did not finish with a pass or a fail

    @property
    def rescued(self) -> bool:
        return self.passes * 2 > self.trials


@dataclass(frozen=True)
class Bisection:
    culprit_step: int | None  # the step whose recorded answer sealed the failure
    probes: list[Probe] = field(default_factory=list)
    stopped: str | None = None  # why no step was found


def bisect_steps(recorded_calls: int, probe: Callable[[int], Probe]) -> Bisection:
    """Search steps 1 to recorded_calls + 1, where forking at recorded_calls + 1 replays every
    recorded answer. Both ends are probed first, so a search starts only between a step that
    rescues the trial and one that does not."""
    first, last = probe(1), None
    probes = [first]
    stopped = _end_problem(first, rescued_expected=True)
    if stopped is None:
        last = probe(recorded_calls + 1)
        probes.append(last)
        stopped = _end_problem(last, rescued_expected=False)
    if stopped is not None:
        return Bisection(None, probes, stopped)
    low, high = 1, recorded_calls + 1
    while high - low > 1:
        middle = (low + high) // 2
        result = probe(middle)
        probes.append(result)
        if result.problem:
            return Bisection(None, probes, f"the forks at step {middle} {result.problem}")
        low, high = (middle, high) if result.rescued else (low, middle)
    return Bisection(low, probes)


def _end_problem(result: Probe, *, rescued_expected: bool) -> str | None:
    if result.problem:
        return f"the forks at step {result.step} {result.problem}"
    if result.rescued == rescued_expected:
        return None
    if rescued_expected:
        return "the rescue config fails the task from the first step too, so no step is to blame"
    return (
        "the rescue config passes even with every recorded answer kept, so the failure is not "
        "in the model's answers (the agent code or config differs in how it acts on them)"
    )
