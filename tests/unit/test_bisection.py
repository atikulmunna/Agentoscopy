"""Bisect: the step where a failed trial went wrong, found by forking it with a rescue config."""

import anthropic
import pytest
import yaml
from conftest import BASE_TASK

from agentoscopy.adapters.base import AgentResult
from agentoscopy.bisection import Probe, bisect_steps
from agentoscopy.cli.main import main
from agentoscopy.gateway.mock import mock_message
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import load_task
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend

# The search on its own ------------------------------------------------------------------


def prober(rescued_through, trials=3, problem_at=None):
    """Forks rescue the trial at steps up to `rescued_through`; records which steps ran."""
    visited = []

    def probe(step):
        visited.append(step)
        passes = trials if step <= rescued_through else 0
        problem = "did not finish: REPLAY_DIVERGED" if step == problem_at else None
        return Probe(step, passes, trials, f"run-{step}", problem)

    return probe, visited


def test_the_search_finds_the_last_step_that_still_rescues_the_trial():
    probe, visited = prober(rescued_through=3)

    result = bisect_steps(8, probe)

    assert result.culprit_step == 3
    assert visited == [1, 9, 5, 3, 4]  # both ends, then halving


def test_one_recorded_call_needs_only_the_two_ends():
    probe, visited = prober(rescued_through=1)

    assert bisect_steps(1, probe).culprit_step == 1
    assert visited == [1, 2]


def test_no_step_is_blamed_when_the_rescue_config_fails_from_the_start():
    probe, visited = prober(rescued_through=0)

    result = bisect_steps(8, probe)

    assert result.culprit_step is None and visited == [1]
    assert "fails the task from the first step" in result.stopped


def test_no_step_is_blamed_when_keeping_every_answer_still_passes():
    probe, _ = prober(rescued_through=9)

    result = bisect_steps(8, probe)

    assert result.culprit_step is None
    assert "not in the model's answers" in result.stopped


def test_a_probe_that_does_not_finish_stops_the_search():
    probe, visited = prober(rescued_through=3, problem_at=5)

    result = bisect_steps(8, probe)

    assert result.culprit_step is None and visited == [1, 9, 5]
    assert result.stopped == "the forks at step 5 did not finish: REPLAY_DIVERGED"


@pytest.mark.parametrize(
    ("passes", "trials", "rescued"), [(2, 3, True), (1, 3, False), (1, 2, False)]
)
def test_a_step_rescues_the_trial_when_most_forks_pass(passes, trials, rescued):
    assert Probe(1, passes, trials, "run").rescued is rescued


# End to end ------------------------------------------------------------------------------


class AnswerFollowingAgent:
    """Makes four model calls and fixes app.py at the end, unless an answer was "break": then
    it writes a broken fix. A careless config gets "break" at step 3; a careful one never."""

    def __init__(self):
        self._style = None

    async def setup(self, config):
        self._style = config.params["style"]

    async def run(self, task, sandbox, model_endpoint, budget, recorder):
        client = anthropic.AsyncAnthropic(
            base_url=model_endpoint.base_url, api_key=model_endpoint.token
        )
        broken = False
        try:
            for step in range(1, 5):
                prompt = f"{self._style} step {step}"
                message = await client.messages.create(
                    model="mock", max_tokens=50, messages=[{"role": "user", "content": prompt}]
                )
                broken = broken or message.content[0].text == "break"
        finally:
            await client.close()
        await sandbox.write_file("app.py", b"broken" if broken else b"fixed")
        return AgentResult(final_message="done")

    async def teardown(self):
        self._style = None


def answer_break_at_careless_step_3(body, seed_key):
    message = mock_message(body, seed_key)
    text = "break" if body["messages"][-1]["content"] == "careless step 3" else "ok"
    return {**message, "content": [{"type": "text", "text": text}]}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture
def project(tmp_path, write_task, tasks_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "agentoscopy.cli.run.DockerBackend", lambda: FakeBackend(grader=passes_when_fixed)
    )
    monkeypatch.setattr("agentoscopy.gateway.server.mock_message", answer_break_at_careless_step_3)
    store = Store(tmp_path / "home" / "agentoscopy.db")
    write_task({**BASE_TASK, "id": "task-a"})
    store.save_task_version(load_task(tasks_dir, "task-a"), "fake-image", [], [])
    yield store, tmp_path
    store.close()


def agent(project, style):
    path = project[1] / f"{style}.yaml"
    entrypoint = f"{__name__}:AnswerFollowingAgent"
    path.write_text(
        yaml.safe_dump(
            {
                "name": style,
                "adapter": "python",
                "entrypoint": entrypoint,
                "params": {"style": style},
            }
        )
    )
    return path


def cli(project, *args):
    root = project[1]
    return main([*args, "--home", str(root / "home"), "--tasks-dir", str(root / "tasks")])


def failed_trial(project):
    store = project[0]
    assert cli(project, "run", "--task", "task-a", "--agent", str(agent(project, "careless")),
               "--trials", "1") == 0  # fmt: skip
    (trial,) = store.trial_rows(store.list_runs()[0]["run_id"])
    assert trial["outcome"] == "fail"
    return trial


def test_bisect_names_the_step_whose_answer_sealed_the_failure(project, capsys):
    trial = failed_trial(project)

    code = cli(project, "bisect", trial["trial_id"], "--agent", str(agent(project, "careful")),
               "--trials", "1")  # fmt: skip

    out = capsys.readouterr().out
    assert code == 0
    assert "step 3: the trial can still be rescued by taking over at step 3" in out
    for step, verdict in [
        (1, "1 of 1 passed, rescued"),
        (3, "1 of 1 passed, rescued"),
        (4, "0 of 1 passed, not rescued"),
        (5, "0 of 1 passed, not rescued"),
    ]:
        assert f"fork at step {step}: {verdict} (run " in out  # fmt: skip
    assert f"#/side/{trial['trial_id']}/" in out
    store = project[0]
    probes = [run for run in store.list_runs() if "bisect" in run["labels"]]
    assert sorted(int(run["labels"]["fork_at"]) for run in probes) == [1, 3, 4, 5]


def test_a_rescue_config_that_fails_too_blames_no_step(project, capsys):
    trial = failed_trial(project)

    code = cli(project, "bisect", trial["trial_id"], "--agent", str(agent(project, "careless")),
               "--trials", "1")  # fmt: skip

    assert code == 1
    assert "no step found: the rescue config fails the task from the first step" in (
        capsys.readouterr().out
    )


def test_bisect_stops_when_its_budget_runs_out(project, capsys):
    trial = failed_trial(project)
    # Each fork reserves the task's $1 cap; the first probe's spend leaves less than that.
    code = cli(project, "bisect", trial["trial_id"], "--agent", str(agent(project, "careful")),
               "--trials", "1", "--budget-usd", "1.0001")  # fmt: skip

    out = capsys.readouterr().out
    assert code == 1
    assert "fork at step 1: 1 of 1 passed, rescued" in out
    assert "no step found: the forks at step 5 were skipped: the bisection budget ran out" in out


def test_a_budget_below_the_trial_cap_is_refused_before_any_fork(project, capsys):
    trial = failed_trial(project)
    runs_before = len(project[0].list_runs())

    code = cli(project, "bisect", trial["trial_id"], "--agent", str(agent(project, "careful")),
               "--budget-usd", "0.5")  # fmt: skip

    assert code == 2
    assert "BUDGET_BELOW_TRIAL_CAP" in capsys.readouterr().err
    assert len(project[0].list_runs()) == runs_before


def test_only_a_failed_trial_can_be_bisected(project, capsys):
    store = project[0]
    careful = str(agent(project, "careful"))
    assert cli(project, "run", "--task", "task-a", "--agent", careful, "--trials", "1") == 0
    (passed,) = store.trial_rows(store.list_runs()[0]["run_id"])

    assert cli(project, "bisect", passed["trial_id"], "--agent", careful) == 2
    assert "NOT_FAILED" in capsys.readouterr().err
    assert cli(project, "bisect", "no-such-trial", "--agent", careful) == 2
