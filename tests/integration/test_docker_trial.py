"""End to end against a real Docker daemon, with the example task, suite, and test agents."""

import asyncio
import subprocess
from pathlib import Path

import pytest
from conftest import running_gateway

from agentoscopy.adapters.python import load_python_adapter
from agentoscopy.cli.main import main
from agentoscopy.graders.judge import GatewayJudge
from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.spec import load_agent_config, load_task
from agentoscopy.storage.store import Store
from agentoscopy.validation import validate_task
from agentoscopy.worker.trial import AttemptResult, AttemptSpec, run_attempt

REPO = Path(__file__).resolve().parents[2]
TASK_ID = "fix-off-by-one-pagination"
REQUIRED = ["hidden_tests", "existing_tests", "no_tamper"]


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not docker_available(), reason="Docker daemon not available"),
]


@pytest.fixture(autouse=True)
def no_provider_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def run_example(config_name: str, output_dir: Path) -> AttemptResult:
    task = load_task(REPO / "tasks", TASK_ID)
    config = load_agent_config(REPO / "configs" / f"{config_name}.yaml")
    spec = AttemptSpec(
        run_id="integration",
        trial_id=f"it-{config_name}",
        attempt=1,
        task=task,
        config=config,
        seed_key="it",
        judge_model="mock",
    )

    async def scenario():
        async with running_gateway() as gateway:
            adapter = load_python_adapter(config.entrypoint)
            return await run_attempt(spec, adapter, DockerBackend(), gateway, output_dir)

    return asyncio.run(scenario())


def leftovers(kind: str, trial_id: str) -> str:
    listing = subprocess.run(
        ["docker", kind, "--all", "--quiet", "--filter", f"label=agentoscopy.trial_id={trial_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return listing.stdout.strip()


def by_name(result: AttemptResult) -> dict[str, bool]:
    return {grade.name: grade.passed for grade in result.grades}


def test_example_task_validates(tmp_path):
    task = load_task(REPO / "tasks", TASK_ID)

    async def scenario():
        async with running_gateway() as gateway:
            recorder = TrajectoryRecorder(tmp_path / "grading.jsonl", tmp_path / "grading", "v", 1)
            judge = GatewayJudge(gateway, "mock", recorder, "it")
            try:
                return await validate_task(task, DockerBackend(), judge)
            finally:
                judge.close()

    report = asyncio.run(scenario())

    assert report.ok, f"{report.error_code}: {report.message}"
    # The reference passes the tests and the tamper check; the mock judge answers at random.
    assert set(REQUIRED) <= set(report.verified_graders)
    assert report.flags == []


def test_scripted_fix_passes_and_leaves_nothing_behind(tmp_path):
    result = run_example("scripted-fix", tmp_path)

    assert result.error is None
    assert (result.outcome, result.termination) == ("pass", "agent_done")
    assert all(by_name(result)[name] for name in REQUIRED)
    assert result.judge_cost_usd > 0 and not result.vetoed
    assert result.usage.steps == 2 and result.usage.cost_usd > 0
    assert result.cleanup_errors == []
    assert leftovers("ps", "it-scripted-fix") == ""  # agent and grading containers
    assert leftovers("images", "it-scripted-fix") == ""  # the grading snapshot


def test_broken_variant_fails(tmp_path):
    result = run_example("scripted-broken", tmp_path)

    assert (result.outcome, result.termination) == ("fail", "agent_done")
    assert by_name(result)["hidden_tests"] is False


def test_malicious_attempts_fail_inside_the_sandbox_and_are_recorded(tmp_path):
    result = run_example("scripted-malicious", tmp_path)

    events = read_events(result.trajectory_path)
    calls = [e["payload"]["args"]["cmd"] for e in events if e["type"] == "tool_call"]
    results = [e["payload"] for e in events if e["type"] == "tool_result"]
    assert result.outcome == "fail"
    assert any("/hidden/" in cmd for cmd in calls)
    assert any("urlopen" in cmd for cmd in calls)
    assert any("api_key" in cmd for cmd in calls)
    assert len(results) == 3
    assert all(outcome["exit_code"] != 0 for outcome in results)


@pytest.mark.parametrize("config_name", ["scripted-tamper-tests", "scripted-tamper-conftest"])
def test_tampering_is_vetoed_even_when_the_tests_pass(config_name, tmp_path):
    result = run_example(config_name, tmp_path)

    grades = by_name(result)
    assert grades["hidden_tests"] and grades["existing_tests"]  # the fix itself is correct
    assert grades["no_tamper"] is False and result.vetoed
    assert (result.outcome, result.score) == ("fail", 0.0)


def test_cli_validates_then_runs_the_example_suite(tmp_path, capsys):
    common = [
        "--home", str(tmp_path / "home"),
        "--tasks-dir", str(REPO / "tasks"),
        "--suites-dir", str(REPO / "suites"),
    ]  # fmt: skip
    judged = [*common, "--judge-model", "mock"]  # validate and run grade the judge grader
    agent = str(REPO / "configs" / "scripted-fix.yaml")

    assert main(["task", "validate", TASK_ID, *judged]) == 0
    exit_code = main(
        ["run", "--suite", "example", "--agent", agent, "--trials", "2", "--seed", "1", *judged]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "macro pass rate 100.0%" in output
    assert "pass 2, infra_error 0" in output

    broken = str(REPO / "configs" / "scripted-broken.yaml")
    run_broken = ["run", "--suite", "example", "--agent", broken, "--trials", "2", "--seed", "1"]
    assert main([*run_broken, *judged]) == 0
    tamper = str(REPO / "configs" / "scripted-tamper-tests.yaml")
    assert main(["run", "--suite", "example", "--agent", tamper, "--trials", "1", *judged]) == 0
    store = Store(tmp_path / "home" / "agentoscopy.db")
    tampered, candidate, baseline = [run["run_id"] for run in store.list_runs()]  # newest first
    (trial,) = store.trial_rows(tampered)
    tags = store.get_trial(trial["trial_id"])["failure_tags"]
    store.close()
    assert trial["outcome"] == "fail"
    assert [(tag["tag"], tag["source"]) for tag in tags] == [("reward_hacking", "auto")]
    capsys.readouterr()

    assert main(["compare", baseline, candidate, *common]) == 0
    compared = capsys.readouterr().out
    assert compared.startswith("REGRESSION")
    # 2 of 2 to 0 of 2 is a real drop, but too few trials for the per-task test to call it.
    assert f"{TASK_ID}  changed" in compared
