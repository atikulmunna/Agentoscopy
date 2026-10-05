"""Choosing which of a suite's tasks a run includes (FR-RUN-02)."""

import pytest
import yaml
from conftest import BASE_TASK

from agentoscopy.api.common import ApiError
from agentoscopy.api.runs import parse_request
from agentoscopy.cli.main import main
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import PlanError, TaskFilter, select_tasks
from agentoscopy.spec import load_task
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend

TASKS = {
    "easy-strings": {"category": "bugfix", "difficulty": "easy", "tags": ["strings"]},
    "hard-dates": {"category": "bugfix", "difficulty": "hard", "tags": ["dates", "parsing"]},
    "new-feature": {"category": "feature", "difficulty": "medium", "tags": ["strings"]},
}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture
def project(tmp_path, write_task, tasks_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "agentoscopy.cli.run.DockerBackend", lambda: FakeBackend(grader=passes_when_fixed)
    )
    home = tmp_path / "home"
    store = Store(home / "agentoscopy.db")
    for task_id, metadata in TASKS.items():
        write_task({**BASE_TASK, "id": task_id, **metadata})
        store.save_task_version(load_task(tasks_dir, task_id), "fake-image", [], [])
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "all.yaml").write_text(yaml.safe_dump({"name": "all", "tasks": list(TASKS)}))
    agent = tmp_path / "fixer.yaml"
    agent.write_text(
        yaml.safe_dump(
            {
                "name": "fixer",
                "adapter": "python",
                "entrypoint": "agentoscopy.testing.scripted_agent:ScriptedAgent",
                "params": {"script": [{"write": "app.py", "content": "fixed"}]},
            }
        )
    )
    yield store, home, tasks_dir, suites, agent
    store.close()


def run(project, *extra):
    _, home, tasks_dir, suites, agent = project
    return main(
        [
            "run", "--suite", "all", "--agent", str(agent), "--trials", "1", "--seed", "1",
            "--home", str(home), "--tasks-dir", str(tasks_dir), "--suites-dir", str(suites),
            *extra,
        ]
    )  # fmt: skip


def run_tasks(store, run_id):
    return sorted({row["task_id"] for row in store.trial_rows(run_id)})


def test_filters_combine_across_fields_and_widen_within_one(project):
    store = project[0]

    assert run(project, "--tag", "strings", "--tag", "dates", "--difficulty", "easy") == 0
    assert run(project, "--category", "feature") == 0

    narrow, feature = reversed([r["run_id"] for r in store.list_runs()])
    assert run_tasks(store, narrow) == ["easy-strings"]
    assert run_tasks(store, feature) == ["new-feature"]


def test_a_filtered_run_records_no_suite_version(project, capsys):
    store = project[0]

    run(project)  # the whole suite
    run(project, "--tag", "dates")

    filtered, full = store.list_runs()
    assert (full["suite_id"], full["suite_version"]) == ("all", 1)
    assert (filtered["suite_id"], filtered["suite_version"]) == ("all", None)
    assert filtered["labels"] == {"filter": "tag=dates"}
    assert store.current_suite_version("all", [(t, 1) for t in TASKS]) == 1  # untouched
    assert "all (tag=dates)" in capsys.readouterr().out


def test_failed_in_reruns_only_what_did_not_pass(project, tmp_path):
    store = project[0]
    idler = tmp_path / "idler.yaml"
    idler.write_text(
        project[4].read_text().replace('"fixed"', '"broken"').replace("fixed", "broken")
    )
    _, home, tasks_dir, suites, _ = project
    first_args = ["--home", str(home), "--tasks-dir", str(tasks_dir), "--suites-dir", str(suites)]
    tasks = ["--task", "easy-strings", "--task", "hard-dates"]
    assert main(["run", *tasks, "--agent", str(idler), "--trials", "1", *first_args]) == 0
    failed_run = store.list_runs()[0]["run_id"]

    assert run(project, "--failed-in", failed_run) == 0

    rerun = store.list_runs()[0]
    assert run_tasks(store, rerun["run_id"]) == ["easy-strings", "hard-dates"]
    assert rerun["labels"] == {"filter": f"failed-in={failed_run}"}


def test_selections_that_match_nothing_are_refused(project, capsys):
    assert run(project, "--tag", "graphics") == 2
    assert run(project, "--failed-in", "no-such-run") == 2

    err = capsys.readouterr().err
    assert "NO_TASKS_SELECTED: no task matches tag=graphics" in err
    assert "NOT_FOUND: no run no-such-run" in err
    assert project[0].list_runs() == []


def test_a_resumed_run_takes_no_filters(project, capsys):
    store = project[0]
    run(project)
    run_id = store.list_runs()[0]["run_id"]
    _, home, tasks_dir, _, _ = project

    dirs = ["--home", str(home), "--tasks-dir", str(tasks_dir)]
    assert main(["run", "--resume", run_id, "--tag", "dates", *dirs]) == 2
    assert "takes no task filters" in capsys.readouterr().err


def test_select_tasks_keeps_suite_order(project):
    store, _, tasks_dir, _, _ = project

    kept = select_tasks(store, tasks_dir, list(TASKS), TaskFilter(tags=frozenset({"strings"})))

    assert kept == ["easy-strings", "new-feature"]
    with pytest.raises(PlanError, match="NO_TASKS_SELECTED"):
        select_tasks(store, tasks_dir, list(TASKS), TaskFilter(categories=frozenset({"docs"})))


def test_the_api_takes_the_same_filter(tmp_path):
    configs = tmp_path
    (configs / "fixer.yaml").write_text("name: fixer\n")
    body = {
        "agent": "fixer",
        "suite": "all",
        "filter": {"tags": ["strings"], "difficulties": ["easy"], "failed_in": "run-1"},
    }

    request = parse_request(body, configs)

    assert request.task_filter == TaskFilter(
        tags=frozenset({"strings"}), difficulties=frozenset({"easy"}), failed_in="run-1"
    )
    for bad in ({"tags": "strings"}, {"colour": ["red"]}, {"failed_in": 3}, []):
        with pytest.raises(ApiError):
            parse_request({**body, "filter": bad}, configs)
