import pytest
from conftest import BASE_TASK

from agentoscopy.cli.common import EXIT_INVALID_INPUT, EXIT_OK
from agentoscopy.cli.main import main
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.storage.store import NewTrial, RunSettings, Store


@pytest.fixture(autouse=True)
def no_provider_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def agentoscopy(tmp_path, tasks_dir, *args):
    return main(
        [
            *args,
            "--home",
            str(tmp_path / "home"),
            "--tasks-dir",
            str(tasks_dir),
            "--suites-dir",
            str(tmp_path / "suites"),
        ]
    )


def test_run_refuses_unvalidated_tasks(write_task, tasks_dir, tmp_path, capsys):
    write_task()

    exit_code = agentoscopy(
        tmp_path, tasks_dir, "run", "--task", "demo-task", "--agent", "agent.yaml"
    )

    assert exit_code == EXIT_INVALID_INPUT
    assert "UNVALIDATED_TASKS: demo-task" in capsys.readouterr().err


def test_run_reports_a_missing_suite(tasks_dir, tmp_path, capsys):
    exit_code = agentoscopy(tmp_path, tasks_dir, "run", "--suite", "nope", "--agent", "agent.yaml")

    assert exit_code == EXIT_INVALID_INPUT
    assert "cannot read file" in capsys.readouterr().err


def test_run_rejects_a_bad_entrypoint_before_creating_a_run(
    write_task, tasks_dir, tmp_path, capsys
):
    write_task()
    store = Store(tmp_path / "home" / "agentoscopy.db")
    store.save_task_version(load_task(tasks_dir, "demo-task"), "img", [], [])
    store.close()
    agent = tmp_path / "agent.yaml"
    agent.write_text("name: a\nadapter: python\nentrypoint: no_such_pkg.mod:Agent\n")

    exit_code = agentoscopy(
        tmp_path, tasks_dir, "run", "--task", "demo-task", "--agent", str(agent)
    )

    assert exit_code == EXIT_INVALID_INPUT
    assert "cannot import 'no_such_pkg.mod'" in capsys.readouterr().err


def test_suite_and_task_are_mutually_exclusive(tasks_dir, tmp_path):
    with pytest.raises(SystemExit):
        agentoscopy(tmp_path, tasks_dir, "run", "--suite", "s", "--task", "t", "--agent", "a.yaml")


def test_validate_needs_task_ids_or_all(tasks_dir, tmp_path, capsys):
    assert agentoscopy(tmp_path, tasks_dir, "task", "validate") == EXIT_INVALID_INPUT
    assert "--all" in capsys.readouterr().err


def test_task_list_shows_validation_status(write_task, tasks_dir, tmp_path, capsys):
    write_task()
    write_task({"id": "broken-task"})

    assert agentoscopy(tmp_path, tasks_dir, "task", "list") == EXIT_OK

    lines = capsys.readouterr().out.splitlines()
    assert lines == ["broken-task  invalid spec", "demo-task  not validated (changed or new)"]


def test_report_prints_a_stored_run(tasks_dir, tmp_path, capsys):
    store = Store(tmp_path / "home" / "agentoscopy.db")
    config = AgentConfig(name="agent", adapter="python", entrypoint="pkg.mod:Agent")
    settings = RunSettings(None, None, 1, None, 5, 1, "test")
    run_id = store.create_run(config, settings, [NewTrial("task-a", 1, 0, 0, 1.0)])
    store.close()

    assert agentoscopy(tmp_path, tasks_dir, "report", run_id, "--format", "md") == EXIT_OK
    assert f"# Run {run_id}" in capsys.readouterr().out
    assert agentoscopy(tmp_path, tasks_dir, "report", "missing-run") == EXIT_INVALID_INPUT


JUDGED = {"name": "quality", "type": "llm_judge", "rubric": "Minimal?", "required": False}


def judged_task(write_task, tasks_dir, tmp_path):
    write_task({**BASE_TASK, "graders": [*BASE_TASK["graders"], JUDGED]})
    store = Store(tmp_path / "home" / "agentoscopy.db")
    store.save_task_version(load_task(tasks_dir, "demo-task"), "img", [], [])
    store.close()


def test_a_real_judge_without_credentials_is_refused_before_the_run(
    write_task, tasks_dir, tmp_path, capsys
):
    judged_task(write_task, tasks_dir, tmp_path)

    exit_code = agentoscopy(
        tmp_path, tasks_dir, "run", "--task", "demo-task", "--agent", "agent.yaml"
    )

    assert exit_code == EXIT_INVALID_INPUT
    assert "JUDGE_NEEDS_CREDENTIALS" in capsys.readouterr().err
    store = Store(tmp_path / "home" / "agentoscopy.db")
    assert store.list_runs() == []
    store.close()


def test_validate_refuses_a_real_judge_without_credentials(write_task, tasks_dir, tmp_path, capsys):
    write_task({**BASE_TASK, "graders": [*BASE_TASK["graders"], JUDGED]})

    exit_code = agentoscopy(tmp_path, tasks_dir, "task", "validate", "demo-task")

    assert exit_code == EXIT_INVALID_INPUT
    assert "FAIL  demo-task: JUDGE_NEEDS_CREDENTIALS" in capsys.readouterr().out


def test_an_unpriced_judge_model_is_refused(write_task, tasks_dir, tmp_path, capsys):
    judged_task(write_task, tasks_dir, tmp_path)

    exit_code = agentoscopy(
        tmp_path, tasks_dir, "run", "--task", "demo-task", "--agent", "agent.yaml",
        "--judge-model", "gpt-x",
    )  # fmt: skip

    assert exit_code == EXIT_INVALID_INPUT
    assert "UNKNOWN_JUDGE_MODEL" in capsys.readouterr().err


@pytest.mark.parametrize("rate", ["-0.1", "1.5", "lots"])
def test_review_rate_must_be_a_fraction(tasks_dir, tmp_path, rate):
    with pytest.raises(SystemExit):
        agentoscopy(
            tmp_path, tasks_dir, "run", "--task", "t", "--agent", "a.yaml", "--review-rate", rate
        )
