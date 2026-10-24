"""Tasks from a repository's history: `agentoscopy task import`."""

import pytest
from conftest import BUGGY_CALC, FIXED_CALC, NEW_CALC_TEST, OTHER_TEST, calc_repo_with_fix

from agentoscopy.cli.main import main
from agentoscopy.spec import load_task
from agentoscopy.task_import import (
    PYTEST_BUILD,
    ImportOptions,
    TaskImportError,
    import_task,
    is_test_path,
)


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    """One repository for every test here, since commits are slow to make: a root commit,
    the fix of add(), then one commit for each kind of change that cannot become a task."""
    repo, fix = calc_repo_with_fix(tmp_path_factory.mktemp("history") / "repo")
    commits = {"root": repo.git("rev-parse", "HEAD~1"), "fix": fix}
    commits["no-tests"] = repo.commit("code only", {"calc.py": "x = 1\n"})
    commits["no-fix"] = repo.commit("tests only", {"tests/test_more.py": "pass\n"})
    commits["deletes-code"] = repo.commit(
        "drop the readme", {"tests/test_more.py": "pass  # again\n"}, delete=["README.md"]
    )
    commits["name-clash"] = repo.commit(
        "two tests named alike",
        {"tests/a/test_x.py": "pass\n", "tests/b/test_x.py": "pass\n", "calc.py": "x = 2\n"},
    )
    return repo, commits


def test_a_fix_commit_becomes_a_task(history, tmp_path):
    repo, commits = history
    tasks_dir = tmp_path / "tasks"

    imported = import_task(
        repo.path, commits["fix"], tasks_dir, ImportOptions("fix-add", tags=("calc",))
    )

    task = load_task(tasks_dir, "fix-add")
    assert task.spec.instructions == "add() subtracts instead of adding"
    assert (task.spec.category, task.spec.tags) == ("bugfix", ["calc"])
    files = {
        path.relative_to(task.path).as_posix(): path.read_text() for path in task.path.rglob("*.*")
    }
    assert files["fixtures/calc.py"] == BUGGY_CALC  # the code before the fix
    assert files["fixtures/tests/test_other.py"] == OTHER_TEST
    assert "fixtures/README.md" in files
    # The test the fix changed is hidden in its fixed version; its old version, which
    # asserted the bug, is held back from the starting code.
    assert files["hidden/test_calc.py"] == NEW_CALC_TEST
    assert "fixtures/tests/test_calc.py" not in files
    assert files["reference/calc.py"] == FIXED_CALC
    graders = {grader.name: grader for grader in task.spec.graders}
    assert list(graders) == ["hidden_tests", "existing_tests", "no_tamper"]
    assert graders["existing_tests"].run == "python -m unittest discover -s tests -p '*test*.py'"
    assert (imported.fix.tests, imported.fix.code, imported.fixture_files) == (
        ["tests/test_calc.py"], ["calc.py"], 3,
    )  # fmt: skip


def test_the_pytest_runner_installs_pytest_before_any_other_build_step(history, tmp_path):
    repo, commits = history
    options = ImportOptions("fix-add", runner="pytest", build=("pip install -r requirements.txt",))

    import_task(repo.path, commits["fix"], tmp_path / "tasks", options)

    task = load_task(tmp_path / "tasks", "fix-add")
    assert task.spec.environment.build == [PYTEST_BUILD, "pip install -r requirements.txt"]
    assert task.spec.graders[0].run == "python -m pytest -q /hidden"


def test_instructions_can_replace_the_commit_message(history, tmp_path):
    repo, commits = history
    options = ImportOptions("fix-add", instructions="Make add() return the sum.")

    import_task(repo.path, commits["fix"], tmp_path / "tasks", options)

    assert (
        load_task(tmp_path / "tasks", "fix-add").spec.instructions == "Make add() return the sum."
    )


@pytest.mark.parametrize(
    ("commit", "expected"),
    [
        ("root", "IMPORT_NO_PARENT"),
        ("no-tests", "IMPORT_NO_TESTS"),
        ("no-fix", "IMPORT_NO_FIX"),
        ("deletes-code", "IMPORT_UNSUPPORTED_CHANGE"),
        ("name-clash", "IMPORT_TEST_NAME_CLASH"),
    ],
)
def test_commits_that_cannot_become_tasks_are_refused_and_leave_nothing(
    history, tmp_path, commit, expected
):
    repo, commits = history
    tasks_dir = tmp_path / "tasks"

    with pytest.raises(TaskImportError, match=expected):
        import_task(repo.path, commits[commit], tasks_dir, ImportOptions("imported"))

    assert not tasks_dir.exists() or list(tasks_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("ref", "task_id", "expected"),
    [
        ("no-such-commit", "fine", "git rev-parse failed"),
        ("--all", "fine", "not a commit"),
        ("HEAD", "Bad_Id", "invalid task id"),
    ],
)
def test_bad_refs_and_ids_are_refused(history, tmp_path, ref, task_id, expected):
    repo, _ = history

    with pytest.raises(TaskImportError, match=expected):
        import_task(repo.path, ref, tmp_path / "tasks", ImportOptions(task_id))


def test_an_existing_task_is_never_overwritten(history, tmp_path):
    repo, commits = history
    existing = tmp_path / "tasks" / "fix-add"
    existing.mkdir(parents=True)

    with pytest.raises(TaskImportError, match="IMPORT_TASK_EXISTS"):
        import_task(repo.path, commits["fix"], tmp_path / "tasks", ImportOptions("fix-add"))
    assert list(existing.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "is_test"),
    [
        ("tests/test_calc.py", True), ("pkg/test/helpers.py", True), ("test_calc.py", True),
        ("pkg/calc_test.py", True), ("conftest.py", True), ("calc.py", False),
        ("pkg/testing.py", False), ("latest/calc.py", False),
    ],
)  # fmt: skip
def test_test_files_are_recognised_by_folder_or_name(path, is_test):
    assert is_test_path(path) is is_test


def test_the_cli_reports_the_import_and_the_next_step(history, tmp_path, capsys):
    repo, commits = history
    common = ["--tasks-dir", str(tmp_path / "tasks"), "--home", str(tmp_path / "home")]
    command = ["task", "import", str(repo.path), commits["fix"], "--id", "fix-add", *common]

    code = main([*command, "--tag", "calc"])

    out = capsys.readouterr().out
    assert code == 0
    assert "hidden tests: tests/test_calc.py" in out and "reference fix: calc.py" in out
    assert "next: agentoscopy task validate fix-add" in out
    assert main(command) == 2 and "IMPORT_TASK_EXISTS" in capsys.readouterr().err
