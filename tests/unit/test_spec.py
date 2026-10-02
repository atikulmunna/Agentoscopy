from pathlib import Path

import pytest
from conftest import BASE_TASK

from agentoscopy.spec import SpecError, config_hash, load_agent_config, load_suite, load_task


def with_environment(**changes):
    return {**BASE_TASK, "environment": {**BASE_TASK["environment"], **changes}}


def test_loads_valid_task_with_defaults(make_task):
    task = make_task()

    assert task.spec.id == "demo-task"
    assert task.spec.environment.workdir == "/workspace"
    assert task.spec.graders[0].required is True
    assert len(task.content_hash) == 64


def test_content_hash_is_stable_and_tracks_task_files(write_task, tasks_dir):
    task_dir = write_task(files={"fixtures/app.py": "x = 1\n", "hidden/test_app.py": "ok\n"})
    first = load_task(tasks_dir, "demo-task").content_hash
    assert load_task(tasks_dir, "demo-task").content_hash == first

    (task_dir / "fixtures" / "app.py").write_text("x = 2\n", encoding="utf-8")
    after_fixture_change = load_task(tasks_dir, "demo-task").content_hash
    assert after_fixture_change != first

    (task_dir / "hidden" / "test_app.py").write_text("changed\n", encoding="utf-8")
    assert load_task(tasks_dir, "demo-task").content_hash != after_fixture_change


def test_content_hash_ignores_pycache(write_task, tasks_dir):
    task_dir = write_task(files={"fixtures/app.py": "x = 1\n"})
    before = load_task(tasks_dir, "demo-task")

    cache = task_dir / "fixtures" / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"\x00\x01")
    after = load_task(tasks_dir, "demo-task")

    assert after.content_hash == before.content_hash
    assert [path.name for path in after.fixture_files()] == ["app.py"]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (with_environment(base_image="python:3.12-slim"), "environment.base_image"),
        (with_environment(workdir="workspace"), "environment.workdir"),
        (with_environment(network={"allow": ["pypi.org"]}), "environment.network.allow"),
        ({**BASE_TASK, "graders": [{"name": "j", "type": "llm_judge", "run": "x"}]}, "graders.0"),
        ({**BASE_TASK, "owner": "me"}, "owner: Extra inputs are not permitted"),
        ({**BASE_TASK, "graders": []}, "graders"),
    ],
)
def test_invalid_fields_are_reported_with_their_path(write_task, tasks_dir, spec, expected):
    write_task(spec)

    with pytest.raises(SpecError, match=expected.replace(".", r"\.")):
        load_task(tasks_dir, "demo-task")


def test_rejects_task_without_required_grader(write_task, tasks_dir):
    grader = {"name": "tests", "type": "command", "run": "x", "required": False}
    write_task({**BASE_TASK, "graders": [grader]})

    with pytest.raises(SpecError, match="at least one grader must be required"):
        load_task(tasks_dir, "demo-task")


def test_rejects_duplicate_grader_names(write_task, tasks_dir):
    grader = {"name": "tests", "type": "command", "run": "x"}
    write_task({**BASE_TASK, "graders": [grader, grader]})

    with pytest.raises(SpecError, match="grader names must be unique"):
        load_task(tasks_dir, "demo-task")


def test_rejects_id_that_differs_from_directory(write_task, tasks_dir):
    task_dir = write_task()
    task_dir.rename(tasks_dir / "other-name")

    with pytest.raises(SpecError, match="must match the directory name"):
        load_task(tasks_dir, "other-name")


@pytest.mark.parametrize("task_id", ["../escape", "Upper", "", "a/b"])
def test_rejects_unsafe_task_ids(tasks_dir, task_id):
    with pytest.raises(SpecError, match="invalid task id"):
        load_task(tasks_dir, task_id)


def test_reports_missing_and_malformed_files(tasks_dir):
    with pytest.raises(SpecError, match="cannot read file"):
        load_task(tasks_dir, "missing-task")

    task_dir = tasks_dir / "bad-yaml"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(SpecError, match="invalid YAML"):
        load_task(tasks_dir, "bad-yaml")

    (task_dir / "task.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(SpecError, match="expected a mapping"):
        load_task(tasks_dir, "bad-yaml")


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agent.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_agent_config_hash_tracks_params(tmp_path):
    base = "name: a\nadapter: python\nentrypoint: pkg.mod:Agent\n"
    first = load_agent_config(write_config(tmp_path, base + "params: {x: 1}\n"))
    same = load_agent_config(write_config(tmp_path, base + "params: {x: 1}\n"))
    changed = load_agent_config(write_config(tmp_path, base + "params: {x: 2}\n"))

    assert config_hash(first) == config_hash(same)
    assert config_hash(first) != config_hash(changed)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name: a\nadapter: http\nentrypoint: pkg.mod:Agent\n", "adapter"),
        ("name: a\nadapter: python\nentrypoint: not-an-entrypoint\n", "entrypoint"),
        ("adapter: python\nentrypoint: pkg.mod:Agent\n", "name"),
    ],
)
def test_invalid_agent_configs_are_rejected(tmp_path, text, expected):
    with pytest.raises(SpecError, match=expected):
        load_agent_config(write_config(tmp_path, text))


def test_reference_files_are_part_of_the_content_hash(write_task, tasks_dir):
    task_dir = write_task(files={"reference/app.py": "fixed\n"})
    before = load_task(tasks_dir, "demo-task")

    (task_dir / "reference" / "app.py").write_text("fixed differently\n", encoding="utf-8")

    assert load_task(tasks_dir, "demo-task").content_hash != before.content_hash
    assert [path.name for path in before.reference_files()] == ["app.py"]


def write_suite(tmp_path: Path, name: str, text: str) -> Path:
    suites_dir = tmp_path / "suites"
    suites_dir.mkdir(exist_ok=True)
    (suites_dir / f"{name}.yaml").write_text(text, encoding="utf-8")
    return suites_dir


def test_loads_a_suite(tmp_path):
    suites_dir = write_suite(tmp_path, "smoke", "name: smoke\ntasks: [task-a, task-b]\n")

    assert load_suite(suites_dir, "smoke").tasks == ["task-a", "task-b"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name: other\ntasks: [task-a]\n", "must match the file name"),
        ("name: smoke\ntasks: [task-a, task-a]\n", "task ids must be unique"),
        ("name: smoke\ntasks: [Bad_Id]\n", "invalid task ids"),
        ("name: smoke\ntasks: []\n", "tasks"),
    ],
)
def test_invalid_suites_are_rejected(tmp_path, text, expected):
    suites_dir = write_suite(tmp_path, "smoke", text)

    with pytest.raises(SpecError, match=expected):
        load_suite(suites_dir, "smoke")


def test_spelling_out_a_default_value_keeps_the_same_version(write_task, tasks_dir):
    write_task()
    implicit = load_task(tasks_dir, "demo-task").content_hash

    write_task({**BASE_TASK, "critical": False, "category": "uncategorized", "tags": []})

    assert load_task(tasks_dir, "demo-task").content_hash == implicit


def test_every_grader_type_loads_with_its_defaults(make_task):
    graders = [
        *BASE_TASK["graders"],
        {"name": "safe", "type": "forbidden_command", "patterns": [r"git\s+push"]},
        {"name": "careful", "type": "must_read_before_edit", "required": False},
        {"name": "tidy", "type": "max_tool_errors", "max": 0},
        {"name": "quick", "type": "max_steps", "max": 20},
        {"name": "no_tamper", "type": "tamper_check"},
        {"name": "quality", "type": "llm_judge", "rubric": "Minimal?", "required": False},
    ]

    spec = make_task(graders=graders).spec

    by_name = {grader.name: grader for grader in spec.graders}
    assert [grader.type for grader in spec.graders] == [grader["type"] for grader in graders]
    assert (by_name["no_tamper"].allowed_paths, by_name["no_tamper"].protected_paths) == (None, [])
    assert (by_name["quality"].max_cost_usd, by_name["quality"].timeout_s) == (0.25, 180)
    assert spec.has_judges() and not make_task().spec.has_judges()


@pytest.mark.parametrize(
    ("grader", "expected"),
    [
        ({"name": "x", "type": "shell", "run": "x"}, "graders.0"),
        ({"name": "x", "type": "max_tool_errors", "max": -1}, "graders.0.max_tool_errors.max"),
        ({"name": "x", "type": "max_steps", "max": 0}, "graders.0.max_steps.max"),
        ({"name": "x", "type": "llm_judge", "rubric": ""}, "graders.0.llm_judge.rubric"),
        ({"name": "x", "type": "forbidden_command", "patterns": ["("]}, "patterns"),
    ],
)
def test_invalid_graders_are_rejected(write_task, tasks_dir, grader, expected):
    write_task({**BASE_TASK, "graders": [grader]})

    with pytest.raises(SpecError, match=expected.replace(".", r"\.")):
        load_task(tasks_dir, "demo-task")
