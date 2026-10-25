"""Tasks from SWE-bench rows: `agentoscopy task swebench`."""

import json
import urllib.request
from pathlib import Path

import pytest

from agentoscopy import swebench_check
from agentoscopy.cli import swebench as swebench_cli
from agentoscopy.cli.main import main
from agentoscopy.spec import load_task
from agentoscopy.swebench_import import (
    SwebenchImportError,
    import_instance,
    read_rows,
    registry_digest,
    task_id_for,
    with_test_exit_code,
)

LOCALE = "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen"
EVAL_SCRIPT = "\n".join([
    "#!/bin/bash",
    "set -uxo pipefail",
    "source /opt/miniconda3/bin/activate",
    "conda activate testbed",
    LOCALE,
    "git apply -v - <<'EOF_1'",
    "diff --git a/tests/test_v.py b/tests/test_v.py",
    "EOF_1",
    ": '>>>>> Start Test Output'",
    "./tests/runtests.py --verbosity 2 auth_tests.test_validators",
    ": '>>>>> End Test Output'",
    "git checkout abc tests/test_v.py",
])  # fmt: skip
IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
DIGEST = "sha256:" + "a" * 64
ROW = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "image": IMAGE,
    "eval_script": EVAL_SCRIPT,
    "log_parser": "parse_log_django",
    "eval_type": "pass_and_fail",
    "patch": "diff --git a/django/v.py b/django/v.py\n",
    "problem_statement": "Validators allow\r\na trailing newline",
    "FAIL_TO_PASS": '["test_ascii (auth_tests.V)"]',  # Hugging Face's form: a JSON string
    "PASS_TO_PASS": ["test_other (auth_tests.V)"],
    "difficulty": "<15 min fix",
}


def no_lookup(image):
    return DIGEST


def test_a_row_becomes_a_task_that_grades_like_swe_bench(tmp_path):
    imported = import_instance(ROW, tmp_path, no_lookup, tags=("verified",))

    task = load_task(tmp_path, "django-django-11099")
    spec = task.spec
    assert (imported.fail_to_pass, imported.pass_to_pass) == (1, 1)
    assert (spec.category, spec.difficulty, spec.tags) == ("django/django", "easy",
                                                           ["swe-bench", "verified"])  # fmt: skip
    assert spec.instructions.startswith("Validators allow\na trailing newline\n\n")
    assert "/testbed" in spec.instructions
    environment = spec.environment
    assert environment.base_image == f"{IMAGE}@{DIGEST}"
    assert (environment.workdir, environment.build) == ("/testbed", [LOCALE])
    assert environment.env == {"PATH": "/opt/miniconda3/envs/testbed/bin:$PATH"}
    assert [grader.name for grader in spec.graders] == ["swebench_tests", "no_tamper"]
    assert "/hidden/check.py /tmp/swebench-eval.log /hidden/tests.json" in spec.graders[0].run
    hidden = task.hidden_dir
    assert json.loads((hidden / "tests.json").read_text()) == {
        "log_parser": "parse_log_django",
        "FAIL_TO_PASS": ["test_ascii (auth_tests.V)"],
        "PASS_TO_PASS": ["test_other (auth_tests.V)"],
    }
    check = Path(swebench_check.__file__).read_text(encoding="utf-8")
    assert (hidden / "check.py").read_bytes() == check.encode()  # LF endings on any OS
    assert (hidden / "eval.sh").read_bytes() == with_test_exit_code(EVAL_SCRIPT).encode()
    assert task.reference_patch.read_text() == ROW["patch"]
    assert not list(tmp_path.glob(".import-*"))


def test_the_eval_script_records_the_test_exit_code_after_the_output():
    lines = with_test_exit_code(EVAL_SCRIPT).splitlines()

    end = lines.index(": '>>>>> End Test Output'")
    assert lines[end - 2 : end + 2] == [
        "./tests/runtests.py --verbosity 2 auth_tests.test_validators",
        "SWEBENCH_TEST_EXIT_CODE=$?",
        ": '>>>>> End Test Output'",
        'echo ">>>>> Test Exit Code: $SWEBENCH_TEST_EXIT_CODE"',
    ]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"image": ""}, "IMPORT_BAD_ROW: 'django__django-11099' lacks image"),
        ({"log_parser": "parse_log_jest"}, "IMPORT_UNSUPPORTED_PARSER: parse_log_jest"),
        ({"eval_type": "fail_only"}, "eval type fail_only"),
        ({"FAIL_TO_PASS": "[not json"}, "FAIL_TO_PASS is not valid JSON"),
        ({"PASS_TO_PASS": [1, 2]}, "PASS_TO_PASS is not a list of test ids"),
        ({"eval_script": "pytest"}, "IMPORT_UNSUPPORTED_SCRIPT"),
        ({"instance_id": "__"}, "cannot make a task id"),
    ],
)
def test_rows_that_cannot_become_tasks_are_refused(tmp_path, change, problem):
    with pytest.raises(SwebenchImportError, match=problem.replace("(", r"\(")):
        import_instance({**ROW, **change}, tmp_path, no_lookup)

    assert list(tmp_path.iterdir()) == []


def test_an_existing_task_is_not_overwritten(tmp_path):
    import_instance(ROW, tmp_path, no_lookup)

    with pytest.raises(SwebenchImportError, match="IMPORT_TASK_EXISTS"):
        import_instance(ROW, tmp_path, no_lookup)


def test_an_instance_without_pass_to_pass_tests_imports(tmp_path):
    assert import_instance({**ROW, "PASS_TO_PASS": "[]"}, tmp_path, no_lookup).pass_to_pass == 0


@pytest.mark.parametrize(
    ("instance_id", "task_id"),
    [("django__django-11099", "django-django-11099"), ("PyCQA__Flake8.5", "pycqa-flake8-5")],
)
def test_task_ids_are_slugs_of_instance_ids(instance_id, task_id):
    assert task_id_for(instance_id) == task_id


def test_rows_load_from_json_or_json_lines(tmp_path):
    lines, listing = tmp_path / "rows.jsonl", tmp_path / "rows.json"
    lines.write_text(json.dumps(ROW) + "\n\n" + json.dumps({**ROW, "instance_id": "b"}) + "\n")
    listing.write_text(json.dumps([ROW]))

    assert list(read_rows(lines)) == ["django__django-11099", "b"]
    assert list(read_rows(listing)) == ["django__django-11099"]
    listing.write_text('{"not": "a list"}')
    with pytest.raises(SwebenchImportError, match="expected a list of rows"):
        read_rows(listing)
    with pytest.raises(SwebenchImportError, match="cannot read file"):
        read_rows(tmp_path / "missing.jsonl")


class FakeResponse:
    def __init__(self, body=b"", headers=None):
        self._body, self.headers = body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self._body


def test_the_digest_comes_from_the_registry_without_pulling(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if isinstance(request, str):
            return FakeResponse(json.dumps({"token": "t0k"}).encode())
        return FakeResponse(headers={"Docker-Content-Digest": DIGEST})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert registry_digest(IMAGE) == DIGEST
    token_url, manifest = requests
    assert "scope=repository:swebench/sweb.eval.x86_64.django_1776_django-11099:pull" in token_url
    assert manifest.get_method() == "HEAD"
    assert manifest.full_url.endswith("/sweb.eval.x86_64.django_1776_django-11099/manifests/latest")
    assert manifest.get_header("Authorization") == "Bearer t0k"


def test_a_lookup_that_times_out_is_tried_again(monkeypatch):
    failures = iter([OSError("timed out"), OSError("timed out")])

    def urlopen(request, timeout):
        failure = next(failures, None)
        if failure:
            raise failure
        if isinstance(request, str):
            return FakeResponse(b'{"token": "t"}')
        return FakeResponse(headers={"Docker-Content-Digest": DIGEST})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert registry_digest(IMAGE) == DIGEST


@pytest.mark.parametrize(
    ("image", "urlopen", "problem"),
    [
        ("ghcr.io/x/y:latest", None, "is not a Docker Hub image:tag"),
        (IMAGE, OSError("timed out"), "cannot look up .*timed out"),
        (IMAGE, FakeResponse(b'{"token": "t"}'), "the registry gave no digest"),
    ],
)
def test_a_digest_that_cannot_be_found_stops_the_import(monkeypatch, image, urlopen, problem):
    def fake(request, timeout):
        if isinstance(urlopen, Exception):
            raise urlopen
        return urlopen

    monkeypatch.setattr(urllib.request, "urlopen", fake)

    with pytest.raises(SwebenchImportError, match=problem):
        registry_digest(image)


def cli(tmp_path, *args):
    return main(["task", "swebench", *args, "--tasks-dir", str(tmp_path / "tasks"),
                 "--home", str(tmp_path / "home")])  # fmt: skip


def test_the_command_imports_named_rows_and_reports_the_rest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(swebench_cli, "registry_digest", no_lookup)
    rows = tmp_path / "rows.jsonl"
    other = {**ROW, "instance_id": "sympy__sympy-1", "log_parser": "parse_log_jest"}
    rows.write_text(json.dumps(ROW) + "\n" + json.dumps(other) + "\n")

    assert cli(tmp_path, str(rows), "django__django-11099") == 0
    assert "next: agentoscopy task validate django-django-11099" in capsys.readouterr().out
    assert cli(tmp_path, str(rows), "--all") == 2  # one exists now, one is unsupported
    errors = capsys.readouterr().err
    assert "IMPORT_TASK_EXISTS" in errors and "IMPORT_UNSUPPORTED_PARSER" in errors
    assert cli(tmp_path, str(rows), "nope__nope-1") == 2
    assert "not in" in capsys.readouterr().err
    assert cli(tmp_path, str(rows)) == 2
    assert "or pass --all" in capsys.readouterr().err
