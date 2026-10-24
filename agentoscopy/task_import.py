"""Tasks from a repository's history (`agentoscopy task import`).

The commit before a fix becomes the task's starting code, the tests the fix added or changed
become its hidden tests, and the code it changed becomes the reference solution. `agentoscopy
task validate` then checks that the hidden tests fail before the fix and pass with it. Only
a local repository is read, through git; nothing is fetched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from agentoscopy.spec import SLUG_PATTERN, TASK_FILE, TaskSpec

DEFAULT_BASE_IMAGE = (
    "python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016"
)
TEST_DIR_NAMES = ("tests", "test")
# (hidden tests command, visible tests command); hidden tests are flattened into /hidden.
RUNNERS = {
    "unittest": (
        "python -m unittest discover -s /hidden -p '*test*.py'",
        "python -m unittest discover -s {tests} -p '*test*.py'",
    ),
    "pytest": ("python -m pytest -q /hidden", "python -m pytest -q {tests}"),
}
PYTEST_BUILD = "pip install --no-cache-dir pytest"
PROTECTED_PATHS = ["/workspace/tests/**", "/workspace/test/**", "**/conftest.py"]
REGULAR_FILE_MODES = ("100644", "100755")


class TaskImportError(Exception):
    """The commit cannot become a task as it is."""


@dataclass(frozen=True)
class FixCommit:
    commit: str
    parent: str
    message: str
    tests: list[str]  # test files the fix added or changed
    code: list[str]  # other files the fix added or changed


@dataclass(frozen=True)
class ImportOptions:
    task_id: str
    instructions: str | None = None  # default: the fix commit's message
    base_image: str = DEFAULT_BASE_IMAGE
    runner: str = "unittest"
    build: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportedTask:
    path: Path
    fix: FixCommit
    fixture_files: int
    skipped: list[str]  # entries that are not regular files: symlinks, submodules


def import_task(repo: Path, ref: str, tasks_dir: Path, options: ImportOptions) -> ImportedTask:
    """Write tasks_dir/<id>/ from the fix commit `ref`. Nothing is left behind on failure."""
    if not re.fullmatch(SLUG_PATTERN, options.task_id):
        raise TaskImportError(f"invalid task id {options.task_id!r}: use lowercase letters, "
                              "digits, and hyphens")  # fmt: skip
    target = tasks_dir / options.task_id
    if target.exists():
        raise TaskImportError(f"IMPORT_TASK_EXISTS: {target} already exists")
    fix = read_fix_commit(repo, ref)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tasks_dir, prefix=".import-") as scratch:
        draft = Path(scratch) / options.task_id
        # The old versions of tests the fix changed would assert the bug, so the agent never
        # sees them; their fixed versions are the hidden tests.
        fixture_files, skipped = _write_tree(
            repo, fix.parent, draft / "fixtures", held_back=frozenset(fix.tests)
        )
        _write_hidden_tests(repo, fix, draft / "hidden")
        for path in fix.code:
            _write(draft / "reference", path, _blob(repo, fix.commit, path))
        visible = next(
            (name for name in TEST_DIR_NAMES if (draft / "fixtures" / name).is_dir()), None
        )
        spec = task_spec(fix, options, visible)
        (draft / TASK_FILE).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        shutil.move(draft, target)
    return ImportedTask(target, fix, fixture_files, skipped)


def read_fix_commit(repo: Path, ref: str) -> FixCommit:
    if ref.startswith("-"):
        raise TaskImportError(f"not a commit: {ref!r}")
    commit = _text(_git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")).strip()
    parents = _text(_git(repo, "rev-list", "--parents", "-n", "1", commit)).split()[1:]
    if not parents:
        raise TaskImportError(
            f"IMPORT_NO_PARENT: {commit[:12]} is a root commit, so no code comes before it"
        )
    message = _text(_git(repo, "log", "-1", "--format=%B", commit)).strip()
    changes = _git(repo, "diff", "--name-status", "-z", "--no-renames", parents[0], commit)
    tests, code = _classify(changes)
    if not tests:
        raise TaskImportError(
            "IMPORT_NO_TESTS: the fix changes no test files, so nothing would check it"
        )
    if not code:
        raise TaskImportError("IMPORT_NO_FIX: the fix changes only tests")
    return FixCommit(commit, parents[0], message, tests, code)


def is_test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1]
    return (
        any(part in TEST_DIR_NAMES for part in parts[:-1])
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def task_spec(fix: FixCommit, options: ImportOptions, visible_tests: str | None) -> dict[str, Any]:
    """The task.yaml content: hidden tests, the existing tests if any, and the tamper check."""
    hidden_command, visible_command = RUNNERS[options.runner]
    build = [*([PYTEST_BUILD] if options.runner == "pytest" else []), *options.build]
    graders: list[dict[str, Any]] = [
        {"name": "hidden_tests", "type": "command", "run": hidden_command, "weight": 0.8}
    ]
    if visible_tests:
        run = visible_command.format(tests=visible_tests)
        graders.append({"name": "existing_tests", "type": "command", "run": run, "weight": 0.2})
    graders.append(
        {"name": "no_tamper", "type": "tamper_check", "protected_paths": PROTECTED_PATHS,
         "weight": 0}
    )  # fmt: skip
    spec = {
        "id": options.task_id,
        "category": "bugfix",
        "tags": list(options.tags),
        "instructions": options.instructions or fix.message,
        "environment": {"base_image": options.base_image, "workdir": "/workspace"}
        | ({"build": build} if build else {}),
        "budget": {"timeout_s": 300},
        "graders": graders,
    }
    try:
        TaskSpec.model_validate(spec)
    except ValidationError as exc:
        raise TaskImportError(f"the task would not be valid: {exc}") from exc
    return spec


def _classify(name_status: bytes) -> tuple[list[str], list[str]]:
    """(test files, other files) added or changed, from `git diff --name-status -z`."""
    fields = _text(name_status).split("\0")
    tests: list[str] = []
    code: list[str] = []
    for status, path in zip(fields[0::2], fields[1::2], strict=False):
        if is_test_path(path):
            if status in ("A", "M"):
                tests.append(path)
            continue  # a deleted test simply goes away
        if status not in ("A", "M"):
            raise TaskImportError(
                f"IMPORT_UNSUPPORTED_CHANGE: the fix deletes or retypes {path} (status "
                f"{status}); a reference solution can only add or change files"
            )
        code.append(path)
    return tests, code


def _write_tree(
    repo: Path, commit: str, target: Path, held_back: frozenset[str]
) -> tuple[int, list[str]]:
    """Every regular file of the commit but those held back; returns how many were written,
    and what was skipped as not a regular file."""
    written, skipped = 0, []
    for entry in _git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not entry:
            continue
        meta, _, raw_path = entry.partition(b"\t")
        mode, kind, sha = _text(meta).split()
        path = _text(raw_path)
        if path in held_back:
            continue
        if kind != "blob" or mode not in REGULAR_FILE_MODES:
            skipped.append(path)
            continue
        _write(target, path, _git(repo, "cat-file", "blob", sha))
        written += 1
    return written, skipped


def _write_hidden_tests(repo: Path, fix: FixCommit, hidden_dir: Path) -> None:
    names = [PurePosixPath(path).name for path in fix.tests]
    clashes = sorted({name for name in names if names.count(name) > 1})
    if clashes:
        raise TaskImportError(
            "IMPORT_TEST_NAME_CLASH: hidden tests share one folder, and the fix changes "
            f"several test files named {clashes}"
        )
    for path, name in zip(fix.tests, names, strict=True):
        _write(hidden_dir, name, _blob(repo, fix.commit, path))


def _write(base: Path, relative: str, data: bytes) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise TaskImportError(f"unsafe path in the repository: {relative!r}")
    target = base.joinpath(*path.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _blob(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "cat-file", "blob", f"{commit}:{path}")


def _git(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise TaskImportError("git is not installed or not on PATH") from exc
    if result.returncode != 0:
        detail = _text(result.stderr).strip() or f"exit code {result.returncode}"
        raise TaskImportError(f"git {args[0]} failed in {repo}: {detail}")
    return result.stdout


def _text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskImportError(
            "the repository has a file name or message that is not UTF-8"
        ) from exc
