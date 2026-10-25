"""Tasks from SWE-bench rows (`agentoscopy task swebench`).

Each row of SWE-bench (or Lite, or Verified) names the Docker image that holds its repository
at /testbed with the environment installed, the eval script that applies the hidden tests and
runs them, and the log parser that reads the result. An imported task starts from that image,
pinned by digest; its grader runs the eval script and then `swebench_check`, which grades the
log the way SWE-bench does; and the fix SWE-bench calls gold becomes its reference patch, so
`agentoscopy task validate` checks that the task gives SWE-bench's verdict. Rows come from a
local .json or .jsonl file; the only network request is for each image's digest.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentoscopy import swebench_check
from agentoscopy.sandbox.base import HIDDEN_MOUNT
from agentoscopy.spec import REFERENCE_PATCH, SLUG_PATTERN, TASK_FILE, TaskSpec

# PASS_TO_PASS is not among them: an instance may have no tests that already passed.
REQUIRED_FIELDS = (
    "instance_id", "repo", "image", "eval_script", "log_parser", "eval_type", "patch",
    "problem_statement", "FAIL_TO_PASS",
)  # fmt: skip
WORKDIR = "/testbed"
IMAGE_PYTHON = "/opt/miniconda3/bin/python3"  # the images' conda base, outside the agent's reach
TESTBED_PATH = "/opt/miniconda3/envs/testbed/bin:$PATH"
EVAL_LOG = "/tmp/swebench-eval.log"
GRADER_RUN = (
    f"bash {HIDDEN_MOUNT}/eval.sh > {EVAL_LOG} 2>&1; "
    f"{IMAGE_PYTHON} {HIDDEN_MOUNT}/check.py {EVAL_LOG} {HIDDEN_MOUNT}/tests.json"
)
TEST_EXIT_CODE_VAR = "SWEBENCH_TEST_EXIT_CODE"
# Graders run as the agent's user, so steps of the eval script that need root run once when
# the image is built instead; the script's own copy then fails harmlessly.
ROOT_ONLY_STEP = re.compile(r"\blocale-gen\b")
DIFFICULTY = {"<15 min fix": "easy", "15 min - 1 hour": "medium", "1-4 hours": "hard",
              ">4 hours": "hard"}  # fmt: skip
INSTRUCTIONS = """{problem}

The repository is checked out in /testbed, the working directory, and its Python environment
is first on PATH. There is no network access. Resolve the issue above by changing the
repository's code; tests you cannot see will check the fix.
"""
DOCKER_HUB_IMAGE = re.compile(r"^(?P<repository>[a-z0-9]+(?:[._-][a-z0-9]+)*/[a-z0-9._-]+)"
                              r":(?P<tag>[\w][\w.-]{0,127})$")  # fmt: skip
MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])  # fmt: skip
REGISTRY_TIMEOUT_S = 30
REGISTRY_ATTEMPTS = 3

DigestLookup = Callable[[str], str]


class SwebenchImportError(Exception):
    """A row cannot become a task as it is."""


@dataclass(frozen=True)
class ImportedInstance:
    instance_id: str
    path: Path
    fail_to_pass: int
    pass_to_pass: int


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Rows by instance id, from a JSON list or a JSON Lines file."""
    try:
        text = path.read_text(encoding="utf-8")
        rows = json.loads(text) if path.suffix == ".json" else _json_lines(text)
    except OSError as exc:
        raise SwebenchImportError(f"{path}: cannot read file ({exc.strerror})") from exc
    except json.JSONDecodeError as exc:
        raise SwebenchImportError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SwebenchImportError(f"{path}: expected a list of rows, one object each")
    return {str(row.get("instance_id")): row for row in rows}


def _json_lines(text: str) -> list[Any]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def import_instance(
    row: dict[str, Any], tasks_dir: Path, digest_of: DigestLookup, tags: tuple[str, ...] = ()
) -> ImportedInstance:
    """Write tasks_dir/<id>/ from one row. Nothing is left behind on failure."""
    _check_row(row)
    task_id = task_id_for(row["instance_id"])
    target = tasks_dir / task_id
    if target.exists():
        raise SwebenchImportError(f"IMPORT_TASK_EXISTS: {target} already exists")
    tests = {
        "log_parser": row["log_parser"],
        "FAIL_TO_PASS": _test_list(row, "FAIL_TO_PASS"),
        "PASS_TO_PASS": _test_list(row, "PASS_TO_PASS"),
    }
    spec = task_spec(row, task_id, digest_of(row["image"]), tags)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tasks_dir, prefix=".import-") as scratch:
        draft = Path(scratch) / task_id
        (draft / "hidden").mkdir(parents=True)
        _write(draft / "hidden" / "eval.sh", with_test_exit_code(row["eval_script"]))
        _write(draft / "hidden" / "tests.json", json.dumps(tests, indent=2) + "\n")
        # Read as text, so the copy has the same bytes, and the task the same hash, on any OS.
        _write(draft / "hidden" / "check.py", Path(swebench_check.__file__).read_text("utf-8"))
        _write(draft / REFERENCE_PATCH, row["patch"])
        _write(draft / TASK_FILE, yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
        shutil.move(draft, target)
    return ImportedInstance(
        row["instance_id"], target, len(tests["FAIL_TO_PASS"]), len(tests["PASS_TO_PASS"])
    )


def task_id_for(instance_id: str) -> str:
    """django__django-11099 becomes django-django-11099."""
    task_id = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    if not re.fullmatch(SLUG_PATTERN, task_id):
        raise SwebenchImportError(f"cannot make a task id from {instance_id!r}")
    return task_id


def task_spec(
    row: dict[str, Any], task_id: str, digest: str, tags: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The task.yaml content: the row's image and problem, and its tests as the grader."""
    build = [line for line in row["eval_script"].splitlines() if ROOT_ONLY_STEP.search(line)]
    difficulty = DIFFICULTY.get(row.get("difficulty") or "")
    spec = {
        "id": task_id,
        "category": row["repo"],
        **({"difficulty": difficulty} if difficulty else {}),
        "tags": ["swe-bench", *tags],
        "instructions": INSTRUCTIONS.format(problem=_one_newline(row["problem_statement"]).strip()),
        "environment": {
            "base_image": f"{row['image']}@{digest}",
            "workdir": WORKDIR,
            "env": {"PATH": TESTBED_PATH},
            **({"build": build} if build else {}),
            "resources": {"cpu": 2, "memory_mb": 4096},
        },
        "budget": {"max_steps": 100, "timeout_s": 3600},
        "graders": [
            {"name": "swebench_tests", "type": "command", "run": GRADER_RUN, "timeout_s": 3600},
            {"name": "no_tamper", "type": "tamper_check", "protected_paths": ["**/conftest.py"],
             "weight": 0},
        ],
    }  # fmt: skip
    try:
        TaskSpec.model_validate(spec)
    except ValidationError as exc:
        raise SwebenchImportError(f"the task would not be valid: {exc}") from exc
    return spec


def with_test_exit_code(eval_script: str) -> str:
    """The eval script, also recording the test command's own exit status after the output
    it is graded on, as SWE-bench's harness does, so check.py can catch a log that claims
    passes the run did not have."""
    lines = eval_script.split("\n")
    for index, line in enumerate(lines):
        if swebench_check.END_TEST_OUTPUT in line:
            recorded = [
                f"{TEST_EXIT_CODE_VAR}=$?",
                line,
                f'echo "{swebench_check.TEST_EXIT_CODE}: ${TEST_EXIT_CODE_VAR}"',
            ]
            return "\n".join([*lines[:index], *recorded, *lines[index + 1 :]])
    raise SwebenchImportError("IMPORT_UNSUPPORTED_SCRIPT: the eval script marks no test output")


def registry_digest(image: str) -> str:
    """The digest a Docker Hub image's tag points to, from the registry, without pulling."""
    match = DOCKER_HUB_IMAGE.fullmatch(image)
    if not match:
        raise SwebenchImportError(f"IMPORT_IMAGE: {image!r} is not a Docker Hub image:tag")
    for attempt in range(1, REGISTRY_ATTEMPTS + 1):
        try:
            digest = _manifest_digest(match["repository"], match["tag"])
            break
        except OSError as exc:  # a timeout or a dropped connection may pass
            if attempt == REGISTRY_ATTEMPTS:
                raise SwebenchImportError(f"IMPORT_IMAGE: cannot look up {image}: {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise SwebenchImportError(f"IMPORT_IMAGE: cannot look up {image}: {exc}") from exc
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise SwebenchImportError(f"IMPORT_IMAGE: the registry gave no digest for {image}")
    return digest


def _manifest_digest(repository: str, tag: str) -> str:
    token_url = (
        "https://auth.docker.io/token?service=registry.docker.io"
        f"&scope=repository:{repository}:pull"
    )
    manifest = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}", method="HEAD"
    )
    manifest.add_header("Accept", MANIFEST_TYPES)
    with urllib.request.urlopen(token_url, timeout=REGISTRY_TIMEOUT_S) as response:
        manifest.add_header("Authorization", f"Bearer {json.load(response)['token']}")
    with urllib.request.urlopen(manifest, timeout=REGISTRY_TIMEOUT_S) as response:
        return response.headers.get("Docker-Content-Digest", "")


def _check_row(row: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_FIELDS if not row.get(name)]
    if missing:
        raise SwebenchImportError(
            f"IMPORT_BAD_ROW: {row.get('instance_id')!r} lacks {', '.join(missing)}; rows from "
            "SWE-bench 5 or later carry the image, eval script, and log parser"
        )
    if row["log_parser"] not in swebench_check.PARSERS:
        raise SwebenchImportError(
            f"IMPORT_UNSUPPORTED_PARSER: {row['log_parser']}; only Python repositories are "
            "supported"
        )
    if row["eval_type"] != "pass_and_fail":
        raise SwebenchImportError(f"IMPORT_UNSUPPORTED_PARSER: eval type {row['eval_type']}")


def _test_list(row: dict[str, Any], field: str) -> list[str]:
    """Test ids; rows from Hugging Face carry them as a JSON string, local ones as a list."""
    value = row.get(field) or []
    try:
        tests = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise SwebenchImportError(f"IMPORT_BAD_ROW: {field} is not valid JSON: {exc}") from exc
    if not isinstance(tests, list) or not all(isinstance(test, str) for test in tests):
        raise SwebenchImportError(f"IMPORT_BAD_ROW: {field} is not a list of test ids")
    return tests


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _one_newline(text: str) -> str:
    """Issue text from GitHub often ends lines with \r\n; the agent sees plain newlines."""
    return text.replace("\r\n", "\n")
