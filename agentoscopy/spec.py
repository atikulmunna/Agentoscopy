"""Task and agent config specs: schema validation, loading, and content hashing.

Only fields the harness acts on are accepted; unknown fields are rejected, not ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

TASK_FILE = "task.yaml"
SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
IGNORED_DIR_NAMES = frozenset({"__pycache__"})

SpecModel = TypeVar("SpecModel", bound=BaseModel)


class SpecError(Exception):
    """A task or agent config failed to load or validate. Messages include field paths."""


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Network(_Spec):
    allow: list[str] = Field(default_factory=list)

    @field_validator("allow")
    @classmethod
    def _deny_all_only(cls, allow: list[str]) -> list[str]:
        if allow:
            raise ValueError(
                "network allowlists are not supported yet: sandboxes have no network, so "
                "install what a task needs in environment.build, which runs with network"
            )
        return allow


class Resources(_Spec):
    cpu: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=1024, gt=0)


class Environment(_Spec):
    base_image: str
    workdir: str = "/workspace"
    build: list[str] = Field(default_factory=list)
    setup: list[str] = Field(default_factory=list)
    network: Network = Field(default_factory=Network)
    resources: Resources = Field(default_factory=Resources)

    @field_validator("base_image")
    @classmethod
    def _pinned_by_digest(cls, image: str) -> str:
        if "@sha256:" not in image:
            raise ValueError("must be pinned by digest, e.g. python:3.12-slim@sha256:<digest>")
        return image

    @field_validator("workdir")
    @classmethod
    def _absolute_without_spaces(cls, workdir: str) -> str:
        if not workdir.startswith("/") or any(char.isspace() for char in workdir):
            raise ValueError("must be an absolute path without whitespace")
        return workdir


class Budget(_Spec):
    max_steps: int = Field(default=50, gt=0)  # model calls through the gateway
    max_tokens: int = Field(default=2_000_000, gt=0)  # cumulative across calls
    max_cost_usd: float = Field(default=1.0, gt=0)
    timeout_s: int = Field(default=600, gt=0)  # excludes time spent in provider backoff


class GraderSpecBase(_Spec):
    """Fields every grader shares. `required` graders must pass; `weight` feeds the score."""

    name: str = Field(min_length=1)
    required: bool = True
    weight: float = Field(default=1.0, ge=0)


class CommandGraderSpec(GraderSpecBase):
    """Passes when a shell command exits 0 in the grading sandbox."""

    type: Literal["command"]
    run: str = Field(min_length=1)
    timeout_s: int = Field(default=300, gt=0)


class ForbiddenCommandSpec(GraderSpecBase):
    """Fails if any command the agent ran matches one of the regular expressions."""

    type: Literal["forbidden_command"]
    patterns: list[str] = Field(min_length=1)

    @field_validator("patterns")
    @classmethod
    def _compiles(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{pattern!r} is not a valid regular expression: {exc}") from exc
        return patterns


class MustReadBeforeEditSpec(GraderSpecBase):
    """Fails if the agent overwrote an existing file without reading it first."""

    type: Literal["must_read_before_edit"]


class MaxToolErrorsSpec(GraderSpecBase):
    """Fails if more than `max` tool calls failed (non-zero exit, timeout, or error)."""

    type: Literal["max_tool_errors"]
    max: int = Field(ge=0)


class MaxStepsSpec(GraderSpecBase):
    """Fails if the agent made more than `max` model calls."""

    type: Literal["max_steps"]
    max: int = Field(gt=0)


class TamperCheckSpec(GraderSpecBase):
    """Fails, and vetoes the whole trial, if the agent changed files it must not touch.

    Any change outside `allowed_paths` (default: the workdir and /tmp), or inside
    `protected_paths`, counts. Paths are globs over absolute container paths.
    """

    type: Literal["tamper_check"]
    allowed_paths: list[str] | None = None
    protected_paths: list[str] = Field(default_factory=list)


class LlmJudgeSpec(GraderSpecBase):
    """A pinned model applies the rubric to the agent's final state (FR-GRD-05)."""

    type: Literal["llm_judge"]
    rubric: str = Field(min_length=1)
    max_cost_usd: float = Field(default=0.25, gt=0)  # the judge's own budget, per trial
    timeout_s: int = Field(default=180, gt=0)


GraderSpec = Annotated[
    CommandGraderSpec
    | ForbiddenCommandSpec
    | MustReadBeforeEditSpec
    | MaxToolErrorsSpec
    | MaxStepsSpec
    | TamperCheckSpec
    | LlmJudgeSpec,
    Field(discriminator="type"),
]


class TaskSpec(_Spec):
    id: str = Field(pattern=SLUG_PATTERN)
    instructions: str = Field(min_length=1)
    category: str = "uncategorized"  # metadata for slicing results (FR-AGG-04)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    tags: list[str] = Field(default_factory=list)
    critical: bool = False  # a regression here fails a comparison outright (FR-TASK-09)
    environment: Environment
    budget: Budget = Field(default_factory=Budget)
    graders: list[GraderSpec] = Field(min_length=1)
    aggregation: Literal["all_required_pass"] = "all_required_pass"

    @model_validator(mode="after")
    def _graders_are_usable(self) -> TaskSpec:
        names = [grader.name for grader in self.graders]
        if len(names) != len(set(names)):
            raise ValueError("grader names must be unique")
        if not any(grader.required for grader in self.graders):
            raise ValueError("at least one grader must be required, otherwise every trial passes")
        return self

    def has_judges(self) -> bool:
        return any(grader.type == "llm_judge" for grader in self.graders)


class AgentConfig(_Spec):
    name: str = Field(min_length=1)
    adapter: Literal["python"]
    entrypoint: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    params: dict[str, Any] = Field(default_factory=dict)


class SuiteSpec(_Spec):
    name: str = Field(pattern=SLUG_PATTERN)
    tasks: list[str] = Field(min_length=1)

    @field_validator("tasks")
    @classmethod
    def _valid_unique_ids(cls, tasks: list[str]) -> list[str]:
        invalid = [task_id for task_id in tasks if not re.fullmatch(SLUG_PATTERN, task_id)]
        if invalid:
            raise ValueError(f"invalid task ids: {invalid}")
        if len(set(tasks)) != len(tasks):
            raise ValueError("task ids must be unique")
        return tasks


@dataclass(frozen=True)
class Task:
    spec: TaskSpec
    path: Path
    content_hash: str

    @property
    def fixtures_dir(self) -> Path:
        return self.path / "fixtures"

    @property
    def hidden_dir(self) -> Path:
        return self.path / "hidden"

    @property
    def reference_dir(self) -> Path:
        """Reference solution: files that overlay the workdir (WF-01 reference check)."""
        return self.path / "reference"

    def fixture_files(self) -> list[Path]:
        return _files_under(self.fixtures_dir)

    def hidden_files(self) -> list[Path]:
        return _files_under(self.hidden_dir)

    def reference_files(self) -> list[Path]:
        return _files_under(self.reference_dir)


def load_task(tasks_dir: Path, task_id: str) -> Task:
    _check_slug("task id", task_id)
    task_dir = tasks_dir / task_id
    task_file = task_dir / TASK_FILE
    spec = _validate(TaskSpec, _read_yaml(task_file), task_file)
    if spec.id != task_id:
        raise SpecError(f"{task_file}: id {spec.id!r} must match the directory name {task_id!r}")
    return Task(spec=spec, path=task_dir, content_hash=_task_hash(spec, task_dir))


def load_agent_config(path: Path) -> AgentConfig:
    return _validate(AgentConfig, _read_yaml(path), path)


def load_suite(suites_dir: Path, name: str) -> SuiteSpec:
    _check_slug("suite name", name)
    path = suites_dir / f"{name}.yaml"
    suite = _validate(SuiteSpec, _read_yaml(path), path)
    if suite.name != name:
        raise SpecError(f"{path}: name {suite.name!r} must match the file name {name!r}")
    return suite


def _check_slug(kind: str, value: str) -> None:
    if not re.fullmatch(SLUG_PATTERN, value):
        raise SpecError(f"invalid {kind} {value!r}: use lowercase letters, digits, and hyphens")


def config_hash(config: AgentConfig) -> str:
    return _sha256_json(config.model_dump(mode="json"))


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"{path}: cannot read file ({exc.strerror})") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a mapping at the top level")
    return data


def _validate(model: type[SpecModel], data: dict[str, Any], path: Path) -> SpecModel:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = [f"{_field_path(error['loc'])}: {error['msg']}" for error in exc.errors()]
        raise SpecError(f"{path}: invalid spec\n  " + "\n  ".join(problems)) from exc


def _field_path(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc) or "(root)"


def _task_hash(spec: TaskSpec, task_dir: Path) -> str:
    """SHA-256 over the normalised spec plus every fixture, hidden, and reference file."""
    # Defaults are left out, so adding an optional field later does not version every task.
    digest = hashlib.sha256(
        _sha256_json(spec.model_dump(mode="json", exclude_defaults=True)).encode()
    )
    for folder in ("fixtures", "hidden", "reference"):
        for file in _files_under(task_dir / folder):
            relative = file.relative_to(task_dir).as_posix()
            file_digest = hashlib.sha256(file.read_bytes()).hexdigest()
            digest.update(f"{relative}\0{file_digest}\n".encode())
    return digest.hexdigest()


def _files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files = (
        path
        for path in root.rglob("*")
        if path.is_file() and not IGNORED_DIR_NAMES.intersection(path.relative_to(root).parts)
    )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
