"""`agentoscopy ci` end to end through the CLI, with fake sandboxes, and the GitHub comment
client against a local stand-in for the GitHub API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import yaml
from conftest import BASE_TASK

from agentoscopy.cli.main import main
from agentoscopy.github import (
    MARKER,
    GitHubError,
    PullRequest,
    pull_request_from_env,
    upsert_comment,
)
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import load_task
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend

TOKEN = "secret-token"
FIX = {"write": "app.py", "content": "fixed"}
AGENTS = {
    "fixer": [{"model_calls": 1}, FIX],
    "idler": [{"model_calls": 1}],
}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture(autouse=True)
def fake_sandboxes(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "agentoscopy.cli.run.DockerBackend", lambda: FakeBackend(grader=passes_when_fixed)
    )


@pytest.fixture
def project(tmp_path, write_task, tasks_dir):
    """Two validated tasks in a suite, and agent configs that fix or ignore the bug."""
    home = tmp_path / "home"
    store = Store(home / "agentoscopy.db")
    for task_id in ("task-a", "task-b"):
        write_task({**BASE_TASK, "id": task_id})
        store.save_task_version(load_task(tasks_dir, task_id), "fake-image", [], [])
    store.close()
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "smoke.yaml").write_text(
        yaml.safe_dump({"name": "smoke", "tasks": ["task-a", "task-b"]})
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    for name, script in AGENTS.items():
        config = {
            "name": name,
            "adapter": "python",
            "entrypoint": "agentoscopy.testing.scripted_agent:ScriptedAgent",
            "params": {"script": script},
        }
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(config))
    return tmp_path, home, tasks_dir, suites, configs


def ci(project, agent, *extra):
    root, home, tasks_dir, suites, configs = project
    return main(
        [
            "ci", "--suite", "smoke", "--agent", str(configs / f"{agent}.yaml"),
            "--baseline", "main", "--trials", "4", "--seed", "1",
            "--comment-file", str(root / "comment.md"),
            "--home", str(home), "--tasks-dir", str(tasks_dir), "--suites-dir", str(suites),
            *extra,
        ]
    )  # fmt: skip


def baseline_agent(project, agent="fixer"):
    return ["--baseline-agent", str(project[4] / f"{agent}.yaml")]


def runs(project):
    store = Store(project[1] / "agentoscopy.db")
    try:
        return store.list_runs()  # newest first
    finally:
        store.close()


def comment(project):
    return (project[0] / "comment.md").read_text(encoding="utf-8")


def test_without_a_baseline_one_is_run_first(project):
    assert ci(project, "fixer", *baseline_agent(project)) == 0

    candidate, baseline = runs(project)
    assert baseline["labels"] == {"branch": "main", "ci": "baseline"}
    assert candidate["labels"] == {"ci": "candidate"}
    assert {candidate["status"], baseline["status"]} == {"completed"}
    assert comment(project).startswith("## Agentoscopy: No significant change")


def test_a_recent_baseline_is_reused(project):
    ci(project, "fixer", *baseline_agent(project))

    assert ci(project, "fixer") == 0
    assert [run["labels"].get("ci") for run in runs(project)] == [
        "candidate",
        "candidate",
        "baseline",
    ]


def test_a_missing_baseline_without_a_baseline_agent_is_a_harness_error(project, capsys):
    assert ci(project, "fixer") == 2
    assert "BASELINE_MISSING" in capsys.readouterr().err
    assert runs(project) == []


def test_a_stale_baseline_is_not_used(project, capsys):
    ci(project, "fixer", *baseline_agent(project))

    assert ci(project, "fixer", "--max-baseline-age", "0s") == 2
    assert "BASELINE_MISSING" in capsys.readouterr().err


def test_a_regression_fails_the_check(project):
    assert ci(project, "idler", *baseline_agent(project)) == 1

    text = comment(project)
    assert text.startswith("## Agentoscopy: Regression")
    assert "| task-a | changed |" in text or "| task-a | regressed |" in text


def test_the_override_label_reports_a_regression_without_failing(project):
    assert ci(project, "idler", *baseline_agent(project), "--override") == 0

    assert "eval-override" in comment(project)
    assert runs(project)[0]["labels"]["override"] == "eval-override"


def test_a_candidate_that_hits_its_budget_reports_it(project):
    ci(project, "fixer", *baseline_agent(project))

    # The cap is a whole trial's max_cost_usd, so after one paid call nothing else fits.
    assert ci(project, "fixer", "--budget-usd", "1") == 3
    assert "hit its budget" in comment(project)


# GitHub comments ------------------------------------------------------------------------


class FakeGitHub(BaseHTTPRequestHandler):
    """Issue comments on pull request 7 of owner/repo, as the GitHub REST API serves them."""

    comments: list[dict] = []
    seen: list[str] = []

    def log_message(self, *args):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        self.seen.append(f"{self.command} {urlparse(self.path).path}")
        if self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._reply(401, {"message": "Bad credentials"})
        return False

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def do_GET(self):
        if not self._authorized():
            return
        query = parse_qs(urlparse(self.path).query)
        size, page = int(query["per_page"][0]), int(query["page"][0])
        self._reply(200, self.comments[(page - 1) * size : page * size])

    def do_POST(self):
        if not self._authorized():
            return
        number = 1 + sum(c["id"] < 1000 for c in self.comments)  # seeded ones start at 1000
        created = {"id": number, "body": self._body()["body"], "html_url": f"https://gh/c/{number}"}
        self.comments.append(created)
        self._reply(201, created)

    def do_PATCH(self):
        if not self._authorized():
            return
        comment_id = int(self.path.rsplit("/", 1)[1])
        target = next(c for c in self.comments if c["id"] == comment_id)
        target["body"] = self._body()["body"]
        self._reply(200, target)


@pytest.fixture
def github():
    FakeGitHub.comments, FakeGitHub.seen = [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_the_comment_is_added_once_then_updated(github):
    pull = PullRequest("owner/repo", 7, TOKEN, github)
    FakeGitHub.comments.extend(
        {"id": 1000 + i, "body": "unrelated", "html_url": "x"} for i in range(150)
    )

    first = upsert_comment(pull, "first verdict")
    second = upsert_comment(pull, "second verdict")

    ours = [c for c in FakeGitHub.comments if MARKER in c["body"]]
    assert first == second and len(ours) == 1
    assert ours[0]["body"] == f"{MARKER}\nsecond verdict"
    assert "PATCH /repos/owner/repo/issues/comments/1" in FakeGitHub.seen


def test_github_errors_are_reported_without_the_token(github):
    pull = PullRequest("owner/repo", 7, "wrong-token", github)

    with pytest.raises(GitHubError) as error:
        upsert_comment(pull, "verdict")

    assert "401" in str(error.value) and "wrong-token" not in str(error.value)


def test_the_pull_request_comes_from_the_actions_environment(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 12}}))
    env = {
        "GITHUB_TOKEN": TOKEN,
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_PATH": str(event),
    }

    assert pull_request_from_env(env) == PullRequest("owner/repo", 12, TOKEN)
    assert pull_request_from_env(env, number=3).number == 3
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        pull_request_from_env({**env, "GITHUB_TOKEN": ""})
    for bad in ("../evil", "owner/..", "owner/.", "own.er/repo", "owner"):
        with pytest.raises(GitHubError, match="owner/name"):
            pull_request_from_env({**env, "GITHUB_REPOSITORY": bad})
    assert pull_request_from_env({**env, "GITHUB_REPOSITORY": "my-org/my.repo_2"}).number == 12
    with pytest.raises(GitHubError, match="--pr"):
        pull_request_from_env({**env, "GITHUB_EVENT_PATH": str(tmp_path / "push.json")})


def test_ci_posts_its_comment_on_the_pull_request(project, github, tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7}}))
    for name, value in {
        "GITHUB_TOKEN": TOKEN,
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_API_URL": github,
    }.items():
        monkeypatch.setenv(name, value)

    assert ci(project, "idler", *baseline_agent(project), "--github") == 1

    (posted,) = FakeGitHub.comments
    assert posted["body"].startswith(f"{MARKER}\n## Agentoscopy: Regression")
