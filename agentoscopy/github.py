"""Pull request comments for `agentoscopy ci` (FR-CI-03), through the GitHub REST API.

The comment is sticky: a hidden marker identifies it, so later pushes update one comment
instead of adding another. Inside GitHub Actions everything needed comes from the environment
the runner sets up (GITHUB_TOKEN must be passed in by the workflow).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER = "<!-- agentoscopy-ci -->"
DEFAULT_API_URL = "https://api.github.com"
PAGE_SIZE = 100
MAX_PAGES = 30
TIMEOUT_S = 30
# Owners are letters, digits, and hyphens; names may also hold dots and underscores.
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9-]+/(?!\.{1,2}$)[A-Za-z0-9_.-]+$")


class GitHubError(Exception):
    """The comment could not be posted."""


@dataclass(frozen=True)
class PullRequest:
    repository: str  # owner/name
    number: int
    token: str
    api_url: str = DEFAULT_API_URL


def pull_request_from_env(env: Mapping[str, str], number: int | None = None) -> PullRequest:
    """The pull request a GitHub Actions job runs for, unless `number` names another one."""
    token, repository = env.get("GITHUB_TOKEN"), env.get("GITHUB_REPOSITORY", "")
    if not token:
        raise GitHubError("GITHUB_TOKEN is not set; pass it to the step that runs agentoscopy ci")
    if not REPOSITORY_PATTERN.match(repository):
        raise GitHubError("GITHUB_REPOSITORY must name the repository as owner/name")
    if number is None:
        number = _event_pull_request(env.get("GITHUB_EVENT_PATH"))
    return PullRequest(repository, number, token, env.get("GITHUB_API_URL") or DEFAULT_API_URL)


def upsert_comment(pull: PullRequest, body: str) -> str:
    """Update this tool's earlier comment on the pull request, or add one. Returns its URL."""
    body = f"{MARKER}\n{body}"
    existing = _find_comment(pull)
    if existing is not None:
        path = f"/repos/{pull.repository}/issues/comments/{existing}"
        return _request(pull, "PATCH", path, {"body": body})["html_url"]
    path = f"/repos/{pull.repository}/issues/{pull.number}/comments"
    return _request(pull, "POST", path, {"body": body})["html_url"]


def _find_comment(pull: PullRequest) -> int | None:
    for page in range(1, MAX_PAGES + 1):
        path = (
            f"/repos/{pull.repository}/issues/{pull.number}/comments"
            f"?per_page={PAGE_SIZE}&page={page}"
        )
        comments = _request(pull, "GET", path)
        for comment in comments:
            if MARKER in (comment.get("body") or ""):
                return int(comment["id"])
        if len(comments) < PAGE_SIZE:
            return None
    return None


def _request(pull: PullRequest, method: str, path: str, payload: Any = None) -> Any:
    request = urllib.request.Request(
        pull.api_url.rstrip("/") + path,
        method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {pull.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agentoscopy",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise GitHubError(f"GitHub answered {exc.code} to {method} {path.split('?')[0]}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GitHubError(f"cannot reach GitHub: {exc}") from None


def _event_pull_request(event_path: str | None) -> int:
    try:
        event = json.loads(Path(event_path or "").read_text(encoding="utf-8"))
        return int(event["pull_request"]["number"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GitHubError(
            "cannot tell which pull request this is: not a pull_request event; pass --pr"
        ) from exc
