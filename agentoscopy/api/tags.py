"""Tagging failed trials by hand (FR-FA-01), from a short taxonomy or free-form."""

from __future__ import annotations

import re

from aiohttp import web

from agentoscopy.api.common import STORE, ApiError, json_body
from agentoscopy.reporting import NotFound

TAXONOMY = {
    "wrong_fix": "Changed the code, but the change does not solve the problem",
    "incomplete": "Solved part of the problem",
    "broke_existing_behavior": "Broke something that used to work",
    "misread_task": "Solved a different problem from the one asked",
    "tool_misuse": "Used its tools wrongly or wastefully",
    "looped": "Repeated the same actions without progress",
    "gave_up": "Stopped without a real attempt",
    "reward_hacking": "Gamed the graders instead of doing the task",
    "environment_issue": "Failed because of the task or its environment, not the agent",
}
TAG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
MAX_NOTE = 500


def add_routes(app: web.Application) -> None:
    app.router.add_get("/failure-tags", _taxonomy)
    app.router.add_post("/trials/{trial_id}/tags", _add)
    app.router.add_delete("/trials/{trial_id}/tags/{tag}", _remove)


def normalize(tag: str) -> str:
    """'Wrong fix' and 'wrong-fix' both become wrong_fix."""
    return re.sub(r"[\s-]+", "_", tag.strip().lower())


async def _taxonomy(request: web.Request) -> web.Response:
    tags = [{"tag": tag, "description": text} for tag, text in TAXONOMY.items()]
    return web.json_response({"taxonomy": tags})


async def _add(request: web.Request) -> web.Response:
    store = request.app[STORE]
    trial = _failed_trial(request)
    body = await json_body(request)
    tag = normalize(body["tag"]) if isinstance(body.get("tag"), str) else ""
    if not TAG_PATTERN.match(tag):
        raise ApiError(
            400,
            "BAD_REQUEST",
            "tag must be 2 to 40 letters, digits, or underscores, starting with a letter",
        )
    note = body.get("note")
    if note is not None and not (isinstance(note, str) and len(note) <= MAX_NOTE):
        raise ApiError(400, "BAD_REQUEST", f"note must be text of at most {MAX_NOTE} characters")
    store.tag_by_hand(trial["trial_id"], tag, (note or "").strip() or None)
    return web.json_response({"failure_tags": _tags(request)}, status=201)


async def _remove(request: web.Request) -> web.Response:
    store = request.app[STORE]
    trial = _failed_trial(request)
    tag = store.failure_tag(trial["trial_id"], request.match_info["tag"])
    if tag is None:
        raise NotFound(f"trial has no tag {request.match_info['tag']}")
    if tag["source"] != "human":
        raise ApiError(
            409, "AUTOMATIC_TAG", "tags the harness added stay, as evidence of what it found"
        )
    store.remove_failure_tag(trial["trial_id"], tag["tag"])
    return web.json_response({"failure_tags": _tags(request)})


def _failed_trial(request: web.Request) -> dict:
    trial = request.app[STORE].get_trial(request.match_info["trial_id"])
    if trial is None:
        raise NotFound(f"no trial {request.match_info['trial_id']}")
    if trial["outcome"] != "fail":
        raise ApiError(
            422, "NOT_FAILED", f"only failed trials are tagged; this one is {trial['outcome']}"
        )
    return trial


def _tags(request: web.Request) -> list[dict]:
    return request.app[STORE].get_trial(request.match_info["trial_id"])["failure_tags"]
