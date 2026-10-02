"""Human review queue and judge calibration endpoints (WF-09)."""

from __future__ import annotations

from typing import Any

from aiohttp import web

from agentoscopy.api.common import HOME, STORE, ApiError, int_query, json_body
from agentoscopy.reporting import NotFound, calibration, review_item
from agentoscopy.stats.calibration import KAPPA_THRESHOLD, MIN_REVIEWS

DEFAULT_QUEUE = 10
MAX_QUEUE = 100
MAX_NOTE = 2000
MAX_REVIEWER = 100
MAX_ID = 200
DEFAULT_REVIEWER = "local"


def add_routes(app: web.Application) -> None:
    app.router.add_get("/review/queue", _queue)
    app.router.add_post("/review/queue", _enqueue)
    app.router.add_post("/review/{review_id}", _submit)
    app.router.add_get("/calibration", _calibration)


async def _queue(request: web.Request) -> web.Response:
    """Pending items in priority order, without the judge's verdict (FR-REV-01, FR-REV-02)."""
    store, home = request.app[STORE], request.app[HOME]
    limit = min(int_query(request, "limit", DEFAULT_QUEUE), MAX_QUEUE)
    items = [review_item(store, home, review) for review in store.review_queue(limit)]
    return web.json_response({"pending": store.pending_reviews(), "items": items})


async def _enqueue(request: web.Request) -> web.Response:
    """Add a finished trial to the queue by hand, as a trial audit or for one judge grader."""
    body = await json_body(request)
    trial_id = _text(body, "trial_id", MAX_ID)
    grader_name = _text(body, "grader_name", MAX_ID) or ""
    if not trial_id:
        raise ApiError(400, "BAD_REQUEST", "trial_id is required")
    store = request.app[STORE]
    trial = store.get_trial(trial_id)
    if trial is None:
        raise NotFound(f"no trial {trial_id}")
    if trial["outcome"] not in ("pass", "fail"):
        raise ApiError(422, "NOT_REVIEWABLE", f"trial outcome is {trial['outcome']}")
    if grader_name:
        grade = store.grade_for(trial_id, trial["attempt"], grader_name)
        if grade is None or grade["kind"] != "judge":
            raise ApiError(422, "NOT_A_JUDGE_GRADER", f"{grader_name!r} graded no judge verdict")
    review_id = store.queue_review(trial_id, trial["attempt"], grader_name, "manual")
    if review_id is None:
        raise ApiError(409, "ALREADY_QUEUED", "this item is already in the manual queue")
    return web.json_response({"review_id": review_id}, status=201)


async def _submit(request: web.Request) -> web.Response:
    """Record a review, then reveal the judge's verdict and the trial outcome (WF-09 step 4)."""
    body = await json_body(request)
    passed, override = body.get("passed"), body.get("override", False)
    if not isinstance(passed, bool) or not isinstance(override, bool):
        raise ApiError(400, "BAD_REQUEST", "passed (and override, if given) must be true or false")
    store = request.app[STORE]
    review = store.get_review(request.match_info["review_id"])
    if review is None:
        raise NotFound(f"no review {request.match_info['review_id']}")
    recorded = store.submit_review(
        review["review_id"],
        reviewer=_text(body, "reviewer", MAX_REVIEWER) or DEFAULT_REVIEWER,
        passed=passed,
        score=_score(body),
        note=_text(body, "note", MAX_NOTE),
        override=override,
    )
    if not recorded:
        raise ApiError(409, "ALREADY_REVIEWED", "this item has already been reviewed")
    trial = store.get_trial(review["trial_id"]) or {}
    grader = review["grader_name"]
    return web.json_response(
        {
            "review_id": review["review_id"],
            "judge": store.grade_for(review["trial_id"], review["attempt"], grader)
            if grader
            else None,
            "outcome": trial.get("outcome"),
            "original_outcome": trial.get("original_outcome"),
        }
    )


async def _calibration(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "kappa_threshold": KAPPA_THRESHOLD,
            "min_reviews": MIN_REVIEWS,
            "judges": calibration(request.app[STORE]),
        }
    )


def _text(body: dict[str, Any], name: str, max_length: int) -> str | None:
    value = body.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > max_length:
        raise ApiError(400, "BAD_REQUEST", f"{name} must be a string of at most {max_length}")
    return value.strip() or None


def _score(body: dict[str, Any]) -> float | None:
    score = body.get("score")
    if score is None:
        return None
    if isinstance(score, bool) or not isinstance(score, int | float) or not 0 <= score <= 1:
        raise ApiError(400, "BAD_REQUEST", "score must be a number from 0 to 1")
    return float(score)
