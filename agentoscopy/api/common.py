"""Pieces shared by the API route modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiohttp import web

from agentoscopy.storage.store import Store

STORE = web.AppKey("store", Store)
HOME = web.AppKey("home", Path)
TOKEN = web.AppKey("token", str)
ALLOWED_HOSTS = web.AppKey("allowed_hosts", frozenset)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


def int_query(request: web.Request, name: str, default: int) -> int:
    try:
        value = int(request.query.get(name, default))
    except ValueError as exc:
        raise ApiError(400, "BAD_REQUEST", f"{name} must be an integer") from exc
    if value < 0:
        raise ApiError(400, "BAD_REQUEST", f"{name} must not be negative")
    return value


async def json_body(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise ApiError(400, "BAD_REQUEST", "the request body must be JSON") from exc
    if not isinstance(body, dict):
        raise ApiError(400, "BAD_REQUEST", "the request body must be a JSON object")
    return body
