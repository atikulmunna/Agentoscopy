"""Results database (§14.3, SQLite): task versions, suites, configs, runs, trials, grades,
reviews, and failure tags.

`Store` is assembled from one part per concern, each in its own module and all sharing one
connection, so callers keep a single object whichever part they use.
"""

from __future__ import annotations

from pathlib import Path

from agentoscopy.storage.maintenance import Maintenance
from agentoscopy.storage.records import (
    IN_FLIGHT_STATES,
    Claim,
    ClaimedTrial,
    NewTrial,
    RunSettings,
    TaskVersion,
)
from agentoscopy.storage.reports import RunReports
from agentoscopy.storage.reviews import ReviewRecords
from agentoscopy.storage.runs import TOPPED_UP, RunRecords
from agentoscopy.storage.schema import SCHEMA_VERSION, StoreError, open_database
from agentoscopy.storage.trial_queue import LEASE_S, WORKER_LOST, TrialQueue

__all__ = [
    "IN_FLIGHT_STATES",
    "LEASE_S",
    "SCHEMA_VERSION",
    "TOPPED_UP",
    "WORKER_LOST",
    "Claim",
    "ClaimedTrial",
    "NewTrial",
    "RunSettings",
    "Store",
    "StoreError",
    "TaskVersion",
]


class Store(RunRecords, TrialQueue, RunReports, ReviewRecords, Maintenance):
    def __init__(self, path: Path) -> None:
        super().__init__(open_database(path))
