"""What people record about trials: failure tags (FR-FA-01) and human review (WF-09)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from agentoscopy.storage.database import Database, timestamp

REVIEW_PRIORITY = "CASE sample_source WHEN 'low_confidence' THEN 0 WHEN 'random' THEN 1 ELSE 2 END"


class ReviewRecords(Database):
    # Failure tags -------------------------------------------------------------------------

    def add_failure_tag(
        self, trial_id: str, tag: str, source: str, note: str | None = None
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO failure_tags (trial_id, tag, source, note) VALUES (?, ?, ?, ?)",
            (trial_id, tag, source, note),
        )

    def tag_by_hand(self, trial_id: str, tag: str, note: str | None) -> None:
        """A reviewer's tag (FR-FA-01). Tagging again updates the note; an automatic tag of
        the same name keeps its source."""
        self._conn.execute(
            "INSERT INTO failure_tags (trial_id, tag, source, note) VALUES (?, ?, 'human', ?) "
            "ON CONFLICT (trial_id, tag) DO UPDATE SET note = excluded.note",
            (trial_id, tag, note),
        )

    def failure_tag(self, trial_id: str, tag: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT tag, source, note FROM failure_tags WHERE trial_id = ? AND tag = ?",
            (trial_id, tag),
        ).fetchone()
        return dict(row) if row else None

    def remove_failure_tag(self, trial_id: str, tag: str) -> None:
        self._conn.execute(
            "DELETE FROM failure_tags WHERE trial_id = ? AND tag = ?", (trial_id, tag)
        )

    def failure_tag_counts(self, run_id: str) -> dict[str, int]:
        """How many of the run's trials carry each tag, most common first."""
        rows = self._conn.execute(
            "SELECT f.tag, COUNT(*) AS trials FROM failure_tags f JOIN trials t USING (trial_id) "
            "WHERE t.run_id = ? GROUP BY f.tag ORDER BY trials DESC, f.tag",
            (run_id,),
        )
        return {row["tag"]: row["trials"] for row in rows}

    # Human review (WF-09) ------------------------------------------------------------------

    def queue_review(
        self, trial_id: str, attempt: int, grader_name: str, source: str
    ) -> str | None:
        """Queue a review item; returns its id, or None if that item is already queued."""
        review_id = str(uuid.uuid4())
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO reviews (review_id, trial_id, attempt, grader_name, "
            "sample_source, queued_at) VALUES (?, ?, ?, ?, ?, ?)",
            (review_id, trial_id, attempt, grader_name, source, timestamp()),
        )
        return review_id if cursor.rowcount == 1 else None

    def review_queue(self, limit: int) -> list[dict[str, Any]]:
        """Pending items: low-confidence judge verdicts first, then random samples (FR-REV-01)."""
        rows = self._conn.execute(
            "SELECT r.*, t.run_id, t.task_id, t.task_version FROM reviews r "
            "JOIN trials t USING (trial_id) WHERE r.reviewed_at IS NULL "
            f"ORDER BY {REVIEW_PRIORITY}, r.queued_at, r.rowid LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def pending_reviews(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE reviewed_at IS NULL"
        ).fetchone()[0]

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT r.*, t.run_id, t.task_id, t.task_version FROM reviews r "
            "JOIN trials t USING (trial_id) WHERE r.review_id = ?",
            (review_id,),
        ).fetchone()
        return dict(row) if row else None

    def submit_review(
        self,
        review_id: str,
        *,
        reviewer: str,
        passed: bool,
        score: float | None,
        note: str | None,
        override: bool,
    ) -> bool:
        """Record a review once. An override replaces the trial's outcome with the reviewer's,
        keeps the original, and drops the run's stored summary so it is recomputed (FR-REV-06)."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE reviews SET reviewer = ?, passed = ?, score = ?, note = ?, override = ?, "
                "reviewed_at = ? WHERE review_id = ? AND reviewed_at IS NULL",
                (reviewer, passed, score, note, override, timestamp(), review_id),
            )
            if cursor.rowcount != 1:
                return False
            if override:
                trial_id, run_id = conn.execute(
                    "SELECT t.trial_id, t.run_id FROM reviews r JOIN trials t USING (trial_id) "
                    "WHERE r.review_id = ?",
                    (review_id,),
                ).fetchone()
                conn.execute(
                    "UPDATE trials SET original_outcome = COALESCE(original_outcome, outcome), "
                    "outcome = ? WHERE trial_id = ? AND outcome IN ('pass', 'fail')",
                    ("pass" if passed else "fail", trial_id),
                )
                conn.execute("DELETE FROM run_summaries WHERE run_id = ?", (run_id,))
            return True

    def grade_for(self, trial_id: str, attempt: int, grader_name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT grader_name, kind, score, passed, rationale, metadata FROM grades "
            "WHERE trial_id = ? AND attempt = ? AND grader_name = ?",
            (trial_id, attempt, grader_name),
        ).fetchone()
        if row is None:
            return None
        return {**dict(row), "passed": bool(row["passed"]), "metadata": json.loads(row["metadata"])}

    def calibration_rows(self) -> list[dict[str, Any]]:
        """Every finished review of a judge grader, paired with the judge's own verdict."""
        rows = self._conn.execute(
            "SELECT r.review_id, r.trial_id, r.grader_name, r.sample_source, "
            "r.passed AS human_passed, "
            "g.passed AS judge_passed, g.metadata, t.task_id FROM reviews r "
            "JOIN grades g ON g.trial_id = r.trial_id AND g.attempt = r.attempt "
            "AND g.grader_name = r.grader_name "
            "JOIN trials t ON t.trial_id = r.trial_id "
            "WHERE r.reviewed_at IS NOT NULL AND r.grader_name != '' AND g.kind = 'judge' "
            "ORDER BY r.reviewed_at",
        )
        return [
            {
                "review_id": row["review_id"],
                "trial_id": row["trial_id"],
                "task_id": row["task_id"],
                "grader_name": row["grader_name"],
                "judge_model": json.loads(row["metadata"]).get("judge_model"),
                "sample_source": row["sample_source"],
                "human_passed": bool(row["human_passed"]),
                "judge_passed": bool(row["judge_passed"]),
            }
            for row in rows
        ]
