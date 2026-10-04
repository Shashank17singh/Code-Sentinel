"""Human-in-the-loop decision log.

The system flags, a human decides. Every decision is appended to a JSON file together
with the agent report it was based on, giving an audit trail.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from codesentinel.config import REVIEWS_FILE

DECISIONS = {"c": "cleared", "f": "follow-up interview", "e": "escalated"}


def load_reviews(path: Path | None = None) -> list[dict]:
    path = path or REVIEWS_FILE  # resolved at call time so it can be overridden
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_review(
    problem_id: str,
    candidate: str,
    report: str,
    decision: str,
    note: str = "",
    reviewer: str = "",
    path: Path | None = None,
) -> dict:
    path = path or REVIEWS_FILE
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "problem": problem_id,
        "candidate": candidate,
        "reviewer": reviewer,
        "agent_report": report,
        "decision": decision,
        "note": note,
    }
    reviews = load_reviews(path)
    reviews.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: never leave a half-written file behind.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(reviews, fh, indent=2)
    os.replace(tmp, path)
    return record
