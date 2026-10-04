"""Problem loader.

Layout of ``data/problems/<problem_id>/``::

    problem.md         the question
    starter.py         base file handed to every candidate (ignored when comparing)
    tests.json         optional hidden tests: [{"input": "...", "output": "..."}]
    submissions/*.py   candidate code (file stem = candidate name)
    references/*.py    AI-generated reference solutions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from codesentinel.config import PROBLEMS_DIR


@dataclass(frozen=True)
class Problem:
    id: str
    statement: str
    starter: str
    submissions: dict[str, str] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)
    tests: list[dict] = field(default_factory=list)

    def with_submission(self, name: str, code: str) -> Problem:
        """Return a copy that includes one extra (e.g. uploaded) submission."""
        return replace(self, submissions={**self.submissions, name: code})


def _read_py_folder(folder: Path) -> dict[str, str]:
    if not folder.exists():
        return {}
    return {f.stem: f.read_text(encoding="utf-8") for f in sorted(folder.glob("*.py"))}


def list_problems(problems_dir: Path = PROBLEMS_DIR, with_submissions: bool = False) -> list[str]:
    ids = sorted(p.name for p in problems_dir.iterdir() if p.is_dir())
    if with_submissions:
        ids = [i for i in ids if any((problems_dir / i / "submissions").glob("*.py"))]
    return ids


def load_problem(problem_id: str, problems_dir: Path = PROBLEMS_DIR) -> Problem:
    folder = problems_dir / problem_id
    if not folder.is_dir():
        raise ValueError(f"Unknown problem '{problem_id}'. Options: {list_problems(problems_dir)}")

    tests_file = folder / "tests.json"
    return Problem(
        id=problem_id,
        statement=(folder / "problem.md").read_text(encoding="utf-8"),
        starter=(folder / "starter.py").read_text(encoding="utf-8"),
        submissions=_read_py_folder(folder / "submissions"),
        references=_read_py_folder(folder / "references"),
        tests=json.loads(tests_file.read_text(encoding="utf-8")) if tests_file.exists() else [],
    )
