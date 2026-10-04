"""Class-wide analysis: run every cheap, offline check once and share the results.

``ClassAnalysis`` is computed lazily and cached on the instance, so the scan table, the
pair viewer and the agent's tools all reuse the same numbers (and the test suite is run
only once per class).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from codesentinel import moss
from codesentinel.ai_style import heuristic_score
from codesentinel.problems import Problem
from codesentinel.reference import reference_match
from codesentinel.same_bug import Signature, same_bug_for, same_bug_groups, wrong_answers


@dataclass
class StudentSummary:
    student: str
    top_classmate: str | None
    top_similarity: float
    same_bug_with: list[str]
    ref_match: float
    ref_name: str | None
    style_score: float
    reasons: list[str]

    @property
    def flagged(self) -> bool:
        return bool(self.reasons)


class ClassAnalysis:
    def __init__(self, problem: Problem):
        self.problem = problem

    @cached_property
    def pairs(self) -> list[dict]:
        return moss.check_class(self.problem)

    @cached_property
    def signatures(self) -> dict[str, Signature]:
        return wrong_answers(self.problem)

    @cached_property
    def groups(self) -> list[dict]:
        return same_bug_groups(self.signatures)

    def classmates(self, student: str, top: int = 3) -> list[dict]:
        return moss.matches_for(self.pairs, student, top=top)

    def same_bug(self, student: str) -> dict:
        return same_bug_for(self.signatures, student)

    def reference(self, student: str) -> dict:
        return reference_match(self.problem, self.problem.submissions[student])

    def style(self, student: str) -> dict:
        return heuristic_score(self.problem.submissions[student], self.problem.starter)

    def summary(self, student: str) -> StudentSummary:
        top = self.classmates(student, top=1)
        bug = self.same_bug(student)["same_wrong_output_as"]
        ref = self.reference(student)
        style = self.style(student)

        reasons = []
        if top and top[0]["similarity"] >= moss.FLAG_AT:
            reasons.append("copy")
        if bug:
            reasons.append("same bug")
        if ref["flag"]:
            reasons.append("AI reference")
        if style["flag"]:
            reasons.append("AI style")

        return StudentSummary(
            student=student,
            top_classmate=top[0]["classmate"] if top else None,
            top_similarity=top[0]["similarity"] if top else 0.0,
            same_bug_with=bug,
            ref_match=ref["best_match"],
            ref_name=ref["best_reference"],
            style_score=style["score"],
            reasons=reasons,
        )

    def summaries(self) -> list[StudentSummary]:
        return [self.summary(s) for s in self.problem.submissions]
