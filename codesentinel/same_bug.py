"""Same-bug check: "same mistake, same wrong output?"

Two people can independently write a correct solution. It is very unlikely that two
people independently make the *same* mistake and produce the *same* wrong output on
the same hidden tests. Candidates are grouped by their wrong-output signature.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from codesentinel.problems import Problem
from codesentinel.sandbox import run_code

Signature = tuple  # tuple of (test_index, test_input, wrong_output)


def wrong_answers(problem: Problem, max_workers: int = 8) -> dict[str, Signature]:
    """Map candidate -> signature of wrong outputs. All tests passing gives an empty tuple."""
    if not problem.tests:
        return {}

    def evaluate(item: tuple[str, str]) -> tuple[str, Signature]:
        student, code = item
        wrong = []
        for n, test in enumerate(problem.tests):
            got = run_code(code, test["input"])
            if got != test["output"]:
                wrong.append((n, test["input"].strip(), got))
        return student, tuple(wrong)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(evaluate, problem.submissions.items()))


def same_bug_groups(signatures: dict[str, Signature]) -> list[dict]:
    groups: dict[Signature, list[str]] = {}
    for student, signature in signatures.items():
        if signature:  # only candidates with wrong answers
            groups.setdefault(signature, []).append(student)

    return [
        {
            "students": students,
            "wrong_tests": [{"input": inp, "their_output": got} for _, inp, got in signature],
        }
        for signature, students in groups.items()
        if len(students) >= 2
    ]


def same_bug_for(signatures: dict[str, Signature], student: str) -> dict:
    """Who else made exactly the same mistakes as ``student``?"""
    if not signatures:
        return {"tests_available": False, "same_wrong_output_as": []}
    mine = signatures[student]
    if not mine:
        return {"tests_available": True, "passes_all_tests": True, "same_wrong_output_as": []}
    return {
        "tests_available": True,
        "passes_all_tests": False,
        "failed_tests": [{"input": inp, "output": got} for _, inp, got in mine],
        "same_wrong_output_as": [s for s, sig in signatures.items() if sig == mine and s != student],
    }
