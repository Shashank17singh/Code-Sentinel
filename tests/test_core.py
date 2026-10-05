import os

import pytest

from codesentinel import moss
from codesentinel.ai_style import heuristic_score
from codesentinel.analysis import ClassAnalysis
from codesentinel.problems import list_problems, load_problem
from codesentinel.reference import extract_code, reference_match
from codesentinel.sandbox import run_code
from codesentinel.tokenizer import normalize, tokenize


def test_normalisation_ignores_names_and_comments():
    a = "sum += arr[i]   # adding"
    b = "total += nums[j]"
    assert normalize(a) == normalize(b) == "ID += ID [ ID ]"


def test_builtins_kept_only_when_called():
    assert normalize("len(x)") == "len ( ID )"
    assert normalize("len = 3") == "ID = NUM"


def test_string_prefix_is_a_string_not_identifier():
    assert [t.kind for t in tokenize('f"hi {x}"')] == ["STR"]


def test_winnow_guarantee_on_identical_code():
    code = "for i in range(10):\n    total += i * 2\nprint(total)\n"
    fp = moss.fingerprint(code)
    assert moss.compare(fp, fp)["score"] == 1.0


def test_renaming_does_not_hide_a_copy():
    a = "def f(s):\n    seen = {}\n    best = 0\n    for i, ch in enumerate(s):\n        best = max(best, i)\n    return best\n"
    b = a.replace("seen", "d").replace("best", "ans").replace("ch", "c")
    assert moss.compare(moss.fingerprint(a), moss.fingerprint(b))["score"] == 1.0


def test_known_copy_pair_is_top_of_longest_substring():
    pairs = moss.check_class(load_problem("longest_substring"))
    assert {pairs[0]["a"], pairs[0]["b"]} <= {"alice", "bob", "eshan", "ishaan"}
    assert pairs[0]["score"] >= moss.FLAG_AT


def test_pair_between_orients_lines():
    problem = load_problem("longest_substring")
    pairs = moss.check_class(problem)
    p = moss.pair_between(pairs, pairs[0]["b"], pairs[0]["a"])
    assert p["a"] == pairs[0]["b"] and p["lines_a"] == pairs[0]["lines_b"]


def test_heuristic_flags_ai_like_code_and_not_messy_code():
    ai_like = (
        "def length_of_longest_substring(text: str) -> int:\n"
        '    """Return the length of the longest substring without repeats."""\n'
        "    if not text:\n        return 0\n"
        "    last_seen = {}\n    start = 0\n    longest = 0\n"
        "    for index, char in enumerate(text):\n"
        "        if char in last_seen and last_seen[char] >= start:\n            start = last_seen[char] + 1\n"
        "        last_seen[char] = index\n        longest = max(longest, index - start + 1)\n"
        '    return longest\n\n\nif __name__ == "__main__":\n    print(length_of_longest_substring(input()))\n'
    )
    messy = "s=input()\nl=0\nr=0\nm=0\nd={}\n#print(d)\nfor c in s:\n  r+=1\n  m=max(m,r-l)\nprint(m)\n"
    assert heuristic_score(ai_like)["flag"] is True
    assert heuristic_score(messy)["flag"] is False


def test_heuristic_handles_syntax_errors():
    assert heuristic_score("def (:")["flag"] is False


def test_extract_code_picks_largest_block_and_drops_think():
    text = "<think>x</think>```python\nprint(1)\n```\ntext\n```python\nprint(1)\nprint(2)\n```"
    assert extract_code(text) == "print(1)\nprint(2)\n"


def test_sandbox_runs_and_limits():
    assert run_code("print(int(input()) * 2)", "21\n") == "42"
    assert run_code("raise SystemExit(1)") == "ERROR"
    assert run_code("while True: pass", timeout=1) == "TIMEOUT"


def test_sandbox_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    out = run_code("import os; print(os.environ.get('GROQ_API_KEY'))")
    assert out == "None"


@pytest.mark.skipif(os.name != "posix", reason="memory limits are POSIX-only")
def test_sandbox_memory_limit():
    assert run_code("x = bytearray(2 * 1024**3)") == "ERROR"


def test_class_analysis_known_findings():
    analysis = ClassAnalysis(load_problem("longest_substring"))
    assert analysis.groups and {"alice", "bob"} <= set(analysis.groups[0]["students"])
    assert analysis.same_bug("karan")["passes_all_tests"] is True
    summaries = {s.student: s for s in analysis.summaries()}
    assert summaries["alice"].flagged and not summaries["karan"].flagged
    assert "AI reference" in summaries["farah"].reasons


def test_reference_match_without_references_is_graceful():
    problem = load_problem("longest_substring")
    problem = type(problem)(problem.id, problem.statement, problem.starter)
    assert reference_match(problem, "print(1)")["available"] is False


def test_problem_listing():
    assert {"longest_substring", "max_subarray"} <= set(list_problems(with_submissions=True))
    with pytest.raises(ValueError):
        load_problem("nope")
