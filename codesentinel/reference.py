"""Reference match: "is this just the ChatGPT answer?"

Problem: if 20 candidates used the same chatbot their code will look alike, and
MOSS's common-code filter would discard it as "common".

Solution: ask several AI models for solutions (different models, several runs, some
temperature) and treat them as fake candidates. Every real candidate is then compared
against those references *without* the common-code filter: looking like the AI answer
is exactly the signal we want.
"""

from __future__ import annotations

import re
from typing import Any

from groq import BadRequestError

from codesentinel.config import (
    PROBLEMS_DIR,
    REFERENCE_FLAG_AT,
    REFERENCE_MODELS,
    REFERENCE_RUNS_PER_MODEL,
    REFERENCE_TEMPERATURE,
)
from codesentinel.llm import LLMClient
from codesentinel.moss import compare, fingerprint
from codesentinel.problems import Problem


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    code = max(blocks, key=len) if blocks else text
    return code.strip() + "\n"


def ask_for_solution(llm: LLMClient, prompt: str, model: str, temperature: float = REFERENCE_TEMPERATURE) -> str:
    for _ in range(3):
        try:
            response = llm.chat(
                [
                    {"role": "system", "content": "You have no tools. Answer in plain markdown."},
                    {"role": "user", "content": prompt},
                ],
                label="references",
                model=model,
                temperature=temperature,
            )
            return extract_code(response.choices[0].message.content)
        except BadRequestError:
            # Some models occasionally invent a tool call even without tools; Groq rejects it.
            continue
    raise RuntimeError(f"{model} kept failing")


def reference_prompt(problem: Problem) -> str:
    # What a candidate would copy-paste: the question plus the starter code.
    return (
        f"{problem.statement}\n\n"
        f"Starter code:\n```python\n{problem.starter}```\n\n"
        "Write the complete Python solution. Reply with the code in a single ```python block."
    )


def generate_references(llm: LLMClient, problem: Problem, force: bool = False, log=print) -> None:
    folder = PROBLEMS_DIR / problem.id / "references"
    folder.mkdir(parents=True, exist_ok=True)

    for model in REFERENCE_MODELS:
        short = model.split("/")[-1]
        for run in range(1, REFERENCE_RUNS_PER_MODEL + 1):
            path = folder / f"{short}_{run}.py"
            if path.exists() and not force:
                log(f"  cached   {path.name}")
                continue
            try:
                path.write_text(ask_for_solution(llm, reference_prompt(problem), model), encoding="utf-8")
                log(f"  wrote    {path.name}")
            except Exception as exc:  # keep going: one failing model should not stop the batch
                log(f"  failed   {path.name}: {exc}")


def reference_match(problem: Problem, code: str) -> dict[str, Any]:
    """Compare ``code`` with every AI reference solution of the problem."""
    if not problem.references:
        return {"available": False, "best_reference": None, "best_match": 0.0,
                "any_reference_coverage": 0.0, "matched_lines": [], "flag": False}

    mine = fingerprint(code)
    ignore = fingerprint(problem.starter).hashes  # only the starter code is ignored here

    best: dict[str, Any] = {"reference": None, "score": 0.0, "lines": []}
    union: set[int] = set()
    for name, ref_code in problem.references.items():
        ref = fingerprint(ref_code)
        result = compare(mine, ref, ignore)
        union |= (mine.hashes - ignore) & ref.hashes
        # pct_a = share of *my* code that appears in this AI answer
        if result["pct_a"] > best["score"]:
            best = {"reference": name, "score": result["pct_a"], "lines": result["lines_a"]}

    total = len(mine.hashes - ignore)
    return {
        "available": True,
        "best_reference": best["reference"],
        "best_match": round(best["score"], 2),
        "any_reference_coverage": round(len(union) / total, 2) if total else 0.0,
        "matched_lines": best["lines"],
        "flag": best["score"] >= REFERENCE_FLAG_AT,
    }
