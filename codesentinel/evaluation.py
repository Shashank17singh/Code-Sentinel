"""Evaluation harness: "would a clean human coder get flagged?"

Building a detector is easy. *Measuring how wrong it is* is the real work.

Dataset (the label is the folder name)::

    data/evals/dataset/ai/            written by an AI         (positive)
    data/evals/dataset/human_clean/   human, tidy code         (negative)
    data/evals/dataset/human_messy/   human, messy code        (negative)

File names look like ``<problem_id>__<anything>.py``; the problem id selects the
reference solutions. Drop your own file into ``human_clean/`` to see whether you get flagged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from codesentinel.ai_style import heuristic_score, llm_judge
from codesentinel.config import EVALS_DIR
from codesentinel.llm import LLMClient
from codesentinel.problems import Problem, load_problem
from codesentinel.reference import reference_match

DATASET = EVALS_DIR / "dataset"
LLM_CACHE = EVALS_DIR / "llm_cache.json"
LABELS = ["ai", "human_clean", "human_messy"]


def load_dataset(dataset: Path = DATASET) -> list[dict]:
    samples = []
    for label in LABELS:
        for f in sorted((dataset / label).glob("*.py")):
            samples.append(
                {
                    "name": f"{label}/{f.stem}",
                    "label": label,
                    "is_ai": label == "ai",
                    "problem": f.stem.split("__")[0],
                    "code": f.read_text(encoding="utf-8"),
                }
            )
    return samples


def _load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cached_llm_judge(llm: LLMClient, sample: dict, cache: dict, cache_path: Path = LLM_CACHE) -> dict:
    key = hashlib.sha1(sample["code"].encode()).hexdigest()
    if key not in cache:
        problem = load_problem(sample["problem"])
        cache[key] = llm_judge(llm, sample["code"], problem.statement)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[key]


def run_detectors(
    samples: list[dict],
    llm: LLMClient | None = None,
    use_cached_llm: bool = False,
    cache_path: Path = LLM_CACHE,
) -> list[dict]:
    """Score every sample.

    With ``llm`` the judge runs on cache misses (and costs tokens); with
    ``use_cached_llm=True`` only already-cached judgements are used (free).
    """
    cache = _load_cache(cache_path)
    problems: dict[str, Problem] = {}

    for s in samples:
        problem = problems.setdefault(s["problem"], load_problem(s["problem"]))
        h = heuristic_score(s["code"], problem.starter)
        r = reference_match(problem, s["code"])
        s["scores"] = {"heuristic": h["score"], "reference": r["best_match"]}
        s["flags"] = {"heuristic": h["flag"], "reference": r["flag"]}

        verdict = None
        if llm is not None:
            verdict = cached_llm_judge(llm, s, cache, cache_path)
        elif use_cached_llm:
            verdict = cache.get(hashlib.sha1(s["code"].encode()).hexdigest())
        if verdict is not None:
            s["scores"]["llm_judge"] = verdict["ai_likelihood"]
            s["flags"]["llm_judge"] = verdict["flag"]

    for s in samples:
        # Combined detector: at least two independent detectors must agree.
        s["flags"]["combined (2+ agree)"] = sum(s["flags"].values()) >= 2
    return samples


def metrics(samples: list[dict], detector: str) -> dict:
    scored = [s for s in samples if detector in s["flags"]]
    tp = sum(1 for s in scored if s["is_ai"] and s["flags"][detector])
    fn = sum(1 for s in scored if s["is_ai"] and not s["flags"][detector])
    fp = sum(1 for s in scored if not s["is_ai"] and s["flags"][detector])
    tn = sum(1 for s in scored if not s["is_ai"] and not s["flags"][detector])

    def fp_in(label: str) -> tuple[int, int]:
        group = [s for s in scored if s["label"] == label]
        return sum(1 for s in group if s["flags"][detector]), len(group)

    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,  # of those flagged, how many really are AI
        "recall": tp / (tp + fn) if tp + fn else 0.0,     # of all AI samples, how many were caught
        "fpr": fp / (fp + tn) if fp + tn else 0.0,        # of all humans, how many were wrongly flagged
        "fp_clean": fp_in("human_clean"),
        "fp_messy": fp_in("human_messy"),
    }


def detectors_of(samples: list[dict]) -> list[str]:
    return list(samples[0]["flags"]) if samples else []
