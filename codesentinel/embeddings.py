"""Embeddings vs MOSS: the same job done the "AI engineering" way (optional extra).

Every submission is turned into a vector by a code-embedding model, then compared with
cosine similarity.

    - Embeddings capture *logic* (a loop rewritten as recursion still looks similar)
    - MOSS is fast, free and shows *which lines* matched (explainable)

Install the extra first:  pip install "codesentinel[embeddings]"
The model (jinaai/jina-embeddings-v2-base-code, ~640 MB) is downloaded on first use.
"""

from __future__ import annotations

from itertools import combinations

from codesentinel.problems import Problem

EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"

_model = None


def embed(codes: list[str]):
    try:
        import numpy as np
        from fastembed import TextEmbedding  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Embeddings need the optional extra: pip install "codesentinel[embeddings]"') from exc

    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    vectors = np.array(list(_model.embed(codes)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # normalised: dot product == cosine


def cosine_pairs(submissions: dict[str, str]) -> dict[tuple[str, str], float]:
    names = list(submissions)
    vectors = embed([submissions[n] for n in names])
    sims = vectors @ vectors.T
    return {(names[i], names[j]): float(sims[i, j]) for i, j in combinations(range(len(names)), 2)}


def compare_embeddings(problem: Problem, top: int = 15) -> list[tuple[str, str, float]]:
    cos = cosine_pairs(problem.submissions)
    return sorted(((a, b, s) for (a, b), s in cos.items()), key=lambda t: t[2], reverse=True)[:top]
