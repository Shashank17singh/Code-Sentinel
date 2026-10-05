"""MOSS-style similarity: "who copied from whom?"

Pipeline::

    code -> tokens (tokenizer.py) -> k-grams -> hashes -> winnowing -> fingerprints
    compare two fingerprint sets -> similarity %

Reference: Schleimer, Wilkerson, Aiken, "Winnowing: Local Algorithms for Document
Fingerprinting" (SIGMOD 2003), the algorithm behind Stanford's MOSS.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations

from codesentinel.config import COMMON_THRESHOLD, KGRAM_SIZE, SIMILARITY_FLAG_AT, WINNOW_WINDOW
from codesentinel.problems import Problem
from codesentinel.tokenizer import Token, tokenize

K = KGRAM_SIZE
W = WINNOW_WINDOW
FLAG_AT = SIMILARITY_FLAG_AT


# --- k-grams and hashing ------------------------------------------------------
# Every overlapping window of K normalised tokens (e.g. "ID = ID + NUM") becomes one
# number. Real MOSS uses a Karp-Rabin rolling hash for speed; a plain hash is enough here.


def hash_kgram(norms: list[str]) -> int:
    digest = hashlib.blake2b(" ".join(norms).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def kgram_hashes(tokens: list[Token], k: int = K) -> list[int]:
    norms = [t.norm for t in tokens]
    return [hash_kgram(norms[i : i + k]) for i in range(len(norms) - k + 1)]


# --- winnowing ----------------------------------------------------------------
# Storing every hash is wasteful, so from each window of W hashes only the minimum is
# kept. Guarantee: any shared run of W + K - 1 tokens is detected.


def winnow(hashes: list[int], w: int = W) -> list[tuple[int, int]]:
    """Return ``[(hash, position)]`` where position is the token index the k-gram starts at."""
    if not hashes:
        return []
    if len(hashes) < w:
        pos = min(range(len(hashes)), key=lambda i: hashes[i])
        return [(hashes[pos], pos)]

    picked: list[tuple[int, int]] = []
    last_pos = -1
    for start in range(len(hashes) - w + 1):
        window = range(start, start + w)
        # minimum hash; ties go to the right-most ("robust winnowing" from the paper)
        pos = min(window, key=lambda i: (hashes[i], -i))
        if pos != last_pos:
            picked.append((hashes[pos], pos))
            last_pos = pos
    return picked


@dataclass
class Fingerprint:
    tokens: list[Token]
    positions: dict[int, list[int]] = field(default_factory=dict)  # hash -> token positions

    @property
    def hashes(self) -> set[int]:
        return set(self.positions)

    def lines_for(self, h: int) -> set[int]:
        """Source lines covered by the k-grams with this hash."""
        lines: set[int] = set()
        for pos in self.positions.get(h, []):
            lines.update(t.line for t in self.tokens[pos : pos + K])
        return lines


def fingerprint(code: str) -> Fingerprint:
    tokens = tokenize(code)
    fp = Fingerprint(tokens)
    for h, pos in winnow(kgram_hashes(tokens)):
        fp.positions.setdefault(h, []).append(pos)
    return fp


# --- comparison ---------------------------------------------------------------


def compare(a: Fingerprint, b: Fingerprint, ignore: set[int] | frozenset[int] = frozenset()) -> dict:
    ha = a.hashes - ignore
    hb = b.hashes - ignore
    shared = ha & hb

    pct_a = len(shared) / len(ha) if ha else 0.0  # share of A that appears in B
    pct_b = len(shared) / len(hb) if hb else 0.0  # share of B that appears in A

    lines_a: set[int] = set()
    lines_b: set[int] = set()
    for h in shared:
        lines_a |= a.lines_for(h)
        lines_b |= b.lines_for(h)

    return {
        "score": max(pct_a, pct_b),
        "pct_a": pct_a,
        "pct_b": pct_b,
        "shared": len(shared),
        "lines_a": sorted(lines_a),
        "lines_b": sorted(lines_b),
    }


# --- common-code filtering ----------------------------------------------------
# Ignore (a) the starter code every candidate received and (b) fingerprints that appear
# in many submissions: on an easy question everybody writes the same few lines.


def common_hashes(fps: dict[str, Fingerprint], starter: str, threshold: int | None = COMMON_THRESHOLD) -> set[int]:
    ignore = set(fingerprint(starter).hashes)

    if threshold is not None:
        counts: dict[int, int] = {}
        for fp in fps.values():
            for h in fp.hashes:
                counts[h] = counts.get(h, 0) + 1
        ignore |= {h for h, c in counts.items() if c > threshold}

    return ignore


def check_class(problem: Problem, use_filter: bool = True) -> list[dict]:
    """Compare every pair of candidates, most similar first."""
    fps = {name: fingerprint(code) for name, code in problem.submissions.items()}
    ignore = common_hashes(fps, problem.starter, COMMON_THRESHOLD if use_filter else None)

    pairs = []
    for a, b in combinations(fps, 2):
        pairs.append({"a": a, "b": b, **compare(fps[a], fps[b], ignore)})
    return sorted(pairs, key=lambda p: p["score"], reverse=True)


def matches_for(pairs: list[dict], student: str, top: int = 3) -> list[dict]:
    """The candidate's most similar classmates, given the output of :func:`check_class`."""
    mine_pairs = [p for p in pairs if student in (p["a"], p["b"])]
    out = []
    for p in mine_pairs[:top]:
        me = "a" if p["a"] == student else "b"
        other = "b" if me == "a" else "a"
        out.append(
            {
                "classmate": p[other],
                "similarity": round(p["score"], 2),
                "my_code_matched": round(p[f"pct_{me}"], 2),
                "their_code_matched": round(p[f"pct_{other}"], 2),
                "my_matched_lines": p[f"lines_{me}"],
            }
        )
    return out


def pair_between(pairs: list[dict], a: str, b: str) -> dict:
    """Return the pair record oriented so that ``lines_a`` belongs to ``a``."""
    pair = next(p for p in pairs if {p["a"], p["b"]} == {a, b})
    if pair["a"] != a:
        pair = {
            **pair,
            "a": a,
            "b": b,
            "pct_a": pair["pct_b"],
            "pct_b": pair["pct_a"],
            "lines_a": pair["lines_b"],
            "lines_b": pair["lines_a"],
        }
    return pair


def side_by_side(problem: Problem, pair: dict) -> str:
    """Plain-text side-by-side view; ``>>`` marks matched lines (used by the CLI)."""
    a, b = pair["a"], pair["b"]
    left = problem.submissions[a].splitlines()
    right = problem.submissions[b].splitlines()
    width = max((len(line) for line in left), default=0) + 2

    rows = [f"\n{a} vs {b}: {pair['score']:.0%} similar  (>> = matched line)\n", f"     {a:<{width}}      {b}"]
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        ml = ">>" if i + 1 in pair["lines_a"] else "  "
        mr = ">>" if i + 1 in pair["lines_b"] else "  "
        rows.append(f"{ml} {i + 1:>2} {l:<{width}} {mr} {i + 1:>2} {r}")
    return "\n".join(rows)
