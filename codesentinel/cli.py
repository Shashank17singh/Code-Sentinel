"""Command-line interface:  ``python -m codesentinel <command> ...``"""

from __future__ import annotations

import argparse
import os
import sys

from codesentinel import APP_NAME, __version__
from codesentinel.analysis import ClassAnalysis
from codesentinel.config import AGENT_MAX_ATTEMPTS, DEFAULT_MODEL
from codesentinel.llm import LLMClient, MissingAPIKeyError
from codesentinel.problems import list_problems, load_problem


def _llm(args) -> LLMClient:
    return LLMClient(os.getenv("GROQ_API_KEY"), model=getattr(args, "model", None) or DEFAULT_MODEL)


def cmd_scan(args) -> None:
    problem = load_problem(args.problem)
    analysis = ClassAnalysis(problem)
    print(f"\nClass scan: {problem.id} ({len(problem.submissions)} submissions)\n")
    print(f"{'student':<10}{'top classmate':>18}{'same bug with':>16}{'AI ref':>8}{'AI style':>10}   send to agent?")
    for s in analysis.summaries():
        classmate = f"{s.top_classmate} {s.top_similarity:.0%}" if s.top_classmate else "-"
        bug = f"{s.same_bug_with[0]} +{len(s.same_bug_with) - 1}" if len(s.same_bug_with) > 1 else (
            s.same_bug_with[0] if s.same_bug_with else "-")
        verdict = "YES: " + ", ".join(s.reasons) if s.reasons else ""
        print(f"{s.student:<10}{classmate:>18}{bug:>16}{s.ref_match:>8.0%}{s.style_score:>10.2f}   {verdict}")
    print(f"\nInvestigate one candidate:  python -m codesentinel agent {problem.id} <student>")
    print("Remember: a flag means 'ask a question', not 'cheater'.")


def cmd_moss(args) -> None:
    from codesentinel import moss

    problem = load_problem(args.problem)
    pairs = moss.check_class(problem, use_filter=not args.no_filter)
    if args.show:
        print(moss.side_by_side(problem, moss.pair_between(pairs, *args.show)))
        return
    label = "WITHOUT" if args.no_filter else f"WITH (threshold={moss.COMMON_THRESHOLD})"
    print(f"MOSS report for '{problem.id}' -- common-code filter {label}")
    print(f"\n{'pair':<22}{'similarity':>11}{'A matched':>11}{'B matched':>11}{'shared fp':>11}")
    for p in pairs[: args.top]:
        flag = "  <-- FLAG" if p["score"] >= moss.FLAG_AT else ""
        print(f"{p['a'] + ' - ' + p['b']:<22}{p['score']:>10.0%}{p['pct_a']:>11.0%}{p['pct_b']:>11.0%}{p['shared']:>11}{flag}")


def cmd_same_bug(args) -> None:
    analysis = ClassAnalysis(load_problem(args.problem))
    if not analysis.problem.tests:
        print("This problem has no tests.json, so the same-bug check is unavailable.")
        return
    if not analysis.groups:
        print("No two candidates made the same mistake.")
    for g in analysis.groups:
        print(f"\nSame wrong answers: {', '.join(g['students'])}")
        for t in g["wrong_tests"]:
            print(f"    input {t['input']!r:<14} -> all produced {t['their_output']!r}")


def cmd_reference(args) -> None:
    from codesentinel.reference import generate_references

    problem = load_problem(args.problem)
    if args.action == "generate":
        generate_references(_llm(args), problem, force=args.force)
        return
    analysis = ClassAnalysis(problem)
    print(f"\nCandidate vs {len(problem.references)} AI solutions ({problem.id})\n")
    print(f"{'student':<10}{'best AI match':>15}{'coverage':>10}   closest reference")
    for student in problem.submissions:
        r = analysis.reference(student)
        flag = "  <-- FLAG" if r["flag"] else ""
        print(f"{student:<10}{r['best_match']:>15.0%}{r['any_reference_coverage']:>10.0%}   {r['best_reference']}{flag}")


def cmd_ai_style(args) -> None:
    from codesentinel.ai_style import judge

    problem = load_problem(args.problem)
    llm = _llm(args) if args.llm else None
    students = [args.student] if args.student else list(problem.submissions)
    for student in students:
        r = judge(problem.submissions[student], problem.statement, problem.starter, llm=llm)
        h = r["heuristic"]
        print(f"\n{student}: heuristic {h['score']:.2f}{'  <-- FLAG' if h['flag'] else ''}")
        for s in h["signals"]:
            print(f"    {s}")
        if "llm" in r:
            print(f"  LLM judge {r['llm']['ai_likelihood']:.2f}{'  <-- FLAG' if r['llm']['flag'] else ''}: {r['llm']['reasoning']}")
    if llm:
        print("\n--- Token usage ---\n" + llm.usage.summary())


def cmd_agent(args) -> None:
    from codesentinel.agent import investigate
    from codesentinel.reviews import DECISIONS, save_review

    problem = load_problem(args.problem)
    if args.student not in problem.submissions:
        sys.exit(f"Unknown candidate '{args.student}'. Options: {', '.join(problem.submissions)}")
    llm = _llm(args)

    print(f"\nInvestigating '{args.student}' on '{problem.id}'\n")
    result = investigate(
        ClassAnalysis(problem), args.student, llm, max_attempts=args.max_attempts, on_event=lambda m: print(f"  [agent] {m}")
    )
    print("\n" + "=" * 60 + f"\nAGENT REPORT: {args.student}\n" + "=" * 60)
    print(result.report)
    print("\n--- Token usage ---\n" + llm.usage.summary())

    if args.no_review:
        return
    print("\n" + "=" * 60 + "\nHUMAN REVIEW  (system flags, human decides)\n" + "=" * 60)
    try:
        choice = input("Decision -- [c]lear / [f]ollow-up interview / [e]scalate: ").strip().lower()
        note = input("Note (optional): ").strip()
    except EOFError:
        print("(no reviewer input, decision not saved)")
        return
    decision = DECISIONS.get(choice[:1], "undecided")
    save_review(problem.id, args.student, result.report, decision, note)
    print(f"Saved: {args.student} -> {decision}")


def cmd_evals(args) -> None:
    from codesentinel import evaluation as ev

    samples = ev.load_dataset()
    counts = {label: sum(1 for s in samples if s["label"] == label) for label in ev.LABELS}
    print(f"Dataset: {len(samples)} samples  {counts}")

    llm = _llm(args) if args.llm else None
    ev.run_detectors(samples, llm=llm, use_cached_llm=not args.llm)
    detectors = ev.detectors_of(samples)

    if args.verbose:
        cols = [d for d in detectors if d != "combined (2+ agree)"]
        print(f"\n{'sample':<42}" + "".join(f"{d[:10]:>12}" for d in cols) + "   flagged?")
        for s in samples:
            row = "".join(f"{s['scores'].get(d, float('nan')):>12.2f}" for d in cols)
            print(f"{s['name']:<42}{row}   {'YES' if s['flags']['combined (2+ agree)'] else ''}")

    print(f"\n{'detector':<22}{'precision':>10}{'recall':>8}{'FPR':>7}{'FP clean':>10}{'FP messy':>10}")
    for d in detectors:
        m = ev.metrics(samples, d)
        clean, messy = f"{m['fp_clean'][0]}/{m['fp_clean'][1]}", f"{m['fp_messy'][0]}/{m['fp_messy'][1]}"
        print(f"{d:<22}{m['precision']:>10.0%}{m['recall']:>8.0%}{m['fpr']:>7.0%}{clean:>10}{messy:>10}")

    combined = "combined (2+ agree)"
    print(f"\nHumans flagged by the combined detector: {[s['name'] for s in samples if not s['is_ai'] and s['flags'][combined]] or 'none'}")
    print(f"AI samples missed by the combined detector: {[s['name'] for s in samples if s['is_ai'] and not s['flags'][combined]] or 'none'}")
    if llm:
        print("\n--- Token usage ---\n" + llm.usage.summary())


def cmd_embeddings(args) -> None:
    from codesentinel.embeddings import compare_embeddings

    problem = load_problem(args.problem)
    print(f"\nEmbedding similarity for '{problem.id}'\n")
    for a, b, s in compare_embeddings(problem, top=args.top):
        print(f"{a + ' - ' + b:<22}{s:>8.2f}")


def cmd_generate_samples(args) -> None:
    from codesentinel.config import EVALS_DIR, REFERENCE_MODELS
    from codesentinel.reference import ask_for_solution

    prompts = [
        "solve this in python\n\n{statement}\n\nReply with the code in a single ```python block.",
        "{statement}\n\ngive me python code for this. it should read input and print output. "
        "Reply with the code in a single ```python block.",
        "I have an online assessment question:\n\n{statement}\n\nWrite an efficient Python solution. "
        "Reply with the code in a single ```python block.",
    ]
    llm = _llm(args)

    def write(path, problem, n):
        if path.exists():
            print(f"  cached   {path.name}")
            return
        model = REFERENCE_MODELS[n % len(REFERENCE_MODELS)]
        path.write_text(ask_for_solution(llm, prompts[n % len(prompts)].format(statement=problem.statement), model), encoding="utf-8")
        print(f"  wrote    {path.name}  ({model})")

    out = EVALS_DIR / "dataset" / "ai"
    out.mkdir(parents=True, exist_ok=True)
    for pid in ("longest_substring", "two_sum", "valid_parentheses"):
        problem = load_problem(pid)
        for n in range(4):
            write(out / f"{pid}__ai{n + 1}.py", problem, n + 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codesentinel", description=f"{APP_NAME} {__version__}: integrity triage for coding assessments")
    sub = p.add_subparsers(dest="command", required=True)
    problems = list_problems()

    def add(name, func, help_, problem=True):
        sp = sub.add_parser(name, help=help_)
        if problem:
            sp.add_argument("problem", choices=problems)
        sp.set_defaults(func=func)
        return sp

    add("scan", cmd_scan, "offline scan of the whole class (no LLM calls)")

    sp = add("moss", cmd_moss, "pairwise MOSS similarity")
    sp.add_argument("--no-filter", action="store_true", help="disable the common-code filter")
    sp.add_argument("--show", nargs=2, metavar=("A", "B"), help="show two candidates side by side")
    sp.add_argument("--top", type=int, default=12)

    add("same-bug", cmd_same_bug, "group candidates that share the same wrong outputs")

    sp = add("reference", cmd_reference, "compare candidates with AI reference solutions")
    sp.add_argument("action", choices=["generate", "match"])
    sp.add_argument("--force", action="store_true", help="regenerate even if cached")

    sp = add("ai-style", cmd_ai_style, "heuristic (and optional LLM) AI-style check")
    sp.add_argument("student", nargs="?", help="omit for the whole class")
    sp.add_argument("--llm", action="store_true", help="also run the LLM judge (uses tokens)")

    sp = add("agent", cmd_agent, "LLM investigation of one candidate + human review")
    sp.add_argument("student")
    sp.add_argument("--no-review", action="store_true", help="skip the human review prompt")
    sp.add_argument("--max-attempts", type=int, default=AGENT_MAX_ATTEMPTS)

    sp = add("evals", cmd_evals, "measure precision / recall / false-positive rate", problem=False)
    sp.add_argument("--llm", action="store_true", help="run the LLM judge on cache misses (uses tokens)")
    sp.add_argument("-v", "--verbose", action="store_true")

    sp = add("embeddings", cmd_embeddings, "embedding similarity (needs the 'embeddings' extra)")
    sp.add_argument("--top", type=int, default=15)

    add("generate-samples", cmd_generate_samples, "generate AI samples for the eval dataset", problem=False)
    return p


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # keep Windows terminals from crashing on unicode
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except MissingAPIKeyError as exc:
        sys.exit(f"error: {exc}")
