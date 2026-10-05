"""CodeSentinel - Streamlit front end.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import html
import os
from dataclasses import replace
from typing import Iterable

import pandas as pd  # type: ignore
import streamlit as st

from codesentinel import APP_NAME, __version__, moss
from codesentinel import evaluation as ev
from codesentinel import reviews as review_log
from codesentinel.agent import investigate
from codesentinel.ai_style import judge
from codesentinel.analysis import ClassAnalysis
from codesentinel.config import AVAILABLE_MODELS, DEFAULT_MODEL
from codesentinel.llm import LLMClient
from codesentinel.problems import list_problems, load_problem
from codesentinel.reference import reference_match
from codesentinel.reviews import DECISIONS, load_reviews, save_review
from codesentinel.same_bug import wrong_answers

st.set_page_config(page_title=APP_NAME, layout="wide")

RISK_BADGE = {"LOW": ":green-badge[LOW]", "MEDIUM": ":orange-badge[MEDIUM]", "HIGH": ":red-badge[HIGH]"}
UPLOADED = "uploaded"


# ============================================================
# Cached computations
# ============================================================


@st.cache_resource(show_spinner="Analysing submissions...", ttl=3600, max_entries=8)
def get_analysis(problem_id: str) -> ClassAnalysis:
    analysis = ClassAnalysis(load_problem(problem_id))
    _ = analysis.pairs, analysis.signatures  # warm the expensive parts once
    return analysis


@st.cache_data(show_spinner=False, ttl=3600)
def scan_table(problem_id: str) -> pd.DataFrame:
    rows = []
    for s in get_analysis(problem_id).summaries():
        rows.append(
            {
                "Candidate": s.student,
                "Closest classmate": s.top_classmate or "-",
                "Similarity": round(s.top_similarity * 100),
                "Same wrong output as": ", ".join(s.same_bug_with) or "-",
                "AI reference match": round(s.ref_match * 100),
                "AI style score": round(s.style_score * 100),
                "Verdict": ("Flagged: " + ", ".join(s.reasons)) if s.reasons else "No flags",
                "Flagged": s.flagged,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Scoring the evaluation dataset...", ttl=3600)
def evaluation_results() -> list[dict]:
    return ev.run_detectors(ev.load_dataset(), use_cached_llm=True)


# ============================================================
# Helpers
# ============================================================


def render_code(code: str, matched_lines: Iterable[int] = (), max_height: int = 460) -> None:
    """Show code with matched lines highlighted."""
    matched = set(matched_lines)
    rows = []
    for n, line in enumerate(code.splitlines(), 1):
        bg = "background:rgba(255,193,7,.28);" if n in matched else ""
        rows.append(
            f'<div style="{bg}white-space:pre;padding:0 .5rem;">'
            f'<span style="opacity:.45;display:inline-block;width:2.2em;user-select:none;">{n}</span>'
            f"{html.escape(line) or '&nbsp;'}</div>"
        )
    st.html(
        f'<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.82rem;line-height:1.5;'
        f"border:1px solid rgba(128,128,128,.35);border-radius:.5rem;padding:.5rem 0;"
        f'overflow:auto;max-height:{max_height}px;background:rgba(128,128,128,.07);">{"".join(rows)}</div>'
    )


def secret_api_key() -> str | None:
    """Key from Streamlit secrets or the environment (never from code)."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets file configured
        key = None
    return key or os.getenv("GROQ_API_KEY")


def get_llm(api_key: str | None, model: str) -> LLMClient | None:
    """One LLM client per browser session; token usage survives model changes."""
    state = st.session_state
    if not api_key:
        state.pop("llm", None)
        return None
    llm = state.get("llm")
    if llm is None or state.get("llm_key") != api_key:
        llm = LLMClient(api_key, model=model)
        state.llm, state.llm_key = llm, api_key
    llm.model = model
    return llm


def pct(x: float) -> str:
    return f"{x:.0%}"


# ============================================================
# Sidebar
# ============================================================


def sidebar() -> tuple[str, LLMClient | None]:
    with st.sidebar:
        st.title(APP_NAME)
        st.caption("Evidence-based integrity triage for coding assessments")

        problems = list_problems(with_submissions=True)
        problem_id = st.selectbox("Assessment problem", problems, format_func=lambda p: p.replace("_", " ").title())
        with st.expander("Problem statement"):
            st.markdown(load_problem(problem_id).statement)

        st.divider()
        st.subheader("Groq API")
        api_key = secret_api_key()
        if api_key:
            st.success("API key loaded from server configuration")
        else:
            api_key = st.text_input(
                "Groq API key", type="password", placeholder="gsk_...",
                help="Used only for this browser session and never stored.",
            ).strip()
            st.caption("Free keys: console.groq.com/keys. Offline checks work without a key.")

        default_index = AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0
        model = st.selectbox("Model", AVAILABLE_MODELS, index=default_index)
        llm = get_llm(api_key, model)

        if llm and llm.usage.total_calls:
            c1, c2 = st.columns(2)
            c1.metric("LLM calls", llm.usage.total_calls)
            c2.metric("Tokens", f"{llm.usage.total_tokens:,}")

        st.divider()
        st.caption("A flag is a reason to ask a question, not proof of misconduct. A human always decides.")
        st.caption(f"{APP_NAME} v{__version__}")
    return problem_id, llm


# ============================================================
# Tab 1: class scan
# ============================================================


def tab_scan(problem_id: str) -> None:
    analysis = get_analysis(problem_id)
    df = scan_table(problem_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", len(df))
    c2.metric("Flagged", int(df["Flagged"].sum()))
    c3.metric("Same-bug groups", len(analysis.groups) if analysis.problem.tests else "n/a")
    c4.metric("AI reference solutions", len(analysis.problem.references))

    only_flagged = st.toggle("Show flagged candidates only", value=False)
    view = df[df["Flagged"]] if only_flagged else df
    event = st.dataframe(
        view.drop(columns="Flagged"),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Similarity": st.column_config.ProgressColumn(
                "Similarity to closest classmate", format="%d%%", min_value=0, max_value=100
            ),
            "AI reference match": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "AI style score": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
        },
    )
    st.caption("Select a row for details. Checks: classmate copying (MOSS), identical wrong outputs, "
               "similarity to AI-generated reference solutions, and AI-style heuristics.")

    if analysis.problem.tests and analysis.groups:
        with st.expander(f"Candidates sharing identical wrong outputs ({len(analysis.groups)} group(s))"):
            for g in analysis.groups:
                st.markdown(f"**{', '.join(g['students'])}** produced the same wrong output on:")
                st.table(pd.DataFrame(g["wrong_tests"]))

    rows = event.selection.rows
    if rows:
        student = view.iloc[rows[0]]["Candidate"]
        st.subheader(f"Details: {student}")
        ref = analysis.reference(student)
        style = analysis.style(student)
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Code** (highlighted = overlaps with AI reference `{ref['best_reference']}`)")
            render_code(analysis.problem.submissions[student], ref["matched_lines"])
        with right:
            st.markdown("**Closest classmates**")
            st.dataframe(pd.DataFrame(analysis.classmates(student)).drop(columns="my_matched_lines"),
                         hide_index=True, width="stretch")
            st.markdown(f"**AI-style signals** (score {style['score']:.2f})")
            for signal in style["signals"] or ["no notable signals"]:
                st.write(f"- {signal}")


# ============================================================
# Tab 2: pair comparison
# ============================================================


def tab_pair(problem_id: str) -> None:
    analysis = get_analysis(problem_id)
    names = list(analysis.problem.submissions)
    top = analysis.pairs[0]

    c1, c2 = st.columns(2)
    a = c1.selectbox("Candidate A", names, index=names.index(top["a"]))
    options_b = [n for n in names if n != a]
    b = c2.selectbox("Candidate B", options_b, index=options_b.index(top["b"]) if top["b"] in options_b else 0)

    pair = moss.pair_between(analysis.pairs, a, b)
    m1, m2, m3 = st.columns(3)
    m1.metric("Similarity", pct(pair["score"]), help="Higher of the two directional scores")
    m2.metric(f"{a} found in {b}", pct(pair["pct_a"]))
    m3.metric(f"{b} found in {a}", pct(pair["pct_b"]))
    if pair["score"] >= moss.FLAG_AT:
        st.warning("Above the review threshold. Check whether the problem is simple enough for this overlap to be natural.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{a}**")
        render_code(analysis.problem.submissions[a], pair["lines_a"])
    with right:
        st.markdown(f"**{b}**")
        render_code(analysis.problem.submissions[b], pair["lines_b"])
    st.caption("Highlighted lines share normalised token fingerprints. Renaming variables or changing comments does not hide a match.")


# ============================================================
# Tab 3: AI investigator + human review
# ============================================================


def tab_agent(problem_id: str, llm: LLMClient | None) -> None:
    analysis = get_analysis(problem_id)
    df = scan_table(problem_id)

    if llm is None:
        st.info("Add your Groq API key in the sidebar to run the AI investigator.")
        return

    flagged = df[df["Flagged"]]["Candidate"].tolist()
    order = flagged + [n for n in df["Candidate"] if n not in flagged]
    student = st.selectbox("Candidate to investigate", order, format_func=lambda n: f"{n}  (flagged)" if n in flagged else n)
    key = (problem_id, student)
    reports = st.session_state.setdefault("reports", {})

    if st.button("Run investigation", type="primary"):
        with st.status("Investigating...", expanded=True) as status:
            try:
                reports[key] = investigate(analysis, student, llm, on_event=lambda m: st.write(m))
                status.update(label="Investigation complete", state="complete", expanded=False)
            except Exception as exc:  # network / quota / invalid key
                status.update(label="Investigation failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")

    result = reports.get(key)
    if result is None:
        st.caption("The agent picks which checks to run and writes an evidence report. It never decides that someone cheated.")
        return

    st.markdown(f"### Agent report: {student}   {RISK_BADGE.get(result.risk, ':gray-badge[UNKNOWN]')}")
    st.text(result.report)
    with st.expander(f"Tool results ({len(result.trace)})"):
        for step in result.trace:
            st.markdown(f"**{step['tool']}**")
            st.json(step["result"], expanded=False)

    st.divider()
    st.subheader("Human review")
    with st.form(f"review-{problem_id}-{student}"):
        reviewer = st.text_input("Reviewer name (optional)")
        choice = st.radio("Decision", list(DECISIONS.values()), horizontal=True)
        note = st.text_area("Note (optional)")
        if st.form_submit_button("Save decision"):
            save_review(problem_id, student, result.report, choice, note, reviewer)
            st.success(f"Saved: {student} -> {choice}")

    reviews = load_reviews()
    if reviews:
        with st.expander(f"Decision log ({len(reviews)})"):
            log = pd.DataFrame(reviews).drop(columns="agent_report", errors="ignore")
            st.dataframe(log, hide_index=True, width="stretch")
            st.download_button("Download full log (JSON)", review_log.REVIEWS_FILE.read_bytes(), "reviews.json", "application/json")
            st.caption("On hosted deployments the file system may be ephemeral: download the log or mount persistent storage.")


# ============================================================
# Tab 4: analyse a pasted / uploaded submission
# ============================================================


def tab_custom(problem_id: str, llm: LLMClient | None) -> None:
    analysis = get_analysis(problem_id)
    problem = analysis.problem

    uploaded = st.file_uploader("Upload a .py file", type=["py"])
    code = st.text_area(
        "...or paste Python code",
        value=uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else "",
        height=260,
        placeholder="import sys\n\n...",
    )
    c1, c2 = st.columns(2)
    run_tests = c1.checkbox(
        "Run hidden tests", value=False, disabled=not problem.tests,
        help="Executes the code in a restricted subprocess (timeout, memory limit, no secrets). "
             + ("" if problem.tests else "This problem has no tests."),
    )
    use_llm = c2.checkbox("Also run the LLM judge", value=False, disabled=llm is None,
                          help="Uses Groq tokens." if llm else "Requires a Groq API key.")

    if not st.button("Analyse submission", type="primary", disabled=not code.strip()):
        return

    st.subheader("Results")
    style = judge(code, problem.statement, problem.starter, llm=llm if use_llm else None)
    ref = reference_match(problem, code)
    pairs = moss.check_class(problem.with_submission(UPLOADED, code))
    top = moss.matches_for(pairs, UPLOADED, top=3)

    m1, m2, m3 = st.columns(3)
    m1.metric("Closest classmate", f"{top[0]['classmate']} ({pct(top[0]['similarity'])})" if top else "-")
    m2.metric("AI reference match", pct(ref["best_match"]), help=f"Closest: {ref['best_reference']}")
    m3.metric("AI style (heuristic)", f"{style['heuristic']['score']:.2f}")

    if run_tests:
        mine = wrong_answers(replace(problem, submissions={UPLOADED: code}))[UPLOADED]
        if not mine:
            st.success("Passes all hidden tests.")
        else:
            twins = [s for s, sig in analysis.signatures.items() if sig == mine]
            st.error(f"Fails {len(mine)} test(s).")
            if twins:
                st.warning(f"Produces exactly the same wrong outputs as: {', '.join(twins)}")

    if "llm" in style:
        st.markdown(f"**LLM judge: {style['llm']['ai_likelihood']:.2f}**: {style['llm']['reasoning']}")
    st.markdown("**Heuristic signals**")
    for signal in style["heuristic"]["signals"] or ["no notable signals"]:
        st.write(f"- {signal}")

    st.markdown(f"**Your code** (highlighted = overlaps with AI reference `{ref['best_reference']}`)")
    render_code(code, ref["matched_lines"])


# ============================================================
# Tab 5: evaluation
# ============================================================


def tab_evals() -> None:
    st.markdown(
        "Building a detector is easy; **measuring how wrong it is** is the real work. "
        "The labelled dataset has AI-written code, tidy human code and messy human code. "
        "A good detector catches AI samples without flagging tidy humans."
    )
    samples = evaluation_results()
    detectors = ev.detectors_of(samples)

    rows = []
    for d in detectors:
        m = ev.metrics(samples, d)
        rows.append(
            {
                "Detector": d,
                "Precision": round(m["precision"] * 100),
                "Recall": round(m["recall"] * 100),
                "False-positive rate": round(m["fpr"] * 100),
                "Tidy humans flagged": f"{m['fp_clean'][0]}/{m['fp_clean'][1]}",
                "Messy humans flagged": f"{m['fp_messy'][0]}/{m['fp_messy'][1]}",
            }
        )
    pct_col = st.column_config.NumberColumn(format="%d%%")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Precision": pct_col, "Recall": pct_col, "False-positive rate": pct_col})

    combined = "combined (2+ agree)"
    st.write(f"**Humans wrongly flagged by the combined detector:** "
             f"{', '.join(s['name'] for s in samples if not s['is_ai'] and s['flags'][combined]) or 'none'}")
    st.write(f"**AI samples missed by the combined detector:** "
             f"{', '.join(s['name'] for s in samples if s['is_ai'] and not s['flags'][combined]) or 'none'}")

    with st.expander("Per-sample scores"):
        score_cols = [d for d in detectors if d != combined]
        table = pd.DataFrame(
            [{"Sample": s["name"], **{d: s["scores"].get(d) for d in score_cols}, "Flagged": s["flags"][combined]} for s in samples]
        )
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption("LLM-judge scores come from a cached run (data/evals/llm_cache.json); regenerate with `python -m codesentinel evals --llm`.")


# ============================================================
# Tab 6: how it works
# ============================================================


def tab_about() -> None:
    st.markdown(
        f"""
**{APP_NAME}** gathers independent pieces of evidence and leaves the decision to a human.

| Check | Question it answers | Strength |
|---|---|---|
| **MOSS fingerprinting** | Do two candidates share unusual code structure? Names, comments and whitespace are normalised away. | Strong when high |
| **Same-bug detection** | Did two candidates fail the same hidden tests with identical wrong outputs? | Strong, rarely coincidental |
| **AI reference match** | Does the code overlap with solutions generated by several AI models? (no common-code filter) | Strong when very high |
| **AI-style analysis** | Does the code *look* AI-written? Heuristics plus an LLM judge. | Weak alone |
| **Investigator agent** | A Groq LLM chooses checks, weighs the evidence and drafts follow-up questions about real lines. | Report only |

**Responsible use**

- Detectors have false positives. Tidy, textbook code is not evidence of misconduct, and short problems naturally look alike.
- Use flags to decide *whom to ask a question*, for example a short follow-up interview about specific lines, never to decide guilt.
- Run the evaluation tab on your own data before relying on any threshold.

**Security notes**

- The hidden-test runner executes submitted code in a restricted subprocess (isolated interpreter, timeout, memory limit, scrubbed environment).
  It is defence in depth, not a hardened sandbox: deploy behind a container or VM when handling untrusted code.
- Your Groq key is kept only in the browser session (or in server secrets) and is never written to disk.
        """
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    problem_id, llm = sidebar()
    st.title(f"{problem_id.replace('_', ' ').title()}")
    st.caption("Class integrity review")

    scan, pair, agent, custom, evals, about = st.tabs(
        ["Class scan", "Compare pair", "AI investigator", "Analyse a submission", "Evaluation", "How it works"]
    )
    with scan:
        tab_scan(problem_id)
    with pair:
        tab_pair(problem_id)
    with agent:
        tab_agent(problem_id, llm)
    with custom:
        tab_custom(problem_id, llm)
    with evals:
        tab_evals()
    with about:
        tab_about()


main()
