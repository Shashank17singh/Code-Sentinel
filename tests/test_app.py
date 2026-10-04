from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def test_app_renders_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = run_app()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Candidates"] == "14"
    assert any("Groq API key" in t.label for t in at.sidebar.text_input)


def test_agent_tab_prompts_for_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = run_app()
    assert any("Groq API key" in i.value for i in at.info)


def test_switching_problem(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = run_app()
    at.sidebar.selectbox[0].select("max_subarray").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Candidates"] == "8"


def test_analyse_submission_flow(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = run_app()
    at.text_area[0].set_value("s = input()\nbest = 0\nprint(best)\n").run()
    next(b for b in at.button if b.label == "Analyse submission").click().run()
    assert not at.exception
    assert any(m.label == "AI reference match" for m in at.metric)


def test_agent_flow_and_human_review(monkeypatch, tmp_path):
    from codesentinel import agent as agent_mod
    from codesentinel import reviews as reviews_mod

    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(reviews_mod, "REVIEWS_FILE", tmp_path / "reviews.json")
    monkeypatch.setattr(
        agent_mod,
        "investigate",
        lambda analysis, student, llm, on_event=None: agent_mod.AgentResult(
            report="RISK: MEDIUM\nEVIDENCE:\n- test", trace=[{"tool": "check_same_bug", "result": {"ok": True}}]
        ),
    )
    at = run_app()
    next(b for b in at.button if b.label == "Run investigation").click().run()
    assert not at.exception
    assert any("MEDIUM" in m.value for m in at.markdown)

    next(b for b in at.button if b.label == "Save decision").click().run()
    assert not at.exception
    saved = reviews_mod.load_reviews()
    assert len(saved) == 1 and saved[0]["decision"] == "cleared"
