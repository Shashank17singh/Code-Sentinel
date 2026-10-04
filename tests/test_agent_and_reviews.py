import json
from types import SimpleNamespace

from codesentinel.agent import TOOLS, investigate
from codesentinel.analysis import ClassAnalysis
from codesentinel.problems import load_problem
from codesentinel.reviews import load_reviews, save_review


class FakeToolCall:
    def __init__(self, name, id_="call_1"):
        self.id = id_
        self.function = SimpleNamespace(name=name, arguments="{}")

    def model_dump(self):
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": "{}"}}


def reply(content="", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeLLM:
    """Scripted stand-in for LLMClient."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, *, label, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        return self.replies.pop(0)


def test_agent_runs_tools_then_reports():
    llm = FakeLLM(
        [
            reply(tool_calls=[FakeToolCall("check_classmate_similarity", "a"), FakeToolCall("check_same_bug", "b")]),
            reply("RISK: HIGH\nEVIDENCE:\n- alice and bob match 100%"),
        ]
    )
    events = []
    result = investigate(ClassAnalysis(load_problem("longest_substring")), "alice", llm, on_event=events.append)

    assert result.risk == "HIGH" and result.completed
    assert [t["tool"] for t in result.trace] == ["check_classmate_similarity", "check_same_bug"]
    assert result.trace[0]["result"]["top_matches"][0]["classmate"] == "bob"
    # tool results are fed back to the model as JSON
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["class_size"] == 14
    assert llm.calls[0]["kwargs"]["tools"] == TOOLS
    assert any("Turn 1" in e for e in events)


def test_agent_treats_candidate_code_as_untrusted():
    llm = FakeLLM([reply("RISK: LOW")])
    investigate(ClassAnalysis(load_problem("longest_substring")), "karan", llm)
    system, user = llm.calls[0]["messages"][:2]
    assert "untrusted" in system["content"] and "<code>" in user["content"]


def test_agent_gives_up_gracefully_and_disables_tools_on_last_turn():
    llm = FakeLLM([reply(tool_calls=[FakeToolCall("check_same_bug")]) for _ in range(3)])
    result = investigate(ClassAnalysis(load_problem("longest_substring")), "alice", llm, max_attempts=3)
    assert result.risk == "UNKNOWN" and not result.completed
    assert "tools" not in llm.calls[-1]["kwargs"]


def test_unknown_tool_is_reported_not_crashed():
    llm = FakeLLM([reply(tool_calls=[FakeToolCall("rm_rf")]), reply("RISK: LOW")])
    result = investigate(ClassAnalysis(load_problem("longest_substring")), "karan", llm)
    assert result.trace[0]["result"] == "Unknown tool rm_rf"


def test_reviews_roundtrip(tmp_path):
    path = tmp_path / "nested" / "reviews.json"
    assert load_reviews(path) == []
    save_review("longest_substring", "alice", "RISK: HIGH", "escalated", "see lines 3-5", "Reviewer", path)
    save_review("longest_substring", "bob", "RISK: LOW", "cleared", path=path)
    reviews = load_reviews(path)
    assert [r["candidate"] for r in reviews] == ["alice", "bob"]
    assert reviews[0]["decision"] == "escalated" and reviews[0]["reviewer"] == "Reviewer"


def test_reviews_survive_corrupt_file(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text("{not json")
    assert load_reviews(path) == []
