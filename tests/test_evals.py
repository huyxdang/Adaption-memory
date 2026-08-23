"""Offline tests: loaders against the real data files, scoring logic against
known cases, and judge plumbing against a fake LLM. No model server needed."""

import json
from pathlib import Path

import pytest

from adaption_memory.evals import beam, locomo, longmemeval
from adaption_memory.interface import Session, Turn
from adaption_memory.systems.full_history import FullHistorySystem

DATA = Path(__file__).resolve().parent.parent / "data"


class FakeLLM:
    """Returns canned responses; records prompts for assertions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat(self, messages, **kw):
        self.prompts.append(messages)
        return self.responses.pop(0) if self.responses else self.responses_default

    responses_default = "fake"


# ---------- loaders ----------

@pytest.mark.skipif(not (DATA / "longmemeval_oracle.json").exists(),
                    reason="data not downloaded")
def test_longmemeval_loader():
    instances = longmemeval.load_instances(DATA / "longmemeval_oracle.json")
    assert len(instances) == 500
    sessions = longmemeval.sessions_of(instances[0])
    assert sessions and sessions[0].turns[0].role in ("user", "assistant")
    assert sessions[0].date


@pytest.mark.skipif(not (DATA / "locomo10.json").exists(),
                    reason="data not downloaded")
def test_locomo_loader():
    samples = locomo.load_samples(DATA / "locomo10.json")
    assert len(samples) == 10
    sessions = locomo.sessions_of(samples[0])
    assert len(sessions) >= 2
    assert sessions[0].date and sessions[0].turns[0].content


@pytest.mark.skipif(not (DATA / "beam_100k.parquet").exists(),
                    reason="data not downloaded")
def test_beam_loader():
    convs = beam.load_conversations(DATA / "beam_100k.parquet")
    assert len(convs) == 20
    sessions = beam.sessions_of(convs[0])
    assert len(sessions) > 1
    # sessions must preserve every chat message
    n_msgs = sum(len(b) for b in convs[0]["chat"])
    assert sum(len(s.turns) for s in sessions) == n_msgs
    qs = convs[0]["questions"]
    assert set(qs) == set(beam.QUESTION_TYPES)
    assert all(q["rubric"] for v in qs.values() for q in v)


# ---------- locomo scoring (official semantics) ----------

def test_locomo_f1_exact():
    assert locomo.score_one("7 May 2023", "7 May 2023", 2) == 1.0


def test_locomo_f1_partial():
    assert 0 < locomo.score_one("she went in May 2023", "7 May 2023", 2) < 1.0


def test_locomo_open_domain_takes_first_semicolon_field():
    assert locomo.score_one("Psychology", "Psychology; counseling", 3) == \
        locomo.score_one("Psychology", "Psychology", 3)


def test_locomo_multihop_partial_f1():
    s = locomo.score_one("painting, hiking", "painting, hiking, baking", 1)
    assert 0.5 < s < 1.0


def test_locomo_adversarial():
    assert locomo.score_one("No information available.", "", 5) == 1.0
    assert locomo.score_one("It was in Paris.", "", 5) == 0.0


# ---------- longmemeval judge plumbing ----------

def test_longmemeval_prompt_selection():
    p = longmemeval.get_anscheck_prompt("knowledge-update", "Q?", "A", "R")
    assert "updated answer" in p and "Q?" in p and "A" in p and "R" in p
    p_abs = longmemeval.get_anscheck_prompt("multi-session", "Q?", "A", "R",
                                            abstention=True)
    assert "unanswerable" in p_abs


def test_longmemeval_judge_stage(tmp_path):
    instances = [{"question_id": "q1", "question_type": "multi-session",
                  "question": "Q?", "answer": "42"},
                 {"question_id": "q2_abs", "question_type": "multi-session",
                  "question": "Q?", "answer": "unknowable"}]
    hyp = tmp_path / "answers.jsonl"
    hyp.write_text(json.dumps({"question_id": "q1", "hypothesis": "42"}) + "\n"
                   + json.dumps({"question_id": "q2_abs", "hypothesis": "no idea"}) + "\n")
    judge = FakeLLM(["yes", "no"])
    out = tmp_path / "judged.jsonl"
    longmemeval.run_judge(judge, instances, hyp, out)
    summary = longmemeval.report(instances, out, tmp_path / "summary.json")
    assert summary["accuracy"] == 0.5
    assert summary["per_type"]["abstention"]["accuracy"] == 0.0


# ---------- beam judging ----------

def test_beam_judge_rubric_scores():
    judge = FakeLLM(['{"score": 1.0, "reason": "ok"}',
                     '{"score": 0.5, "reason": "partial"}'])
    r = beam.judge_rubric(judge, "Q?", ["item1", "item2"], "resp")
    assert r["llm_judge_score"] == 0.75
    sent = judge.prompts[0][0]["content"]
    assert "item1" in sent and "resp" in sent and "Q?" in sent
    assert "<rubric_item>" not in sent and "<question>" not in sent


def test_beam_judge_handles_garbage_json():
    judge = FakeLLM(["score one hundred percent!!"])
    r = beam.judge_rubric(judge, "Q?", ["item"], "resp")
    assert 0.0 <= r["llm_judge_score"] <= 1.0


def test_beam_event_ordering_tau_perfect_and_reversed():
    class EqualityLLM:
        def chat(self, messages, **kw):
            body = messages[-1]["content"]
            first = body.split("First snippet:")[1].split("Second snippet:")[0].strip()
            second = body.split("Second snippet:")[1].strip()
            return "YES" if first == second else "NO"

    rubric = ["event A", "event B", "event C"]
    assert beam.event_ordering_tau(EqualityLLM(), rubric,
                                   "event A\nevent B\nevent C") == 1.0
    assert beam.event_ordering_tau(EqualityLLM(), rubric,
                                   "event C\nevent B\nevent A") == 0.0


# ---------- full-history baseline ----------

def test_full_history_prompt_contains_sessions():
    llm = FakeLLM(["the answer"])
    system = FullHistorySystem(llm)
    system.ingest(Session("s1", "2023/04/10", [Turn("user", "my car broke"),
                                               Turn("assistant", "sorry!")]))
    out = system.answer("What broke?", "2023/05/01", "Answer concisely.")
    assert out == "the answer"
    body = llm.prompts[0][1]["content"]
    assert "my car broke" in body and "2023/04/10" in body and "What broke?" in body
