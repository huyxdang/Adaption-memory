"""Offline tests: loaders against the real data files, scoring logic against
known cases, and judge plumbing against a fake LLM. No model server needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from adaption_memory.evals import beam, locomo, longmemeval, mini
from adaption_memory.evals.provenance import (ensure_config, file_fingerprint,
                                               require_config)
from adaption_memory.evals.run import endpoint_identity
from adaption_memory.interface import Session, Turn
from adaption_memory.llm import LLM, Usage
from adaption_memory.systems import REGISTRY
from adaption_memory.systems.adaptive import AdaptiveMemorySystem, AtomicMemory
from adaption_memory.systems.full_history import FullHistorySystem

DATA = Path(__file__).resolve().parent.parent / "data"


class FakeLLM:
    """Returns canned responses; records prompts for assertions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.kwargs = []

    def chat(self, messages, **kw):
        self.prompts.append(messages)
        self.kwargs.append(kw)
        return self.responses.pop(0) if self.responses else self.responses_default

    responses_default = "fake"


@pytest.mark.parametrize(("mode", "expected"), [
    ("json-schema", "schema"),
    ("json-object", "object"),
    ("prompt-only", None),
])
def test_llm_uses_configured_structured_output_mode(mode, expected):
    class Completions:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content='{"ok": true}')
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice], usage=None)

    completions = Completions()
    llm = LLM.__new__(LLM)
    llm.model = "fake"
    llm.temperature = 0.0
    llm.max_tokens = 64
    llm.reasoning_effort = None
    llm.structured_output = mode
    llm.usage = Usage()
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    response_format = {"type": "json_object"}

    assert llm.chat([], response_format=response_format) == '{"ok": true}'
    assert completions.kwargs["max_tokens"] == 64
    assert "max_completion_tokens" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs
    if expected == "schema":
        assert completions.kwargs["response_format"] is response_format
    elif expected == "object":
        assert completions.kwargs["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in completions.kwargs


@pytest.mark.parametrize("reasoning_effort", [None, "medium"])
def test_llm_omits_temperature_for_gpt5_model(reasoning_effort):
    class Completions:
        kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content="ready")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=None
            )

    completions = Completions()
    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.6-luna"
    llm.temperature = 0.0
    llm.max_tokens = 64
    llm.reasoning_effort = reasoning_effort
    llm.structured_output = "json-schema"
    llm.usage = Usage()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    assert llm.chat([]) == "ready"
    assert "temperature" not in completions.kwargs
    assert completions.kwargs["max_completion_tokens"] == 64
    assert "max_tokens" not in completions.kwargs
    if reasoning_effort:
        assert completions.kwargs["reasoning_effort"] == reasoning_effort
    else:
        assert "reasoning_effort" not in completions.kwargs


def test_llm_does_not_forward_openai_key_to_local_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-test-key")
    local = LLM("qwen3:1.7b", base_url="http://localhost:11434/v1")
    hosted = LLM("gpt-5.6-luna")

    assert local.client.api_key == "EMPTY"
    assert hosted.client.api_key == "openai-secret-test-key"


def test_llm_resolves_openai_base_url_without_leaking_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-test-key")

    llm = LLM("gpt-5.6-luna")

    assert str(llm.client.base_url).rstrip("/") == \
        "https://gateway.example.test/v1"
    assert llm.client.api_key == "EMPTY"


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


@pytest.mark.skipif(not all((DATA / name).exists() for name in [
    "longmemeval_s_cleaned.json", "longmemeval_oracle.json",
    "locomo10.json", "beam_100k.parquet",
]), reason="data not downloaded")
def test_preflight_tier_generation_native_loaders_and_determinism(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = mini.generate_suite(DATA, first)
    mini.generate_suite(DATA, second)

    smoke_lme = longmemeval.load_instances(first / "smoke/longmemeval.json")
    assert len(smoke_lme) == len(mini.LONGMEMEVAL_TYPES) + 1
    assert set(i["question_type"] for i in smoke_lme) == \
        set(mini.LONGMEMEVAL_TYPES)
    assert sum("_abs" in i["question_id"] for i in smoke_lme) == 1
    assert all(set(i["answer_session_ids"]) <= set(i["haystack_session_ids"])
               for i in smoke_lme)

    smoke_locomo = locomo.load_samples(first / "smoke/locomo.json")
    assert len(smoke_locomo) == 1
    assert {q["category"] for q in smoke_locomo[0]["qa"]} == \
        set(locomo.CATEGORY_NAMES)
    n_sessions = len(locomo.sessions_of(smoke_locomo[0]))
    assert all(1 <= int(evidence.split(":")[0][1:]) <= n_sessions
               for q in smoke_locomo[0]["qa"] for evidence in q["evidence"])

    smoke_beam = beam.load_conversations(first / "smoke/beam.parquet")
    assert len(smoke_beam) == 1
    assert set(smoke_beam[0]["questions"]) == set(beam.QUESTION_TYPES)
    assert all(len(smoke_beam[0]["questions"][qtype]) == 1
               for qtype in beam.QUESTION_TYPES)
    smoke_manifest = json.loads(
        (first / "smoke/manifest.json").read_text(encoding="utf-8")
    )
    assert [selection["question_id"] for selection in
            smoke_manifest["benchmarks"]["longmemeval"]["selections"]] == [
        "6f9b354f", "1b9b7252", "b0479f84", "099778bb",
        "gpt4_2c50253f", "2698e78f", "0ddfec37_abs",
    ]
    assert smoke_manifest["benchmarks"]["locomo"]["mini"][
        "source_sample_id"
    ] == "conv-44"
    assert [selection["source_question_index"] for selection in
            smoke_manifest["benchmarks"]["locomo"]["selections"]] == [
        32, 4, 43, 115, 125,
    ]
    assert smoke_manifest["benchmarks"]["beam"]["mini"][
        "source_conversation_id"
    ] == "9"
    assert [selection["source_question_index"] for selection in
            smoke_manifest["benchmarks"]["beam"]["selections"]] == [
        0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
    ]

    signal_lme = longmemeval.load_instances(first / "signal/longmemeval.json")
    assert len(signal_lme) == mini.SIGNAL_QUESTIONS_PER_BENCHMARK
    signal_lme_counts = {
        question_type: sum(item["question_type"] == question_type
                           for item in signal_lme)
        for question_type in mini.LONGMEMEVAL_TYPES
    }
    assert min(signal_lme_counts.values()) >= mini.SIGNAL_MINIMUM_PER_STRATUM
    assert sum("_abs" in item["question_id"] for item in signal_lme) >= \
        mini.SIGNAL_MINIMUM_PER_STRATUM
    assert all(set(item["answer_session_ids"]) <=
               set(item["haystack_session_ids"]) for item in signal_lme)

    signal_locomo = locomo.load_samples(first / "signal/locomo.json")
    assert len(signal_locomo) == mini.SIGNAL_LOCOMO_SOURCES
    signal_locomo_questions = [question for sample in signal_locomo
                               for question in sample["qa"]]
    assert len(signal_locomo_questions) == mini.SIGNAL_QUESTIONS_PER_BENCHMARK
    assert min(sum(question["category"] == category
                   for question in signal_locomo_questions)
               for category in locomo.CATEGORY_NAMES) >= \
        mini.SIGNAL_MINIMUM_PER_STRATUM
    assert all(
        1 <= int(evidence.split(":")[0][1:]) <= len(locomo.sessions_of(sample))
        for sample in signal_locomo
        for question in sample["qa"]
        for evidence in question.get("evidence", [])
    )

    signal_beam = beam.load_conversations(first / "signal/beam.parquet")
    assert len(signal_beam) == mini.SIGNAL_BEAM_SOURCES
    assert all(len(conversation["questions"][question_type]) == 1
               for conversation in signal_beam
               for question_type in beam.QUESTION_TYPES)
    assert sum(len(conversation["questions"][question_type])
               for conversation in signal_beam
               for question_type in beam.QUESTION_TYPES) == \
        mini.SIGNAL_QUESTIONS_PER_BENCHMARK

    signal_manifest = json.loads(
        (first / "signal/manifest.json").read_text(encoding="utf-8")
    )
    assert signal_manifest["benchmarks"]["longmemeval"]["mini"][
        "full_context_cases"
    ] == len(mini.LONGMEMEVAL_TYPES)
    assert {
        question_type: sum(
            selection["full_context"]
            and selection["question_type"] == question_type
            for selection in signal_manifest["benchmarks"]["longmemeval"][
                "selections"
            ]
        )
        for question_type in mini.LONGMEMEVAL_TYPES
    } == {question_type: 1 for question_type in mini.LONGMEMEVAL_TYPES}
    assert signal_manifest["benchmarks"]["longmemeval"]["mini"][
        "character_retention"
    ] >= 0.4
    assert all(
        selection["source_context"] == selection["mini_context"]
        if selection["full_context"]
        else len(selection["distractor_session_ids"])
             == mini.SIGNAL_LONGMEMEVAL_DISTRACTORS
        for selection in signal_manifest["benchmarks"]["longmemeval"][
            "selections"
        ]
    )
    assert signal_manifest["benchmarks"]["locomo"]["mini"][
        "character_retention"
    ] == 1.0
    assert all(source["source_context"] == source["context"]
               for source in signal_manifest["benchmarks"]["locomo"][
                   "mini"
               ]["sources"])
    assert signal_manifest["benchmarks"]["beam"]["mini"][
        "character_retention"
    ] == 1.0
    assert all(source["source_context"] == source["context"]
               for source in signal_manifest["benchmarks"]["beam"][
                   "mini"
               ]["sources"])
    assert all(
        details["representativeness"]["total_variation_distance"] < 0.06
        for details in signal_manifest["benchmarks"].values()
    )

    assert manifest["deterministic"] is True
    assert set(manifest["source_data"]) == {
        "longmemeval_s_cleaned.json", "longmemeval_oracle.json",
        "locomo10.json", "beam_100k.parquet",
    }
    assert all(len(fingerprint["sha256"]) == 64 and fingerprint["bytes"] > 0
               for fingerprint in manifest["source_data"].values())
    assert manifest["tiers"]["smoke"]["questions"] == 22
    assert manifest["tiers"]["signal"]["questions"] == 90
    assert (first / "manifest.json").exists()
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*") if path.is_file()
    } == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }


def test_result_provenance_rejects_config_mismatch(tmp_path):
    data = tmp_path / "mini.json"
    data.write_text("[]", encoding="utf-8")
    out = tmp_path / "results"
    config = {"mode": "smoke", "data": file_fingerprint(data)}

    ensure_config(out, "dataset.json", config, ("answers.jsonl",))
    assert json.loads((out / "dataset.json").read_text()) == config
    with pytest.raises(SystemExit, match="configuration mismatch"):
        ensure_config(out, "dataset.json", {**config, "mode": "signal"},
                      ("answers.jsonl",))


def test_result_provenance_rejects_unverified_artifact(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "answers.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="unverified artifacts"):
        ensure_config(out, "dataset.json", {"mode": "mini"},
                      ("answers.jsonl",))


def test_stage_provenance_requires_config_and_artifacts(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    with pytest.raises(SystemExit, match="missing result provenance"):
        require_config(out, "answer-config.json", ("answers.jsonl",))

    (out / "answer-config.json").write_text(
        '{"model": "test"}\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="missing required result artifacts"):
        require_config(out, "answer-config.json", ("answers.jsonl",))

    (out / "answers.jsonl").write_text("{}\n", encoding="utf-8")
    assert require_config(
        out, "answer-config.json", ("answers.jsonl",)
    ) == {"model": "test"}


def test_endpoint_identity_drops_credentials_and_query_tokens():
    assert endpoint_identity(
        "https://user:password@example.test:8443/v1?token=secret#fragment"
    ) == "https://example.test:8443/v1"


# ---------- locomo scoring (official semantics) ----------

def test_locomo_f1_exact():
    assert locomo.score_one("7 May 2023", "7 May 2023", 2) == 1.0


def test_locomo_f1_partial():
    assert 0 < locomo.score_one("she went in May 2023", "7 May 2023", 2) < 1.0


def test_locomo_open_domain_takes_first_semicolon_field():
    assert locomo.score_one("Psychology", "Psychology; counseling", 3) == \
        locomo.score_one("Psychology", "Psychology", 3)
    assert locomo.oracle_answer({
        "category": 3,
        "answer": "Psychology; counseling",
    }) == "Psychology"


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
    longmemeval.run_judge(judge, instances, hyp, out, max_tokens=77)
    assert judge.kwargs[0]["max_tokens"] == 77
    summary = longmemeval.report(instances, out, tmp_path / "summary.json")
    assert summary["accuracy"] == 0.5
    assert summary["per_type"]["abstention"]["accuracy"] == 0.0


# ---------- beam judging ----------

def test_beam_judge_rubric_scores():
    judge = FakeLLM(['{"score": 1.0, "reason": "ok"}',
                     '{"score": 0.5, "reason": "partial"}'])
    r = beam.judge_rubric(judge, "Q?", ["item1", "item2"], "resp",
                          max_tokens=77)
    assert r["llm_judge_score"] == 0.75
    sent = judge.prompts[0][0]["content"]
    assert "item1" in sent and "resp" in sent and "Q?" in sent
    assert "<rubric_item>" not in sent and "<question>" not in sent
    assert judge.kwargs[0]["max_tokens"] == 77


def test_beam_judge_handles_garbage_json():
    judge = FakeLLM(["score one hundred percent!!"])
    r = beam.judge_rubric(judge, "Q?", ["item"], "resp")
    assert 0.0 <= r["llm_judge_score"] <= 1.0


def test_beam_event_ordering_oracle_emits_one_event_per_line():
    question = {
        "ordering_tested": ["first", "second"],
        "rubric": ["event A", "event B"],
        "answer": "First event A, then event B.",
    }
    assert beam.oracle_answer(question) == "event A\nevent B"


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


# ---------- adaptive write-time memory ----------

class UsageFakeLLM(FakeLLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.usage = Usage()

    def chat(self, messages, **kw):
        self.usage.calls += 1
        self.usage.prompt_tokens += 10
        self.usage.completion_tokens += 2
        return super().chat(messages, **kw)


def test_adaptive_memory_appends_and_supersedes():
    llm = UsageFakeLLM([
        json.dumps({
            "narratives": [{"content": "Sam moved to Paris for a new role."}],
            "atomics": [{"key": "sam.current_city", "value": "Paris",
                         "context": "Moved for work", "supersedes": None}],
        }),
        json.dumps({
            "narratives": [{"content": "Sam later relocated from Paris to Lyon."}],
            "atomics": [{"key": "sam.current_city", "value": "Lyon",
                         "context": "Relocated after changing teams",
                         "supersedes": "a1"}],
        }),
        "Sam lives in Lyon.",
    ])
    system = AdaptiveMemorySystem(llm)
    system.ingest(Session("s1", "2025-01-01", [Turn("user", "I moved to Paris")]))
    system.ingest(Session("s2", "2026-01-01", [Turn("user", "I moved to Lyon")]))

    assert [memory.value for memory in system.atomics] == ["Paris", "Lyon"]
    assert [memory.key for memory in system.atomics] == ["sam.current_city"] * 2
    assert system.atomics[0].superseded_by == "a2"
    assert system.atomics[1].active
    assert len(system.narratives) == 2
    response_format = llm.kwargs[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True

    answer = system.answer("Where does Sam live now?")
    assert answer == "Sam lives in Lyon."
    prompt = llm.prompts[-1][1]["content"]
    assert "sam.current_city = Lyon" in prompt
    assert "active" in prompt and "superseded by a2" in prompt


def test_adaptive_memory_deduplicates_exact_fact_and_handles_bad_json():
    llm = UsageFakeLLM([
        '{"narratives": [], "atomics": '
        '[{"key": "user.timezone", "value": "UTC+7"}]}',
        '{narratives: [], atomics: [{key: user.timezone, value: UTC+7}]}',
        "not json at all",
    ])
    system = AdaptiveMemorySystem(llm)
    session = Session("s1", None, [Turn("user", "My timezone is UTC+7")])
    system.ingest(session)
    system.ingest(session)
    system.ingest(session)
    assert len(system.atomics) == 1
    assert system.atomics[0].value == "UTC+7"


def test_adaptive_system_is_registered():
    assert REGISTRY["adaptive"] is AdaptiveMemorySystem


def test_adaptive_memory_rejects_placeholder_metadata_and_malformed_keys():
    system = AdaptiveMemorySystem(UsageFakeLLM([]))
    system._append_atomics([
        {"key": "stable.semantic.key", "value": "placeholder"},
        {"key": "session.date", "value": "2026-08-24"},
        {"key": "transcript.turn_count", "value": "4"},
        {"key": "conversation.id", "value": "s1"},
        {"key": "Sam.current_city", "value": "Lyon"},
        {"key": "missing_dot", "value": "bad"},
        {"key": "sam.current city", "value": "bad"},
        {"key": "sam.current_city", "value": "Lyon"},
    ], Session("s1", None, []))

    assert [(memory.key, memory.value) for memory in system.atomics] == [
        ("sam.current_city", "Lyon"),
    ]


def test_adaptive_memory_rejects_mismatched_supersession_reference():
    system = AdaptiveMemorySystem(UsageFakeLLM([]))
    session = Session("s1", None, [])
    system._append_atomics([{
        "key": "sam.current_city",
        "value": "Paris",
        "context": "explicit city",
        "supersedes": None,
    }], session)
    system._append_atomics([{
        "key": "sam.job",
        "value": "designer",
        "context": "explicit job",
        "supersedes": "a1",
    }], session)

    assert [(memory.key, memory.value, memory.active)
            for memory in system.atomics] == [
        ("sam.current_city", "Paris", True),
    ]


@pytest.mark.parametrize(("source_date", "expected"), [
    ("1:56 pm on 8 May, 2023", "7 May 2023"),
    ("2023/05/20 (Sat) 02:21", "19 May 2023"),
    ("March-15-2024", "14 March 2024"),
    ("2024-03-15T09:30:00", "14 March 2024"),
])
def test_adaptive_resolves_yesterday_across_benchmark_date_formats(
        source_date, expected):
    assert AdaptiveMemorySystem._resolve_relative_date(
        "person.event_date", "wrong date", "yesterday", source_date,
    ) == expected


def test_adaptive_does_not_rewrite_unscoped_or_negated_yesterday():
    resolve = AdaptiveMemorySystem._resolve_relative_date
    source_date = "1:56 pm on 8 May, 2023"
    assert resolve(
        "person.event_date", "20 July 2023", "not yesterday", source_date,
    ) == "20 July 2023"
    assert resolve(
        "person.event_date",
        "20 July 2023",
        "joined an activist group; support group was yesterday",
        source_date,
    ) == "20 July 2023"
    assert resolve(
        "caroline.lgbtq_support_group_date",
        "8 May 2023",
        "Caroline: I went to a LGBTQ support group yesterday",
        source_date,
    ) == "7 May 2023"


def test_adaptive_prompts_distinguish_event_and_source_session_dates():
    llm = UsageFakeLLM([
        json.dumps({
            "narratives": [{
                "content": "Caroline attended an LGBTQ support group on 7 May 2023.",
            }],
            "atomics": [{
                "key": "caroline.lgbtq_support_group_date",
                "value": "8 May 2023",
                "context": "said yesterday",
                "supersedes": None,
            }, {
                "key": "caroline.lgbtq_activist_group_join_date",
                "value": "20 July 2023",
                "context": "joined a different activist group last Tuesday",
                "supersedes": None,
            }, {
                "key": "caroline.lgbtq_group_date",
                "value": "11 July 2023",
                "context": "attended a later LGBTQ conference",
                "supersedes": None,
            }],
        }),
        "7 May 2023",
    ])
    system = AdaptiveMemorySystem(llm)
    system.ingest(Session(
        "s1",
        "1:56 pm on 8 May, 2023",
        [Turn("Caroline", "I went to an LGBTQ support group yesterday.")],
    ))

    assert "Resolve explicit relative" in llm.prompts[0][0]["content"]
    assert "Session date: 1:56 pm on 8 May, 2023" in llm.prompts[0][1]["content"]
    assert system.atomics[0].value == "7 May 2023"

    question = "When did Caroline go to the LGBTQ support group?"
    assert system.answer(question) == "7 May 2023"
    assert len(llm.prompts) == 1
    atomics, narratives = system.retrieve(question)
    rendered = system._render_memories(atomics, narratives)
    assert "caroline.lgbtq_support_group_date = 7 May 2023" in rendered
    assert "source_session_date=1:56 pm on 8 May, 2023" in rendered
    assert "; date=" not in rendered


def test_adaptive_exact_date_requires_active_specific_match():
    system = AdaptiveMemorySystem(UsageFakeLLM([]))
    old = AtomicMemory(
        "a1", "caroline.lgbtq_support_group_date", "7 May 2023", "short",
        "s1", "8 May 2023", superseded_by="a2",
    )
    current = AtomicMemory(
        "a2", "caroline.lgbtq_support_group_date", "8 May 2023",
        "many unrelated context tokens that lower its retrieval density",
        "s2", "9 May 2023",
    )
    loose = AtomicMemory(
        "a3", "caroline.lgbtq_group_date", "11 July 2023", "later event",
        "s3", "12 July 2023",
    )
    question = "When did Caroline go to the LGBTQ support group?"

    assert system._exact_date_answer(question, [old, loose, current]) == \
        "8 May 2023"


def test_adaptive_does_not_implicitly_overwrite_existing_key():
    system = AdaptiveMemorySystem(UsageFakeLLM([]))
    session = Session("s1", None, [])
    system._append_atomics([{
        "key": "caroline.lgbtq_support_group_date",
        "value": "7 May 2023",
        "context": "original event",
        "supersedes": None,
    }], session)
    system._append_atomics([{
        "key": "caroline.lgbtq_support_group_date",
        "value": "11 July 2023",
        "context": "different later event",
        "supersedes": None,
    }], Session("s2", None, []))

    assert [(memory.value, memory.active) for memory in system.atomics] == [
        ("7 May 2023", True),
    ]


def test_adaptive_write_context_is_bounded():
    system = AdaptiveMemorySystem(UsageFakeLLM([]), max_active_context=3)
    session = Session("seed", None, [])
    system._append_atomics([
        {"key": f"fact.{i}", "value": str(i)} for i in range(10)
    ], session)
    context = system._active_context("Tell me about fact 2")
    assert len(context) == 3
    assert any(memory.key == "fact.2" for memory in context)
    assert any(memory.key == "fact.9" for memory in context)


# ---------- write-time usage accounting ----------

class CostSystem:
    def __init__(self):
        self.current = Usage()

    def reset(self):
        self.current = Usage()

    def ingest(self, session):
        self.current.calls += 1
        self.current.prompt_tokens += 10

    def answer(self, question, question_date=None, instruction=None):
        self.current.calls += 1
        self.current.prompt_tokens += 2
        return "answer"

    def usage(self):
        return self.current.snapshot()


def test_locomo_attributes_write_usage_once(tmp_path):
    sample = {
        "sample_id": "sample",
        "conversation": {
            "session_1": [{"speaker": "Sam", "text": "hello"}],
        },
        "qa": [
            {"question": "q1", "answer": "a1", "category": 4},
            {"question": "q2", "answer": "a2", "category": 4},
        ],
    }
    out = tmp_path / "answers.jsonl"
    locomo.run_answers(CostSystem, [sample], out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["usage"]["calls"] == 2
    assert rows[0]["usage"]["prompt_tokens"] == 12
    assert rows[1]["usage"]["calls"] == 1
    assert rows[1]["usage"]["prompt_tokens"] == 2


def test_beam_attributes_write_usage_once(tmp_path):
    conversation = {
        "conversation_id": "conv",
        "chat": [[{"time_anchor": "2026-01-01", "role": "user",
                   "content": "hello"}]],
        "questions": {
            "abstention": [
                {"question": "q1", "rubric": ["r1"]},
                {"question": "q2", "rubric": ["r2"]},
            ],
        },
    }
    out = tmp_path / "answers.jsonl"
    beam.run_answers(CostSystem, [conversation], out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["usage"]["calls"] == 2
    assert rows[0]["usage"]["prompt_tokens"] == 12
    assert rows[1]["usage"]["calls"] == 1
    assert rows[1]["usage"]["prompt_tokens"] == 2


def test_locomo_resume_does_not_reattribute_write_usage(tmp_path):
    sample = {
        "sample_id": "sample",
        "conversation": {
            "session_1": [{"speaker": "Sam", "text": "hello"}],
        },
        "qa": [
            {"question": "q1", "answer": "a1", "category": 4},
            {"question": "q2", "answer": "a2", "category": 4},
        ],
    }
    out = tmp_path / "answers.jsonl"
    out.write_text(json.dumps({
        "question_id": "sample:0", "category": 4, "hypothesis": "answer",
        "usage": {"calls": 2, "prompt_tokens": 12, "completion_tokens": 0},
    }) + "\n")
    locomo.run_answers(CostSystem, [sample], out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[1]["question_id"] == "sample:1"
    assert rows[1]["usage"]["calls"] == 1
    assert rows[1]["usage"]["prompt_tokens"] == 2
