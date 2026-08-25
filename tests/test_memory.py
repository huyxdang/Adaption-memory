import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from adaption_memory.interface import Session, Turn
from adaption_memory.llm import Usage
from adaption_memory.memory.answer import MemoryAnswerer
from adaption_memory.memory.budget import BudgetExceeded, SpendTracker
from adaption_memory.memory.extract import (LOCAL_INFERENCE_REVISION,
                                             LOCAL_MAX_RECORDS,
                                             LOCAL_MAX_TOKENS,
                                             MemoryExtractor)
from adaption_memory.memory.judge import MemoryJudge
from adaption_memory.memory.prompts import (extraction_schema,
                                            fewshot_messages,
                                            production_prompt)
from adaption_memory.memory.retrieve import HybridRetriever
from adaption_memory.memory.store import MemoryStore, Record
from adaption_memory.overnight import create_signal_split


class FakeEmbedder:
    def encode(self, texts):
        return [self.encode_one(text) for text in texts]

    def encode_one(self, text):
        vector = np.zeros(4, dtype=np.float32)
        vector[hash(text) % 4] = 1.0
        return vector


class FakeTrackedLLM:
    def __init__(self, model, response, reasoning_effort="none"):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.response = response
        self.usage = Usage()
        self.chat_calls = 0

    def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.usage.calls += 1
        self.usage.prompt_tokens += 11
        self.usage.completion_tokens += 3
        self.usage.latency_ms += 7.5
        return self.response


def make_record(record_id, content, *, supersedes_id=None, value=0):
    vector = np.zeros(4, dtype=np.float32)
    vector[value] = 1.0
    return Record(
        id=record_id, session_id="s1", type="atomic", content=content,
        entities=["Noor"], created_at="2026-01-01",
        supersedes_id=supersedes_id, embedding=vector,
    )


def test_store_is_append_only_and_roundtrips_embeddings(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    record = make_record("old", "Noor lives in Hanoi.")
    store.add(record)
    loaded = store.get("old")
    assert loaded is not None
    assert loaded.as_dict() == record.as_dict()
    assert np.array_equal(loaded.embedding, record.embedding)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connection.execute(
            "UPDATE records SET content = 'changed' WHERE id = 'old'"
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connection.execute("DELETE FROM records WHERE id = 'old'")
    store.close()


def test_extractor_validation_enforces_candidate_and_exact_numbers():
    extractor = MemoryExtractor.__new__(MemoryExtractor)
    extractor.format_name = "F1"
    session = Session("s", "2026-01-01", [Turn("Noor", "Budget is $4,800")])
    parsed = {"records": [
        {"type": "atomic", "content": "Budget is $4,800.",
         "entities": ["Noor"], "supersedes_id": "shown"},
        {"type": "atomic", "content": "Budget is $9,999.",
         "entities": ["Noor"], "supersedes_id": None},
        {"type": "narrative", "content": "Noor approved a budget.",
         "entities": ["Noor"], "supersedes_id": "invented"},
    ]}
    valid, rejected = extractor._validate(
        parsed, "Noor: Budget is $4,800", {"shown"}, session
    )
    assert len(valid) == 1
    assert len(rejected) == 2
    assert any("exact-string" in reason
               for item in rejected for reason in item["reasons"])
    assert any("not shown" in reason
               for item in rejected for reason in item["reasons"])


def test_extractor_checkpoint_replays_usage_without_spend_callback(tmp_path):
    store = MemoryStore(tmp_path / "store.sqlite3")
    checkpoint = tmp_path / "extractions.jsonl"
    session = Session("s1", "2026-01-01", [Turn("user", "Remember tea.")])
    first_llm = FakeTrackedLLM("qwen3:4b", '{"r":[]}')
    first = MemoryExtractor(
        first_llm, store, FakeEmbedder(), checkpoint,
        fewshot=False, arm="qwen3-4b-zeroshot",
    )
    first.extract(session)
    saved = json.loads(checkpoint.read_text().splitlines()[0])
    assert saved["usage"]["calls"] == 1

    resumed_llm = FakeTrackedLLM("qwen3:4b", '{"r":[]}')
    resumed = MemoryExtractor(
        resumed_llm, store, FakeEmbedder(), checkpoint,
        fewshot=False, arm="qwen3-4b-zeroshot",
    )
    resumed.extract(session)
    resumed.extract(session)
    assert resumed_llm.chat_calls == 0
    assert resumed_llm.usage.snapshot() == first_llm.usage.snapshot()
    store.close()


def test_local_extractor_has_bounded_output(tmp_path):
    store = MemoryStore(tmp_path / "store.sqlite3")
    extractor = MemoryExtractor(
        FakeTrackedLLM("qwen3:4b", '{"r":[]}'),
        store, FakeEmbedder(), tmp_path / "extractions.jsonl",
        fewshot=False,
    )
    schema = extraction_schema("F1", max_records=extractor.max_records)
    compact_schema = extraction_schema(
        "F1", max_records=extractor.max_records, compact=True
    )
    records = compact_schema["json_schema"]["schema"]["properties"]["r"]
    assert extractor.max_tokens == LOCAL_MAX_TOKENS
    assert LOCAL_INFERENCE_REVISION == "compact-v2"
    assert records["maxItems"] == LOCAL_MAX_RECORDS
    assert records["items"]["properties"]["e"]["maxItems"] == 3
    store.close()


def test_semantically_invalid_records_are_dropped_without_model_retry(tmp_path):
    store = MemoryStore(tmp_path / "store.sqlite3")
    response = json.dumps({"r": [
        {"t": "n", "c": "Noor chose tea.",
         "e": ["Noor"], "s": None},
        {"t": "a", "c": "Noor chose 99 teas.",
         "e": ["Noor"], "s": None},
    ]})
    llm = FakeTrackedLLM("qwen3:4b", response)
    extractor = MemoryExtractor(
        llm, store, FakeEmbedder(), tmp_path / "extractions.jsonl",
        fewshot=False,
    )
    result = extractor.extract(
        Session("s1", "2026-01-01", [Turn("user", "Noor chose tea.")])
    )
    assert llm.chat_calls == 1
    assert [record.content for record in result.records] == ["Noor chose tea."]
    assert len(result.rejected) == 1
    assert result.repaired is False
    store.close()


def test_truncated_compact_tail_preserves_complete_records_without_retry(tmp_path):
    store = MemoryStore(tmp_path / "store.sqlite3")
    response = (
        '{"r":[{"t":"n","c":"Noor chose tea.","e":["Noor"],"s":null},'
        '{"t":"n","c":"unfinished tail'
    )
    llm = FakeTrackedLLM("qwen3:4b", response)
    extractor = MemoryExtractor(
        llm, store, FakeEmbedder(), tmp_path / "extractions.jsonl",
        fewshot=False,
    )
    result = extractor.extract(
        Session("s1", "2026-01-01", [Turn("user", "Noor chose tea.")])
    )
    assert llm.chat_calls == 1
    assert [record.content for record in result.records] == ["Noor chose tea."]
    assert len(result.rejected) == 1
    assert result.repaired is False
    store.close()


def test_answer_and_judge_checkpoints_replay_usage(tmp_path):
    answer_llm = FakeTrackedLLM("gpt-5.6-luna", "Tea")
    MemoryAnswerer(answer_llm, tmp_path / "answers.jsonl", "arm").answer(
        "What?", []
    )
    resumed_answer_llm = FakeTrackedLLM("gpt-5.6-luna", "unused")
    answer, _ = MemoryAnswerer(
        resumed_answer_llm, tmp_path / "answers.jsonl", "arm"
    ).answer("What?", [])
    assert answer == "Tea"
    assert resumed_answer_llm.chat_calls == 0
    assert resumed_answer_llm.usage.snapshot() == answer_llm.usage.snapshot()

    judge_llm = FakeTrackedLLM(
        "gpt-5.6-luna", '{"label":true,"reason":"correct"}'
    )
    kwargs = dict(
        question_id="q1", benchmark="beam", question="What?",
        reference="Tea", response="Tea",
    )
    MemoryJudge(judge_llm, tmp_path / "judge.jsonl").judge(**kwargs)
    resumed_judge_llm = FakeTrackedLLM("gpt-5.6-luna", "unused")
    judgement = MemoryJudge(
        resumed_judge_llm, tmp_path / "judge.jsonl"
    ).judge(**kwargs)
    assert judgement["label"] is True
    assert resumed_judge_llm.chat_calls == 0
    assert resumed_judge_llm.usage.snapshot() == judge_llm.usage.snapshot()


def test_hybrid_retrieval_demotes_stale_and_expands_successor(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add(make_record("old", "Noor lives in Hanoi.", value=0))
    store.add(make_record("new", "Noor lives in Da Nang.",
                          supersedes_id="old", value=1))
    retriever = HybridRetriever(
        store, FakeEmbedder(), tmp_path / "retrievals.jsonl",
        k=1, demotion_factor=0.1,
    )
    results = retriever.retrieve("Where does Noor live now?")
    ids = [item.record.id for item in results]
    assert ids == ["new"]
    checkpoint = [json.loads(line) for line in
                  (tmp_path / "retrievals.jsonl").read_text().splitlines()]
    assert checkpoint[0]["input_hash"]
    store.close()


def test_spend_tracker_hard_cap_and_atomic_updates(tmp_path):
    tracker = SpendTracker(tmp_path / "spend.json", cap_usd=0.001)
    tracker.on_call({
        "model": "gpt-5.6-luna", "prompt_tokens": 100,
        "completion_tokens": 10, "reasoning_tokens": 0,
        "latency_ms": 12.5, "cost_usd": 0.000032,
    })
    assert json.loads((tmp_path / "spend.json").read_text())["calls"] == 1
    with pytest.raises(BudgetExceeded):
        tracker.before_call({
            "model": "gpt-5.6-luna",
            "messages": [{"content": "x" * 100_000}],
            "max_tokens": 1000,
        })


def test_spend_trackers_reload_under_file_lock(tmp_path):
    path = tmp_path / "spend.json"
    first = SpendTracker(path, cap_usd=1)
    second = SpendTracker(path, cap_usd=1)
    event = {
        "model": "gpt-5.6-luna", "prompt_tokens": 10,
        "completion_tokens": 2, "reasoning_tokens": 0,
        "latency_ms": 1.0, "cost_usd": 0.0000044,
    }
    first.on_call(event)
    second.on_call(event)
    saved = json.loads(path.read_text())
    assert saved["calls"] == 2
    assert saved["prompt_tokens"] == 20


def test_signal_split_is_stratified_and_complete():
    split = create_signal_split()
    for details in split["benchmarks"].values():
        assert len(details["dev"]) + len(details["holdout"]) == 30
        assert not (set(details["dev"]) & set(details["holdout"]))


@pytest.mark.parametrize("format_name", ["F1", "F2", "F3", "F4"])
def test_each_format_has_five_fewshot_examples(format_name):
    messages = fewshot_messages(format_name)
    assert len(messages) == 10
    outputs = [json.loads(message["content"])
               for message in messages if message["role"] == "assistant"]
    assert len(outputs) == 5
    types = {record["type"] for output in outputs
             for record in output["records"]}
    if format_name == "F4":
        assert types <= {"memory"}
    else:
        assert types <= {"atomic", "narrative"}
    if format_name == "F2":
        assert all(isinstance(record["content"], dict)
                   for output in outputs for record in output["records"]
                   if record["type"] == "atomic")


def test_coverage_prompt_is_explicit_and_f4_only():
    prompt = production_prompt("F4", "coverage")
    assert "Preserve every explicit fact" in prompt
    assert 'type="memory"' in prompt
    with pytest.raises(ValueError, match="only for F4"):
        production_prompt("F1", "coverage")


def test_validated_prompt_excludes_metadata_dates_and_derived_values():
    prompt = production_prompt("F4", "validated")
    assert "Never copy, reformat, or mention that metadata date" in prompt
    assert "Never calculate or infer durations" in prompt
    assert "non-empty string" in prompt


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "mini" / "smoke"
         / "locomo.json").exists(),
    reason="smoke tier data not generated",
)
def test_fastloop_runs_offline_and_resumes_from_checkpoints(tmp_path, monkeypatch):
    from adaption_memory import overnight
    from adaption_memory.memory import system as memory_system

    extraction = json.dumps({"records": [{
        "type": "narrative",
        "content": "The speakers discussed their plans.",
        "entities": ["speakers"],
        "supersedes_id": None,
    }]})
    made = []

    def fake_make_llm(model, **kwargs):
        llm = FakeTrackedLLM(model, extraction)
        made.append(llm)
        return llm

    monkeypatch.setattr(overnight, "make_llm", fake_make_llm)
    monkeypatch.setattr(memory_system, "LocalEmbedder", FakeEmbedder)
    monkeypatch.setattr(overnight, "RESULTS", tmp_path)
    tracker = SpendTracker(tmp_path / "spend.json", cap_usd=40.0)

    first = overnight.run_fastloop(
        arm="luna-target", tracker=tracker,
        benchmarks=("locomo",), limit=1,
    )
    locomo_summary = first["benchmarks"]["locomo"]
    assert locomo_summary["questions"] > 0
    assert locomo_summary["store_recall"] is not None
    first_calls = sum(llm.chat_calls for llm in made)
    assert first_calls > 0

    run_dir = tmp_path / "fastloop" / "luna-target" / Path(first["run_dir"]).name
    assert (run_dir / "locomo" / "summary.json").exists()
    assert (run_dir / "locomo" / "misses.jsonl").exists()

    made.clear()
    second = overnight.run_fastloop(
        arm="luna-target", tracker=tracker,
        benchmarks=("locomo",), limit=1,
    )
    assert sum(llm.chat_calls for llm in made) == 0
    assert second["benchmarks"]["locomo"]["store_recall"] == \
        locomo_summary["store_recall"]


def test_fastloop_never_answers():
    from adaption_memory.overnight import _NoAnswerLLM
    with pytest.raises(RuntimeError):
        _NoAnswerLLM().chat([{"role": "user", "content": "hi"}])


def test_fastloop_config_hash_tracks_prompt_content(monkeypatch):
    from adaption_memory import overnight
    base = overnight.fastloop_config_hash("luna-target", "F1", "base")
    assert base == overnight.fastloop_config_hash("luna-target", "F1", "base")
    assert base != overnight.fastloop_config_hash("qwen3-4b-zeroshot", "F1", "base")
    assert base != overnight.fastloop_config_hash("luna-target", "F4", "base")
    monkeypatch.setattr(overnight, "production_prompt",
                        lambda *args, **kwargs: "edited prompt")
    assert base != overnight.fastloop_config_hash("luna-target", "F1", "base")


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "mini" / "smoke"
         / "longmemeval.json").exists(),
    reason="smoke tier data not generated",
)
def test_fastloop_runs_conversations_concurrently_and_isolated(tmp_path, monkeypatch):
    from adaption_memory import overnight
    from adaption_memory.memory import system as memory_system

    extraction = json.dumps({"records": [{
        "type": "narrative",
        "content": "The user shared some plans.",
        "entities": ["user"],
        "supersedes_id": None,
    }]})

    def fake_make_llm(model, **kwargs):
        return FakeTrackedLLM(model, extraction)

    monkeypatch.setattr(overnight, "make_llm", fake_make_llm)
    monkeypatch.setattr(memory_system, "LocalEmbedder", FakeEmbedder)
    monkeypatch.setattr(overnight, "RESULTS", tmp_path)
    tracker = SpendTracker(tmp_path / "spend.json", cap_usd=40.0)

    result = overnight.run_fastloop(
        arm="luna-target", tracker=tracker,
        benchmarks=("longmemeval",), limit=3, workers=2,
    )
    summary = result["benchmarks"]["longmemeval"]
    assert summary["conversations"] == 3
    assert summary["questions"] == 3
    assert result["workers"] == 2
    assert result["sequential_estimate_seconds"] >= 0
    run_dir = tmp_path / "fastloop" / "luna-target" / Path(result["run_dir"]).name
    checkpoint_dirs = sorted(
        path.name for path in (run_dir / "longmemeval" / "ckpt").iterdir()
    )
    assert len(checkpoint_dirs) == 3
    stores = list((run_dir / "longmemeval" / "stores").glob("*.sqlite3"))
    assert len(stores) == 3


def _make_extractor(tmp_path, emission, response, candidates=()):
    from adaption_memory.memory.extract import MemoryExtractor
    store = MemoryStore(tmp_path / f"{emission}.sqlite3")
    for record in candidates:
        store.add(record)
    llm = FakeTrackedLLM("qwen3:4b", response)
    extractor = MemoryExtractor(
        llm, store, FakeEmbedder(), tmp_path / f"{emission}.jsonl",
        fewshot=False, format_name="F1", arm="qwen3-4b-zeroshot",
        emission=emission,
    )
    return extractor, store


def test_lines_emission_parses_and_maps_updates(tmp_path):
    old = make_record("old-1", "Noor lives in Hanoi on 2026-01-02.", value=0)
    raw = ("A: Noor moved on 2026-03-01.\n"
           "garbage line without a prefix\n"
           "N updates 1: Noor relocated because her team moved.\n")
    extractor, store = _make_extractor(tmp_path, "lines", raw, [old])
    result = extractor.extract(Session("s2", "2026-03-02", [
        Turn("user", "Noor moved on 2026-03-01. Her team moved."),
    ]))
    contents = {record.content: record for record in result.records}
    assert "Noor moved on 2026-03-01." in contents
    narrative = contents["Noor relocated because her team moved."]
    assert narrative.type == "narrative"
    assert narrative.supersedes_id == "old-1"
    assert narrative.entities == ["Noor"]
    store.close()


def test_simple_emission_maps_numbered_updates(tmp_path):
    old = make_record("old-9", "Noor lives in Hanoi.", value=0)
    raw = json.dumps({"records": [
        {"type": "atomic", "content": "Noor lives in Da Nang.", "updates": 1},
        {"type": "atomic", "content": "Meeting on 2026-05-05.", "updates": 7},
    ]})
    extractor, store = _make_extractor(tmp_path, "simple", raw, [old])
    result = extractor.extract(Session("s2", None, [
        Turn("user", "Noor lives in Da Nang now. Meeting on 2026-05-05."),
    ]))
    by_content = {record.content: record for record in result.records}
    assert by_content["Noor lives in Da Nang."].supersedes_id == "old-9"
    # out-of-range updates keeps the fact but drops the link
    assert by_content["Meeting on 2026-05-05."].supersedes_id is None
    store.close()


def test_lines_emission_none_output(tmp_path):
    extractor, store = _make_extractor(tmp_path, "lines", "NONE")
    result = extractor.extract(Session("s1", None, [Turn("user", "hi")]))
    assert result.records == [] and result.schema_valid
    store.close()


def test_derive_entities_skips_stopwords():
    from adaption_memory.memory.extract import derive_entities
    assert derive_entities("The team met Audrey and Andrew in Hanoi today.") \
        == ["Audrey", "Andrew", "Hanoi"]


def test_interaction_chunks_preserve_order():
    from adaption_memory.overnight import interaction_chunks
    session = Session("s7", "2026-01-01",
                      [Turn("user", f"t{i}") for i in range(5)])
    chunks = interaction_chunks(session)
    assert [c.session_id for c in chunks] == ["s7#i0", "s7#i1", "s7#i2"]
    assert [len(c.turns) for c in chunks] == [2, 2, 1]
    assert chunks[0].turns[0].content == "t0"


def test_fastloop_config_hash_covers_new_dimensions():
    from adaption_memory.overnight import fastloop_config_hash
    base = fastloop_config_hash("qwen3-4b-zeroshot", "F1", "base")
    assert base != fastloop_config_hash("qwen3-4b-zeroshot", "F1", "base",
                                        emission="simple")
    assert base != fastloop_config_hash("qwen3-4b-zeroshot", "F1", "base",
                                        granularity="interaction")
    assert base != fastloop_config_hash("qwen3-4b-zeroshot", "F1", "base",
                                        local_thinking=True)
    assert base != fastloop_config_hash("qwen3-4b-zeroshot", "F1",
                                        "coverage-f1")


def test_simple_emission_tolerates_malformed_updates(tmp_path):
    raw = json.dumps({"records": [
        {"type": "atomic", "content": "Rent is $900.", "updates": ""},
        {"type": "atomic", "content": "Lease ends 2026-09-01.", "updates": "x"},
    ]})
    extractor, store = _make_extractor(tmp_path, "simple", raw)
    result = extractor.extract(Session("s1", None, [
        Turn("user", "Rent is $900. Lease ends 2026-09-01."),
    ]))
    assert {record.content for record in result.records} == \
        {"Rent is $900.", "Lease ends 2026-09-01."}
    assert all(record.supersedes_id is None for record in result.records)
    store.close()
