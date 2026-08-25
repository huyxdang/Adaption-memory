import json
import sqlite3

import numpy as np
import pytest

from adaption_memory.interface import Session, Turn
from adaption_memory.llm import Usage
from adaption_memory.memory.answer import MemoryAnswerer
from adaption_memory.memory.budget import BudgetExceeded, SpendTracker
from adaption_memory.memory.extract import (LOCAL_MAX_RECORDS,
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
    assert records["maxItems"] == LOCAL_MAX_RECORDS
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
