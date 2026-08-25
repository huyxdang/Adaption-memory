"""Resumable execution harness for ``OVERNIGHT_PLAN.md``.

The harness intentionally exposes only smoke and signal tiers. There is no
full-tier command path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from adaption_memory.evals import beam, locomo, longmemeval
from adaption_memory.evals.common import (add_usage, append_jsonl, read_jsonl,
                                          usage_delta)
from adaption_memory.interface import Session
from adaption_memory.llm import LLM
from adaption_memory.memory.budget import BudgetExceeded, SpendTracker
from adaption_memory.memory.extract import LOCAL_MAX_RECORDS, LOCAL_MAX_TOKENS
from adaption_memory.memory.judge import MemoryJudge
from adaption_memory.memory.report import build_report
from adaption_memory.memory.sft import build_sft
from adaption_memory.memory.system import WriteTimeMemorySystem


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mini"
RESULTS = ROOT / "results"
ARMS = {
    "qwen3-4b-zeroshot": {
        "model": "qwen3:4b", "base_url": "http://127.0.0.1:11434/v1",
        "fewshot": False, "structured_output": "json-schema",
    },
    "qwen3-4b-fewshot": {
        "model": "qwen3:4b", "base_url": "http://127.0.0.1:11434/v1",
        "fewshot": True, "structured_output": "json-schema",
    },
    "luna-target": {
        "model": "gpt-5.6-luna", "base_url": None,
        "fewshot": True, "structured_output": "json-schema",
    },
}
BENCHMARKS = ("longmemeval", "locomo", "beam")


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    reference: str
    date: str | None = None
    instruction: str | None = None
    locomo_category: int | None = None


@dataclass(frozen=True)
class Conversation:
    id: str
    sessions: list[Session]
    questions: list[Question]


def load_local_env(path: Path = ROOT / ".env.local") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"").strip("'")


def load_conversations(tier: str, benchmark: str) -> list[Conversation]:
    if tier not in {"smoke", "signal"}:
        raise ValueError("overnight runs are restricted to smoke and signal")
    if benchmark == "longmemeval":
        instances = longmemeval.load_instances(DATA / tier / "longmemeval.json")
        return [Conversation(
            id=str(instance["question_id"]),
            sessions=longmemeval.sessions_of(instance),
            questions=[Question(
                id=str(instance["question_id"]),
                category=str(instance["question_type"]),
                text=instance["question"],
                reference=str(instance["answer"]),
                date=instance.get("question_date"),
                instruction=longmemeval.ANSWER_INSTRUCTION,
            )],
        ) for instance in instances]
    if benchmark == "locomo":
        samples = locomo.load_samples(DATA / tier / "locomo.json")
        out = []
        for sample in samples:
            questions = []
            for index, item in enumerate(sample["qa"]):
                category = int(item["category"])
                reference = (str(item.get("answer", ""))
                             if category != 5 else "No information available.")
                questions.append(Question(
                    id=locomo.question_id(sample["sample_id"], index),
                    category=locomo.CATEGORY_NAMES[category],
                    text=item["question"], reference=reference,
                    instruction=locomo.ANSWER_INSTRUCTION,
                    locomo_category=category,
                ))
            out.append(Conversation(
                id=str(sample["sample_id"]),
                sessions=locomo.sessions_of(sample), questions=questions,
            ))
        return out
    if benchmark == "beam":
        conversations = beam.load_conversations(DATA / tier / "beam.parquet")
        out = []
        for conversation in conversations:
            questions = []
            for qtype in beam.QUESTION_TYPES:
                for index, item in enumerate(conversation["questions"].get(qtype, [])):
                    questions.append(Question(
                        id=beam.question_id(conversation["conversation_id"], qtype, index),
                        category=qtype,
                        text=item["question"],
                        reference=beam.oracle_answer(item),
                    ))
            out.append(Conversation(
                id=str(conversation["conversation_id"]),
                sessions=beam.sessions_of(conversation), questions=questions,
            ))
        return out
    raise ValueError(f"unknown benchmark: {benchmark}")


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in value)


def make_llm(model: str, *, base_url: str | None, tracker: SpendTracker,
             max_tokens: int, structured_output: str = "json-schema") -> LLM:
    return LLM(
        model=model, base_url=base_url,
        api_key=(os.getenv("OPENAI_API_KEY") if base_url is None else "ollama"),
        max_tokens=max_tokens,
        # Ollama's OpenAI-compatible endpoint maps this field to Qwen's
        # native `think: false`, preventing reasoning-only empty responses.
        reasoning_effort="none",
        structured_output=structured_output,
        before_call=tracker.before_call,
        on_call=tracker.on_call,
    )


def ensure_baselines() -> dict:
    destination = RESULTS / "baselines"
    destination.mkdir(parents=True, exist_ok=True)
    index = {
        "schema": 1,
        "tier": "signal subsample",
        "primary": "full-history-luna-none",
        "reason": "Matches the fixed Luna none answerer and Luna none judge in the overnight plan.",
        "benchmarks": {},
        "legacy_plan_numbers": {
            "longmemeval": {"metric": "accuracy", "value": 0.9000,
                            "configuration": "Luna medium answers / high judge"},
            "beam": {"metric": "overall", "value": 0.6595,
                     "configuration": "Luna medium answers / high judge"},
            "locomo": {"metric": "token_f1", "value": 0.4466,
                       "configuration": "Luna medium answers"},
        },
    }
    for benchmark in BENCHMARKS:
        source = RESULTS / "signal" / benchmark / "full-history-luna-none"
        target = destination / benchmark
        target.mkdir(parents=True, exist_ok=True)
        mapping = {
            "answers.jsonl": "predictions.jsonl",
            "summary.json": "summary.json",
            "answer-config.json": "answer-config.json",
        }
        if benchmark != "locomo":
            mapping.update({
                "judged.jsonl": "official-judged.jsonl",
                "judge-config.json": "judge-config.json",
            })
        missing = [name for name in mapping if not (source / name).exists()]
        if missing:
            index["benchmarks"][benchmark] = {"available": False, "missing": missing}
            continue
        for source_name, target_name in mapping.items():
            shutil.copy2(source / source_name, target / target_name)
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        index["benchmarks"][benchmark] = {
            "available": True,
            "predictions": str((target / "predictions.jsonl").relative_to(ROOT)),
            "summary": summary,
        }
    (destination / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index


def create_signal_split() -> dict:
    path = RESULTS / "splits" / "signal.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    split = {"schema": 1, "seed": "overnight-signal-v1", "benchmarks": {}}
    for benchmark in BENCHMARKS:
        strata: dict[str, list[str]] = defaultdict(list)
        for conversation in load_conversations("signal", benchmark):
            for question in conversation.questions:
                strata[question.category].append(question.id)
        dev, holdout = [], []
        for category, question_ids in sorted(strata.items()):
            ordered = sorted(question_ids, key=lambda question_id: hashlib.sha256(
                f"overnight-signal-v1:{benchmark}:{category}:{question_id}".encode()
            ).hexdigest())
            if len(ordered) == 1:
                dev.extend(ordered)
                continue
            dev_count = min(len(ordered) - 1, max(1, round(len(ordered) * 0.6)))
            dev.extend(ordered[:dev_count])
            holdout.extend(ordered[dev_count:])
        split["benchmarks"][benchmark] = {
            "dev": sorted(dev), "holdout": sorted(holdout),
            "strata": {key: len(value) for key, value in sorted(strata.items())},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def selected_question_ids(tier: str, benchmark: str,
                          split_name: str | None) -> set[str] | None:
    if tier != "signal" or split_name is None:
        return None
    split = create_signal_split()
    return set(split["benchmarks"][benchmark][split_name])


def normalized(text: str) -> str:
    return " ".join(locomo.normalize_answer(str(text)).split())


def recall_proxy(reference: str, category: str, memory_text: str) -> bool | None:
    if category in {"adversarial", "abstention"} or not reference.strip():
        return None
    haystack = normalized(memory_text)
    candidates = [normalized(part) for part in reference.split(";") if part.strip()]
    return any(candidate and candidate in haystack for candidate in candidates)


def supersession_rank_ok(system: WriteTimeMemorySystem, category: str) -> bool | None:
    if category not in {"knowledge-update", "knowledge_update",
                        "contradiction_resolution"}:
        return None
    positions = {item.record.id: index for index, item in enumerate(system.last_retrieved)}
    pairs = []
    for stale in system.store.all():
        for current in system.store.successor_chain(stale.id):
            pairs.append((stale.id, current.id))
    if not pairs:
        return False
    for stale_id, current_id in pairs:
        if current_id in positions and (
            stale_id not in positions or positions[current_id] < positions[stale_id]
        ):
            return True
    return False


def run_arm(*, tier: str, arm: str, tracker: SpendTracker,
            format_name: str = "F1", split_name: str | None = None,
            prompt_revision: str = "base",
            retrieval_k: int = 12, dense_weight: float = 1.0,
            bm25_weight: float = 1.0, demotion_factor: float = 0.3,
            benchmarks: tuple[str, ...] = BENCHMARKS) -> dict:
    if tier not in {"smoke", "signal"}:
        raise ValueError("full tier is intentionally unsupported")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if prompt_revision not in {"base", "coverage", "validated"}:
        raise ValueError(f"unknown extractor prompt: {prompt_revision}")
    if prompt_revision != "base" and format_name != "F4":
        raise ValueError("optimized prompts are defined only for F4")
    arm_config = ARMS[arm]
    run_name = format_name if split_name is None else f"{format_name}-{split_name}"
    if prompt_revision != "base":
        run_name = f"{run_name}-{prompt_revision}"
    run_root = RESULTS / "overnight" / tier / arm / run_name
    overall = {"tier": tier, "arm": arm, "format": format_name,
               "split": split_name, "prompt_revision": prompt_revision,
               "inference": {
                   "extractor_max_tokens": (
                       LOCAL_MAX_TOKENS
                       if arm_config["model"].startswith("qwen3") else 1400
                   ),
                   "extractor_max_records": (
                       LOCAL_MAX_RECORDS
                       if arm_config["model"].startswith("qwen3") else None
                   ),
                   "repair_policy": "malformed-top-level-only",
               },
               "retrieval": {"k": retrieval_k,
                             "dense_weight": dense_weight,
                             "bm25_weight": bm25_weight,
                             "demotion_factor": demotion_factor},
               "benchmarks": {}}
    for benchmark in benchmarks:
        benchmark_dir = run_root / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        answers_path = benchmark_dir / "answers.jsonl"
        judged_path = benchmark_dir / "judged.jsonl"
        answer_rows = {row["question_id"]: row for row in read_jsonl(answers_path)}
        judged_rows = {row["question_id"]: row for row in read_jsonl(judged_path)}
        selected = selected_question_ids(tier, benchmark, split_name)
        extraction_results = []

        judge_llm = make_llm("gpt-5.6-luna", base_url=None,
                             tracker=tracker, max_tokens=250)
        judge = MemoryJudge(judge_llm, benchmark_dir / "judge_calls.jsonl")
        for conversation in load_conversations(tier, benchmark):
            questions = [question for question in conversation.questions
                         if selected is None or question.id in selected]
            if not questions:
                continue
            extractor_llm = make_llm(
                arm_config["model"], base_url=arm_config["base_url"],
                tracker=tracker, max_tokens=1400,
                structured_output=arm_config["structured_output"],
            )
            answer_llm = make_llm("gpt-5.6-luna", base_url=None,
                                  tracker=tracker, max_tokens=1200)
            system = WriteTimeMemorySystem(
                extractor_llm=extractor_llm, answer_llm=answer_llm,
                store_path=benchmark_dir / "stores" / f"{safe_name(conversation.id)}.sqlite3",
                checkpoint_dir=benchmark_dir,
                arm=arm, fewshot=arm_config["fewshot"], format_name=format_name,
                prompt_revision=prompt_revision,
                retrieval_k=retrieval_k, dense_weight=dense_weight,
                bm25_weight=bm25_weight, demotion_factor=demotion_factor,
            )
            before_ingest = system.usage()
            for session in conversation.sessions:
                system.ingest(session)
            write_usage = usage_delta(before_ingest, system.usage())
            extraction_results.extend(system.extractions)
            memory_text = "\n".join(record.content for record in system.store.all())
            for question in questions:
                if question.id not in answer_rows:
                    before_answer = system.usage()
                    started = time.perf_counter()
                    hypothesis = system.answer(
                        question.text, question.date, question.instruction
                    )
                    wall_ms = (time.perf_counter() - started) * 1000
                    answer_usage = usage_delta(before_answer, system.usage())
                    usage = add_usage(write_usage, answer_usage)
                    write_usage = {}
                    row = {
                        "question_id": question.id,
                        "conversation_id": conversation.id,
                        "category": question.category,
                        "question": question.text,
                        "reference": question.reference,
                        "hypothesis": hypothesis,
                        "answer_input_hash": system.last_answer_hash,
                        "retrieved_ids": [item.record.id for item in system.last_retrieved],
                        "recall_proxy": recall_proxy(
                            question.reference, question.category, memory_text
                        ),
                        "supersession_accuracy": supersession_rank_ok(
                            system, question.category
                        ),
                        "usage": usage,
                        "wall_clock_ms": round(wall_ms, 3),
                        "locomo_category": question.locomo_category,
                    }
                    append_jsonl(answers_path, row)
                    answer_rows[question.id] = row
                if question.id not in judged_rows:
                    answer_row = answer_rows[question.id]
                    judgement = judge.judge(
                        question_id=question.id, benchmark=benchmark,
                        question=question.text, reference=question.reference,
                        response=answer_row["hypothesis"],
                    )
                    append_jsonl(judged_path, judgement)
                    judged_rows[question.id] = judgement
            system.close()

        active_answer_rows = [row for row in answer_rows.values()
                              if selected is None or row["question_id"] in selected]
        active_judged_rows = [row for row in judged_rows.values()
                              if selected is None or row["question_id"] in selected]
        labels = [1 if row["label"] else 0 for row in active_judged_rows]
        recall_values = [row["recall_proxy"] for row in active_answer_rows
                         if row.get("recall_proxy") is not None]
        supersession_values = [row["supersession_accuracy"] for row in active_answer_rows
                               if row.get("supersession_accuracy") is not None]
        usage = {}
        for row in active_answer_rows:
            usage = add_usage(usage, row.get("usage", {}))
        benchmark_summary = {
            "benchmark": benchmark,
            "n": len(active_answer_rows),
            "judge_accuracy": round(sum(labels) / len(labels), 4) if labels else None,
            "extraction_recall_proxy": (
                round(sum(recall_values) / len(recall_values), 4)
                if recall_values else None
            ),
            "supersession_accuracy": (
                round(sum(supersession_values) / len(supersession_values), 4)
                if supersession_values else None
            ),
            "schema_validity": (
                round(sum(result.schema_valid for result in extraction_results)
                      / len(extraction_results), 4)
                if extraction_results else 1.0
            ),
            "rejected_records": sum(len(result.rejected)
                                    for result in extraction_results),
            "usage": usage,
        }
        if benchmark == "locomo":
            lexical_scores = [locomo.score_one(
                row["hypothesis"], row["reference"], int(row["locomo_category"])
            ) for row in active_answer_rows]
            benchmark_summary["token_f1_secondary"] = (
                round(sum(lexical_scores) / len(lexical_scores), 4)
                if lexical_scores else None
            )
        (benchmark_dir / "summary.json").write_text(
            json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8"
        )
        overall["benchmarks"][benchmark] = benchmark_summary

    accuracies = [summary["judge_accuracy"]
                  for summary in overall["benchmarks"].values()
                  if summary["judge_accuracy"] is not None]
    overall["macro_judge_accuracy"] = (
        round(sum(accuracies) / len(accuracies), 4) if accuracies else None
    )
    overall["signal_subsample_caveat"] = tier == "signal"
    (run_root / "summary.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="utf-8"
    )
    return overall


def run_preflight(tracker: SpendTracker) -> dict:
    baselines = ensure_baselines()
    split = create_signal_split()
    availability = {
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "qwen_model": "qwen3:4b",
        "qwen_endpoint": "http://127.0.0.1:11434/v1",
        "data": {
            tier: {benchmark: (DATA / tier / (
                "beam.parquet" if benchmark == "beam" else f"{benchmark}.json"
            )).exists() for benchmark in BENCHMARKS}
            for tier in ("smoke", "signal")
        },
        "full_tier_command_available": False,
        "baselines": baselines,
        "signal_split": {
            benchmark: {name: len(values) for name, values in details.items()
                        if name in {"dev", "holdout"}}
            for benchmark, details in split["benchmarks"].items()
        },
    }
    # One-token local endpoint ping.
    local = make_llm("qwen3:4b", base_url="http://127.0.0.1:11434/v1",
                     tracker=tracker, max_tokens=1,
                     structured_output="prompt-only")
    try:
        local.chat([{"role": "user", "content": "/no_think Reply READY"}],
                   max_tokens=1)
        availability["qwen_endpoint_responds"] = True
    except Exception as exc:
        availability["qwen_endpoint_responds"] = False
        availability["qwen_error"] = f"{type(exc).__name__}: {exc}"

    if not availability["openai_key_present"]:
        availability["pipeline_trace"] = {
            "completed": False, "reason": "OPENAI_API_KEY missing"
        }
    else:
        conversation = load_conversations("smoke", "longmemeval")[0]
        question = conversation.questions[0]
        preflight_dir = RESULTS / "preflight_trace"
        extractor_llm = make_llm("gpt-5.6-luna", base_url=None,
                                 tracker=tracker, max_tokens=1400)
        answer_llm = make_llm("gpt-5.6-luna", base_url=None,
                              tracker=tracker, max_tokens=1200)
        system = WriteTimeMemorySystem(
            extractor_llm=extractor_llm, answer_llm=answer_llm,
            store_path=preflight_dir / "store.sqlite3",
            checkpoint_dir=preflight_dir,
            arm="luna-target", fewshot=True,
        )
        # Exactly one session, as required by Phase 0.
        system.ingest(conversation.sessions[0])
        hypothesis = system.answer(question.text, question.date,
                                   question.instruction)
        judge_llm = make_llm("gpt-5.6-luna", base_url=None,
                             tracker=tracker, max_tokens=250)
        judgement = MemoryJudge(
            judge_llm, preflight_dir / "judge_calls.jsonl"
        ).judge(question_id=question.id, benchmark="longmemeval",
                question=question.text, reference=question.reference,
                response=hypothesis)
        availability["pipeline_trace"] = {
            "completed": True,
            "session_id": conversation.sessions[0].session_id,
            "records_stored": system.store.count(),
            "records_retrieved": len(system.last_retrieved),
            "answer_nonempty": bool(hypothesis.strip()),
            "judge_label": judgement["label"],
        }
        system.close()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "preflight.json").write_text(
        json.dumps(availability, indent=2) + "\n", encoding="utf-8"
    )
    return availability


def rescore_baselines(tracker: SpendTracker) -> dict:
    ensure_baselines()
    summaries = {}
    for benchmark in BENCHMARKS:
        base = RESULTS / "baselines" / benchmark
        predictions = {row["question_id"]: row
                       for row in read_jsonl(base / "predictions.jsonl")}
        judged_path = base / "mem0-judged.jsonl"
        judged = {row["question_id"]: row for row in read_jsonl(judged_path)}
        judge_llm = make_llm("gpt-5.6-luna", base_url=None,
                             tracker=tracker, max_tokens=250)
        judge = MemoryJudge(judge_llm, base / "mem0-judge-calls.jsonl")
        lexical = []
        for conversation in load_conversations("signal", benchmark):
            for question in conversation.questions:
                prediction = predictions.get(question.id)
                if prediction is None:
                    continue
                if benchmark == "locomo":
                    lexical.append(locomo.score_one(
                        prediction["hypothesis"], question.reference,
                        int(question.locomo_category),
                    ))
                if question.id not in judged:
                    row = judge.judge(
                        question_id=question.id, benchmark=benchmark,
                        question=question.text, reference=question.reference,
                        response=prediction["hypothesis"],
                    )
                    append_jsonl(judged_path, row)
                    judged[question.id] = row
        labels = [1 if row["label"] else 0 for row in judged.values()]
        summary = {
            "benchmark": benchmark,
            "tier": "signal subsample",
            "n": len(labels),
            "judge_accuracy": (
                round(sum(labels) / len(labels), 4) if labels else None
            ),
            "configuration": "full-history Luna none answers / Luna none Mem0-style judge",
        }
        if lexical:
            summary["token_f1_secondary"] = round(
                sum(lexical) / len(lexical), 4
            )
        (base / "mem0-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries[benchmark] = summary
    output = {"tier": "signal subsample", "benchmarks": summaries}
    (RESULTS / "baselines" / "mem0-summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--tier", required=True, choices=["smoke", "signal"])
    run_parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    run_parser.add_argument("--format", default="F1", choices=["F1", "F2", "F3", "F4"])
    run_parser.add_argument("--split", choices=["dev", "holdout"])
    run_parser.add_argument("--extractor-prompt", default="base",
                            choices=["base", "coverage", "validated"])
    run_parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    subparsers.add_parser("rescore-baselines")
    sft_parser = subparsers.add_parser("build-sft")
    sft_parser.add_argument("--format", default="F1",
                            choices=["F1", "F2", "F3", "F4"])
    sft_parser.add_argument("--extractor-prompt", default="base",
                            choices=["base", "coverage", "validated"])
    subparsers.add_parser("report")
    args = parser.parse_args()
    tracker = SpendTracker(RESULTS / "spend.json", cap_usd=40.0)
    try:
        if args.command == "preflight":
            output = run_preflight(tracker)
        elif args.command == "run":
            output = run_arm(
                tier=args.tier, arm=args.arm, tracker=tracker,
                format_name=args.format, split_name=args.split,
                prompt_revision=args.extractor_prompt,
                benchmarks=tuple(args.benchmark or BENCHMARKS),
            )
        elif args.command == "rescore-baselines":
            output = rescore_baselines(tracker)
        elif args.command == "build-sft":
            output = build_sft(
                root=ROOT,
                format_name=args.format,
                prompt_revision=args.extractor_prompt,
                source_run=(RESULTS / "overnight" / "signal" / "luna-target"
                            / (f"{args.format}-dev"
                               if args.extractor_prompt == "base"
                               else f"{args.format}-dev-{args.extractor_prompt}")),
            )
        else:
            output = build_report(ROOT)
    except BudgetExceeded as exc:
        raise SystemExit(f"BUDGET STOP: {exc}") from exc
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
