"""Build deterministic smoke and signal preflight versions of all evals.

The smoke tier is a category-complete plumbing check. The signal tier samples
90 questions across multiple contexts, follows the source category mix subject
to at least three examples per stratum, and preserves full-context stress cases.
Manifests record every selection and quantify the remaining differences from
the full benchmarks.
"""

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from adaption_memory.evals import beam, locomo, longmemeval
from adaption_memory.evals.provenance import file_fingerprint

LONGMEMEVAL_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]
LONGMEMEVAL_DISTRACTORS = 2
LOCOMO_DISTRACTORS = 3
BEAM_DISTRACTORS = 4
SIGNAL_QUESTIONS_PER_BENCHMARK = 30
SIGNAL_MINIMUM_PER_STRATUM = 3
SIGNAL_LONGMEMEVAL_DISTRACTORS = 10
SIGNAL_LOCOMO_SOURCES = 5
SIGNAL_BEAM_SOURCES = 3

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _median_pick(items: list[dict], metrics: Iterable[Callable[[dict], float]],
                 identity: Callable[[dict], str]) -> dict:
    """Pick the item nearest the multivariate median with a stable tiebreak."""
    metrics = list(metrics)
    centers = [statistics.median(metric(item) for item in items)
               for metric in metrics]
    spans = [max(metric(item) for item in items)
             - min(metric(item) for item in items)
             for metric in metrics]

    def distance(item: dict):
        normalized = [abs(metric(item) - center) / (span or 1)
                      for metric, center, span in zip(metrics, centers, spans)]
        return sum(normalized), identity(item)

    return min(items, key=distance)


def _evenly_spaced(values: list, count: int) -> list:
    """Choose stable values spread across their existing order."""
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    positions = [round(i * (len(values) - 1) / (count - 1))
                 for i in range(count)]
    return [values[position] for position in positions]


def _quantile_picks(items: list, count: int, key: Callable,
                    identity: Callable) -> list:
    """Pick stable quantile centers without over-weighting extreme outliers."""
    if count <= 0:
        return []
    if count > len(items):
        raise ValueError(f"cannot select {count} unique items from {len(items)}")
    ordered = sorted(items, key=lambda item: (key(item), identity(item)))
    positions = [round((index + 0.5) * len(ordered) / count - 0.5)
                 for index in range(count)]
    return [ordered[position] for position in positions]


def _proportional_counts(full_counts: dict, total: int,
                         minimum: int) -> dict:
    """Closest deterministic proportional allocation with a stratum floor."""
    if total < minimum * len(full_counts):
        raise ValueError("total is too small for the requested stratum minimum")
    full_total = sum(full_counts.values())
    ideal = {key: total * value / full_total
             for key, value in full_counts.items()}
    allocated = {
        key: max(minimum, math.floor(ideal[key]))
        for key in full_counts
    }
    while sum(allocated.values()) < total:
        key = max(full_counts, key=lambda item: (
            ideal[item] - allocated[item], full_counts[item], str(item)
        ))
        allocated[key] += 1
    while sum(allocated.values()) > total:
        candidates = [key for key in full_counts if allocated[key] > minimum]
        if not candidates:
            raise ValueError("could not satisfy proportional allocation")
        key = max(candidates, key=lambda item: (
            allocated[item] - ideal[item], -full_counts[item], str(item)
        ))
        allocated[key] -= 1
    return allocated


def _distribution_metrics(full_counts: dict, selected_counts: dict) -> dict:
    keys = list(full_counts)
    full_total = sum(full_counts.values())
    selected_total = sum(selected_counts.values())
    full_proportions = {str(key): full_counts[key] / full_total for key in keys}
    selected_proportions = {
        str(key): selected_counts.get(key, 0) / selected_total for key in keys
    }
    total_variation = 0.5 * sum(
        abs(full_proportions[str(key)] - selected_proportions[str(key)])
        for key in keys
    )
    return {
        "full_proportions": full_proportions,
        "selected_proportions": selected_proportions,
        "total_variation_distance": total_variation,
    }


def _longmemeval_context(instance: dict) -> dict[str, int]:
    return {
        "sessions": len(instance["haystack_sessions"]),
        "turns": sum(len(session) for session in instance["haystack_sessions"]),
        "characters": sum(
            len(turn["content"])
            for session in instance["haystack_sessions"]
            for turn in session
        ),
    }


def _build_longmemeval_smoke(cleaned_path: Path,
                             oracle_path: Path) -> tuple[list[dict], dict]:
    cleaned = longmemeval.load_instances(cleaned_path)
    oracle = longmemeval.load_instances(oracle_path)
    cleaned_by_id = {item["question_id"]: item for item in cleaned}

    selected_oracle = []
    for question_type in LONGMEMEVAL_TYPES:
        candidates = [item for item in oracle
                      if item["question_type"] == question_type
                      and "_abs" not in item["question_id"]]
        selected_oracle.append(_median_pick(
            candidates,
            [lambda item: len(item["haystack_sessions"]),
             lambda item: _longmemeval_context(item)["characters"]],
            lambda item: item["question_id"],
        ))
    abstention = [item for item in oracle if "_abs" in item["question_id"]]
    selected_oracle.append(_median_pick(
        abstention,
        [lambda item: len(item["haystack_sessions"]),
         lambda item: _longmemeval_context(item)["characters"]],
        lambda item: item["question_id"],
    ))

    mini, selections = [], []
    for evidence_item in selected_oracle:
        source = cleaned_by_id[evidence_item["question_id"]]
        source_ids = list(source["haystack_session_ids"])
        support = list(evidence_item["answer_session_ids"])
        missing = set(support) - set(source_ids)
        if missing:
            raise ValueError(
                f"LongMemEval support sessions missing for {source['question_id']}: "
                f"{sorted(missing)}"
            )
        distractor_candidates = [sid for sid in source_ids if sid not in support]
        distractors = _evenly_spaced(distractor_candidates,
                                      LONGMEMEVAL_DISTRACTORS)
        retained = set(support + distractors)
        positions = [index for index, sid in enumerate(source_ids)
                     if sid in retained]
        item = dict(source)
        item["answer_session_ids"] = support
        item["haystack_session_ids"] = [source_ids[index] for index in positions]
        item["haystack_dates"] = [source["haystack_dates"][index]
                                   for index in positions]
        item["haystack_sessions"] = [source["haystack_sessions"][index]
                                      for index in positions]
        mini.append(item)
        selections.append({
            "question_id": source["question_id"],
            "question_type": source["question_type"],
            "abstention": "_abs" in source["question_id"],
            "source_context": _longmemeval_context(source),
            "mini_context": _longmemeval_context(item),
            "support_session_ids": support,
            "distractor_session_ids": distractors,
        })

    manifest = {
        "source_files": [cleaned_path.name, oracle_path.name],
        "strategy": (
            "One median evidence-context example per question type, plus one "
            "abstention example; retain every annotated support session and "
            f"{LONGMEMEVAL_DISTRACTORS} evenly spaced distractor sessions."
        ),
        "full": {
            "instances": len(cleaned),
            "question_types": dict(sorted(Counter(
                item["question_type"] for item in cleaned).items())),
            "abstentions": sum("_abs" in item["question_id"] for item in cleaned),
        },
        "mini": {
            "instances": len(mini),
            "question_types": dict(sorted(Counter(
                item["question_type"] for item in mini).items())),
            "abstentions": sum("_abs" in item["question_id"] for item in mini),
            "context": {
                key: sum(_longmemeval_context(item)[key] for item in mini)
                for key in ("sessions", "turns", "characters")
            },
        },
        "selections": selections,
    }
    return mini, manifest


def _locomo_context(sample: dict) -> dict[str, int]:
    sessions = locomo.sessions_of(sample)
    return {
        "sessions": len(sessions),
        "turns": sum(len(session.turns) for session in sessions),
        "characters": sum(len(turn.content) for session in sessions
                          for turn in session.turns),
    }


def _question_text_size(question: dict) -> int:
    answer = question.get("answer", question.get("adversarial_answer", ""))
    return len(question["question"]) + len(str(answer))


def _evidence_sessions(question: dict) -> set[int]:
    return {int(match.group(1))
            for evidence in question.get("evidence", [])
            for match in re.finditer(r"\bD(\d+):\d+\b", evidence)}


def _remap_evidence(evidence: str, session_map: dict[int, int]) -> str:
    def replace(match: re.Match) -> str:
        old_session = int(match.group(1))
        return f"D{session_map[old_session]}:{match.group(2)}"

    return re.sub(r"\bD(\d+):(\d+)\b", replace, evidence)


def _build_locomo_smoke(path: Path) -> tuple[list[dict], dict]:
    samples = locomo.load_samples(path)
    eligible = [sample for sample in samples
                if set(question["category"] for question in sample["qa"])
                == set(locomo.CATEGORY_NAMES)]
    source = _median_pick(
        eligible,
        [lambda sample: _locomo_context(sample)["sessions"],
         lambda sample: _locomo_context(sample)["turns"],
         lambda sample: _locomo_context(sample)["characters"]],
        lambda sample: sample["sample_id"],
    )

    selected_questions = []
    selections = []
    for category in sorted(locomo.CATEGORY_NAMES):
        all_in_category = [question for sample in samples for question in sample["qa"]
                           if question["category"] == category]
        target_evidence = statistics.median(
            len(question.get("evidence", [])) for question in all_in_category
        )
        target_size = statistics.median(
            _question_text_size(question) for question in all_in_category
        )
        candidates = [(index, question) for index, question in enumerate(source["qa"])
                      if question["category"] == category]
        index, question = min(
            candidates,
            key=lambda pair: (
                abs(len(pair[1].get("evidence", [])) - target_evidence),
                abs(_question_text_size(pair[1]) - target_size),
                pair[0],
            ),
        )
        selected_questions.append(dict(question))
        selections.append({
            "category": category,
            "category_name": locomo.CATEGORY_NAMES[category],
            "source_question_index": index,
            "question": question["question"],
            "source_evidence": list(question.get("evidence", [])),
        })

    required_sessions = set().union(
        *(_evidence_sessions(question) for question in selected_questions)
    )
    source_context = _locomo_context(source)
    source_session_numbers = list(range(1, source_context["sessions"] + 1))
    missing = required_sessions - set(source_session_numbers)
    if missing:
        raise ValueError(f"LoCoMo evidence references missing sessions: {sorted(missing)}")
    distractors = _evenly_spaced(
        [number for number in source_session_numbers
         if number not in required_sessions],
        LOCOMO_DISTRACTORS,
    )
    retained = sorted(required_sessions | set(distractors))
    session_map = {old: new for new, old in enumerate(retained, start=1)}

    conversation = {
        key: source["conversation"][key]
        for key in ("speaker_a", "speaker_b")
        if key in source["conversation"]
    }
    for old, new in session_map.items():
        conversation[f"session_{new}"] = source["conversation"][f"session_{old}"]
        date_key = f"session_{old}_date_time"
        if date_key in source["conversation"]:
            conversation[f"session_{new}_date_time"] = source["conversation"][date_key]

    for question in selected_questions:
        question["evidence"] = [
            _remap_evidence(item, session_map)
            for item in question.get("evidence", [])
        ]
    mini_sample = {
        "sample_id": source["sample_id"],
        "conversation": conversation,
        "qa": selected_questions,
    }
    manifest = {
        "source_files": [path.name],
        "strategy": (
            "One conversation nearest the corpus median context shape; one "
            "median-sized question per official category; retain every "
            f"evidence session and {LOCOMO_DISTRACTORS} evenly spaced distractors."
        ),
        "full": {
            "samples": len(samples),
            "questions": sum(len(sample["qa"]) for sample in samples),
            "categories": dict(sorted(Counter(
                question["category"] for sample in samples
                for question in sample["qa"]
            ).items())),
        },
        "mini": {
            "samples": 1,
            "questions": len(selected_questions),
            "categories": dict(sorted(Counter(
                question["category"] for question in selected_questions
            ).items())),
            "source_sample_id": source["sample_id"],
            "source_context": source_context,
            "context": _locomo_context(mini_sample),
            "source_session_numbers": retained,
            "evidence_session_numbers": sorted(required_sessions),
            "distractor_session_numbers": distractors,
        },
        "selections": selections,
    }
    return [mini_sample], manifest


def _beam_context(conversation: dict) -> dict[str, int]:
    sessions = beam.sessions_of(conversation)
    return {
        "sessions": len(sessions),
        "messages": sum(len(batch) for batch in conversation["chat"]),
        "characters": sum(len(message["content"])
                          for batch in conversation["chat"] for message in batch),
    }


def _tokens(text: str) -> set[str]:
    return {token for token in _WORD.findall(text.lower())
            if token not in _STOPWORDS and len(token) > 1}


def _best_message(flat_messages: list[tuple[int, int, dict]], query: str):
    query_tokens = _tokens(query)

    def rank(positioned):
        batch_index, message_index, message = positioned
        message_tokens = _tokens(message["content"])
        overlap = len(query_tokens & message_tokens)
        score = overlap / math.sqrt(max(len(message_tokens), 1))
        return score, overlap, message["role"] == "user", -batch_index, -message_index

    return max(flat_messages, key=rank)


def _paired_positions(chat: list[list[dict]], positions: set[tuple[int, int]]) \
        -> set[tuple[int, int]]:
    expanded = set(positions)
    for batch_index, message_index in list(positions):
        batch = chat[batch_index]
        message = batch[message_index]
        if (message["role"] == "user" and message_index + 1 < len(batch)
                and batch[message_index + 1]["role"] == "assistant"):
            expanded.add((batch_index, message_index + 1))
        elif (message["role"] == "assistant" and message_index > 0
              and batch[message_index - 1]["role"] == "user"):
            expanded.add((batch_index, message_index - 1))
    return expanded


def _build_beam_smoke(path: Path, output_path: Path) -> dict:
    conversations = beam.load_conversations(path)
    source = _median_pick(
        conversations,
        [lambda conversation: _beam_context(conversation)["sessions"],
         lambda conversation: _beam_context(conversation)["messages"],
         lambda conversation: _beam_context(conversation)["characters"]],
        lambda conversation: conversation["conversation_id"],
    )

    selected_questions, selections = {}, []
    for question_type in beam.QUESTION_TYPES:
        corpus_questions = [question for conversation in conversations
                            for question in conversation["questions"][question_type]]
        target_rubrics = statistics.median(
            len(question["rubric"]) for question in corpus_questions
        )
        target_size = statistics.median(
            len(question["question"]) for question in corpus_questions
        )
        candidates = source["questions"][question_type]
        source_index, selected = min(
            enumerate(candidates),
            key=lambda pair: (
                abs(len(pair[1]["rubric"]) - target_rubrics),
                abs(len(pair[1]["question"]) - target_size),
                pair[0],
            ),
        )
        selected_questions[question_type] = [selected]
        selections.append({
            "question_type": question_type,
            "source_question_index": source_index,
            "question": selected["question"],
            "rubric_items": len(selected["rubric"]),
        })

    flat_messages = [(batch_index, message_index, message)
                     for batch_index, batch in enumerate(source["chat"])
                     for message_index, message in enumerate(batch)]
    support_positions = set()
    for questions in selected_questions.values():
        question = questions[0]
        rubric = question["rubric"] or [""]
        for rubric_item in rubric:
            batch_index, message_index, _ = _best_message(
                flat_messages, f"{question['question']} {rubric_item}"
            )
            support_positions.add((batch_index, message_index))
    support_positions = _paired_positions(source["chat"], support_positions)

    distractor_candidates = [
        (batch_index, message_index)
        for batch_index, message_index, _ in flat_messages
        if (batch_index, message_index) not in support_positions
    ]
    distractor_seeds = set(_evenly_spaced(distractor_candidates, BEAM_DISTRACTORS))
    distractor_positions = (_paired_positions(source["chat"], distractor_seeds)
                            - support_positions)
    seeded_positions = support_positions | distractor_positions
    selected_batches = {batch_index for batch_index, _ in seeded_positions}
    anchor_seeds = {
        (batch_index, message_index)
        for batch_index in selected_batches
        for message_index, message in enumerate(source["chat"][batch_index])
        if message.get("time_anchor")
    }
    anchor_positions = (_paired_positions(source["chat"], anchor_seeds)
                        - seeded_positions)
    retained = seeded_positions | anchor_positions
    mini_chat = [
        [message for message_index, message in enumerate(batch)
         if (batch_index, message_index) in retained]
        for batch_index, batch in enumerate(source["chat"])
    ]
    mini_chat = [batch for batch in mini_chat if batch]

    mini_conversation = {
        "conversation_id": source["conversation_id"],
        "chat": mini_chat,
        "questions": selected_questions,
    }
    table = pq.read_table(path)
    rows = table.to_pylist()
    row = next(row for row in rows
               if str(row["conversation_id"]) == source["conversation_id"])
    row["chat"] = mini_chat
    row["probing_questions"] = repr(selected_questions)
    mini_table = pa.Table.from_pylist([row], schema=table.schema)
    pq.write_table(mini_table, output_path, compression="zstd")

    message_at = {(batch_index, message_index): message
                  for batch_index, message_index, message in flat_messages}
    return {
        "source_files": [path.name],
        "strategy": (
            "One conversation nearest the corpus median context shape; one "
            "median-sized question per memory ability; rubric-guided lexical "
            "support messages with adjacent dialogue turns, plus "
            f"{BEAM_DISTRACTORS} evenly spaced distractor seeds."
        ),
        "full": {
            "conversations": len(conversations),
            "questions": sum(
                len(conversation["questions"][question_type])
                for conversation in conversations
                for question_type in beam.QUESTION_TYPES
            ),
            "question_types": beam.QUESTION_TYPES,
        },
        "mini": {
            "conversations": 1,
            "questions": len(beam.QUESTION_TYPES),
            "question_types": beam.QUESTION_TYPES,
            "source_conversation_id": source["conversation_id"],
            "source_context": _beam_context(source),
            "context": _beam_context(mini_conversation),
            "support_message_ids": sorted({
                message_at[position]["id"] for position in support_positions
            }),
            "distractor_message_ids": sorted({
                message_at[position]["id"] for position in distractor_positions
            }),
            "anchor_message_ids": sorted({
                message_at[position]["id"] for position in anchor_positions
            }),
        },
        "selections": selections,
    }


def _build_longmemeval_signal(cleaned_path: Path,
                              oracle_path: Path) -> tuple[list[dict], dict]:
    cleaned = longmemeval.load_instances(cleaned_path)
    oracle = longmemeval.load_instances(oracle_path)
    cleaned_by_id = {item["question_id"]: item for item in cleaned}
    full_counts = Counter(item["question_type"] for item in cleaned)
    target_counts = _proportional_counts(
        {question_type: full_counts[question_type]
         for question_type in LONGMEMEVAL_TYPES},
        SIGNAL_QUESTIONS_PER_BENCHMARK,
        SIGNAL_MINIMUM_PER_STRATUM,
    )

    full_abstentions = Counter(
        item["question_type"] for item in oracle
        if "_abs" in item["question_id"]
    )
    target_abstentions = max(
        SIGNAL_MINIMUM_PER_STRATUM,
        round(
            SIGNAL_QUESTIONS_PER_BENCHMARK
            * sum(full_abstentions.values()) / len(cleaned)
        ),
    )
    abstention_types = sorted(
        (question_type for question_type in LONGMEMEVAL_TYPES
         if full_abstentions[question_type]),
        key=lambda question_type: (
            -(target_counts[question_type]
              * full_abstentions[question_type]
              / full_counts[question_type]),
            LONGMEMEVAL_TYPES.index(question_type),
        ),
    )[:target_abstentions]
    abstention_targets = Counter(abstention_types)

    selected_by_type: dict[str, list[dict]] = {}
    for question_type in LONGMEMEVAL_TYPES:
        candidates = [item for item in oracle
                      if item["question_type"] == question_type]
        abstentions = [item for item in candidates
                       if "_abs" in item["question_id"]]
        answerable = [item for item in candidates
                      if "_abs" not in item["question_id"]]
        abstention_count = abstention_targets[question_type]
        answerable_count = target_counts[question_type] - abstention_count
        selected = _quantile_picks(
            answerable,
            answerable_count,
            key=lambda item: (
                len(item["haystack_sessions"]),
                _longmemeval_context(item)["characters"],
            ),
            identity=lambda item: item["question_id"],
        )
        selected += _quantile_picks(
            abstentions,
            abstention_count,
            key=lambda item: (
                len(item["haystack_sessions"]),
                _longmemeval_context(item)["characters"],
            ),
            identity=lambda item: item["question_id"],
        )
        selected_by_type[question_type] = selected

    mini, selections = [], []
    for question_type in LONGMEMEVAL_TYPES:
        selected = selected_by_type[question_type]
        full_context_item = _median_pick(
            selected,
            [lambda item: len(item["haystack_sessions"]),
             lambda item: _longmemeval_context(item)["characters"]],
            lambda item: item["question_id"],
        )
        for evidence_item in selected:
            source = cleaned_by_id[evidence_item["question_id"]]
            source_ids = list(source["haystack_session_ids"])
            support = list(evidence_item["answer_session_ids"])
            missing = set(support) - set(source_ids)
            if missing:
                raise ValueError(
                    "LongMemEval support sessions missing for "
                    f"{source['question_id']}: {sorted(missing)}"
                )
            is_full_context = evidence_item is full_context_item
            if is_full_context:
                distractors = [sid for sid in source_ids if sid not in support]
                positions = list(range(len(source_ids)))
            else:
                distractor_candidates = [sid for sid in source_ids
                                         if sid not in support]
                distractors = _evenly_spaced(
                    distractor_candidates, SIGNAL_LONGMEMEVAL_DISTRACTORS
                )
                retained = set(support + distractors)
                positions = [index for index, sid in enumerate(source_ids)
                             if sid in retained]
            item = dict(source)
            item["answer_session_ids"] = support
            item["haystack_session_ids"] = [source_ids[index]
                                               for index in positions]
            item["haystack_dates"] = [source["haystack_dates"][index]
                                      for index in positions]
            item["haystack_sessions"] = [source["haystack_sessions"][index]
                                         for index in positions]
            mini.append(item)
            selections.append({
                "question_id": source["question_id"],
                "question_type": source["question_type"],
                "abstention": "_abs" in source["question_id"],
                "full_context": is_full_context,
                "source_context": _longmemeval_context(source),
                "mini_context": _longmemeval_context(item),
                "support_session_ids": support,
                "distractor_session_ids": distractors,
            })

    selected_counts = Counter(item["question_type"] for item in mini)
    source_context = {
        key: sum(selection["source_context"][key] for selection in selections)
        for key in ("sessions", "turns", "characters")
    }
    retained_context = {
        key: sum(selection["mini_context"][key] for selection in selections)
        for key in ("sessions", "turns", "characters")
    }
    return mini, {
        "source_files": [cleaned_path.name, oracle_path.name],
        "strategy": (
            "Thirty proportionally stratified questions with at least three "
            "per question type and at least three abstentions; one "
            "median full-context stress case per type, while other cases keep "
            "all support sessions and "
            f"{SIGNAL_LONGMEMEVAL_DISTRACTORS} spread distractor sessions."
        ),
        "full": {
            "instances": len(cleaned),
            "question_types": dict(full_counts),
            "abstentions": sum(full_abstentions.values()),
        },
        "mini": {
            "instances": len(mini),
            "question_types": dict(selected_counts),
            "abstentions": sum("_abs" in item["question_id"] for item in mini),
            "full_context_cases": sum(selection["full_context"]
                                      for selection in selections),
            "source_context": source_context,
            "context": retained_context,
            "character_retention": (
                retained_context["characters"] / source_context["characters"]
            ),
        },
        "representativeness": _distribution_metrics(
            {key: full_counts[key] for key in LONGMEMEVAL_TYPES},
            {key: selected_counts[key] for key in LONGMEMEVAL_TYPES},
        ),
        "selections": selections,
    }


def _build_locomo_signal(path: Path) -> tuple[list[dict], dict]:
    samples = locomo.load_samples(path)
    full_counts = Counter(
        question["category"] for sample in samples for question in sample["qa"]
    )
    target_counts = _proportional_counts(
        {category: full_counts[category]
         for category in sorted(locomo.CATEGORY_NAMES)},
        SIGNAL_QUESTIONS_PER_BENCHMARK,
        SIGNAL_MINIMUM_PER_STRATUM,
    )
    sources = _quantile_picks(
        samples,
        SIGNAL_LOCOMO_SOURCES,
        key=lambda sample: (
            _locomo_context(sample)["characters"],
            _locomo_context(sample)["turns"],
        ),
        identity=lambda sample: sample["sample_id"],
    )
    selected_by_source = {sample["sample_id"]: [] for sample in sources}
    selections = []
    for category in sorted(locomo.CATEGORY_NAMES):
        category_sources = [
            source for source in sources
            if any(question["category"] == category for question in source["qa"])
        ]
        base, remainder = divmod(target_counts[category], len(category_sources))
        allocations = [base] * len(category_sources)
        offset = (category - 1) % len(category_sources)
        for index in range(remainder):
            allocations[(offset + index) % len(category_sources)] += 1
        for source, count in zip(category_sources, allocations):
            candidates = [(index, question)
                          for index, question in enumerate(source["qa"])
                          if question["category"] == category]
            picked = _quantile_picks(
                candidates,
                count,
                key=lambda pair: (
                    len(pair[1].get("evidence", [])),
                    _question_text_size(pair[1]),
                ),
                identity=lambda pair: str(pair[0]),
            )
            for source_index, question in picked:
                selected_by_source[source["sample_id"]].append(dict(question))
                selections.append({
                    "source_sample_id": source["sample_id"],
                    "category": category,
                    "category_name": locomo.CATEGORY_NAMES[category],
                    "source_question_index": source_index,
                    "question": question["question"],
                    "source_evidence": list(question.get("evidence", [])),
                })

    mini = []
    source_details = []
    for source in sources:
        questions = selected_by_source[source["sample_id"]]
        mini_sample = {
            "sample_id": source["sample_id"],
            "conversation": dict(source["conversation"]),
            "qa": questions,
        }
        mini.append(mini_sample)
        source_details.append({
            "source_sample_id": source["sample_id"],
            "source_context": _locomo_context(source),
            "context": _locomo_context(mini_sample),
            "questions": len(questions),
        })

    selected_counts = Counter(
        question["category"] for sample in mini for question in sample["qa"]
    )
    return mini, {
        "source_files": [path.name],
        "strategy": (
            "Five conversations spread across the corpus context-size "
            "distribution; thirty proportionally stratified questions with "
            "at least three per category; every selected conversation is "
            "retained in full."
        ),
        "full": {
            "samples": len(samples),
            "questions": sum(len(sample["qa"]) for sample in samples),
            "categories": dict(full_counts),
        },
        "mini": {
            "samples": len(mini),
            "questions": sum(len(sample["qa"]) for sample in mini),
            "categories": dict(selected_counts),
            "full_context_cases": len(mini),
            "sources": source_details,
            "context": {
                key: sum(_locomo_context(sample)[key] for sample in mini)
                for key in ("sessions", "turns", "characters")
            },
            "character_retention": 1.0,
        },
        "representativeness": _distribution_metrics(
            {key: full_counts[key] for key in sorted(locomo.CATEGORY_NAMES)},
            {key: selected_counts[key] for key in sorted(locomo.CATEGORY_NAMES)},
        ),
        "selections": selections,
    }


def _build_beam_signal(path: Path, output_path: Path) -> dict:
    conversations = beam.load_conversations(path)
    sources = _quantile_picks(
        conversations,
        SIGNAL_BEAM_SOURCES,
        key=lambda conversation: (
            _beam_context(conversation)["characters"],
            _beam_context(conversation)["messages"],
        ),
        identity=lambda conversation: conversation["conversation_id"],
    )
    table = pq.read_table(path)
    rows_by_id = {str(row["conversation_id"]): row
                  for row in table.to_pylist()}
    selected_rows = []
    source_details = []
    selections = []
    for source in sources:
        selected_questions = {}
        for question_type in beam.QUESTION_TYPES:
            corpus_questions = [
                question for conversation in conversations
                for question in conversation["questions"][question_type]
            ]
            target_rubrics = statistics.median(
                len(question["rubric"]) for question in corpus_questions
            )
            target_size = statistics.median(
                len(question["question"]) for question in corpus_questions
            )
            source_index, selected = min(
                enumerate(source["questions"][question_type]),
                key=lambda pair: (
                    abs(len(pair[1]["rubric"]) - target_rubrics),
                    abs(len(pair[1]["question"]) - target_size),
                    pair[0],
                ),
            )
            selected_questions[question_type] = [selected]
            selections.append({
                "source_conversation_id": source["conversation_id"],
                "question_type": question_type,
                "source_question_index": source_index,
                "question": selected["question"],
                "rubric_items": len(selected["rubric"]),
            })
        row = dict(rows_by_id[source["conversation_id"]])
        row["chat"] = source["chat"]
        row["probing_questions"] = repr(selected_questions)
        selected_rows.append(row)
        source_details.append({
            "source_conversation_id": source["conversation_id"],
            "source_context": _beam_context(source),
            "context": _beam_context({**source, "questions": selected_questions}),
            "questions": len(beam.QUESTION_TYPES),
        })
    mini_table = pa.Table.from_pylist(selected_rows, schema=table.schema)
    pq.write_table(mini_table, output_path, compression="zstd")

    full_counts = {
        question_type: sum(
            len(conversation["questions"][question_type])
            for conversation in conversations
        )
        for question_type in beam.QUESTION_TYPES
    }
    selected_counts = {question_type: len(sources)
                       for question_type in beam.QUESTION_TYPES}
    return {
        "source_files": [path.name],
        "strategy": (
            "Three conversations spread across the corpus context-size "
            "distribution; one median-shaped question per memory ability in "
            "each conversation; every selected conversation is retained in full."
        ),
        "full": {
            "conversations": len(conversations),
            "questions": sum(full_counts.values()),
            "question_types": full_counts,
        },
        "mini": {
            "conversations": len(sources),
            "questions": sum(selected_counts.values()),
            "question_types": selected_counts,
            "full_context_cases": len(sources),
            "sources": source_details,
            "context": {
                key: sum(_beam_context(source)[key] for source in sources)
                for key in ("sessions", "messages", "characters")
            },
            "character_retention": 1.0,
        },
        "representativeness": _distribution_metrics(
            full_counts, selected_counts
        ),
        "selections": selections,
    }


def generate_suite(data_dir: str | Path = "data",
                   output_dir: str | Path = "data/mini") -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    source_names = (
        "longmemeval_s_cleaned.json",
        "longmemeval_oracle.json",
        "locomo10.json",
        "beam_100k.parquet",
    )
    source_data = {
        name: file_fingerprint(data_dir / name) for name in source_names
    }
    for obsolete_name in ("longmemeval.json", "locomo.json", "beam.parquet"):
        (output_dir / obsolete_name).unlink(missing_ok=True)
    smoke_dir = output_dir / "smoke"
    signal_dir = output_dir / "signal"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)

    smoke_longmemeval, smoke_longmemeval_manifest = _build_longmemeval_smoke(
        data_dir / "longmemeval_s_cleaned.json",
        data_dir / "longmemeval_oracle.json",
    )
    _write_json(smoke_dir / "longmemeval.json", smoke_longmemeval)

    smoke_locomo, smoke_locomo_manifest = _build_locomo_smoke(
        data_dir / "locomo10.json"
    )
    _write_json(smoke_dir / "locomo.json", smoke_locomo)

    smoke_beam_manifest = _build_beam_smoke(
        data_dir / "beam_100k.parquet", smoke_dir / "beam.parquet"
    )
    smoke_manifest = {
        "suite": "preflight-smoke-v2",
        "tier": "smoke",
        "purpose": (
            "Category-complete plumbing and major-regression check. Scores are "
            "not estimates of official benchmark performance."
        ),
        "deterministic": True,
        "source_data": source_data,
        "benchmarks": {
            "longmemeval": smoke_longmemeval_manifest,
            "locomo": smoke_locomo_manifest,
            "beam": smoke_beam_manifest,
        },
    }
    _write_json(smoke_dir / "manifest.json", smoke_manifest)

    signal_longmemeval, signal_longmemeval_manifest = _build_longmemeval_signal(
        data_dir / "longmemeval_s_cleaned.json",
        data_dir / "longmemeval_oracle.json",
    )
    _write_json(signal_dir / "longmemeval.json", signal_longmemeval)

    signal_locomo, signal_locomo_manifest = _build_locomo_signal(
        data_dir / "locomo10.json"
    )
    _write_json(signal_dir / "locomo.json", signal_locomo)

    signal_beam_manifest = _build_beam_signal(
        data_dir / "beam_100k.parquet", signal_dir / "beam.parquet"
    )
    signal_manifest = {
        "suite": "preflight-signal-v2",
        "tier": "signal",
        "purpose": (
            "A repeated, distribution-aware preflight signal before official "
            "full-dataset runs. It is substantially more representative than "
            "the smoke tier but is not a confidence-bounded estimate of the "
            "official benchmark score."
        ),
        "deterministic": True,
        "source_data": source_data,
        "benchmarks": {
            "longmemeval": signal_longmemeval_manifest,
            "locomo": signal_locomo_manifest,
            "beam": signal_beam_manifest,
        },
    }
    _write_json(signal_dir / "manifest.json", signal_manifest)

    manifest = {
        "suite": "preflight-v2",
        "purpose": (
            "Two deterministic local tiers: smoke for very fast plumbing checks "
            "and signal for repeated, distribution-aware model evaluation."
        ),
        "deterministic": True,
        "source_data": source_data,
        "tiers": {
            "smoke": {
                "directory": "smoke",
                "manifest": "smoke/manifest.json",
                "questions": sum(
                    details["mini"].get("instances", details["mini"].get("questions"))
                    for details in smoke_manifest["benchmarks"].values()
                ),
            },
            "signal": {
                "directory": "signal",
                "manifest": "signal/manifest.json",
                "questions": sum(
                    details["mini"].get("instances", details["mini"].get("questions"))
                    for details in signal_manifest["benchmarks"].values()
                ),
            },
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                        help="directory containing the three full benchmark files")
    parser.add_argument("--out", default="data/mini",
                        help="output directory for the mini suite")
    args = parser.parse_args()
    manifest = generate_suite(args.data_dir, args.out)
    print(f"Wrote deterministic preflight tiers to {Path(args.out)}")
    for tier, details in manifest["tiers"].items():
        print(f"- {tier}: {details['questions']} scored questions/instances")


if __name__ == "__main__":
    main()
