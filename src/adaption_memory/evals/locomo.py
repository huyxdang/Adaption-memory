"""LoCoMo adapter (github.com/snap-research/locomo).

Answer stage: one MemorySystem per conversation sample (10 samples, ~2k
questions total), sessions ingested in order, every question asked against
the same populated system. Scoring: the official task_eval/evaluation.py
lexical metrics reproduced verbatim — stemmed token F1 for categories 2/3/4,
multi-answer partial F1 for category 1, and the abstention substring check
for category 5. No judge model needed.
"""

import json
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
import regex
from nltk.stem import PorterStemmer
from tqdm import tqdm

from adaption_memory.evals.common import append_jsonl, done_ids, read_jsonl, write_summary
from adaption_memory.interface import Session, Turn

ps = PorterStemmer()

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
                  4: "single-hop", 5: "adversarial"}

ANSWER_INSTRUCTION = (
    "Answer with a short phrase based only on the conversations. If the "
    "question cannot be answered from the conversations, reply exactly: "
    "No information available."
)


def load_samples(path: str | Path) -> list[dict]:
    return json.load(open(path))


def sessions_of(sample: dict) -> list[Session]:
    conv = sample["conversation"]
    out = []
    i = 1
    while f"session_{i}" in conv:
        turns = []
        for t in conv[f"session_{i}"]:
            text = t["text"]
            if t.get("blip_caption"):
                text += f' [shares a photo: {t["blip_caption"]}]'
            turns.append(Turn(role=t["speaker"], content=text))
        out.append(Session(session_id=f"session_{i}",
                           date=conv.get(f"session_{i}_date_time"),
                           turns=turns))
        i += 1
    return out


def question_id(sample_id: str, qi: int) -> str:
    return f"{sample_id}:{qi}"


def run_answers(make_system, samples: list[dict], out_path: Path,
                limit: int | None = None, oracle: bool = False) -> None:
    done = done_ids(out_path, "question_id")
    for sample in samples:
        qa = sample["qa"][:limit] if limit else sample["qa"]
        todo = [(i, q) for i, q in enumerate(qa)
                if question_id(sample["sample_id"], i) not in done]
        if not todo:
            continue
        if not oracle:
            system = make_system()
            system.reset()
            for session in sessions_of(sample):
                system.ingest(session)
        for i, q in tqdm(todo, desc=f'locomo:{sample["sample_id"]}'):
            if oracle:
                gold = q.get("answer", "Not mentioned in the conversation")
                hyp, usage = str(gold), {}
            else:
                before = system.llm.usage.snapshot()
                hyp = system.answer(q["question"], instruction=ANSWER_INSTRUCTION)
                after = system.llm.usage.snapshot()
                usage = {k: after[k] - before[k] for k in after}
            append_jsonl(out_path, {
                "question_id": question_id(sample["sample_id"], i),
                "category": q["category"], "hypothesis": hyp, "usage": usage})


# --- scoring: verbatim from LoCoMo task_eval/evaluation.py ---

def normalize_answer(s):
    s = s.replace(',', "")

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    from collections import Counter
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([f1_score(p, gt) for p in predictions])
                    for gt in ground_truths])


def score_one(hypothesis: str, answer, category: int) -> float:
    answer = str(answer)
    if category == 3:
        answer = answer.split(';')[0].strip()
    if category in [2, 3, 4]:
        return f1_score(hypothesis, answer)
    if category == 1:
        return float(multi_answer_f1(hypothesis, answer))
    if category == 5:
        low = hypothesis.lower()
        return 1.0 if ('no information available' in low or 'not mentioned' in low) else 0.0
    raise ValueError(f"unknown category {category}")


def report(samples: list[dict], hyp_path: Path, out_path: Path) -> dict:
    gold = {}
    for sample in samples:
        for i, q in enumerate(sample["qa"]):
            gold[question_id(sample["sample_id"], i)] = q
    per_cat, scores = defaultdict(list), []
    for h in read_jsonl(hyp_path):
        q = gold[h["question_id"]]
        # category 5 has no 'answer' field requirement; guard for missing keys
        s = score_one(h["hypothesis"], q.get("answer", ""), q["category"])
        scores.append(s)
        per_cat[q["category"]].append(s)
    summary = {
        "benchmark": "locomo",
        "n": len(scores),
        "f1": round(float(np.mean(scores)), 4) if scores else None,
        "per_category": {
            CATEGORY_NAMES[c]: {"n": len(v), "score": round(float(np.mean(v)), 4)}
            for c, v in sorted(per_cat.items())},
    }
    write_summary(out_path, summary)
    return summary
