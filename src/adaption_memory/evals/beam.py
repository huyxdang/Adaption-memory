"""BEAM adapter (github.com/mohammadtavakoli78/BEAM, ICLR 2026).

Data: HF parquet split (100K by default) — each row is one conversation whose
`chat` is a list of batches of messages carrying time anchors, plus 20
probing questions (2 per memory ability) with grading rubrics.

Answer stage: consecutive messages sharing a time anchor become one Session;
the question is asked plainly, matching the official long-context path which
just appends the question to the chat.

Judge stage: the official compute_metrics.py scheme — every rubric item is
scored 0/0.5/1 by the unified judge prompt (stored verbatim in
prompts/beam_unified_judge.txt) and averaged; event_ordering is reported as
normalized Kendall's tau over LLM-aligned event lists, exactly as
report_results.py does. Deviation from the official code: we substitute the
<question> placeholder in the judge prompt (the official runner leaves it
unreplaced), since the prompt's responsiveness rules depend on it.
"""

import ast
import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from json_repair import repair_json
from scipy.stats import kendalltau
from tqdm import tqdm

from adaption_memory.evals.common import append_jsonl, done_ids, read_jsonl, write_summary
from adaption_memory.interface import Session, Turn
from adaption_memory.llm import LLM

QUESTION_TYPES = ["abstention", "contradiction_resolution", "event_ordering",
                  "information_extraction", "instruction_following",
                  "knowledge_update", "multi_session_reasoning",
                  "preference_following", "summarization", "temporal_reasoning"]

UNIFIED_JUDGE_PROMPT = (resources.files("adaption_memory.evals")
                        / "prompts" / "beam_unified_judge.txt").read_text()


def load_conversations(path: str | Path) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    out = []
    for row in rows:
        out.append({
            "conversation_id": str(row["conversation_id"]),
            "chat": row["chat"],
            "questions": ast.literal_eval(row["probing_questions"]),
        })
    return out


def sessions_of(conv: dict) -> list[Session]:
    """Group consecutive messages that share a time anchor into one session."""
    out = []
    current, anchor = [], None
    for batch in conv["chat"]:
        for msg in batch:
            if msg["time_anchor"] != anchor and current:
                out.append(Session(session_id=f"s{len(out)+1}", date=anchor,
                                   turns=current))
                current = []
            anchor = msg["time_anchor"]
            current.append(Turn(role=msg["role"], content=msg["content"]))
    if current:
        out.append(Session(session_id=f"s{len(out)+1}", date=anchor, turns=current))
    return out


def question_id(conv_id: str, qtype: str, qi: int) -> str:
    return f"{conv_id}:{qtype}:{qi}"


def oracle_answer(q: dict) -> str:
    """Gold text for --system oracle; the gold field name varies by type."""
    for key in ("ideal_response", "ideal_answer", "ideal_summary", "answer",
                "expected_compliance"):
        if q.get(key):
            return "\n".join(q[key]) if isinstance(q[key], list) else str(q[key])
    return "\n".join(q["rubric"])


def run_answers(make_system, conversations: list[dict], out_path: Path,
                limit: int | None = None, oracle: bool = False) -> None:
    done = done_ids(out_path, "question_id")
    for conv in conversations[:limit]:
        todo = [(qtype, qi, q)
                for qtype in QUESTION_TYPES
                for qi, q in enumerate(conv["questions"].get(qtype, []))
                if question_id(conv["conversation_id"], qtype, qi) not in done]
        if not todo:
            continue
        if not oracle:
            system = make_system()
            system.reset()
            for session in sessions_of(conv):
                system.ingest(session)
        for qtype, qi, q in tqdm(todo, desc=f'beam:conv{conv["conversation_id"]}'):
            if oracle:
                hyp, usage = oracle_answer(q), {}
            else:
                before = system.llm.usage.snapshot()
                hyp = system.answer(q["question"])
                after = system.llm.usage.snapshot()
                usage = {k: after[k] - before[k] for k in after}
            append_jsonl(out_path, {
                "question_id": question_id(conv["conversation_id"], qtype, qi),
                "question_type": qtype, "hypothesis": hyp, "usage": usage})


# --- judging: official compute_metrics.py scheme ---

def parse_judge_json(response: str) -> dict:
    try:
        return json.loads(repair_json(response))
    except Exception:
        return {"score": 0.0, "reason": f"unparseable judge output: {response[:200]}"}


def judge_rubric(judge: LLM, question: str, rubric: list[str],
                 hypothesis: str) -> dict:
    responses, score = [], 0.0
    for item in rubric:
        prompt = (UNIFIED_JUDGE_PROMPT
                  .replace("<question>", question)
                  .replace("<rubric_item>", item)
                  .replace("<llm_response>", hypothesis))
        parsed = parse_judge_json(judge.chat(
            [{"role": "user", "content": prompt}], max_tokens=500))
        try:
            score += float(parsed.get("score", 0.0))
        except (TypeError, ValueError):
            pass
        responses.append(parsed)
    return {"llm_judge_score": score / len(rubric) if rubric else 0.0,
            "llm_judge_responses": responses}


# verbatim system message from compute_metrics.llm_equivalence
EQUIVALENCE_SYSTEM_MSG = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """


def llm_equivalence(judge: LLM, first: str, second: str) -> bool:
    resp = judge.chat([
        {"role": "system", "content": EQUIVALENCE_SYSTEM_MSG},
        {"role": "user", "content": f"""First snippet: {first} \n
                       Second snippet: {second}
                    """},
    ], max_tokens=10)
    return "yes" in resp.lower()


def align_with_llm(judge: LLM, reference: list[str], system: list[str]):
    used, system_out = set(), []
    for s in system:
        matched = None
        for idx, r in enumerate(reference):
            if idx in used:
                continue
            if llm_equivalence(judge, r, s):
                matched = idx
                break
        if matched is not None:
            system_out.append(reference[matched])
            used.add(matched)
        else:
            system_out.append(s)
    return reference, system_out


def event_ordering_tau(judge: LLM, rubric: list[str], hypothesis: str) -> float:
    system_list = [line for line in hypothesis.split("\n") if line.strip()]
    if not system_list:
        return 0.0
    reference_canon, system_canon = align_with_llm(judge, rubric, system_list)
    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(to_rank(reference_canon), to_rank(system_canon),
                          variant="b", method="auto")
    return (tau_b + 1) / 2 if tau_b is not None and not np.isnan(tau_b) else 0.0


def run_judge(judge: LLM, conversations: list[dict], hyp_path: Path,
              out_path: Path) -> None:
    questions = {}
    for conv in conversations:
        for qtype in QUESTION_TYPES:
            for qi, q in enumerate(conv["questions"].get(qtype, [])):
                questions[question_id(conv["conversation_id"], qtype, qi)] = q
    done = done_ids(out_path, "question_id")
    for h in tqdm(read_jsonl(hyp_path), desc="beam:judge"):
        qid = h["question_id"]
        if qid in done or qid not in questions:
            continue
        q = questions[qid]
        result = judge_rubric(judge, q["question"], q["rubric"], h["hypothesis"])
        if h["question_type"] == "event_ordering":
            result["tau_norm"] = event_ordering_tau(judge, q["rubric"],
                                                    h["hypothesis"])
        append_jsonl(out_path, {"question_id": qid,
                                "question_type": h["question_type"], **result})


def report(judged_path: Path, out_path: Path) -> dict:
    per_type = defaultdict(list)
    for j in read_jsonl(judged_path):
        # official report_results.py: event_ordering counts tau_norm,
        # everything else counts llm_judge_score
        if j["question_type"] == "event_ordering":
            per_type[j["question_type"]].append(j.get("tau_norm", 0.0))
        else:
            per_type[j["question_type"]].append(j["llm_judge_score"])
    type_scores = {t: round(float(np.mean(v)), 4) for t, v in sorted(per_type.items())}
    summary = {
        "benchmark": "beam",
        "n": sum(len(v) for v in per_type.values()),
        "overall": round(float(np.mean(list(type_scores.values()))), 4)
        if type_scores else None,
        "per_type": {t: {"n": len(per_type[t]), "score": s}
                     for t, s in type_scores.items()},
    }
    write_summary(out_path, summary)
    return summary
