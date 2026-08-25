"""LongMemEval adapter (github.com/xiaowu0162/LongMemEval).

Answer stage: one MemorySystem instance per question; ingest the haystack
sessions in order, then ask. Judge stage: the official evaluate_qa.py judge
prompts, reproduced verbatim, yes/no parsed the same way (label = "yes" in
response.lower()).
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from adaption_memory.evals.common import (append_jsonl, done_ids, read_jsonl,
                                          usage_delta, write_summary)
from adaption_memory.interface import Session, Turn
from adaption_memory.llm import LLM

ANSWER_INSTRUCTION = (
    "Answer concisely based on the conversation history. If the history does "
    "not contain the information needed, say so."
)


def load_instances(path: str | Path) -> list[dict]:
    return json.load(open(path))


def sessions_of(instance: dict) -> list[Session]:
    out = []
    for sid, date, turns in zip(instance["haystack_session_ids"],
                                instance["haystack_dates"],
                                instance["haystack_sessions"]):
        out.append(Session(
            session_id=str(sid), date=date,
            turns=[Turn(role=t["role"], content=t["content"]) for t in turns],
        ))
    return out


def run_answers(make_system, instances: list[dict], out_path: Path,
                limit: int | None = None, oracle: bool = False) -> None:
    done = done_ids(out_path, "question_id")
    for inst in tqdm(instances[:limit], desc="longmemeval:answer"):
        if inst["question_id"] in done:
            continue
        if oracle:
            hyp, usage = str(inst["answer"]), {}
        else:
            system = make_system()
            system.reset()
            before = system.usage()
            for session in sessions_of(inst):
                system.ingest(session)
            hyp = system.answer(inst["question"], inst["question_date"],
                                ANSWER_INSTRUCTION)
            after = system.usage()
            usage = usage_delta(before, after)
        append_jsonl(out_path, {"question_id": inst["question_id"],
                                "hypothesis": hyp, "usage": usage})


# --- judge prompts: verbatim from LongMemEval src/evaluation/evaluate_qa.py ---

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def run_judge(judge: LLM, instances: list[dict], hyp_path: Path,
              out_path: Path, max_tokens: int = 10) -> None:
    by_id = {i["question_id"]: i for i in instances}
    done = done_ids(out_path, "question_id")
    for h in tqdm(read_jsonl(hyp_path), desc="longmemeval:judge"):
        qid = h["question_id"]
        if qid in done or qid not in by_id:
            continue
        inst = by_id[qid]
        prompt = get_anscheck_prompt(
            inst["question_type"], inst["question"], inst["answer"],
            h["hypothesis"], abstention="_abs" in qid)
        resp = judge.chat([{"role": "user", "content": prompt}],
                          max_tokens=max_tokens)
        append_jsonl(out_path, {"question_id": qid,
                                "label": "yes" in resp.lower(),
                                "judge_response": resp})


def report(instances: list[dict], judged_path: Path, out_path: Path) -> dict:
    by_id = {i["question_id"]: i for i in instances}
    judged = read_jsonl(judged_path)
    per_type = defaultdict(list)
    for j in judged:
        qtype = by_id[j["question_id"]]["question_type"]
        per_type[qtype].append(1 if j["label"] else 0)
        if "_abs" in j["question_id"]:
            per_type["abstention"].append(1 if j["label"] else 0)
    labels = [1 if j["label"] else 0 for j in judged]
    summary = {
        "benchmark": "longmemeval",
        "n": len(labels),
        "accuracy": round(sum(labels) / len(labels), 4) if labels else None,
        "per_type": {t: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
                     for t, v in sorted(per_type.items())},
    }
    write_summary(out_path, summary)
    return summary
