"""CLI for running the three benchmark evals.

    uv run eval <longmemeval|locomo|beam> [--stage answer|judge|report|all]
        [--system full-history|oracle] [--limit N]
        [--model M --base-url URL]            # answering model (default: Ollama)
        [--judge-model M --judge-base-url URL]  # judge model (default: OpenAI)

Answer stage writes results/<bench>/<system>/answers.jsonl (resumable — rerun
to continue). Judge stage writes judged.jsonl, report writes summary.json.
`--system oracle` copies the gold answer as the hypothesis, to validate the
judge/scoring plumbing end to end (should score ~1.0).
"""

import argparse
import os
from pathlib import Path

from adaption_memory.evals import beam, locomo, longmemeval
from adaption_memory.llm import LLM
from adaption_memory.systems.full_history import FullHistorySystem

DATA_FILES = {
    "longmemeval": "data/longmemeval_s_cleaned.json",
    "locomo": "data/locomo10.json",
    "beam": "data/beam_100k.parquet",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench", choices=["longmemeval", "locomo", "beam"])
    p.add_argument("--stage", default="all",
                   choices=["answer", "judge", "report", "all"])
    p.add_argument("--system", default="full-history",
                   choices=["full-history", "oracle"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap instances (longmemeval/beam) or questions per sample (locomo)")
    p.add_argument("--data", default=None, help="path to the benchmark data file")
    p.add_argument("--out", default=None, help="output dir (default results/<bench>/<system>)")
    p.add_argument("--model", default=os.getenv("ANSWER_MODEL", "qwen3:8b"))
    p.add_argument("--base-url", default=os.getenv("ANSWER_BASE_URL",
                                                   "http://localhost:11434/v1"))
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    p.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL"))
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args()

    data_path = Path(args.data or DATA_FILES[args.bench])
    out_dir = Path(args.out or f"results/{args.bench}/{args.system}")
    answers = out_dir / "answers.jsonl"
    judged = out_dir / "judged.jsonl"
    summary = out_dir / "summary.json"
    stages = ["answer", "judge", "report"] if args.stage == "all" else [args.stage]
    oracle = args.system == "oracle"

    def make_system():
        llm = LLM(args.model, base_url=args.base_url,
                  api_key=os.getenv("ANSWER_API_KEY", "ollama"),
                  max_tokens=args.max_tokens)
        return FullHistorySystem(llm)

    def make_judge() -> LLM:
        if not args.judge_model:
            raise SystemExit("--judge-model (or JUDGE_MODEL env) is required for the judge stage")
        return LLM(args.judge_model, base_url=args.judge_base_url,
                   api_key=os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"))

    if args.bench == "longmemeval":
        instances = longmemeval.load_instances(data_path)
        if "answer" in stages:
            longmemeval.run_answers(make_system, instances, answers,
                                    limit=args.limit, oracle=oracle)
        if "judge" in stages:
            longmemeval.run_judge(make_judge(), instances, answers, judged)
        if "report" in stages:
            longmemeval.report(instances, judged, summary)

    elif args.bench == "locomo":
        samples = locomo.load_samples(data_path)
        if "answer" in stages:
            locomo.run_answers(make_system, samples, answers,
                               limit=args.limit, oracle=oracle)
        if "judge" in stages:
            print("locomo is scored lexically; no judge stage needed")
        if "report" in stages:
            locomo.report(samples, answers, summary)

    elif args.bench == "beam":
        conversations = beam.load_conversations(data_path)
        if "answer" in stages:
            beam.run_answers(make_system, conversations, answers,
                             limit=args.limit, oracle=oracle)
        if "judge" in stages:
            beam.run_judge(make_judge(), conversations, answers, judged)
        if "report" in stages:
            beam.report(judged, summary)


if __name__ == "__main__":
    main()
