"""CLI for running the three benchmark evals.

    uv run eval <longmemeval|locomo|beam> [--tier smoke|signal]
        [--stage answer|judge|report|all]
        [--system <registered system>|oracle] [--limit N]
        [--model M --base-url URL]            # answering model (default: Ollama)
        [--judge-model M --judge-base-url URL]  # judge model (default: OpenAI)

Answer stage writes results/[<tier>/]<bench>/<system>/answers.jsonl (resumable
— rerun to continue). Judge stage writes judged.jsonl, report writes
summary.json.
`--system oracle` copies the gold answer as the hypothesis, to validate the
judge/scoring plumbing end to end (should score ~1.0).
"""

import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit

from adaption_memory.evals import beam, locomo, longmemeval
from adaption_memory.evals.provenance import (ensure_config, file_fingerprint,
                                               require_config)
from adaption_memory.llm import LLM
from adaption_memory.systems import REGISTRY

DATA_FILES = {
    "longmemeval": "data/longmemeval_s_cleaned.json",
    "locomo": "data/locomo10.json",
    "beam": "data/beam_100k.parquet",
}

TIER_DATA_FILES = {
    tier: {
        "longmemeval": f"data/mini/{tier}/longmemeval.json",
        "locomo": f"data/mini/{tier}/locomo.json",
        "beam": f"data/mini/{tier}/beam.parquet",
    }
    for tier in ("smoke", "signal", "half")
}

RESULT_ARTIFACTS = ("answers.jsonl", "judged.jsonl", "summary.json")


def endpoint_identity(url: str | None) -> str:
    """Identify an endpoint without persisting credentials or query tokens."""
    if not url:
        return "openai-default"
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return f"{parsed.scheme}://{host}{parsed.path.rstrip('/')}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench", choices=["longmemeval", "locomo", "beam"])
    p.add_argument("--stage", default="all",
                   choices=["answer", "judge", "report", "all"])
    p.add_argument("--system", default="full-history",
                   choices=[*sorted(REGISTRY), "oracle"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap instances (longmemeval/beam) or questions per sample (locomo)")
    p.add_argument("--data", default=None, help="path to the benchmark data file")
    p.add_argument("--tier", choices=sorted(TIER_DATA_FILES), default=None,
                   help="use a generated preflight tier instead of full data")
    p.add_argument("--out", default=None, help="output dir (default results/<bench>/<system>)")
    p.add_argument("--model", default=os.getenv("ANSWER_MODEL", "qwen3:4b"))
    p.add_argument("--base-url", default=os.getenv("ANSWER_BASE_URL",
                                                   "http://localhost:11434/v1"))
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    p.add_argument("--judge-base-url", default=(
        os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    ))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--judge-max-tokens", type=int, default=None,
                   help="judge completion budget (adapter default if omitted)")
    p.add_argument("--reasoning-effort",
                   choices=["none", "low", "medium", "high", "xhigh", "max"],
                   default=os.getenv("REASONING_EFFORT"),
                   help="optional thinking level for compatible models")
    p.add_argument("--judge-reasoning-effort",
                   choices=["none", "low", "medium", "high", "xhigh", "max"],
                   default=os.getenv("JUDGE_REASONING_EFFORT"),
                   help="judge thinking level (defaults to --reasoning-effort)")
    p.add_argument("--structured-output",
                   choices=["json-schema", "json-object", "prompt-only"],
                   default=os.getenv("STRUCTURED_OUTPUT_MODE", "json-schema"),
                   help="answer-endpoint structured output capability")
    args = p.parse_args()

    if args.tier and args.data:
        p.error("--tier and --data are mutually exclusive")
    data_path = Path(TIER_DATA_FILES[args.tier][args.bench] if args.tier
                     else args.data or DATA_FILES[args.bench])
    if args.tier and not data_path.exists():
        raise SystemExit(
            "preflight tiers not found; generate them with: uv run make-mini"
        )
    default_out = (f"results/{args.tier}/{args.bench}/{args.system}" if args.tier
                   else f"results/{args.bench}/{args.system}")
    out_dir = Path(args.out or default_out)
    answers = out_dir / "answers.jsonl"
    judged = out_dir / "judged.jsonl"
    summary = out_dir / "summary.json"
    stages = ["answer", "judge", "report"] if args.stage == "all" else [args.stage]
    oracle = args.system == "oracle"
    dataset_config = {
        "schema": 1,
        "benchmark": args.bench,
        "mode": args.tier or "full",
        "system": args.system,
        "data": file_fingerprint(data_path),
    }
    ensure_config(out_dir, "dataset.json", dataset_config, RESULT_ARTIFACTS)

    if "answer" in stages:
        answer_config = ({"oracle": True} if oracle else {
            "model": args.model,
            "endpoint": endpoint_identity(args.base_url),
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "structured_output": args.structured_output,
        })
        ensure_config(out_dir, "answer-config.json", answer_config,
                      ("answers.jsonl",))

    judge_max_tokens = args.judge_max_tokens
    if judge_max_tokens is None:
        judge_max_tokens = 10 if args.bench == "longmemeval" else 500
    if "judge" in stages and args.bench != "locomo":
        if not args.judge_model:
            raise SystemExit(
                "--judge-model (or JUDGE_MODEL env) is required for the judge stage"
            )
        judge_config = {
            "model": args.judge_model,
            "endpoint": endpoint_identity(args.judge_base_url),
            "max_tokens": judge_max_tokens,
            "reasoning_effort": (args.judge_reasoning_effort
                                 or args.reasoning_effort),
        }
        ensure_config(out_dir, "judge-config.json", judge_config,
                      ("judged.jsonl",))

    def make_system():
        llm = LLM(args.model, base_url=args.base_url,
                  api_key=os.getenv("ANSWER_API_KEY"),
                  max_tokens=args.max_tokens,
                  reasoning_effort=args.reasoning_effort,
                  structured_output=args.structured_output)
        return REGISTRY[args.system](llm)

    def make_judge() -> LLM:
        if not args.judge_model:
            raise SystemExit("--judge-model (or JUDGE_MODEL env) is required for the judge stage")
        return LLM(args.judge_model, base_url=args.judge_base_url,
                   api_key=os.getenv("JUDGE_API_KEY"),
                   reasoning_effort=(args.judge_reasoning_effort
                                     or args.reasoning_effort))

    if args.bench == "longmemeval":
        instances = longmemeval.load_instances(data_path)
        if "answer" in stages:
            longmemeval.run_answers(make_system, instances, answers,
                                    limit=args.limit, oracle=oracle)
        if "judge" in stages:
            require_config(out_dir, "answer-config.json", ("answers.jsonl",))
            longmemeval.run_judge(
                make_judge(), instances, answers, judged,
                max_tokens=judge_max_tokens,
            )
        if "report" in stages:
            require_config(out_dir, "answer-config.json", ("answers.jsonl",))
            require_config(out_dir, "judge-config.json", ("judged.jsonl",))
            longmemeval.report(instances, judged, summary)

    elif args.bench == "locomo":
        samples = locomo.load_samples(data_path)
        if "answer" in stages:
            locomo.run_answers(make_system, samples, answers,
                               limit=args.limit, oracle=oracle)
        if "judge" in stages:
            print("locomo is scored lexically; no judge stage needed")
        if "report" in stages:
            require_config(out_dir, "answer-config.json", ("answers.jsonl",))
            locomo.report(samples, answers, summary)

    elif args.bench == "beam":
        conversations = beam.load_conversations(data_path)
        if "answer" in stages:
            beam.run_answers(make_system, conversations, answers,
                             limit=args.limit, oracle=oracle)
        if "judge" in stages:
            require_config(out_dir, "answer-config.json", ("answers.jsonl",))
            beam.run_judge(
                make_judge(), conversations, answers, judged,
                max_tokens=judge_max_tokens,
            )
        if "report" in stages:
            require_config(out_dir, "answer-config.json", ("answers.jsonl",))
            require_config(out_dir, "judge-config.json", ("judged.jsonl",))
            beam.report(judged, summary)


if __name__ == "__main__":
    main()
