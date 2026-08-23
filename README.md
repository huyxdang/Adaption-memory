# Adaption-memory

Reproduction of Adaption Labs' **"Better Agent Memory Starts Before Retrieval"**
— <https://adaptionlabs.ai/blog/agent-memory-write-time>

> Bargav Jagatha (Modelling Resident) and Sudip Roy (Co-founder) — August 13, 2026

---

## The thesis

The bottleneck in agent memory is **not retrieval — it is what gets written**.
The highest-leverage optimization happens at write time, when information is
extracted into memory, not later when it is searched for.

### Why write time, not retrieval

The naive approach is to resend the full conversation history on every turn.
That makes cost a function of relationship length rather than question
difficulty — the tokens spent on a single turn come to track how long the agent
has known the user rather than how hard the current question is.

No retrieval strategy can recover a fact that was never extracted, or repair a
fact that was flattened into a lossy summary on the way in. Retrieval quality is
capped by write quality.

## Two memory representations

A single representation forces a bad trade: rich summaries lose exact values,
and exact values lose the reasoning around them. The system writes both.

| | Preserves | Good for |
|---|---|---|
| **Narrative memory** | causal context — the reasoning behind a decision or a change | *why* something is true, how a preference formed |
| **Atomic memory** | exact facts, the kind that has to survive verbatim | identifiers, numbers, names, dates, settings |

Retrieval draws on whichever the question needs.

## Evolutionary state tracking

Facts change. The system **defaults to appending** rather than overwriting or
deleting. When a value evolves, the new state is added and **marks what it
superseded**, so retrieval can favor what is true now while the earlier version
stays on record.

This is what makes knowledge-update and preference-change questions answerable:
the history of a fact is itself information.

## Benchmarks

| Benchmark | Context size | Role |
|---|---|---|
| LongMemEval | ~5K tokens | multi-session baseline |
| LoCoMo | ~5K tokens | multi-session baseline |
| BEAM | ~100K tokens | long-conversation evaluation |

Baselines: **full-history inference** (whole conversation in the prompt) and a
**Mem0 OSS** reproduction run under an identical evaluation harness, with the
same answering model throughout.

### vs. full-history inference

| Metric | Result |
|---|---|
| LongMemEval accuracy | **90.6%** vs 60.6% |
| Tokens processed (long conversations) | **97% fewer** |
| Tokens processed (standard) | **57% fewer** |
| Mean latency | **45% lower** |
| p95 latency (long conversations) | **67% lower** |

### vs. Mem0 OSS

| Benchmark | This approach | Mem0 OSS |
|---|---|---|
| LongMemEval | **90.6%** | 71.6% |
| LoCoMo | **88.2%** | 82.2% |
| BEAM | **61.3%** | 39.7% |
| Knowledge-update accuracy | **94.9%** | 75.6% |
| Preference retention | **96.7%** | 76.7% |

## Takeaway

Memory should *reduce* context, not accumulate it. Done right, it decouples
relationship length from inference cost — accuracy goes up while tokens and
latency go down — and that comes from representation choices made at write time,
not from retrieval sophistication alone.

## Status

The three eval harnesses are built and tested; the memory system itself is
next. `--system full-history` reproduces the article's baseline column;
`--system oracle` echoes gold answers to validate the scoring plumbing.

## Running the evals

Data (gitignored, ~300MB total):

```
mkdir -p data && cd data
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
curl -L  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o locomo10.json
curl -L  https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/100K-00000-of-00001.parquet -o beam_100k.parquet
```

Run (answering model defaults to Ollama at `localhost:11434`, judge is any
OpenAI-compatible endpoint):

```
uv run eval locomo      --system full-history --model qwen3:8b
uv run eval longmemeval --system full-history --judge-model gpt-5.6-luna
uv run eval beam        --system full-history --judge-model gpt-5.6-luna
```

Stages are resumable (`--stage answer|judge|report|all`); results land in
`results/<bench>/<system>/`. Offline tests: `uv run pytest`.

### Scoring fidelity

Each harness reproduces its official scorer:

- **LongMemEval** — the verbatim judge prompts from `evaluate_qa.py`
  (per-type templates, abstention via `_abs`, label = "yes" in judge output).
- **LoCoMo** — the lexical metrics from `task_eval/evaluation.py`: stemmed
  token F1 (categories 2/3/4, category 3 takes the first `;`-field of gold),
  multi-answer partial F1 (category 1), abstention substring check (category 5).
  No judge model involved.
- **BEAM** — the official rubric scheme: every rubric item judged 0/0.5/1 with
  the verbatim unified judge prompt, averaged; `event_ordering` reported as
  normalized Kendall's tau over LLM-aligned event lists. One deliberate fix:
  we substitute the `<question>` placeholder the official runner leaves
  unreplaced.

## License

MIT — see [LICENSE](LICENSE).
