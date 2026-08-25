# Extraction interface experiments (smoke tier, 2026-08-25)

Autonomous overnight campaign over the write-time extraction interface.
Constant in every non-ablation config: **F1 storage** — narrative + atomic
records, append-only supersession (per the reproduction invariant in
CLAUDE.md). What varied: how the extractor model talks to the store
(emission format), the extraction prompt, extraction frequency, thinking
mode, and extractor model size. Hosted models were unreachable (corporate
network), so all extraction ran locally; scores are smoke-tier (9
conversations, 20 scorable questions) and are **directional evidence only**
— promotion into the canonical pipeline requires the signal-tier gate.

## Metrics

- **kw** — keyword coverage: mean fraction of the reference answer's
  substantive words present in the store (paraphrase-tolerant, free,
  deterministic; the in-loop score).
- **judged** — Claude Sonnet judge reads each question + full store and
  scores contained (1) / partial (0.5) / absent (0). Final arbiter.
- The two agree at the extremes (Qwen control kw 0.23 / judged 0.25; Luna
  kw 0.55 / judged 0.56) but diverge per-question mid-range: kw can credit
  scattered words the judge correctly rejects (e.g. a store with "four
  dogs" but no tattoo record still gets tattoo-question keyword credit).

## Results (macro over LongMemEval / LoCoMo / BEAM smoke)

| Config (qwen3:4b zero-shot unless noted) | kw | judged | rejected records |
|---|---:|---:|---:|
| pointer emission, base prompt (control) | 0.232 | 0.251 | 50 |
| **simple emission**, base prompt | 0.373 | 0.266 | **2** |
| pointer, coverage-f1 prompt | 0.335 | 0.275 | 58 |
| **simple + coverage-f1 (best)** | **0.396** | **0.320** | 21 |
| qwen3:1.7b, simple emission | 0.122 | — | 0 |
| *Luna few-shot pointer (reference)* | *0.546* | *0.561* | *0* |

- **Simple emission** = two required fields (`type`, `content`) plus
  optional `updates: <1-based candidate number>`, resolved by the harness to
  `supersedes_id`; entities derived heuristically from content. Storage
  format unchanged.
- **coverage-f1** = the F1 prompt rewritten with explicit coverage language
  (preserve every fact likely to support a later question; separate records
  per independent fact), keeping the narrative/atomic distinction.

## Negative results (each with evidence in results/fastloop/)

- **lines emission** (no JSON): without a grammar constraint Qwen rambles
  reasoning prose; the parser then picks up template echoes ("A: <exact
  fact>") as records. Store coverage 0.
- **thinking mode**: thinking + json_schema on Ollama returns empty
  content (the documented reason the pipeline pins non-thinking mode).
- **per-interaction extraction**: matched the best LoCoMo coverage (0.54)
  at 129 calls vs 9 — 14x cost for parity on this slice. Untested at
  full-suite scale; only worth revisiting with a much faster extractor.
- **qwen3:1.7b**: clean JSON (0 rejections) but macro 0.12 — extraction
  competence collapses below 4B. 4B stays the local floor.

## Infrastructure landed with the campaign

- Fastloop scores now include paraphrase-tolerant `store_coverage` /
  `retrieved_coverage` (strict substring recall kept as secondary); misses
  are bucketed fact_not_extracted / partially_extracted with per-question
  coverage.
- `adaption_memory/evals/fastloop_judge.py`: dumps per-run judge packets
  (question + reference + store only — never transcripts) and applies
  external judge verdicts into `judged_store_recall.json`.
- Embedder loads fully offline from the local Hugging Face cache
  (fastembed silently revalidated over the network on every process start;
  on this network that was a connection reset).
- `updates` values are parsed defensively (empty/garbage strings drop the
  link, never the record).

## Recommended next steps

1. When hosted access returns, run the **signal tier** with the
   simple + coverage-f1 extractor against the pointer/base control (Luna
   answering + judging, both effort none): the smoke-tier +0.07 judged gain
   must clear the pre-stated >=3-point macro gate on signal-dev and holdout
   before the canonical `run_arm` path adopts the new interface.
2. Re-run the Luna arm with the simple emission — if the teacher also
   improves, the SFT target data improves with it.
3. LongMemEval remains the weakest benchmark for every local config
   (specific values like "Mindful.org", "6:45 AM" never extracted);
   the next prompt iteration should target exact-value retention.
