# Overnight Plan: Write-Time Memory System — Pipeline, Extractor Comparison, SFT Prep, Report

**Canonical runtime (2026-08-25):** The two local extractor arms use the
installed Qwen3-4B model. Qwen3-4B names, commands, result paths, and configs
are the only supported local-extractor interface.

## Context for the agent

We are implementing the write-time agent memory architecture from Adaption Labs'
"Better Agent Memory Starts Before Retrieval"
(https://adaptionlabs.ai/blog/agent-memory-write-time). Core design:

- **Write path** (per session): an LLM extractor reads the session transcript
  PLUS retrieved candidate memories from the store, and emits JSON records.
  Two record types: `narrative` (causal context / reasoning, paraphrased) and
  `atomic` (exact facts that must survive verbatim: dates, figures, IDs,
  constraints). Store is **append-only**: updates are new records carrying a
  `supersedes` pointer to the old record's id. Nothing is ever deleted.
- **Read path** (per question): hybrid retrieval (dense + BM25) over one
  unified index of both record types, superseded records demoted (not
  removed), successors expanded in, top-k (~12) records injected into the
  answering model's prompt.

**Answering model everywhere: `gpt-5.6-luna`, `reasoning.effort: "none"`.**
Never change this — it is held fixed across all arms so results are comparable
to our existing full-history baselines (already computed with Luna; do not
rerun them, load them from `results/baselines/`).

**Extractor arms to compare:**
1. `qwen3-4b-zeroshot` — local Qwen3-4B, instruction prompt only
2. `qwen3-4b-fewshot`  — same + 4–6 curated few-shot examples (see Phase 2)
3. `luna-target`       — gpt-5.6-luna, effort none, same few-shot prompt
   (this is the target/teacher: the quality ceiling and the SFT data source)

**Datasets:** three pre-split tiers already prepared in `data/`:
`smoke` (tiny, end-to-end sanity), `signal` (stratified subsample, decisions),
`full` (complete benchmarks: LongMemEval, LoCoMo, BEAM). NEVER touch the
held-out test splits during any prompt iteration or data generation.

## Binding engineering decision rules

These rules govern how every phase below is implemented. If a phase can be
read in more than one way, choose the interpretation that follows these
rules.

1. **One canonical path, no compatibility layer.** Implement the current
   Qwen3-4B design directly. Remove obsolete model names, commands, aliases,
   result-path fallbacks, and migration shims instead of teaching the system
   to support both old and new behavior. Existing result artifacts remain
   immutable research evidence, but they do not define a supported runtime
   interface.
2. **Grow a working system in gated layers.** The delivery order is:
   one-session trace → F1 smoke pipeline → three-arm smoke comparison →
   signal comparison → format ablation → one-lever optimization → SFT/report.
   A layer may begin only after the previous layer has a runnable artifact and
   its stated gate passes. Later experiments must not destabilize an earlier
   working layer.
3. **Build only what the experiment needs.** Use one CLI, one append-only
   SQLite record table, the fixed F1–F4 format menu, and the explicitly allowed
   optimization levers. Do not add generic provider frameworks, plugin
   systems, migration machinery, speculative configuration, or unused
   extension points.
4. **Keep concerns modular.** Canonical source lives under
   `src/adaption_memory/memory/`: storage, extraction, retrieval, answering,
   judging, checkpoints, budget accounting, SFT export, and reporting remain
   separate modules with small explicit interfaces. The orchestration layer
   composes them; it does not duplicate their logic.
5. **Reuse proven dependencies.** Check the project's installed dependencies,
   documentation, and types before adding code or packages. Prefer a mature
   dependency when it makes the total system smaller or more reliable. Any new
   dependency must have one documented job and replace more custom complexity
   than it introduces.
6. **Make long-term choices now.** On-disk schemas, checkpoint identities,
   result provenance, and module boundaries must be suitable for the eventual
   full run and fine-tuned extractor. Do not ship a temporary path that is
   expected to be replaced after the overnight run.
7. **Adopt established patterns before inventing one.** Before an architectural
   choice, record the relevant established pattern and what we adopt in
   `results/design_notes.md`: the Adaption Labs write path, Mem0 judging,
   Ollama's supported non-thinking/structured-output API, FastEmbed's local
   embedding interface, and `rank_bm25`'s BM25 implementation. Novel behavior
   is allowed only when those patterns do not meet a measured requirement.
8. **Fail closed on incompatible artifacts.** A checkpoint or result directory
   whose dataset/model/prompt/config fingerprint differs from the requested
   run is rejected with a clear instruction to use a new canonical output
   directory. Never silently reinterpret an old artifact.

## Hard constraints (read before doing anything)

- **Tier ceiling: NEVER run the `full` tier.** `signal` is the terminal
  evaluation tier for every arm and every phase tonight — no exceptions, not
  even with budget remaining. `data/full/` is off-limits except for
  train-split session sampling in Phase 4. All reported numbers carry a
  "signal subsample" caveat in the report.
- **Budget cap:** track cumulative API spend in `results/spend.json` after
  every batch. Hard stop at $40 total for the night: if projected phase cost
  exceeds remaining budget, skip to Phase 6 (report) with whatever exists.
- **Checkpoint everything:** every extraction, retrieval, and answer is
  written to disk (JSONL, one row per item, with input hashes) before moving
  on. Every phase must be resumable: on restart, skip items whose output
  already exists. Assume you WILL be interrupted.
- **Fail soft:** phases are ordered so each produces a standalone artifact.
  If a phase fails 3 times, log the failure in `results/FAILURES.md`, mark it
  skipped, and continue to the next phase. Never let one broken arm block the
  report.
- **Local model:** Qwen3-4B is served locally (assume an OpenAI-compatible
  endpoint at `http://localhost:11434/v1`, e.g. Ollama). In Phase 0, verify
  it responds; if it does not, disable both qwen arms, note it in FAILURES.md,
  and run luna-target only. Use non-thinking mode and JSON-constrained
  decoding (response_format json_schema if supported; otherwise strict prompt
  + retry-on-parse-failure, max 2 retries).
- **Judging protocol:** LLM-judge accuracy (not F1) for all three benchmarks,
  judge prompts matching Mem0's published prompts, judge model = gpt-5.6-luna
  effort none. LoCoMo especially: judge accuracy only, F1 may be logged as
  secondary.
- **Logging per question, per arm:** input tokens, output tokens, reasoning
  tokens (if the API returns them), wall-clock latency, dollar cost at list
  price ($0.20/M input, $1.20/M output for Luna). These feed the cost charts.

## Phase 0 — Preflight (do first, ~15 min)

1. Verify env: API key present, local Qwen endpoint responds to a 1-token
   ping, `data/{smoke,signal,full}/` exist, `results/baselines/` contains the
   full-history baseline numbers (LongMemEval 0.900 acc, BEAM 0.6595,
   LoCoMo 0.4466 F1 — LoCoMo needs judge rescoring, see Phase 5).
2. Run ONE session from `smoke` through the entire pipeline end-to-end with
   the luna-target extractor: extract → store → retrieve → answer → judge.
   If this fails, fix the pipeline before anything else. Nothing in later
   phases is worth running until this single trace works.
3. Write `results/preflight.json` recording what's available (qwen up? keys
   valid? baselines found?).
4. Write `results/design_notes.md` with the established product/library
   patterns consulted, the behavior adopted from each, and any measured reason
   for deviating. This is a design gate, not a general research survey.

## Phase 1 — Memory pipeline implementation

Implement as the small canonical package `src/adaption_memory/memory/`, not
notebooks. Complete and verify the F1 end-to-end path before adding F2–F4:

- `store.py` — SQLite, one table `records(id, session_id, type, content,
  entities, created_at, supersedes_id NULLABLE, embedding BLOB)`. Append-only
  API: `add(record)`, `candidates(query_or_session, k)`, no delete/update.
- `extract.py` — builds extractor input = session transcript + top-10
  candidate records for the session's entities; calls the configured
  extractor arm; validates output against the JSON schema below; writes
  accepted records to store. Schema per record:
  `{"type": "narrative"|"atomic", "content": str,
    "entities": [str], "supersedes_id": str|null}`
  Validation: schema-valid; supersedes_id (if present) must exist in the
  candidate set shown to the model; atomic records containing a date/number
  must have that exact substring present in the source session (grep check).
  Malformed top-level JSON: one repair retry, then drop and log.
  Semantically invalid records: keep valid siblings, drop and log only the
  invalid records locally. Never regenerate a complete extraction because one
  record failed deterministic validation. The local Qwen path bounds
  structured output to six concise records and 512 completion tokens per
  session. This is a recorded inference contract, so local iteration cannot
  silently regress to verbose generations.
- `retrieve.py` — hybrid: dense (any small local embedding model, e.g.
  Qwen3-embedding or bge-small via the same local endpoint; cache embeddings)
  + BM25 (rank_bm25), reciprocal-rank fusion. Post-processing: multiply
  superseded records' scores by 0.3; for any retrieved record with a
  successor or predecessor, include the successor. k=12 final.
- `answer.py` — Luna effort none; prompt = question + retrieved records with
  their type labels and timestamps. Same answer-prompt template across all
  arms.
- `judge.py` — Mem0-style judge prompts, Luna effort none.

Sessions within one benchmark conversation are processed IN ORDER so the
store accumulates and candidates are real (this is what makes supersession
trainable and testable).

## Phase 2 — Few-shot examples for the extractor

Hand-write (agent-write) 5 few-shot examples covering, one each:
1. plain atomic extraction (a date + a figure, exact-string discipline)
2. narrative extraction (a decision + its reasoning, no invented details)
3. a supersession: candidate shows old value, session shows new →
   new record with supersedes_id
4. a duplicate: candidate already covers the fact → output empty list
5. a mixed session producing one narrative + two atomic records

Keep them short (~15 lines of transcript each). Store in
`src/adaption_memory/memory/prompts/fewshot.json`. Use the same examples for
the canonical `qwen3-4b-fewshot` and `luna-target` arms so they differ only in
model.

## Phase 3 — Extractor comparison on `smoke` then `signal`

Gate structure — run `smoke` first for all three arms; proceed to `signal`
only for arms whose smoke run passes (schema validity ≥95%, pipeline
completes). On `signal`, for each arm, produce:

- End-to-end judge accuracy per benchmark. Reference line = the full-history
  baseline recomputed ON THE SAME SIGNAL QUESTIONS from its saved
  per-question predictions in `results/baselines/` (re-aggregation only, no
  new generation) — never compare signal-tier memory numbers against the
  full-set baseline aggregate, the subsample can differ by a couple points
  on its own.
- **Direct extractor metrics** (report these prominently — they localize
  failure): extraction recall proxy (fraction of benchmark answers whose
  key string appears in ANY stored record for that conversation) and
  supersession accuracy (on the update-category questions: did the current
  value outrank the stale one in retrieval?)
- Tokens / latency / cost per question

Decision rule to apply and record in the report: if qwen3-4b-fewshot is
within 3 points of luna-target on signal end-to-end accuracy, the fine-tune
is low-priority (prompting suffices); if the gap is >3 points, the SFT
dataset from Phase 4 is the path to close it.

`signal` is the final tier tonight for ALL arms including luna-target — the
`full` tier is never run (see hard constraints). Report all numbers with the
"signal subsample" caveat; the `full` runs happen later, manually, only for
the final chosen configuration.

## Phase 3.4 — Storage format ablation (predefined menu, luna-target arm)

The record format is the biggest single design variable; settle it BEFORE
fine-grained optimization. Test exactly these variants — do NOT invent new
formats beyond this menu:

- **F1 — dual-type baseline** (the Phase 1 design): separate `narrative` and
  `atomic` records, free-text content. This is the control.
- **F2 — structured atomic**: atomic records become key–value with typed
  fields `{"subject", "attribute", "value", "unit"|null, "as_of_date"|null}`;
  narrative records unchanged. Hypothesis: structure improves exact-fact
  retrieval and supersession matching (same subject+attribute = update).
- **F3 — entity-consolidated**: on write, records are grouped per entity;
  retrieval returns an entity's full mini-dossier (all current records for
  that entity) instead of individual records. Hypothesis: multi-fact
  questions improve; risk: context bloat, cost rises.
- **F4 — single-type**: everything stored as one record type, free text
  (i.e., ablate the paper's core narrative/atomic distinction). Hypothesis:
  it LOSES to F1 — this variant exists to measure whether the dual
  representation actually earns its complexity. If F4 ties F1, that is a
  reportable finding against the paper's central design claim.

Protocol: run all four on `signal-dev` only (format changes require full
re-extraction — budget ~$8 for this phase; if projected cost exceeds that,
drop F3 first, then F2, never the F1/F4 pair since that comparison is the
scientific core). Score end-to-end judge accuracy plus the two direct
extractor metrics per format. Winner = highest dev accuracy, ties broken by
tokens-per-question. The winner becomes the schema for Phase 3.5, Phase 4,
and the SFT dataset; the full four-way comparison table goes in the report
as its own section ("Which representation choices matter").

Note: if the winner is not F1, update `extract.py` validation and the SFT
schema accordingly, and say so explicitly in `sft/EVAL_PLAN.md`.

## Phase 3.5 — Iterative optimization of the memory pipeline (luna-target arm)

Goal: push the luna-target pipeline's score as high as possible by iterating
on the pipeline itself, WITHIN the winning format from Phase 3.4 (format is
frozen here; 3.5 tunes prompts and read-path parameters inside it). Strict
rules to keep this honest and affordable:

**Dev/eval split discipline (non-negotiable):** carve `signal` into
`signal-dev` (60%) and `signal-holdout` (40%) BEFORE the first iteration,
stratified by benchmark and question category. ALL iteration happens against
`signal-dev` only. `signal-holdout` is scored exactly twice: once with the
starting config (already have it from Phase 3) and once with the final
chosen config. If the holdout improvement is less than half the dev
improvement, report the holdout number and flag overfitting explicitly.

**Iteration budget:** max 5 iterations, max $10 total for this phase, stop
early if an iteration improves dev accuracy by <1 point. Re-use cached
extractions whenever a change only affects the read path (retrieval or
answer prompt changes do NOT require re-extraction — only extractor-prompt
or record-format changes do; exploit this, read-path iterations are ~10x
cheaper).

**Error-driven, one lever per iteration.** Before changing anything, run
error analysis on the current dev failures: bucket every wrong answer as
(a) fact never extracted — recall proxy fails → lever: extractor prompt /
record format, (b) fact stored but not retrieved → lever: retrieval (k,
fusion weights, supersession demotion factor, embedding of type labels),
(c) fact retrieved but answer wrong → lever: answer prompt (record
formatting, timestamps, type annotations). Change ONLY the lever matching
the largest bucket, one change per iteration, and log the hypothesis +
result in `results/optimization_log.md` so the report can show the
trajectory.

**Allowed levers** (pick from these, do not invent architecture changes at
3am): extractor system prompt wording; record content format (e.g. adding
entity prefixes, date normalization at write time); few-shot example
swaps; retrieval k (8–16), fusion weights, demotion factor (0.1–0.5);
answer-prompt record rendering. **Not allowed:** changing the answering
model or its effort setting, touching `full` or `signal-holdout` mid-loop,
knowledge-graph detours, or any change that breaks the record schema the
SFT dataset depends on.

The winning config becomes the config for Phase 4 data generation and for
any `full` run, and the optimization trajectory (dev score per iteration +
final holdout check) gets its own small chart in the report.

## Phase 4 — SFT dataset generation (distillation prep, no training)

Using luna-target extraction outputs produced with the Phase 3.5 winning
config (re-extract with the final config if the extractor prompt or record
format changed during optimization; otherwise reuse Phase 3 outputs), plus
additional sessions from the TRAIN splits only, up to budget:

1. Collect (input → output) pairs where input = exactly what the extractor
   saw (transcript + candidates) and output = the accepted JSON records.
2. Filter: drop any pair where a record failed validation, where
   supersedes_id was hallucinated, or where an atomic grep-check failed.
   Log the rejection rate.
3. Emit `sft/northstar-style/train.jsonl` in chat-message SFT format
   (system prompt = the production extractor prompt WITHOUT few-shots;
   the model should internalize the behavior, not depend on examples).
4. Emit `sft/config.yaml` — LoRA r=64, alpha=128, cosine schedule peaking
   ~1e-4, target modules q/k/v/o, 3 epochs as starting point, plus a 90/10
   mixture note: reserve a slot for general-purpose examples to be added
   before training.
5. Emit `sft/EVAL_PLAN.md`: the tuned model plugs into the same harness as
   the qwen arms; success = closes ≥70% of the fewshot-vs-luna gap on the
   two direct extractor metrics before any full benchmark run.

Note in the dataset README: examples derive from benchmark TRAIN splits;
list which splits, so contamination is auditable.

## Phase 5 — LoCoMo judge rescore of the existing full-history baseline

Our stored LoCoMo baseline is token-F1 (0.4466) and is NOT comparable to
judge-accuracy numbers. Rescore the saved full-history predictions (they are
in `results/baselines/locomo/predictions.jsonl`) with the Phase-1 judge.
Cheap (judging only, no generation). Output replaces F1 as the headline
LoCoMo baseline; keep F1 as a secondary column.

## Phase 6 — HTML report (always runs, even if earlier phases partially failed)

Single self-contained `report/index.html` (inline CSS/JS, Chart.js from CDN),
structured like the Adaption Labs post — sections:

1. **Motivation** — one paragraph: cost of full-history inference, the
   write-time hypothesis. Link the original post. Do not copy its text;
   write original prose.
2. **Architecture** — an inline SVG of our diagram: write path (session →
   extractor ⇄ store, with "candidates ↑ / new records + supersede links ↓"
   labels), append-only store with narrative + atomic lanes, read path
   (query → retrieval → answerer).
3. **Setup** — models table (answerer: luna/none; extractor arms; judge),
   dataset tiers, judging protocol, held-out discipline.
4. **Results** — charts (all with exact numbers labeled):
   a. Accuracy per benchmark: full-history baseline vs each arm's memory
      pipeline (bar chart; note which tier each number comes from)
   b. Cost: input tokens, output tokens, latency — memory arms as % of
      full-history (mirror the original post's cost chart, with our data)
   c. Extractor-direct metrics: recall proxy + supersession accuracy per arm
   d. The 2026 finding: full-history accuracy with a modern answerer
      matches/exceeds the paper's memory-system numbers (call this out
      honestly: at benchmark scale the case for memory is cost, not accuracy)
5. **Fine-tuning plan** — dataset size, filter rejection rate, LoRA config,
   the gap-closing success criterion, and the decision-rule outcome from
   Phase 3.
6. **Limitations** — signal-tier caveats where applicable, single answering
   model, benchmark-scale vs relationship-scale argument.

Every chart's underlying numbers must also be written to
`report/data/*.json` so nothing is locked inside the HTML.

## Morning summary

Finish by writing `results/MORNING.md`: what completed, what failed and why,
total spend, the three numbers that matter most (luna-target signal accuracy
vs baseline; `qwen3-4b-fewshot` gap; SFT dataset size after filtering), and the
single recommended next action.
