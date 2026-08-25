# Findings

Cumulative experimental findings for this reproduction. Each entry states
what was tested, what was measured, and what we concluded — including
negative results, so nothing gets silently retried. Smoke-tier numbers are
directional; nothing is promoted into the canonical pipeline without the
signal-tier gate (>=3 macro points on signal-dev, confirmed on holdout —
see CLAUDE.md).

## 1. The extraction interface was the bottleneck (2026-08-25)

**Question.** Qwen3-4B extracted almost nothing useful at write time
(judged store recall 0.25 vs Luna's 0.56). Model too small, or interface
too hostile?

**What we varied** (storage stayed canonical F1 — narrative + atomic,
append-only supersession — in every non-ablation config):

- **Emission format.** Old: five fields per record, including entity lists
  and an exact `supersedes_id` copied from candidate JSON. New ("simple"):
  `{type, content}` plus optional `updates: <candidate number>` against a
  numbered list; the harness resolves the number to the real id and
  derives entities. Same stored records, radically easier to emit.
- **Prompt.** Old F1 prompt optimized for precision ("concise", "empty
  list when nothing durable is new"). New ("coverage-f1") keeps the
  narrative/atomic rules but demands coverage: preserve every fact likely
  to support a later question, one record per independent fact.

**Results** (smoke tier, 20 scorable questions; judged = Claude Sonnet
scoring contained/partial/absent over the store):

| Config | Keyword | Judged | Rejected records |
|---|---:|---:|---:|
| Qwen 4B, old interface (control) | 0.232 | 0.251 | 50 |
| Qwen 4B, simple emission | 0.373 | 0.266 | **2** |
| Qwen 4B, coverage prompt | 0.335 | 0.275 | 58 |
| **Qwen 4B, simple + coverage** | **0.396** | **0.320** | 21 |
| Luna, old interface | 0.546 | 0.561 | 1 |
| **Luna, simple + coverage** | **0.706** | **0.638** | 10 |

**Conclusions.**
- The old interface destroyed real extractions: 50 rejected records per
  run were mostly formatting casualties (invented ids, `"null"` as a
  string), not bad facts. The simple emission nearly eliminates this.
- The gain is a *system* property, not a small-model crutch: Luna gained
  as much as Qwen (judged +0.077), and its LongMemEval store recall went
  0.571 -> 0.857. The old interface was capping the teacher too — which
  also means SFT training data should be regenerated with the new
  interface once it passes the gate.
- Signal-tier gate for promotion is in progress (candidate vs control,
  dev split).

## 2. Negative results (all with evidence under results/fastloop/)

- **Plain-lines emission (no JSON).** Without a grammar constraint the
  model rambles reasoning prose and the parser eats its template echoes.
  Store coverage 0.
- **Thinking mode.** Qwen thinking + json_schema on Ollama returns empty
  content. This is why the pipeline pins non-thinking mode.
- **qwen3:1.7b as extractor.** Perfect JSON (0 rejections), macro 0.122 —
  clean output, no judgment. 4B is the local floor.
- **Per-interaction extraction** (extract every exchange instead of every
  session): matched the best LoCoMo coverage at **14x the calls**. Parity
  at 14x cost; only worth revisiting with a much faster extractor.
- **llama-server migration** (prefix caching + speculative decoding),
  2026-08-25 morning: speculative decoding crashes (draft mode) or gains
  nothing (ngram); prefix caching works (~410 static tokens skipped per
  call) but grammar-constrained decode is slower than Ollama's overall:
  98.9s baseline vs 118.6s / 108.4s (without/with flash attention), with
  output drift under flash attention. Migration reverted; Ollama stays
  canonical. Lesson: local latency here is decode token count, not
  serving-stack overhead.

## 3. Speed is concurrency, not serving stacks

Decode is memory-bandwidth-bound and sessions within a conversation are
inherently sequential (each extraction must see the store its predecessors
built — that is the supersession architecture). Conversations, however,
are fully independent. Exploiting that:

- **Fastloop**: conversations pooled across benchmarks — 2.7x measured
  (513s vs 1379s sequential estimate on the full smoke suite).
- **Signal runner**: same pattern (worker-local model clients, lock-guarded
  shared checkpoints, layout unchanged for resume). Also overlaps hosted
  answer/judge latency with local extraction, which previously left the
  GPU idle between conversations.
- **Both gate arms run simultaneously**, each against its own local
  server (a second process-local `ollama serve`) — two decode queues on
  one GPU, converting waits into work: roughly 2x again.
- **Hard floor**: the longest conversation (51 ordered sessions) is
  ~10 minutes of irreducible sequential extraction.

## 4. Metrics: measure the store, and don't trust one score

- **Strict substring recall** (does the gold answer appear verbatim in
  the store) misses paraphrases — it scored stores containing "tattoos of
  her dogs" as failures against gold "Tattoos of her four dogs", and
  floored both Qwen and Luna near zero on LoCoMo while their stores
  differed enormously.
- **Keyword coverage** (fraction of the reference's substantive words in
  the store) is the free in-loop score. It agrees with the LLM judge at
  the extremes (0.23/0.25 and 0.55/0.56) but can over-credit scattered
  words mid-range.
- **LLM judge** (contained / partial / absent over the store) is the
  final arbiter; keyword coverage selects candidates, the judge confirms.
- Write-path scoring works because retrieval currently loses nothing:
  across every config measured, zero questions were stored-but-not-
  retrieved. The entire battle is extraction.

## 5. Open items

- Signal-dev gate verdict for simple + coverage (running).
- Exact-value retention is the remaining weak spot everywhere: specific
  strings like "Mindful.org" and "6:45 AM" are the facts that die even
  under the best configs. Next prompt iteration targets those.
- Regenerate Luna SFT extraction data with the new interface if the gate
  passes.
