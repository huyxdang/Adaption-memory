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

## 2. Supersession links: restating the old fact beats pointing at it

How should the extractor express "this new fact replaces that old one"?
Storage is identical in all three designs — append-only rows where the new
record's `supersedes_id` names its predecessor; a fact that changes twice
becomes a three-row chain (each new row points backward; old rows are never
edited). What varied is only how the model communicates the link:

1. **Pointer id** (original): copy the old record's opaque id out of a
   JSON candidate blob. A transcription task — small models routinely
   botched it (invented ids, `"null"` as a string), and each botch cost a
   whole record to validation.
2. **Numbered updates**: candidates shown as a numbered list; the model
   emits `"updates": 2`. A positional-reference task — better, but models
   are still mediocre at index tracking.
3. **Replaces text** (proposed by Huy): the model restates the replaced
   fact — `{"content": "Huy has 4 dogs.", "replaces": "Huy has 5 dogs."}`.
   A pure language task (copy a sentence you were just shown); the harness
   matches the text against the shown candidates (exact normalized match,
   then token overlap >= 0.6; ties prefer the non-superseded, later-listed
   candidate) and an unmatched link keeps the fact, drops the link.

Qwen3-4B smoke fastloop, all with the coverage prompt:

| Link design | kw macro | Rejected records | Supersession |
|---|---:|---:|---:|
| pointer id | 0.335 | 58 | 1.0 |
| numbered updates | 0.396 | 21 | 1.0 |
| **replaces text** | **0.415** | **13** | 1.0 |

The ordering matches the theory exactly: the more the link expression
looks like language and the less like bookkeeping, the fewer good facts
die on transcription errors. The replaces margin over numbered is small
on 20 questions (directional, not conclusive), but it is also more
auditable — the extraction log literally reads "4 dogs replaces 5 dogs",
so supersession decisions can be reviewed without dereferencing ids.
Design caveats recorded: matching is scoped to the candidates shown in
that call (a chain hop can only target the current belief, which also
resolves duplicate-text chains like 4 -> 5 -> 4 unambiguously), and
within-session flip-flops still collapse to the durable conclusion.
**Signal-tier verdict (teacher tier):** numbered updates passed the
promotion gate — dev 0.5778 vs control 0.4852 (+9.3), holdout 0.5778 vs
control 0.4722 (+10.6), with zero dev-to-holdout drop. Replaces **failed
at signal scale** (dev 0.4537, below the control) despite winning the
Qwen smoke fastloop. Log forensics located the exact mechanism: the
format suppressed linking itself. With identical prompts and record
volume (~3,740 records each), the numbered variant emitted 85 update
links; replaces emitted 39 (1.0%) — restating a sentence is expensive
and carries an implicit accuracy bar, so the model defaulted to null.
Unlinked updates never demote their stale predecessors, so
knowledge-update questions degraded (supersession accuracy 0.0 on
BEAM). The text-matching resolver itself was mostly innocent: 37 of 45
emitted links resolved correctly. Lesson: emission formats change not
just what a model can express but what it bothers to do — a three-
character link gets used, an expensive one gets skipped. Second lesson:
store-coverage metrics cannot see link behavior, and smoke had too few
update questions for supersession accuracy to warn us.
The replaces idea stays open only behind a smarter resolver (e.g.
embedding matching); the promoted interface is simple + coverage-f1.

## 3. Negative results (all with evidence under results/fastloop/)

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

## 4. Speed is concurrency, not serving stacks

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

## 5. Metrics: measure the store, and don't trust one score

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

## 6. The beat-full-history campaign (2026-08-25, evening)

Goal: memory > full-conversation-history on the 90-question signal set,
teacher tier, mean-of-3 passes everywhere (a 3.5-point single-run noise
band was measured first; single runs are screens only).

Accepted levers (all mechanical; every prompt-persuasion lever failed):

| Config (means of 3) | Dev | Holdout |
|---|---:|---:|
| Full-history baseline | 0.5802 | 0.7241 |
| Session start (simple + coverage) | 0.5605 | 0.4926 |
| + coverage-f1-v2 (verbatim values) | 0.5834 | — |
| + chrono rendering | 0.6006 | 0.4945 |
| + supersession labels (chrono-v2) | 0.6123 | 0.4945 |
| **+ cue-gated adaptive-k (champion)** | **0.6315** | **0.5130** |

Rejected: blanket k=20 (-7 dev), type-aware prompt (inconclusive),
chrono-v3 relative-date resolution (LoCoMo up, everything else down),
multi-query retrieval (dev-flat, holdout -5.8 — paraphrase union displaces
relevant records with paraphrase-shaped noise).

**Verdict: split.** Dev is won outright — +5.1 mean, and the champion's
worst draw beats the baseline's best. Holdout is lost decisively: its
composition (aggregation, multi-hop composition, vocabulary-mismatch
lookups, LoCoMo holdout conversations) is long-context's home turf —
the baseline scores 0.861/0.778 on holdout LongMemEval/LoCoMo. Overall:
memory ~0.587 vs baseline ~0.635.

Durable lessons: (1) models need better inputs, not more instructions —
every accepted lever changed what reached the model, every rejected one
told it what to do; (2) dev/holdout difficulty asymmetry on a 90-question
slice is large enough to invert conclusions — the split discipline caught
what would have been a false "we beat full history" claim from dev alone;
(3) answer-stage stochasticity (default temperature, effort none) puts a
~3.5-point noise band on every single run.

## 7. Open items

- Canonical teacher config after the campaign: coverage-f1-v2 prompt,
  simple emission, chrono-v2 answering, cue-gated adaptive-k.
- Remaining holdout gap classes need write-side or architecture-adjacent
  work: complete-list extraction, verbatim-phrase retention, cross-session
  composition (consolidation/summary records would need their own gate
  under the reproduction invariant).
- The 90-question signal slice is the current arbiter; the full benchmarks
  remain unrun and could reorder these conclusions in either direction.
- Exact-value retention is the remaining weak spot everywhere: specific
  strings like "Mindful.org" and "6:45 AM" are the facts that die even
  under the best configs. Next prompt iteration targets those.
- Regenerate Luna SFT extraction data with the new interface if the gate
  passes.
