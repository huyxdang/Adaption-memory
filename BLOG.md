# Agent memory fails at the interface, not the model

*Notes from reproducing Adaption Labs' write-time memory architecture — and the
week the JSON schema turned out to be the bottleneck.*

We set out to reproduce the architecture from ["Better Agent Memory Starts
Before Retrieval"](https://adaptionlabs.ai/blog/agent-memory-write-time): an
LLM extractor that writes two kinds of memory records at conversation time —
narrative records that preserve reasoning, atomic records that preserve exact
facts — into an append-only store where updates supersede rather than
overwrite. The thesis of that article is that the highest-leverage
optimization happens at write time, when information is extracted into
memory, not later when it is searched for.

We can report that the thesis survived our attempt to reproduce it — in a way
we didn't expect. The write path was the bottleneck, but the biggest lever
inside it wasn't the extraction model, the retrieval stack, or the prompt
alone. It was the interface: the shape of the JSON we forced the extractor to
emit.

## Retrieval lost nothing. Extraction lost almost everything.

Our first end-to-end runs were bad, and the failure analysis was unambiguous.
Of 28 bucketed failures on our evaluation slice, 22 were facts that never got
extracted, 3 were retrieval misses, and 3 were answering mistakes. When we
later instrumented the store directly, the retrieval number went to zero:
across every configuration we measured, not a single fact that made it into
the store failed to surface when its question came. The entire battle was
write-time.

So we built the tightest loop we could around that one stage: run extraction
only, then check whether each benchmark question's answer exists in the store
— no answering model, no judge, ~100 seconds per iteration on a laptop. Most
of what follows was found with that loop and confirmed at scale afterward.

## The model was not the bottleneck

The obvious suspect was our small local extractor (Qwen3-4B). So we ran the
same pipeline with a frontier model, and then with an even stronger one. The
two frontier tiers produced *identical* miss sets — capability saturated —
and both still left roughly half the needed facts out of the store.

When even the strongest model you can buy misses half the facts, the problem
is what you asked for, not who you asked.

## What we were asking for was hostile

Our original extraction contract required, per memory: a type, the content, a
list of entity names, and — for updates — the exact ID of the superseded
record, copied out of a JSON blob of candidate memories:

```json
{"type": "atomic", "content": "Huy has 4 dogs.",
 "entities": ["Huy"], "supersedes_id": "rec_8f3ab2"}
```

Small models botched the ID transcription constantly — invented IDs, the
string `"null"`, mangled copies — and every botch cost the whole record to
validation: ~50 good facts rejected per run. The prompt compounded it. It was
precision-tuned: *be concise, don't repeat, output an empty list when nothing
durable is new.* The model obeyed and saved little.

We changed exactly two things:

**A two-field contract.** The model emits type and content, plus an optional
`updates: 2` pointing at a numbered candidate list. Our code resolves the
number to the real record ID, derives entities from the content, stamps
session and date, and embeds the text. Storage still holds the full original
record — the architecture is untouched; the model is just responsible for
less of it.

**A coverage prompt.** *Preserve every explicit fact likely to support a
later question. Emit separate records for independent facts so one detail is
not lost inside a broad summary.* The narrative/atomic distinction and the
copy-numbers-verbatim discipline stay.

Judged store recall (does the store contain the information each benchmark
question needs), before and after, at both model tiers:

| Extractor | Old interface | New interface |
|---|---:|---:|
| Qwen3-4B (local) | 0.25 | 0.32 |
| GPT-5.6 (frontier) | 0.56 | 0.64 |
| — validation rejections per run | ~50 | ~2 |

The frontier model gained as much as the small one. The old interface wasn't
a small-model crutch problem; it was capping everyone.

## Gate it like you mean it

Cheap loops overfit. Earlier in this project, a single-record-type ablation
"beat" the canonical two-type design by 2.2 points on our dev split and
briefly got promoted — a difference of one net answer across 56 questions.
We reverted it and wrote a rule: architecture-level changes need at least +3
points of macro judge accuracy on the dev split, confirmed on a holdout split
the change has never seen.

The new interface went through that gate on the full pipeline — extract,
store, retrieve, answer, judge:

| | Dev (56 q) | Holdout (34 q) |
|---|---:|---:|
| Old interface | 0.485 | 0.472 |
| New interface | **0.578** | **0.578** |

+9.3 on dev, +10.6 on holdout, zero generalization drop. Three times the
threshold that the false positive couldn't clear.

## Formats change what models bother to do

The best finding of the week came from a variant that lost.

Mid-redesign, an appealing idea came up: drop the numbered pointer too, and
let the model express an update by *restating the fact it replaces* —

```json
{"content": "Huy has 4 dogs.", "replaces": "Huy has 5 dogs."}
```

— with the harness matching the restated text back to a stored record. It's
more natural for a language model (copying a sentence beats tracking an
index) and self-documenting in the logs. On the fast loop it looked like the
winner: best recall of any local config, fewer rejections than the numbered
version.

At full-pipeline scale it scored *below the old interface*. The post-mortem
was not what we expected. The text matching was fine — 37 of 45 emitted
links resolved correctly. The problem was that there were only 45 links.
With identical prompts and identical record volume (~3,740 records each),
the numbered variant emitted 85 update links; the restating variant emitted
39. Restating a sentence is work, and it carries an implicit accuracy bar —
so when unsure, the model just wrote `null`. And an unlinked update never
demotes its stale predecessor, so every question about a changed fact became
a coin flip between the old and new value.

A three-character link gets used. An expensive link gets skipped. Emission
formats don't just bound what a model *can* express — they set the price of
each behavior, and models economize.

## What didn't work, so you don't have to

- **Plain-text output (no JSON).** Without a grammar constraint the model
  rambles; the parser ate its own template echoes. Recall: zero.
- **Letting the model think first.** Thinking mode plus JSON-schema
  enforcement returned empty responses on our local stack.
- **A smaller extractor (1.7B).** Perfect JSON, no judgment — recall
  collapsed to a third of the 4B model's.
- **Extracting per message instead of per session.** Same recall, 14× the
  calls.
- **A faster serving stack.** Prefix caching and speculative decoding on
  llama.cpp measured *slower* end-to-end than our boring Ollama baseline —
  local extraction latency is decode token count, and nothing else.
- **Trusting one metric.** Strict substring recall scored a store containing
  "tattoos of her dogs" as a failure against gold "tattoos of her four
  dogs" — and scored two wildly different stores identically. We now run a
  keyword-coverage score in the loop and an LLM judge as the arbiter, and
  the restating-variant episode taught us that neither sees supersession
  link behavior; that needs its own counter.

## Where the reproduction actually stands

Honesty section. The article's headline result is that memory *beats*
sending the full conversation history — 90.6% vs 60.6% on LongMemEval —
with 97% fewer tokens. We are not there. On our 90-question signal
subsample, our best memory configuration scores 0.578 macro against a
full-history baseline around 0.62: memory wins on one benchmark of three
and loses the other two. Our token reduction is real but smaller (~40%).
We have not yet run the full benchmarks, and our local-model distillation
story is unfinished.

What we can confirm from our own data: the write path is where the leverage
lives, exactly as the article argues — we moved end-to-end accuracy nine
points without touching retrieval or answering. The remaining gap to full
history is concentrated in exact-value retention (specific times, URLs,
dates that die during extraction), and that is a write-time problem too.

The architecture reproduces. The numbers, so far, don't — and the distance
between those two sentences is where we're working next.

---

*Everything here is reproducible from the repo: the fast extraction loop,
the gated comparisons, per-run checkpoints, and a findings file with the
negative results included.*
