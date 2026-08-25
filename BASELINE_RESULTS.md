# Baseline results

Measured locally on 2026-08-24 against the deterministic 90-question
`signal` tier. These are pre-final comparison numbers, not estimates of the
official full-benchmark scores.

## Signal: GPT-5.6 Luna medium

Configuration:

- Answer model: `gpt-5.6-luna`
- Answer reasoning: `medium`
- Answer completion cap: 4,096 tokens
- Judge model: `gpt-5.6-luna` for LongMemEval and BEAM
- Judge reasoning: `high`
- Judge completion cap: 2,048 tokens

| System | Benchmark | Completed | Score | Answer calls | Input tokens | Output tokens |
|---|---|---:|---:|---:|---:|---:|
| Full history | LoCoMo | 30/30 | F1 0.4466 | 30 | 605,566 | 897 |
| Full history | LongMemEval | 30/30 | Accuracy 0.9000 | 30 | 1,294,392 | 1,680 |
| Full history | BEAM | 30/30 | Overall 0.6595 | 30 | 3,992,871 | 7,416 |
| Adaptive memory | LoCoMo | 30/30 | F1 0.5133 | 162 | 490,724 | 165,663 |
| Adaptive memory | LongMemEval | 13/30 | Not scored | 270 | 1,104,125 | 146,491 |
| Adaptive memory | BEAM | 30/30 | Overall 0.5914 | 56 | 481,771 | 44,589 |

Judge usage is additional and is not currently persisted by the harness.

The adaptive LongMemEval run was intentionally stopped after 13 completed
answers when the evaluation plan switched back to the smoke tier. Its answers
remain resumable under
`results/signal/longmemeval/adaptive-luna-medium/`; no partial score is reported.

### Comparison

| Benchmark | Full history | Adaptive memory | Adaptive delta |
|---|---:|---:|---:|
| LoCoMo | 0.4466 | 0.5133 | +0.0667 |
| BEAM | 0.6595 | 0.5914 | -0.0681 |

On these completed signal comparisons, adaptive memory improves LoCoMo
temporal and adversarial behavior and BEAM abstention and knowledge updates.
It loses detail on several multi-hop, extraction, summarization, and temporal
BEAM cases. Because the same Luna model performs strongly with full history,
these gaps are evidence about the current memory write/retrieval path rather
than sufficient evidence to escalate the whole evaluation to Terra or Sol.

## Signal: GPT-5.6 Luna none

Fresh full-history-only rerun with the API's explicit
`reasoning_effort="none"` setting for both answers and judges. The answer
completion cap was 4,096 tokens; LongMemEval's direct-label judge used 10
tokens and BEAM's JSON rubric judge used 500.

| Benchmark | Completed | Score | Answer calls | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| LoCoMo | 30/30 | F1 0.4607 | 30 | 605,566 | 333 |
| LongMemEval | 30/30 | Accuracy 0.7667 | 30 | 1,294,392 | 800 |
| BEAM | 30/30 | Overall 0.7464 | 30 | 3,992,871 | 5,476 |

The answer stages total 90 calls, 5,892,829 input tokens, and 6,609 output
tokens. BEAM produced all 77 fixed rubric decisions with zero unparseable judge
outputs; LongMemEval's judge returned only exact `yes` or `no` labels.

### None versus medium/high

| Benchmark | Medium answers, high judge | None answers, none judge | None delta |
|---|---:|---:|---:|
| LoCoMo | 0.4466 | 0.4607 | +0.0141 |
| LongMemEval | 0.9000 | 0.7667 | -0.1333 |
| BEAM | 0.6595 | 0.7464 | +0.0869 |

The judge configuration also changed for LongMemEval and BEAM, so those score
deltas are end-to-end configuration comparisons rather than isolated answer
model effects. The four LongMemEval labels that changed from correct to
incorrect also had materially weaker no-reasoning answers: two omitted user
preference context, one missed the required temporal evidence, and one omitted
one of two gift purchases. The LongMemEval loss therefore cannot be explained
only by judge variance.

Use `none` for the fast iteration path. The approved final-run configuration is
also Luna `none` for answers, paired with an independent Luna `high` judge for
LongMemEval and BEAM. This keeps answer latency low while spending additional
reasoning on grading. The recorded LongMemEval regression remains an explicit
accepted tradeoff of disabling answer reasoning.

## Smoke: GPT-5.6 Luna none

Measured with the full-history system only. Both answer and judge models used
the API's explicit `reasoning_effort="none"` setting. The answer completion cap
was 4,096 tokens; LongMemEval's direct-label judge used 10 tokens and BEAM's
JSON rubric judge used 500.

| Benchmark | Completed | Score | Answer calls | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| LoCoMo | 5/5 | F1 0.3010 | 5 | 44,035 | 53 |
| LongMemEval | 7/7 | Accuracy 0.8571 | 7 | 63,114 | 165 |
| BEAM | 10/10 | Overall 0.6533 | 10 | 192,341 | 2,009 |

The three answer stages total 22 calls, 299,490 input tokens, and 2,227 output
tokens. BEAM produced 27 fixed rubric decisions with zero unparseable judge
outputs. Judge usage is additional and is not persisted by the harness.

The longest job, BEAM answering plus judging, completed in about 84 seconds.
The smoke tier has only one example per category or ability, so these scores
are useful for fast regressions but should not be interpreted as stable model
rankings.
