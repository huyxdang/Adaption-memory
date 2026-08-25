# Extractor SFT evaluation plan

The tuned Qwen3-4B extractor plugs into the same `overnight-memory` harness as
the zero-shot and few-shot arms. The frozen storage format is **F4**.
The frozen extractor prompt revision is **coverage**.
The answerer and judge remain `gpt-5.6-luna` with reasoning effort `none`.

## Gate order

1. Smoke: require at least 95% schema validity and complete pipeline output.
2. Signal-dev: compare extraction recall proxy and supersession accuracy with
   the untuned Qwen3-4B few-shot and Luna-target traces.
3. Signal-holdout: run once only after choosing a checkpoint.
4. Full: remains manual and prohibited by the overnight harness.

## Success criterion

Before any full benchmark run, the tuned extractor must close at least 70% of
the Qwen3-4B-fewshot versus Luna-target gap on both direct extractor metrics.
End-to-end judge accuracy is reported but does not replace that gate.

No training was performed by the overnight plan.
