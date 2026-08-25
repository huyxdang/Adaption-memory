# Northstar-style extractor SFT dataset

This dataset contains 374 validated Luna-target extraction pairs
from `results/overnight/signal/luna-target/F4-dev-coverage` using format F4. Each assistant
target contains only records accepted by the production validator; any session
with a schema, supersession-reference, or exact-value failure is excluded.

## Provenance and contamination boundary

Sources are the benchmark **signal-dev development subsample**, not official
benchmark train splits: LongMemEval, LoCoMo, and BEAM. This repository did not
contain separately identified official train-split files, so no additional
sessions were sampled from the flat full datasets. Do not describe these pairs
as official train-split data. They are suitable for pipeline prototyping, but a
future official full evaluation must keep its held-out questions isolated and
should prefer a replacement dataset built from verified train splits.

- Source run: `results/overnight/signal/luna-target/F4-dev-coverage`
- Extractor prompt: `coverage`
- Accepted pairs: 374
- Rejected pairs: 114
- Rejection rate: 0.2336
- Training started: no
