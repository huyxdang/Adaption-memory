# Project rules

## Git authorship

**Never add Claude as a co-author in this repo.**

Do not append `Co-Authored-By: Claude ...` (or any other Claude/Anthropic
attribution trailer) to commit messages, and do not add "Generated with Claude
Code" footers to commits or PR bodies here. This overrides the default harness
behavior of adding a `Co-Authored-By` trailer.

Commits are authored by the repo owner alone.

## Reproduction invariant

**This repo reproduces the Adaption Labs write-time memory architecture.**
The canonical system is F1: narrative + atomic record types with append-only
supersession. That is the point of the project, not a tunable.

- Single-type formats (F4 and friends) are **ablations**. They may be run,
  measured, and reported — always labeled as ablations — but must never be
  promoted to "the system", used for reproduction SFT data, or presented as
  the project's result, regardless of their scores.
- **Architecture-level changes require a pre-stated minimum effect**: at least
  3 points of macro judge accuracy on the signal dev split, confirmed on the
  holdout split. Differences below that threshold are noise; keep the
  reproduction design. (Context: F4 was once wrongly promoted over F1 on a
  0.5074 vs 0.4852 dev difference — one net answer across 56 questions.)
- If an ablation genuinely beats the reproduction above threshold, that is a
  *finding about the article's claim* to report explicitly — still not a
  license to swap the canonical architecture.
