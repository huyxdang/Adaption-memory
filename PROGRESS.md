# Progress log

## Task: Scan this codebase

**Status:** in progress
**Started:** 2026-08-24 15:24 +07

### Scope

- Map the repository structure, runtime paths, evaluation architecture, tests, dependencies, and current working-tree state.
- Identify what is implemented, what remains incomplete, and any concrete risks or inconsistencies.

### Plan

- Read the project guidance and public contract.
- Trace interfaces, systems, evaluation adapters, and CLI execution.
- Run the offline verification available in the repository.
- Summarize architecture, health, and prioritized findings.

### 2026-08-24 15:24 - Repository surface mapped

**Status:** in progress

**Completed**

- Confirmed this is a small Python 3.13 package with one CLI, three benchmark adapters, one concrete memory system, and one offline test module.
- Confirmed the working tree already contains user changes across the README, interface, eval adapters, runner, and full-history system.

**Evidence**

- `pyproject.toml`, `README.md`, `CLAUDE.md`, `src/adaption_memory/`, and `tests/test_evals.py`.
- `git status --short --branch` reports eight modified tracked files on `main`.

**Decisions**

- Treat all pre-existing modifications as user-owned and perform a read-only scan apart from this progress log.

**Next**

- Trace implementation and tests, then run verification.

**Blockers**

- None.

## Task: Build a representative local mini benchmark suite

**Status:** complete
**Started:** 2026-08-24 19:54 +07

### Scope

- Create deterministic local mini versions of LongMemEval, LoCoMo, and BEAM.
- Preserve each benchmark's main question/category mix and a representative context shape.
- Make selection reproducible and auditable through a generated manifest.
- Validate the minis through offline tests, oracle scoring, and a real hosted-model run.

### Plan

- Measure the full datasets' strata and context-size distributions.
- Implement one generator that produces all three native-format mini datasets.
- Add regression tests and concise generation/run instructions.
- Generate and evaluate the local suite, then record measured runtime and limitations.

**Blockers**

- None.

### 2026-08-24 20:37 - Mini suite generated and live-validated

**Status:** in progress

**Completed**

- Added a deterministic generator for native-format LongMemEval, LoCoMo, and BEAM minis plus an auditable manifest.
- Added `make-mini`, `eval --mini`, isolated `results/mini/` outputs, native-loader coverage tests, and exact generation/run documentation.
- Preserved all official strata: six LongMemEval question types plus abstention, five LoCoMo categories, and ten BEAM memory abilities.
- Retained annotated or rubric-matched support context, deterministic distractors, and BEAM time anchors.
- Corrected BEAM's event-ordering oracle to emit one event per line so the official tau path recognizes gold ordering.
- Added current Chat Completions compatibility for GPT-5.6 (`max_completion_tokens`, default temperature omission), separate judge reasoning/token controls, and secure OpenAI-key fallback.
- Verified the three oracle pipelines at 1.0 and ran the real adaptive system with GPT-5.6 Luna medium plus a high-effort Luna judge.

**Evidence**

- Generated files total under 500 KB: 7 LongMemEval cases, 5 LoCoMo questions, and 10 BEAM questions.
- Oracle results: LoCoMo F1 1.0, LongMemEval accuracy 1.0, BEAM overall 1.0.
- Hosted adaptive results: LoCoMo F1 0.4000, LongMemEval accuracy 0.7143, BEAM overall 0.4267.
- Answer-side hosted usage: 66 calls, 165,865 input tokens, and 37,760 output tokens across the suite.
- The offline suite passes 37 tests; compilation and diff checks pass.

**Next**

- Complete independent code review and final regression verification.

**Blockers**

- None.

### 2026-08-24 21:29 - Task complete

**Status:** complete

**Completed**

- Added hard dataset and model-configuration provenance guards so resumed runs cannot silently mix mini/full data, answer settings, judge settings, or endpoints.
- Corrected provider-specific token-limit routing and verified Ollama now honors the requested local completion cap.
- Kept the supplied OpenAI credential process-local, removed it after the hosted run, and verified it was not written to the workspace.
- Completed independent code review with no remaining critical or important findings.

**Evidence**

- The offline suite passes 42 tests; compilation and diff checks pass.
- Regeneration produced byte-identical mini datasets and manifest hashes.
- A live Qwen request stopped at exactly the configured five-token cap.
- Reusing a mini result directory for full data is rejected with a configuration-mismatch error.
- All three oracle mini pipelines score 1.0, and the hosted Luna baselines reproduce from provenance-checked result directories.

**Next**

- Use the mini suite for rapid iteration, then run the unchanged official datasets for final reporting.

**Blockers**

- None.

## Task: Resume Qwen3 1.7B download

**Status:** complete
**Started:** 2026-08-24 17:54

### 2026-08-24 18:05 - Download resumed and verified

**Status:** complete

**Completed**

- Resumed `ollama pull qwen3:1.7b` from Ollama's cached partial layer.
- Downloaded the remaining data, verified the SHA-256 digest, and wrote the Ollama manifest.
- Confirmed `qwen3:1.7b` is installed alongside `qwen3:0.6b`.
- Ran a local non-thinking chat request and received exactly `READY`.

**Evidence**

- `ollama list` reports `qwen3:1.7b` with ID `8f68893c685c` and size `1.4 GB`.
- The verification request completed successfully in about 1.2 seconds including model load.

**Next**

- Use `qwen3:1.7b` for the next end-to-end memory evaluation.

**Blockers**

- None.

### 2026-08-24 17:54 - Download paused by user

**Status:** blocked

**Completed**

- Switched from the corporate network to the user's hotspot and successfully started `ollama pull qwen3:1.7b`.
- Confirmed the registry block was gone and the 1.4 GB model layer began downloading.
- Stopped the pull on request and verified no `ollama pull qwen3:1.7b` process remains.

**Evidence**

- Pull reached approximately 413 MB of 1.4 GB before stopping.
- Process inspection returned no active matching pull command after cancellation.

**Decisions**

- Leave Ollama's partial download intact so a later pull can resume it.

**Next**

- On a faster connection, rerun `ollama pull qwen3:1.7b`, then verify with `ollama list` and a local inference call.

**Blockers**

- Download intentionally deferred to another turn.

## Task: Find a corporate-network-safe Qwen3 1.7B download route

**Status:** complete
**Started:** 2026-08-24 17:05

### Scope

- Test legitimate alternate distribution paths for the Qwen3 1.7B GGUF without weakening TLS or bypassing company security controls.

### 2026-08-24 17:05 - Distribution endpoints tested

**Status:** blocked

**Completed**

- Confirmed the local Ollama server and `qwen3:0.6b` remain healthy.
- Tested the Ollama registry, Hugging Face, the `hf.co` alias, Hugging Face mirror mode, and ModelScope.
- Identified the official-compatible Q4_K_M GGUF as a 1.28 GB file with published SHA-256 `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`.

**Evidence**

- Ollama registry and ModelScope returned HTTP 503 through the corporate network.
- Hugging Face connections were reset; the public mirror redirected to Hugging Face and its Python client path encountered the corporate self-signed certificate chain.
- Ollama `/api/tags` lists the installed 522 MB `qwen3:0.6b` model.

**Decisions**

- Do not disable TLS verification or use untrusted proxy sites.
- Use a permitted non-corporate network, an IT-approved domain allowlist, or transfer the verified GGUF from another machine; then import it locally into Ollama.

**Next**

- Obtain `Qwen3-1.7B-Q4_K_M.gguf` over a permitted route and import it into Ollama, or continue hardening the 0.6B extraction path meanwhile.

**Blockers**

- The current corporate network blocks all tested trustworthy model distribution endpoints.

### 2026-08-24 17:14 - Native pull confirms policy enforcement

**Status:** blocked

**Completed**

- Retried `ollama pull qwen3:1.7b` through Ollama's native authenticated registry flow.
- Tested the explicitly authorized TLS-disabled mirror route with bounded byte-range probes.

**Evidence**

- Ollama returned HTTP 503 with an explicit company-policy `Application Blocked` page before any model layer downloaded.
- TLS-disabled mirror requests still redirected to the blocked Hugging Face domain, where the connection was reset.

**Decisions**

- TLS verification is not the controlling failure, so weakening it further cannot unblock the download.
- Do not evade or disguise traffic around an explicit corporate security control. Continue only through an approved allowlist, a policy-permitted offline transfer, or the existing local model.

**Next**

- Prepare the offline GGUF import and checksum commands, request an IT allowlist, or harden the installed 0.6B workflow.

**Blockers**

- Company policy explicitly blocks the required model distribution application/domain.

### 2026-08-24 17:25 - Approved access paths prepared

**Status:** in progress

**Completed**

- Added a ready-to-send IT allowlist request with the native Ollama route and Hugging Face's documented Hub/Xet/CDN hostnames.
- Added an offline importer for the 1.28 GB Q4_K_M GGUF that verifies the publisher-provided SHA-256 before creating an Ollama model.

**Evidence**

- `MODEL_ACCESS.md`
- `scripts/import_qwen3_1_7b.sh`
- Hugging Face's official proxy/firewall documentation and Ollama's official GGUF import documentation.

**Decisions**

- Pin the importer to one known artifact and checksum instead of accepting arbitrary model files.
- Keep both external routes ready, but proceed to the installed 0.6B model because allowlisting and physical transfer require action outside this workspace.

**Next**

- Validate importer failure behavior, then harden structured extraction for the available 0.6B model.

**Blockers**

- IT submission and approved-device transfer require user/company-side access not exposed in this task.

### 2026-08-24 17:48 - Local fallback hardened and measured

**Status:** in progress

**Completed**

- Added strict OpenAI-compatible structured output for write-time extraction.
- Added deterministic rejection of placeholder, metadata, malformed, and non-dotted atomic keys.
- Tested and rejected an always-split narrative/atomic extractor because it doubled benchmark cost without improving the sampled LoCoMo score.
- Kept a refined single-call extractor and made active atomic context authoritative for current-state reasons.

**Evidence**

- Offline regression: 22 tests passed; `compileall` and `git diff --check` passed.
- Live Paris-to-Lyon acceptance: 3 calls, 1,284 prompt tokens, 228 completion tokens; active Lyon fact supersedes Paris and the answer cites the team relocation.
- Final one-question LoCoMo slice: 20 calls, 27,636 prompt tokens, 3,519 completion tokens, F1 0.0.
- Discarded split design on the same slice: 39 calls, 50,632 prompt tokens, 7,075 completion tokens, F1 0.0.

**Decisions**

- Keep 0.6B as a fast plumbing and acceptance-test model, not a benchmark-quality substitute for 1.7B.
- Preserve one extraction call per session because the two-call design had no measured benchmark benefit.

**Next**

- Complete independent code review and address any Critical or Important findings.

**Blockers**

- Installing 1.7B still requires IT allowlisting or a policy-approved offline transfer.

### 2026-08-24 18:10 - Task complete

**Status:** complete

**Completed**

- Delivered the IT allowlist request, checksum-pinned offline importer, and documented model-access handoff.
- Delivered strict structured extraction and deterministic atomic-key validation while preserving one extraction call per session.
- Added direct response-format forwarding coverage and expanded metadata/casing rejection tests.
- Attempted two independent read-only code reviews; both reviewer tasks timed out without returning findings, so no review verdict is claimed.

**Evidence**

- Final regression: 23 tests passed.
- `sh -n scripts/import_qwen3_1_7b.sh`, `compileall`, and `git diff --check` passed.
- The importer rejects a non-matching file before `ollama create`.
- Live 0.6B acceptance previously passed in 3 calls; the sampled real LoCoMo result remains F1 0.0 and is documented in the README.

**Decisions**

- Treat 0.6B as a fast local plumbing model only.
- Do not represent the community Q4 quantization as an official Qwen-published artifact.

**Next**

- Submit the allowlist request or transfer the verified GGUF through an approved route; then run `scripts/import_qwen3_1_7b.sh`.

**Blockers**

- The 1.7B bytes cannot be obtained on the current network without company-side allowlisting or approved offline transfer.

### 2026-08-24 16:47 - Model ready; Python localhost blocked in sandbox

**Status:** in progress

**Completed**

- Downloaded and checksum-verified `qwen3:0.6b`; Ollama lists the 522 MB model locally.

**Evidence**

- The first live `AdaptiveMemorySystem` call failed before reaching Ollama with `httpx2.ConnectError: [Errno 1] Operation not permitted` from the managed sandbox.
- The Ollama API remains reachable through `curl`, confirming this is a Python socket permission boundary rather than a server or model failure.

**Decisions**

- Rerun the local Python evaluation outside the socket sandbox without changing project code.

**Next**

- Complete the handcrafted update test, then run a minimal benchmark slice.

**Blockers**

- Requires approved local-network access for the Python evaluator process.

### 2026-08-24 16:50 - Live local evaluation complete

**Status:** complete

**Completed**

- Added optional OpenAI-compatible `reasoning_effort` support and exposed it through the CLI; `none` disables Qwen3 thinking for fast iteration.
- Tightened the extractor prompt after raw 0.6B output copied schema placeholders and stored session metadata.
- Completed a two-session live update test and a one-question LoCoMo slice through the real adaptive system.

**Evidence**

- The improved handcrafted run completed in 6.2 seconds, produced a real `sam.name` atomic, preserved two narratives, and correctly answered that Sam lives in Lyon because his team moved there.
- Disabling thinking reduced completion usage from 593 to 175 tokens in the handcrafted three-call run.
- The LoCoMo slice ingested 19 sessions and answered one temporal question in 20 calls; it predicted `The information is not available.` for gold `7 May 2023`, scoring `0.0` with 25,688 prompt and 4,420 completion tokens.
- Final regression suite: 21 tests passed; compilation and diff checks passed.

**Decisions**

- Treat `qwen3:0.6b` as a fast plumbing model only. It is below the quality floor for evaluating atomic extraction or benchmark accuracy.
- Recommend `qwen3:1.7b` as the next quality iteration, while retaining `reasoning_effort=none` for speed.

**Next**

- Pull `qwen3:1.7b` only when a stronger extraction-quality run is desired.

**Blockers**

- None for local development. Python evaluation commands need approved localhost access in this managed environment.

### 2026-08-24 16:50 - Task complete

**Status:** complete

**Completed**

- Installed Ollama, installed and verified `qwen3:0.6b`, optimized the local iteration path, and ran live adaptive-memory verification.

**Evidence**

- Ollama `0.32.15` serves locally; the model pull passed SHA-256 verification; live project and benchmark calls completed.

**Next**

- None.

**Blockers**

- None.

### 2026-08-24 16:18 - Application installed; CLI link fallback required

**Status:** in progress

**Completed**

- Downloaded and installed the official Ollama application into `/Applications/Ollama.app`.

**Evidence**

- The official installer reached 100% and completed the application move.
- It then failed while creating `/usr/local/bin/ollama` because `sudo` requires an interactive password terminal.

**Decisions**

- Link the installed CLI into the existing user-managed `/opt/homebrew/bin` path instead of adding an interactive-sudo dependency.

**Next**

- Create the Homebrew-path link, launch Ollama, and verify the API.

**Blockers**

- None; the non-sudo link is a safe equivalent on this Homebrew-based Mac.

### 2026-08-24 16:23 - Iteration model reduced

**Status:** in progress

**Completed**

- Confirmed Ollama `0.32.15` is serving at `127.0.0.1:11434`.
- Stopped the interrupted `qwen3:8b` pull after roughly 83 MB of local transfer.

**Decisions**

- Use `qwen3:0.6b` (about 523 MB) for the first live plumbing test; do not treat its accuracy as representative of the memory design.

**Next**

- Pull `qwen3:0.6b`, smoke-test structured extraction, and run the smallest adaptive evaluation.

**Blockers**

- None.

### 2026-08-24 15:41 - Working adaptive memory slice complete

**Status:** in progress

**Completed**

- Added narrative and atomic extraction, exact-fact deduplication, append-only supersession, bounded retrieval, and focused answer prompting.
- Registered the system as `--system adaptive` without coupling benchmark adapters to the concrete implementation.
- Corrected usage accounting so LoCoMo and BEAM include one-time ingestion cost once per logical conversation.
- Added five focused tests; the complete suite now passes 19 tests.

**Evidence**

- `src/adaption_memory/systems/adaptive.py`, `src/adaption_memory/systems/__init__.py`, and shared usage helpers in `src/adaption_memory/evals/common.py`.
- `.venv/bin/python -m pytest -q` completed with `19 passed in 3.26s`.
- CLI help lists `--system {adaptive,full-history,oracle}`.

**Decisions**

- Keep retrieval bounded and deterministic rather than adding a vector store or knowledge graph.
- Attribute shared ingestion usage to the first answer so existing answer JSONL remains the single resumable artifact and aggregate usage stays correct.

**Next**

- Finish documentation, inspect the complete diff, and rerun regression checks.

**Blockers**

- A live model benchmark cannot run locally because no Ollama server or installation is present; offline fake-LLM coverage exercises the complete system contract.

### 2026-08-24 15:52 - Review feedback applied

**Status:** in progress

**Completed**

- Added durable supersession references so variant extractor keys resolve to the prior canonical atomic identity.
- Bounded the active-state context supplied to extraction and added recency coverage alongside lexical relevance.
- Added resume, key-variant, and bounded-context tests; corrected the protocol docstring from three calls to four.

**Evidence**

- Independent review identified prompt-only key stability as an important correctness gap.
- The first bounded-context regression run failed because one-character numeric tokens were filtered, causing `fact 2` to miss; tokenization was corrected to retain numeric tokens.

**Decisions**

- Did not add a crash ledger: result usage represents one completed logical benchmark run, and the current resume path already attributes ingestion once in that logical artifact.
- Retained linear in-memory historical scanning for this first slice while bounding every LLM-facing context; it is simple and adequate for the current benchmark sizes.

**Next**

- Rerun all verification and close the progress record.

**Blockers**

- Live model and isolated wheel builds remain unavailable without external runtime/dependency downloads.

### 2026-08-24 16:03 - Task complete

**Status:** complete

**Completed**

- Delivered a working `adaptive` system with narrative and atomic write-time extraction, durable supersession references, canonical key preservation, bounded active-state and answer contexts, and focused retrieval.
- Corrected one-time ingestion usage attribution across LongMemEval, LoCoMo, and BEAM, including normal resumed logical runs.
- Updated the CLI registry, README, protocol documentation, and regression coverage.
- Applied two rounds of independent review feedback; the final verdict reports no Critical or Important issues and is ready to merge.

**Evidence**

- `.venv/bin/python -m pytest -q`: `21 passed in 2.37s`.
- Focused supersession, context-bound, and resume tests pass.
- `.venv/bin/python -m compileall -q src tests` and `git diff --check` pass.
- CLI help exposes `--system {adaptive,full-history,oracle}`.

**Next**

- Start a compatible local model server and run a small real `adaptive` benchmark slice, then scale only after inspecting extraction quality and usage.

**Blockers**

- No Ollama installation/server is currently available for live inference.
- Offline wheel construction could not be verified because the local environment lacks `pip`/`hatchling` and sandboxed network access prevented fetching the build backend; source imports and package resources were verified directly.

## Task: Install local model and run adaptive evaluation

**Status:** in progress
**Started:** 2026-08-24 16:12 +07

### Scope

- Install Ollama on the Apple Silicon Mac, download `qwen3:8b`, verify its OpenAI-compatible endpoint, and run a small adaptive LoCoMo evaluation slice.

### Plan

- Install the official macOS runtime.
- Pull and smoke-test the 5.2 GB `qwen3:8b` model.
- Run an isolated adaptive benchmark slice and inspect answers, usage, and score.

### 2026-08-24 16:12 - Installation prerequisites verified

**Status:** in progress

**Completed**

- Confirmed Apple Silicon, macOS 26.5.2, Homebrew availability, 109 GiB free disk space, and no existing Ollama binary/application.
- Confirmed the official `qwen3:8b` Ollama artifact is 5.2 GB with a 40K context window.

**Evidence**

- Local system checks and the official Ollama macOS/model pages.

**Decisions**

- Use Ollama's official macOS installer and its local OpenAI-compatible endpoint; no Hugging Face download is needed.

**Next**

- Download, inspect, and run the official installer.

**Blockers**

- None.

## Task: Implement write-time memory

**Status:** in progress
**Started:** 2026-08-24 15:34 +07

### Scope

- Correct write-time usage accounting across shared-conversation benchmark adapters.
- Implement a minimal narrative and atomic memory system with append-only supersession and focused retrieval.
- Register, test, document, and smoke-test the system without requiring a live external model.

### Plan

- Add shared usage arithmetic and adapter tests.
- Add the adaptive system as one concrete module behind the existing protocol.
- Verify extraction, deduplication, supersession, retrieval, prompting, registry selection, and all existing scorers.

### 2026-08-24 15:34 - Source design translated into implementation contract

**Status:** in progress

**Completed**

- Confirmed the source design requires two append-only representations: causal narrative memories and exact atomic facts.
- Defined supersession by stable atomic keys: a changed value appends a new record and marks the prior active record as superseded.
- Chose lightweight lexical retrieval over adding a vector database or knowledge graph, matching the source's finding that extra structure was unnecessary.

**Evidence**

- Adaption Labs, `Better Agent Memory Starts Before Retrieval`, sections on narrative vs atomic memory and appending superseding states.

**Decisions**

- Reuse the existing answering LLM for extraction, consistent with the single-LLM registry seam and the source evaluation setup.
- Keep storage in memory for the first working vertical slice; persistence is outside the current benchmark contract.

**Next**

- Implement shared usage accounting and the adaptive system.

**Blockers**

- None.

### 2026-08-24 15:26 - Runtime architecture traced

**Status:** in progress

**Completed**

- Traced the CLI through all three benchmark adapters and the `MemorySystem` registry.
- Confirmed only `FullHistorySystem` is implemented; narrative memory, atomic memory, evolutionary state tracking, and retrieval are still README-level requirements.
- Confirmed all four benchmark data files are present and existing result artifacts cover only oracle answer/scoring plumbing.

**Evidence**

- `src/adaption_memory/evals/run.py`, `src/adaption_memory/interface.py`, `src/adaption_memory/systems/__init__.py`, and `src/adaption_memory/systems/full_history.py`.
- `data/` is 287 MB; `results/` contains oracle artifacts for BEAM, LoCoMo, and LongMemEval.

**Decisions**

- Verify with the existing `.venv` because `uv run` cannot initialize its user cache inside the managed sandbox.

**Next**

- Run offline tests and CLI smoke checks directly with `.venv/bin/python`; inspect saved result completeness and edge cases.

**Blockers**

- `uv run` is sandbox-blocked at `/Users/tung.vole1/.cache/uv`; the project-local virtual environment provides a non-network fallback.

### 2026-08-24 15:29 - Verification and risk review complete

**Status:** complete

**Completed**

- Ran the full offline test suite successfully: 14 tests passed.
- Verified the CLI, package resources, registry, all real dataset loaders, and limited oracle answer/report flows.
- Identified a measurement inconsistency: LongMemEval includes ingestion in its usage delta, but LoCoMo and BEAM snapshot usage only after ingestion, omitting write-time LLM cost.
- Confirmed the repository has no reproduced full-history or write-time-memory results yet; saved artifacts are partial oracle runs.

**Evidence**

- `.venv/bin/python -m pytest -q` completed with `14 passed in 2.95s`.
- Limited oracle runs produced 20 LoCoMo answers and a perfect lexical report, 3 LongMemEval answers, and 20 BEAM answers under temporary directories.
- Usage snapshots occur before ingestion in `src/adaption_memory/evals/longmemeval.py:52-58`, but after ingestion in `src/adaption_memory/evals/locomo.py:71-84` and `src/adaption_memory/evals/beam.py:97-109`.
- Existing ignored artifacts contain 200/1,986 LoCoMo answers, 10/500 LongMemEval answers, and 400/400 BEAM answers; only LoCoMo has a summary.

**Decisions**

- Prioritize a shared usage-accounting contract before adding a write-time memory system, so benchmark cost comparisons remain valid.
- Treat implementation of narrative/atomic memory and evolutionary state tracking as the next product layer; do not add retrieval abstractions before the write path works end to end.

**Next**

- Fix and test ingestion-cost accounting across all adapters.
- Implement the smallest end-to-end write-time memory system behind the existing protocol and registry.
- Run a small benchmark slice before scaling to full evaluations.

**Blockers**

- None.

### 2026-08-24 15:29 - Task complete

**Status:** complete

**Completed**

- Delivered a structural, runtime, verification, and risk scan of the repository.

**Evidence**

- Source inspection, git diff validation, offline tests, dataset loading, and temporary oracle smoke runs all completed.

**Next**

- None.

## Task: Finish the Qwen3 1.7B evaluation pass

**Status:** complete
**Started:** 2026-08-24 18:09 +07

### Scope

- Run the installed `qwen3:1.7b` through the real adaptive-memory acceptance path.
- Diagnose and fix any remaining correctness gaps exposed by the stronger model.
- Finish with offline regression, live evaluation evidence, and accurate documentation.

### Plan

- Establish the current offline baseline and reproduce the prior LoCoMo temporal slice.
- Inspect extracted memories and answer behavior, then make only evidence-backed fixes.
- Re-run the relevant live comparison and all deterministic verification.

**Blockers**

- None.

### 2026-08-24 19:11 - Temporal failure isolated and corrected

**Status:** in progress

**Completed**

- Reproduced the prior LoCoMo temporal question on `qwen3:1.7b`; the initial full-history-memory answer was 8 May 2023 for gold 7 May 2023 (F1 0.3333).
- Confirmed extraction dropped the relative-time relationship and treated the source-session timestamp as the event date.
- Added write-time relative-date guidance, deterministic `yesterday` resolution, explicit source-session provenance labels, and a high-confidence exact-date answer path.
- Installed and SHA-256-verified `qwen3:4b` for the final quality check.
- Rejected the first exact-date matcher after a full 4B run showed that a later activist-group date could loosely match the support-group question.
- Tightened the shortcut to require an absolute date and at least 80% date-key coverage; added a competing-event regression case.

**Evidence**

- Focused 1.7B live run stores and returns `7 May 2023` in one model call.
- Initial full 4B run returned `last Tuesday` and scored F1 0.0, proving the first shortcut was too permissive.
- Offline suite passes 24 tests; compilation and `git diff --check` pass.
- `ollama list` reports `qwen3:4b` ID `359d7dd4bcda`, size 2.5 GB.

**Next**

- Complete the fresh full 4B rerun, then update documentation and final verification.

**Blockers**

- None.

### 2026-08-24 19:35 - Task complete

**Status:** complete

**Completed**

- Made Qwen3 4B the local default and removed the obsolete corporate-only 1.7B import path.
- Resolved unambiguous `yesterday` event dates across LoCoMo, LongMemEval, BEAM, and ISO source timestamp formats while rejecting negated or unrelated evidence.
- Required explicit same-key supersession references for changed atomic values; implicit and mismatched overwrites are rejected.
- Restricted exact-date answers to active absolute-date records with strong bidirectional event-key matching.
- Added explicit `json-schema`, `json-object`, and `prompt-only` endpoint capability modes.
- Attributed source benchmark tables accurately and documented the local acceptance result without benchmark-wide claims.

**Evidence**

- Fresh official 19-session Qwen3 4B LoCoMo run returned `7 May 2023` for gold `7 May 2023`, F1 1.0.
- Final live usage: 19 calls, 31,151 prompt tokens, and 4,757 completion tokens.
- Final regression: 34 tests passed; compilation, CLI help, and `git diff --check` passed.
- Independent final review found no remaining Critical or Important issues and assessed the work ready to merge.

**Next**

- None.

**Blockers**

- None.

## Task: Add a representative mini signal benchmark tier

**Status:** complete
**Started:** 2026-08-24 22:08 +07

### Scope

- Preserve the existing 22-question mini suite as a very fast smoke test.
- Add a larger deterministic signal tier with repeated coverage per category, multiple source conversations, and substantially fuller contexts.
- Make the CLI, manifests, results, tests, and documentation distinguish smoke, signal, and full runs unambiguously.

### Plan

- Measure the full datasets and define explicit per-benchmark sampling and context-retention targets.
- Implement stratified deterministic selection without changing official benchmark formats or scorers.
- Generate both tiers, validate native loading and oracle scoring, and quantify their size and coverage.
- Run the full regression suite and record the verified handoff.

**Blockers**

- None.

### 2026-08-24 22:24 - Signal sampling design implemented

**Status:** in progress

**Completed**

- Kept the 22-question category-complete smoke tier and added a 90-question signal tier with 30 questions per benchmark.
- Matched LongMemEval and LoCoMo source category proportions subject to a three-example floor; BEAM remains exactly balanced across all ten abilities.
- Selected five full LoCoMo conversations and three full BEAM conversations across their context-size distributions.
- Added six full-context LongMemEval stress cases plus ten spread distractor sessions on the other 24 cases.
- Replaced the ambiguous boolean mini CLI with explicit `--tier smoke|signal` routing and isolated result directories.

**Evidence**

- Signal category total-variation distance from the full distributions is 0.0413 for LongMemEval, 0.0567 for LoCoMo, and 0 for BEAM.
- Signal context retention is 41.2% for LongMemEval and 100% for selected LoCoMo and BEAM conversations.
- Native-format generation succeeds with 30 questions per benchmark; 42 offline tests pass, including byte-identical regeneration.

**Next**

- Update user-facing commands and measured size documentation, then generate workspace artifacts and run oracle pipelines.

**Blockers**

- None.

### 2026-08-24 22:43 - Generated artifacts and oracle pipelines validated

**Status:** in progress

**Completed**

- Generated the final `data/mini/smoke/` and `data/mini/signal/` native artifacts and removed the obsolete flat mini files.
- Broadened LoCoMo source selection across all ten source conversations rather than only conversations containing every category.
- Corrected LoCoMo category-3 oracle hypotheses to match the official scorer's first-semicolon-field rule.
- Ran all three signal-tier oracle pipelines through the real CLI and provenance checks.

**Evidence**

- The first `uv run make-mini` attempt inside the restricted sandbox failed because uv could not open its external cache; the approved cache-enabled rerun succeeded without changing project dependencies.
- Final signal size is 7,990,620 bytes: 90 of 2,886 questions and 2.80% of the primary full-data bytes.
- LoCoMo and LongMemEval oracle results are exactly 1.0 across 30 cases each.
- BEAM oracle is 0.9889: one 1.7B rubric decision scored a present `CSS3` item as zero while its rationale explicitly said the required element was present; event ordering and every other reported type scored 1.0.
- Reusing a signal result directory with `--tier smoke` exits with a dataset-configuration mismatch.

**Next**

- Incorporate independent review feedback, rerun regression checks, and complete the task record.

**Blockers**

- None.

### 2026-08-24 22:45 - Task complete

**Status:** complete

**Completed**

- Enforced at least three LongMemEval abstentions in addition to the six base question-type floors.
- Required earlier-stage configuration and artifacts for judge-only and report-only resumptions.
- Resolved environment-derived OpenAI-compatible judge endpoints before fingerprinting and prevented the global OpenAI key from being forwarded implicitly to custom endpoints.
- Froze the exact 22 smoke selections and added exact full-context and distractor-count regression checks.
- Added SHA-256 and byte-size fingerprints for all four source data files to every generated manifest.
- Documented the signal tier's 836 answer-side calls and lack of a measured wall-time baseline so it is positioned as pre-final rather than a daily loop.

**Evidence**

- Final regression: 44 tests passed; compilation and `git diff --check` pass.
- Final generated signal: 90 questions, 7,990,620 bytes, three abstentions, 41.2% LongMemEval character retention, and full selected LoCoMo/BEAM contexts.
- Report-only resumptions succeed with complete provenance; cross-tier and missing-provenance resumptions are rejected.
- Independent review found no remaining Critical or Important issues and assessed the changes ready to merge.

**Next**

- None.

**Blockers**

- None.

## Task: Establish hosted signal baselines

**Status:** in progress
**Started:** 2026-08-24 22:54 +07

### Scope

- Add the supplied architecture diagram to the repository without altering it.
- Measure the representative signal tier with a fixed hosted-model baseline.
- Start with GPT-5.6 Luna at medium reasoning and escalate only when a controlled comparison shows model weakness.

### Plan

- Verify and inspect the diagram, then place it under `assets/` and reference it from the README.
- Run the signal tier for the full-history baseline before the more expensive adaptive system.
- Keep the answer model fixed at GPT-5.6 Luna medium and use GPT-5.6 Luna high only for judge-scored benchmarks.
- Use a fixed canary subset to compare Terra, and then Sol only if Terra remains materially inadequate.

### 2026-08-24 22:54 - Diagram incorporated and baseline ladder fixed

**Status:** in progress

**Completed**

- Read the diagram as a write-time memory architecture: user sessions flow through an LLM extractor into an append-only store containing narrative and atomic records; new records may supersede old ones; query-time retrieval feeds the LLM answerer.
- Copied `Desktop/diagram2.png` byte-for-byte to `assets/diagram2.png` and linked it from the README architecture section.
- Selected `gpt-5.6-luna` at medium reasoning as the first answer model, with a Luna high judge for LongMemEval and BEAM.
- Defined Terra and Sol as evidence-triggered escalations rather than unconditional full-suite runs.

**Evidence**

- Source and repository copies share SHA-256 `20c2c7e1539fc42782d83e1edd8934323a958063b143d6d8dab25452e1f95987`.
- The PNG is 1541 by 1557 pixels and was visually inspected at original resolution.
- Current official model guidance positions Luna for cost-sensitive high-volume workloads, Terra for balancing intelligence and cost, and Sol for complex professional work.

**Next**

- Run the 90-question signal baseline once a current OpenAI API key is available to the process.

**Blockers**

- `OPENAI_API_KEY`, `ANSWER_API_KEY`, and `JUDGE_API_KEY` are not set in the current environment.

### 2026-08-24 22:56 - Signal run sized and offline verification passed

**Status:** in progress

**Completed**

- Verified the current 90-question signal data and eval CLI without changing the selected cases.
- Sized the full-history signal path at 90 answer calls with about 27.3 million prompt characters, or roughly 6.8 million tokens before provider tokenization.
- Counted 30 LongMemEval judge calls and 77 fixed BEAM rubric calls, plus the output-dependent event-ordering alignment calls.
- Re-ran the complete offline regression suite after adding the diagram reference.

**Evidence**

- Signal answer prompts by benchmark: about 1.54 million estimated tokens for LongMemEval, 0.65 million for LoCoMo, and 4.64 million for BEAM.
- At current input rates, that fixed full-history prompt volume is roughly $1.37 on Luna, $13.66 on Terra, or $27.31 on Sol before completions and judge calls.
- `.venv/bin/python -m pytest -q` passes all 44 tests in 10.66 seconds; `git diff --check` passes.

**Decisions**

- Run Luna across the signal baseline first. If escalation is needed, compare a fixed canary subset on Terra before spending on a full Terra run; only test Sol if Terra remains inadequate.

**Next**

- Start the resumable full-history signal run when a current key is available, then run the adaptive system against the same fixed data and configuration.

### 2026-08-24 23:46 - Signal baseline checkpoint recorded

**Status:** in progress

**Completed**

- Completed all three full-history signal evaluations with GPT-5.6 Luna medium answers and Luna high judges.
- Completed adaptive signal evaluations for LoCoMo and BEAM.
- Stopped the adaptive LongMemEval signal run at 13 of 30 completed answers when the evaluation plan changed to a no-reasoning smoke rerun.
- Recorded the exact scores, answer-side usage, configuration, caveats, and resumable artifact paths in `BASELINE_RESULTS.md`.

**Evidence**

- Full history: LoCoMo F1 0.4466, LongMemEval accuracy 0.9000, and BEAM overall 0.6595.
- Adaptive memory: LoCoMo F1 0.5133 and BEAM overall 0.5914.
- Adaptive LongMemEval has 13 complete answers and no judge or summary artifact; no partial score is reported.
- BEAM produced all 77 fixed rubric decisions with zero unparseable judge outputs in the full-history run.

**Decisions**

- Preserve the medium-reasoning signal artifacts unchanged.
- Run fresh smoke outputs with `reasoning_effort="none"` for both Luna answer and judge calls.

**Next**

- Run the smoke tier for full-history and adaptive systems, then add the measured no-reasoning comparison to `BASELINE_RESULTS.md`.

### 2026-08-24 23:49 - Full-history no-reasoning smoke complete

**Status:** complete

**Completed**

- Narrowed the requested smoke scope to the full-history system only; no adaptive smoke job was launched.
- Ran all three smoke benchmarks with `gpt-5.6-luna` and explicit `reasoning_effort="none"` for answers and judges.
- Recorded scores, answer-side usage, output caps, judge validity, and interpretation limits in `BASELINE_RESULTS.md`.

**Evidence**

- LoCoMo: F1 0.3010 across 5 questions.
- LongMemEval: accuracy 0.8571 across 7 questions; only the temporal case failed.
- BEAM: overall 0.6533 across 10 questions; all 27 rubric judge responses parsed successfully.
- Answer-side total: 22 calls, 299,490 input tokens, and 2,227 output tokens.
- Result provenance records `reasoning_effort="none"` for every answer and judge configuration.

**Next**

- Use these full-history smoke numbers as the fast baseline before deciding whether to run the adaptive smoke comparison.

**Blockers**

- None.

### 2026-08-25 00:14 - Full-history no-reasoning signal complete

**Status:** complete

**Completed**

- Reran all three 30-question signal benchmarks with the full-history system and `gpt-5.6-luna` using explicit `reasoning_effort="none"` for answers and judges.
- Kept the earlier medium/high signal artifacts unchanged in separate result directories.
- Compared the two end-to-end configurations and inspected every LongMemEval label that changed.
- Added the exact scores, usage, judge validity, deltas, and interpretation caveat to `BASELINE_RESULTS.md`.

**Evidence**

- LoCoMo: F1 0.4607, versus 0.4466 for medium reasoning.
- LongMemEval: accuracy 0.7667, versus 0.9000 for medium answers with a high-reasoning judge.
- BEAM: overall 0.7464, versus 0.6595 for medium answers with a high-reasoning judge.
- Answer-side total: 90 calls, 5,892,829 input tokens, and 6,609 output tokens.
- All 77 BEAM rubric responses parsed successfully; LongMemEval judges returned only exact yes/no labels.
- The four LongMemEval regressions contained substantive answer omissions, so the loss is not attributable only to the changed judge reasoning.

**Decisions**

- Keep `none` as the fast iteration path.
- Use Luna `none` answers with Luna `high` judges for the final run; the user explicitly accepted the recorded LongMemEval answer-quality tradeoff.

**Next**

- None.

**Blockers**

- None.

## Task: Execute the overnight write-time memory plan

**Status:** in progress
**Started:** 2026-08-25 00:20 +07

### 2026-08-25 - Local extractor changed to Qwen3-4B

**Status:** in progress

**Completed**

- Verified `.env.local` contains a non-empty OpenAI key without printing it; the file will remain untouched.
- Verified Ollama serves Qwen3 0.6B, 1.7B, and 4B, and the Mac has 18 GiB RAM with 68% system-wide memory free at inspection time.
- Updated `OVERNIGHT_PLAN.md` so both local extractor arms consistently use the installed Qwen3-4B model, per the user's direction.

**Decisions**

- Do not import or download Qwen3-8B; use `qwen3:4b` for the zero-shot and few-shot local arms.
- Preserve Luna answerer and judge settings, tier ceiling, checkpointing, and spend limits unchanged.

**Next**

- Complete the one-session preflight trace, then execute smoke gates and eligible signal runs.

### 2026-08-25 - Phase 0 preflight passed

**Status:** in progress

**Completed**

- Installed and verified local BGE-small dense retrieval plus BM25 without changing the extractor model.
- Implemented the modular append-only memory pipeline with per-call spend tracking and hash-addressed extraction, retrieval, answer, and judge checkpoints.
- Carved and persisted the stratified signal split before any optimization: LongMemEval 18/12, LoCoMo 18/12, and BEAM 20/10 dev/holdout.
- Materialized existing Luna-none signal predictions under `results/baselines/` and retained the plan's medium/high numbers as explicitly non-comparable legacy context.
- Completed the required one-session Luna-target trace: 3 records stored, 3 retrieved, non-empty answer, and completed Luna-none judge call.
- Fixed and revalidated macOS ONNX single-thread lifecycle behavior and checkpoint replay ordering.

**Evidence**

- `results/preflight.json`, `results/preflight_trace/`, `results/spend.json`, `results/splits/signal.json`, and `results/baselines/index.json`.
- 49 offline tests pass.
- Preflight spend after the clean rerun remains far below the $40 cap; the tracker updates atomically after each API call.

**Next**

- Run all three revised arms on smoke and apply the 95% schema-validity gate before signal.

**Blockers**

- None.

### 2026-08-25 - Phase 3 smoke gates passed

**Status:** in progress

**Completed**

- Completed 44 session extractions and 22 answer/judge traces for each of the three revised extractor arms.
- All arms achieved 100% initial JSON schema validity and completed the pipeline, clearing the 95% promotion gate.
- Recorded direct extraction, supersession, latency, token, and cost data beside per-question outputs.

**Evidence**

- Qwen3-4B zero-shot macro judge accuracy: 0.2619; 63 rejected records.
- Qwen3-4B few-shot macro judge accuracy: 0.2810; 31 rejected records.
- Luna-target macro judge accuracy: 0.3619; 0 rejected records.
- Results are under `results/overnight/smoke/<arm>/F1/`.

**Decisions**

- Promote all three arms to signal because the specified gate is schema validity and pipeline completion, not smoke accuracy.
- Run Luna-target signal first to establish the comparison ceiling, then the more decision-relevant Qwen few-shot arm, then zero-shot.

**Next**

- Execute checkpointed signal runs without touching the full tier.

### Scope

- Maintain `OVERNIGHT_PLAN.md` as the canonical, editable execution plan; user
  revisions supersede its originally pasted wording.
- Implement the checkpointed write-time memory pipeline and extractor arms.
- Run smoke gates and signal-only comparisons within the $40 API cap.
- Prepare the SFT dataset artifacts, rescore the LoCoMo baseline, and always produce the HTML report and morning summary.

### Plan

- Complete Phase 0 preflight and reconcile the plan's expected paths with the current repository without touching full evaluation data.
- Implement and verify the append-only store, extraction, hybrid retrieval, answering, judging, checkpointing, and spend accounting.
- Add the five shared few-shot examples and run eligible smoke/signal arms.
- Execute format ablation and bounded dev-only optimization when budget and completed prerequisites permit.
- Generate SFT preparation artifacts, baseline rescore, report data, HTML report, and `results/MORNING.md`.

### 2026-08-25 00:20 - Plan stored verbatim

**Status:** in progress

**Completed**

- Read the complete supplied 305-line plan and copied it unchanged to `OVERNIGHT_PLAN.md`.
- Established `PROGRESS.md` as the durable execution log.

**Evidence**

- The attachment and repository copy share SHA-256 `dd9a5bc0045a95065a2e8c2dd92a2a711d1eeaf1e8186c5bcc5ab201132b89f0`.

**Next**

- Run the Phase 0 environment, model, dataset, and baseline audit before implementing later phases.

**Blockers**

- `.env.local` currently has an empty `OPENAI_API_KEY`; hosted Luna phases will remain unavailable until it is refilled.

### 2026-08-25 - Plan aligned with repository engineering rules

**Status:** in progress

**Completed**

- Converted `OVERNIGHT_PLAN.md` from a verbatim snapshot into the canonical,
  editable execution plan at the user's request.
- Added binding rules for canonical-only interfaces, gated end-to-end layers,
  minimal implementation scope, modular boundaries, dependency reuse,
  durable architecture, established-pattern review, and fail-closed artifact
  provenance.
- Made `src/adaption_memory/memory/` the sole package path and Qwen3-4B the
  sole supported local-extractor identity; no compatibility aliases or
  migrations are planned.
- Added `results/design_notes.md` as a Phase 0 design gate so external patterns
  are recorded before novel architecture is introduced.

**Decisions**

- Existing result files remain immutable experiment evidence, not a reason to
  preserve obsolete runtime paths.
- F2-F4 remain bounded experiment variants and are added only on top of the
  verified F1 pipeline.

**Next**

- Reconcile the implementation with the revised plan, create the design-note
  artifact, and continue only the already-gated signal work.

**Blockers**

- None.

### 2026-08-25 - F4 selected and one-lever optimization defined

**Status:** in progress

**Completed**

- Completed the four-way Luna-target signal-dev representation ablation.
- Selected F4 single-type at 0.5074 macro judge accuracy; F1 scored 0.4852,
  F2 scored 0.3796, and F3 scored 0.4148.
- Classified all 28 wrong F4 dev answers from stored/retrieved evidence: 22
  facts absent from memory, 3 stored but not retrieved, and 3 retrieved but
  answered incorrectly.
- Added one bounded `coverage` revision of the F4 extractor prompt and a unique
  canonical output path; no other pipeline lever changes.
- Started the one-time starting-config F4 holdout evaluation without inspecting
  it for optimization decisions.

**Decisions**

- Freeze F4 for optimization and SFT preparation. The dev result is evidence
  that a single record type currently earns the best accuracy with less schema
  complexity.
- Iteration 1 targets extraction coverage because missing writes account for
  78.6% of dev errors.

**Evidence**

- `results/optimization_log.md`
- `results/overnight/signal/luna-target/F1-dev/summary.json`
- `results/overnight/signal/luna-target/F2-dev/summary.json`
- `results/overnight/signal/luna-target/F3-dev/summary.json`
- `results/overnight/signal/luna-target/F4-dev/summary.json`
- 11 targeted memory tests pass after adding the prompt revision.

**Next**

- Finish the starting holdout, run F4 coverage on dev, apply the 1-point stop
  rule, then score the selected final configuration on holdout exactly once.

**Blockers**

- None.

### 2026-08-25 - Phase 3.5 dev optimization stopped after two iterations

**Status:** in progress

**Completed**

- Iteration 1 changed only F4 extractor coverage wording and improved dev macro
  accuracy from 0.5074 to 0.5982 (+9.08 points).
- Re-bucketed the 23 remaining dev errors: 19 facts not extracted, 1 stored but
  not retrieved, and 3 retrieved but answered incorrectly.
- Traced 185 validator failures to non-verbatim values, often redundant
  session-date metadata, and ran one prompt-only validation-discipline change.
- Iteration 2 reduced rejected records dramatically but scored 0.5648, down
  3.34 points from coverage; the mandatory sub-1-point stop rule fired.

**Decisions**

- Select F4 `coverage` as the final configuration because accuracy is primary.
- Preserve the cleaner `validated` result as an ablation, not a supported final
  path. Invalid coverage sessions will be excluded from SFT data.
- Score `F4-holdout-coverage` once as the second and last holdout run.

**Evidence**

- `results/optimization_log.md`
- `results/overnight/signal/luna-target/F4-dev-coverage/summary.json`
- `results/overnight/signal/luna-target/F4-dev-validated/summary.json`

**Next**

- Complete the final holdout, generate the filtered F4 coverage SFT artifacts,
  and finish the two local signal arms.

**Blockers**

- None.

### 2026-08-25 - Revised plan and implementation verification

**Status:** in progress

**Completed**

- Rechecked the maintained plan and implementation against the repository's
  binding engineering rules: one canonical Qwen3-4B path, no result fallback
  or migration layer, working-layer gates, modular memory concerns, and
  explicit use of established dependencies.
- Confirmed the canonical design decisions and adopted external patterns are
  recorded in `results/design_notes.md`.
- Ran the complete offline regression suite after the plan and prompt changes.

**Evidence**

- 56 tests pass.
- Python compilation and `git diff --check` pass.
- The final F4 coverage holdout and the last Qwen zero-shot signal arm remain
  checkpointed and in progress.

**Next**

- Complete those two signal runs, record the final holdout generalization
  check, regenerate the report, and perform browser QA.

**Blockers**

- None.

### 2026-08-25 - Final optimized holdout completed

**Status:** in progress

**Completed**

- Completed the second and final permitted holdout evaluation with the selected
  F4 `coverage` extractor prompt.
- Compared it with the frozen starting F4 holdout and applied the plan's
  half-the-dev-gain overfitting rule.
- Performed an early desktop browser QA of the generated report: the embedded
  supplied diagram and all four charts load, no horizontal overflow is present,
  and the only console error was a missing favicon. Added an inline empty icon
  to make the final pass clean.

**Evidence**

- Final holdout macro judge accuracy: 0.4889 versus 0.4389 base, a +5.00-point
  improvement.
- Per benchmark: LongMemEval 0.8333, LoCoMo 0.3333, BEAM 0.3000.
- Half of the +9.08-point dev gain is +4.54 points; the +5.00 holdout gain
  clears the overfitting threshold by 0.46 points.
- `outputs/memory-report-desktop.png` records the preliminary visual pass.

**Next**

- Finish Qwen3-4B zero-shot signal, consolidate the three benchmark summaries,
  regenerate final report artifacts, and repeat browser QA.

**Blockers**

- None.

### 2026-08-25 - Checkpoint usage replay made exact

**Status:** in progress

**Completed**

- Measured Qwen's serialized Ollama bottleneck and used the documented
  three-request server setting for the three independent benchmark workers;
  the model, 4K per-worker context, prompts, temperature, and decoding stayed
  fixed.
- Removed the launch environment override immediately after the running server
  inherited it, so the user's next Ollama restart returns to the default.
- Found that resumed LLM checkpoints restored outputs but not their cumulative
  token, latency, and cost deltas. Added usage persistence and one-time replay
  for extractor, answerer, and judge checkpoints without repeating spend
  callbacks.
- Preserved the usage-incomplete zero-shot attempt under an explicitly excluded
  path and restarted the canonical F1 zero-shot signal run cleanly.

**Evidence**

- Ollama worker: `-c 12288 -np 3`, which is three 4K slots, 100% GPU.
- 14 focused memory tests pass, including fresh-process extraction, answer, and
  judge usage replay without duplicate model calls.
- Partial artifact:
  `results/overnight/signal/qwen3-4b-zeroshot/F1-interrupted-usage-incomplete`.

**Next**

- Complete the clean canonical zero-shot signal run, then perform final
  consolidation, full regression tests, report generation, and browser QA.

**Blockers**

- None.

### 2026-08-25 - Local inference bottleneck isolated

**Status:** in progress

**Completed**

- Stopped the two Qwen signal workers at durable checkpoints for isolated
  profiling.
- Confirmed Qwen3-4B is fully GPU-resident on the M3 Pro and a short raw
  generation runs at about 39 tokens/second; the inference engine is healthy.
- Identified harness over-generation as the bottleneck: 378-404 average
  completion tokens per call, up to 900 per call, 1.09-1.50 calls per session,
  and two benchmark processes queueing through one Ollama execution slot.
- Changed deterministic record-validation failures to local drop-and-log while
  preserving valid sibling records. A model repair is now reserved for a
  malformed top-level response.
- Bounded the local structured extraction contract to six concise records and
  512 completion tokens.

**Evidence**

- Historical zero-shot checkpoints show average call latency around 40-42
  seconds under the queued two-worker run.
- LongMemEval had 30 model repairs in 62 completed sessions; LoCoMo had 8 in
  89; BEAM had 13 in 26. Those full second generations were triggered by
  record-level validation failures.

**Next**

- Run focused tests, profile the optimized path on saved representative input,
  and resume only after the measured latency improves materially.

**Blockers**

- None.

### 2026-08-25 - Local compact-v2 inference path validated

**Status:** in progress

**Completed**

- Replaced verbose local JSON field names with the compact `r/t/c/e/s`
  transport and normalized it to canonical records at the extractor boundary.
- Reduced local candidate context from ten full records to six compact
  `i/t/c/d` rows and bounded entities to three names per record.
- Added deterministic partial-tail salvage: every complete record before a
  token-truncated final item is validated and preserved without another model
  call.
- Versioned the local contract as `compact-v2`; its checkpoints cannot collide
  with previous Qwen outputs. Quarantined both prior local attempts under
  explicitly excluded result names.

**Evidence**

- Exact first LoCoMo signal session: 52.8 seconds before versus 10.2 seconds
  after the bounded compact path, a 5.2x speedup.
- Exact difficult smoke session 7: 37.6 seconds, two calls, zero records before
  tail salvage versus 15.8 seconds, one call, six valid records after compact
  candidates and tail salvage.
- Focused memory suite: 17 tests pass, including compact bounds, local drop and
  log, usage replay, and truncated-tail preservation.

**Next**

- Run the clean optimized smoke extraction suite, compare quality, then resume
  signal only if the smoke gate passes.

**Blockers**

- Hosted Luna answer/judge requests currently reset during TLS setup; this does
  not affect local Qwen extraction profiling or checkpoints.

## Task: Lock the reproduction goal and build the extraction fast loop

**Status:** in progress
**Started:** 2026-08-25

### Scope

- Prevent a repeat of the F4 drift: record the F1-canonical invariant and a
  minimum-effect decision rule where every session loads it.
- Tighten the iteration loop: score write-time extraction directly, without
  answering or judging.

### 2026-08-25 - Invariant recorded and fastloop implemented

**Status:** in progress

**Completed**

- Added a "Reproduction invariant" section to `CLAUDE.md`: F1
  (narrative + atomic + supersession) is canonical; single-type formats are
  ablations that must never be promoted; architecture changes require >=3
  macro points on signal-dev confirmed on holdout.
- Added `overnight-memory fastloop`: smoke-tier extraction plus local
  retrieval only, guarded by a fail-closed answering stub so no hosted
  answer or judge call is possible. Reports store recall, retrieved recall,
  supersession, rejections, and writes per-question `misses.jsonl` bucketed
  as `fact_not_extracted` vs `stored_not_retrieved`.
- Run directories are hashed over the resolved prompt, few-shot messages,
  and local inference bounds, so prompt edits start clean and unchanged
  reruns resume from checkpoints.

**Evidence**

- 64 tests pass, including fastloop offline run, checkpoint resume with zero
  model calls, config-hash sensitivity, and the fail-closed answer stub.

**Next**

- Measure a live qwen3-4b-zeroshot fastloop iteration on one LoCoMo smoke
  conversation and record the wall time.

**Blockers**

- None.

### 2026-08-25 - llama-server migration measured and rejected

**Status:** complete

**Completed**

- Attempted the planned local-inference optimization: llama.cpp llama-server
  with prompt-prefix caching and speculative decoding, serving the existing
  Ollama GGUF blobs with a pinned non-thinking chat template.
- Measured a controlled A/B on the same fastloop slice and rejected the
  migration: prefix caching works but is dwarfed by slower
  grammar-constrained decoding, and speculative decoding either crashes
  (draft model) or gains nothing (ngram).
- Reverted the endpoint wiring; Ollama remains the canonical local endpoint.
  Recorded full numbers in `results/FAILURES.md`.

**Evidence**

- Ollama 98.9s vs llama-server 118.6s vs llama-server+flash-attn 108.4s on
  the identical one-conversation LoCoMo fastloop; store_recall identical
  (0.0), rejections 7 vs 7 vs 14.
- Per-call decomposition from server timings: prefill 2-4s at 410-590 tok/s
  with the 410-token static prefix reused; decode ~360 tokens at ~28 ms/token
  under json_schema grammar.
- 64 tests pass after revert; Ollama endpoint healthy.

**Decisions**

- Do not change serving stacks for speed on this machine: decode token count
  dominates, so future latency work targets output size (record shaping) or
  a smaller model (qwen3:1.7b), both gated by fastloop recall.

**Next**

- Read the fastloop rejection log and decide the recall fix (prompt-only
  constraints vs caps), which remains the open question.

**Blockers**

- None.


### 2026-08-25 - Fastloop conversations parallelized

**Status:** complete

**Completed**

- Ran fastloop conversations on a shared thread pool across benchmarks
  (`--workers`, default 3), keeping sessions within a conversation strictly
  sequential. Checkpoints moved to per-conversation directories so threads
  never share files; the config hash schema was bumped so old layouts land in
  separate directories.
- Verified the running Ollama server serves concurrent requests (3 calls in
  4.4 s vs 11.4 s serial).
- Ran the first full-suite fastloop: all nine smoke conversations, 44
  extraction calls, in 513.5 s wall vs a 1378.9 s sequential estimate (2.7x).

**Evidence**

- 65 tests pass, including a concurrent multi-conversation fastloop test
  asserting isolated per-conversation stores and checkpoints.
- Full-suite qwen3-4b-zeroshot recall: LongMemEval 0.14 (1/7), LoCoMo 0.0
  (0/4), BEAM 0.0 (0/9); 19 of 22 scorable facts never extracted; 50 records
  rejected by validation. Every miss is fact_not_extracted; retrieval loses
  nothing that was stored.

**Decisions**

- Iterate prompts on the ~100 s single-conversation form; confirm on the
  ~8.5 min full suite before any signal run.

**Next**

- Read the rejection log and fix extraction recall (prompt-only constraints
  vs caps) against the full-suite miss list.

**Blockers**

- None.
