## v0.15.0 — original Java 110-family corpus and full-coverage CPU training (2026-11-22)

- Expand the Java instruction/code curriculum to **17,536 original parameterized examples**: 17,080 train, 228 validation, 228 test; family-disjoint splits across 110 task types, 40 new types.
- Introduce strict canonical regeneration, source SHA-256 checks and exact duplicate rejection. No arbitrary third-party code is ingested, and no external license is asserted.
- Add balanced-by-family shuffle to ensure early checkpoints cover varied algorithms, and explicit counters for examples/families seen by best and final checkpoints.
- Add deterministic reference Java compilation and independent output oracle tests, local CPU training scripts and experiment evidence. Model-generated Java runtime is **NOT_RUN**.
- Add offline, **unreviewed** Java-repository source inventory/export that skips secrets and marks every external dataset `PENDING` until a human reviews rights. No third-party Java code was actually trained.
- Two real full-pass CPU experiments: each **17,080/17,080** training examples used; validation-selected checkpoint step **1,000**, held-out compiled **4/10** new vs **6/10** continuation, exact code match **0/10** both. The selected best checkpoints do not contain all 17,080 gradients; final checkpoints do.
- Keep training data/weights as separate downloadable artifacts, not Git blobs. Still not a proven Java coding assistant.

## v0.14.0 — expanded original Java corpus and full-pass CPU SFT (2026-11-22)

- Generate **5,436 original parameterized Java 17 code/instruction pairs** across 70 task families: 5,220 train / 108 validation / 108 held-out test, segregated by entire family. Japanese and English task requests. No third-party repository or pretrained model downloaded.
- New deterministic corpus generator and strict audit: source and file hashes, exact regeneration, no duplicate Java source, family-disjoint split and fail-closed license statement (`NOT_LEGALLY_ATTESTED`).
- Replace the prototype ByteBPE encoder's repeated full scans with a rank-ordered heap and correct linked-node invalidation, preserving the old tokenizer ID outputs; randomized old/new parity regression tests added.
- Train a random-initialized ~1.4M-parameter native model on an offline CPU, reporting the **distinct training examples actually used**, optimizer steps, validation-selected checkpoint and held-out compile-only behavior. Keep all generated weights/corpora outside Git.
- Data volume is not algorithmic variety: 5,436 variants belong to just 70 parameterized families. Java behavior, real-world code generation skill and license approval remain unverified.

## v0.13.0 — validation-selected Java SFT and guarded OCI behavior harness (2026-11-22)

- Use validation-only early stopping and retain both selected best and last-step
  checkpoints; do not access held-out test loss before selecting the model.
- Add original 15-case Python behavioral oracles for Java test task families
  (Fibonacci, primality and palindrome). Verify reference output against
  15 trusted project-authored Java source programs on the local JDK.
- Add an explicit-opt-in, local-image-only Docker/Podman behavioral evaluator
  with offline containers, read-only source mount, restricted capabilities,
  CPU/memory/process/time limits and bounded capture. No OCI engine is
  installed in the development environment: generated Java behavior NOT_RUN.
- Actual CPU re-training with unchanged 350-example dataset: from 1,500
  requested steps, early stop at 300, best step 150; test loss **3.7609**
  vs previous **6.0101**, but compile **8/15** vs previous **10/15** and
  exact matches still **0/15**. Improvement in loss is not evidence of
  reliable Java coding.
- Add regression tests, opt-in behavioral evaluation CLI, docs, and experiment
  provenance. No third-party Java data imported, no 1B/3B pretraining.

## v0.12.0 — original Java training dataset and CPU SFT (2026-11-22)

- Add 350 **new original** Java instruction/code examples (320 train, 15 validation,
  15 test), split by disjoint algorithm families, in Japanese and English.
- Add deterministic SHA-256 dataset audit, content and provenance checks, and
  Java compile tests; no third-party code or pretrained model weights used.
- Add bounded local `scripts/train_java_original.py` for genuine random-base
  CPU supervised fine-tuning, held-out loss, and compile-only Java scoring.
- Actual CPU experiment: 774,240 parameters, 1,500 steps, train loss
  6.2038→0.2634, held-out test loss 6.1769→6.0101, 10/15 generated Java
  compiled but 0/15 exact matches. Model can emit *compilable but incorrect*
  Java and exhibits overfitting; runtime program behavior NOT_RUN.
- No Java source downloaded, no cloud GPU, no external API, no 1B/3B model
  trained. Keep generated corpora, tokenizer, and model weights outside Git.

## v0.11.0 — 2026-11-22

- Added original, task-family-separated Java synthetic SFT fixture (`java dataset-build/verify`) with deterministic SHA-256 manifests and strict integrity checks. No external code corpus or license approval is implied.
- Added held-out **compile-only** Java benchmark (`java evaluate`) against an explicitly supplied trusted local SagaLLM checkpoint. Never runs generated Java. Tracks compilability, exact reference equality, and `runtime_behavior=NOT_RUN` separately.
- Added `scripts/java_sft_smoke.py`: real small CPU random-weight model, 2 SFT optimizer steps and held-out loss measurements; no strong Java generation ability is claimed.
- Added regression tests, usage docs and version bump; no 1B/3B training, GPU work, hosted API or paid services.

## v0.10.0 — 2026-11-21

- Add opt-in, **local SagaLLM checkpoint** Java revision backend (`java ai-revise`, `java ai-chat`); generation always comes from a randomly initialized/locally trained SagaTransformer checkpoint, never a hosted or pretrained model fallback.
- Limit weight loading, token context and generation budgets; require tokenizer provenance, use CPU by default and reject malformed model output.
- Compile generated candidates in a temporary directory using `javac -proc:none`; feed bounded compiler diagnostics back to the same model for up to three proposals; never execute untrusted Java automatically.
- Preview unified diffs; interactive mode applies the **exact compiled candidate** only on explicit `APPLY`, with source SHA-256 integrity guards. Model-edited files are protected from later template regeneration.
- Add `java toy-sft` to export five original synthetic Java instruction-response pairs using the existing SFT wire format. This is a pipeline smoke dataset, not a useful standalone Java training corpus.
- Add test coverage for proposals, refusals, retry, JDK absence, source tampering, human approval, toy SFT and an actual tiny checkpoint load/SFT step.
- Real-world Java generation quality, 1B/3B training, real GPU and licensed Java corpus remain NOT_RUN/UNVERIFIED.

## v0.9.0 — 2026-11-21

- Add local interactive Java starter workflow (`sagallm java`) for nine explicit Japanese/English task templates, with no LLM API and no pretrained code model.
- Support conversation-style reviewed modifications (class name, comments, greeting text and FizzBuzz upper bound) with diffs and deliberate `--apply`.
- Add optional JDK compilation with annotation processors disabled; explicit untrusted-code warnings and `--trust-local-code` opt-in for executing the generated Java.
- Prevent overwrite of project directories and detect out-of-band source edits through SHA-256.
- Add Java source/runtime regression tests and CI JDK 21 setup. Arbitrary vibe coding remains unsupported until a capable local model and substantial code training are available.

## v0.8.0 — 2026-11-21

- Added opt-in `sagallm hardware inspect`, `smoke`, and `distributed-smoke` commands, with safe, small synthetic tests and exclusive-create JSON evidence.
- Added CUDA/MPS inventory (driver/VRAM where available), BF16-capability gates, CPU checkpoint roundtrip, and two-process local CPU/Gloo collective readiness.
- Added hardware regression tests and a physical validation guide; never assert `PASS_PHYSICAL` or trained 1B/3B.
- All GPU and NCCL/multi-GPU tests are marked `NOT_RUN` in CPU-only development.

# v0.7.0 — Offline MMLU reference-style CSV adapter

- Added `sagallm mmlu inventory`, `mmlu verify` and `mmlu evaluate` commands. Local dev/test CSVs are verified using a pending-by-default human-reviewed SHA-256 inventory.
- Added a reference-style plain-text, 0–5-shot MMLU prompt builder based on Hendrycks et al. The default mode checks stable single-token answer-label scoring; an explicitly noncomparable suffix-likelihood mode is available.
- Added context-aware shot reduction, round-robin subject sampling, per-subject accuracy, exact test and few-shot dev contamination screening, checkpoint/data provenance checks, and report metadata. Official and certified benchmark claims are explicitly false.
- Added original synthetic fixtures and CPU regression tests; no official MMLU data downloaded.

# Changelog

## 0.6.0 (2026-11-21)

- Add human-attested, purpose-restricted dataset review cards with SHA-256 verification and offline `data verify-license` CLI; cards are not proof of legal usage rights.
- Integrate optional review-card verification into `data prepare`, `data shard` and `benchmark` without breaking legacy CLI usage.
- Fix MMLU-style training contamination checks to scan the original question text, rather than only the generated zero-shot evaluation prompt.
- Report subject-level and macro accuracy, Wilson interval and a stable, clearly non-official evaluation protocol ID.
- Add regression, card validation and CLI tests, status documentation and template card.
- No external datasets, paid services, GPU training, or large-model weights are used.


## 0.5.0 — Local Hugging Face-compatible model export

- Export trusted SagaLLM checkpoints to CPU safetensors and standalone custom Transformers architecture/configuration files.
- Preserve tied embeddings without duplicating shared storage, and include immutable SHA-256 provenance for source weights and tokenizer.
- Map model layer/head parameters to standard Transformers config attributes; use no-cache greedy HF generation by default until interoperable Cache support is verified.
- Emit HF native `tokenizer.json` and AutoTokenizer metadata for native FastBPE; label legacy Python BPE as incompatible with AutoTokenizer rather than silently changing its IDs.
- Protect against accidental directory overwrite, tokenizer mismatch, and large-model allocation; export stays local and never uploads weights.
- GitHub Actions run `44904787682` passed 52 tests; added a second custom-model loading regression for the legacy tokenizer backend and report verified-versus-unverified tokenizer provenance.
- Add optional Hugging Face loading/forward/greedy tests in CI and documentation; no official benchmark or large-model training.

## 0.4.0 — Offline evaluation and optional fast tokenizer

- Added optional Hugging Face Tokenizers-based Rust ByteLevel BPE training from a document iterator (`--backend hf`), with its own one-file tokenizer format and backward-compatible loader for the v0.3 Python BPE format.
- Added a local multiple-choice continuation log-likelihood benchmark harness and MMLU-style four-choice JSONL adapter. Reports include SHA-256 provenance and explicitly **do not** claim official benchmark comparability.
- Added optional contamination screening against immutable prepared training document shards, including their SHA-256 validation and checkpoint corpus matching. Missing corpus proofs are marked `NOT_CHECKED` / `CHECKPOINT_CORPUS_NOT_RECORDED`.
- Added tests, toy benchmark input, and dedicated documentation. GitHub-hosted CPU CI passed 48/48 tests, including the native tokenizer backend (run 44903131319).

## v0.3.0 — 2026-11-21

- Added deterministic second-pass dataset curation with whitespace/Unicode-normalized duplicate removal and SimHash/LSH + 5-gram Jaccard **heuristic** near-duplicate filtering before train/validation split.
- Added conservative sensitive-data checks for email, credentials/private keys, Japanese mobile phone patterns, and Luhn-valid card numbers; manifest records only rejection categories/counts.
- Added offline CPU verification script and least-privilege GitHub Actions CPU workflow (no external LLM API or expensive training). Test extras install both httpx and the newer Starlette TestClient dependency httpx2 on Python 3.11+.
- Added regression tests for cross-split duplicate leakage, source order determinism, privacy screening and CI safeguards.
- Explicitly documented limitations: no comprehensive PII screening, no guaranteed semantic dedup, no training rights verification, CI run status unverified until hosted execution.


## 0.2.0 — 2026-11-21

- Added SQLite-backed exact duplicate removal and deterministic document-level train/validation splits.
- Added JSONL and little-endian uint32 token shards with SHA-256 integrity manifests.
- Added on-disk random batch sampling, validation during pretraining and sharded evaluation.
- Added fixed-schedule, hyperparameter-checked checkpoint resumption and CPU bitwise-repeatability test.
- Added tokenizer/checkpoint hash validation and generation vocabulary restriction.
- Added basic in-process HTTP request rate/concurrency limits, bounded message lengths and cooperative generation deadlines.
- Added unit tests, end-to-end CPU smoke checks and technical documentation for the new data pipeline.

**Important:** The changes are experimental. They do not add a trained 1B/3B model or a production inference service. No source license has been chosen by the repository owner.

## 0.1.0 — 2026-11-21

- Initial native PyTorch decoder-only Transformer with RoPE, RMSNorm, GQA, SwiGLU and KV cache.
- Small offline BPE tokenizer, single-device pretraining/checkpoints, single-turn SFT, simple evaluation and local inference API.