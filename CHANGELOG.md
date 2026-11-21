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