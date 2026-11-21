## v0.3.0 — 2026-11-21

- Added deterministic second-pass dataset curation with whitespace/Unicode-normalized duplicate removal and SimHash/LSH + 5-gram Jaccard **heuristic** near-duplicate filtering before train/validation split.
- Added conservative sensitive-data checks for email, credentials/private keys, Japanese mobile phone patterns, and Luhn-valid card numbers; manifest records only rejection categories/counts.
- Added offline CPU verification script and least-privilege GitHub Actions CPU workflow (no external LLM API or expensive training).
- Added regression tests for cross-split duplicate leakage, source order determinism, privacy screening and CI safeguards.
- Explicitly documented limitations: no comprehensive PII screening, no guaranteed semantic dedup, no training rights verification, CI run status unverified until hosted execution.

# Changelog

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