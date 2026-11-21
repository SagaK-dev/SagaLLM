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