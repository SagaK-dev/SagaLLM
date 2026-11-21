# Implementation status

| Component | Status | Notes |
| --- | --- | --- |
| Transformer / RoPE / GQA / RMSNorm / SwiGLU | IMPLEMENTED | Native PyTorch, CPU test needed |
| Configs tiny / micro / pilot / 1b / 3b | IMPLEMENTED | 1B/3B never loaded in tests |
| Byte-level BPE training | IMPLEMENTED | Python educational backend + optional HF Tokenizers backend; HF backend passed GitHub-hosted CPU verification (v0.4) |
| Single-device pretraining | IMPLEMENTED | CPU / optional CUDA/MPS unverified |
| Atomic checkpoint / resume | IMPLEMENTED | Fixed schedule and hyperparameter checks; bitwise CPU interrupted/uninterrupted test passes |
| CLI generation / KV cache | IMPLEMENTED | No production decoding server |
| Disk-backed dataset shards | PARTIAL | SQLite exact/normalized/heuristic near dedup, JSONL splits, uint32 shards; only basic privacy screening |
| Basic next-token held-out validation | IMPLEMENTED | Model loss, perplexity, accuracy; rejects training split by default |
| GitHub Actions CPU CI | IMPLEMENTED / HOSTED_VERIFIED_V0.5 | v0.5 Python 3.11 CPU workflow passed: 53 tests after the final test update; v0.6 hosted CI pending |
| Standardized benchmarks | PARTIAL | Offline multiple-choice/MMLU-style adapter implemented; official datasets, splits and published scoring protocols NOT_RUN |
| Single-turn SFT | IMPLEMENTED | Assistant-response masking; local JSONL |
| DPO / scalable SFT | NOT_IMPLEMENTED | Phase 4 |
| Local inference HTTP API | PARTIAL | Three endpoints, optional bearer key |
| API rate/concurrency limits | IMPLEMENTED | In-process limiter and cooperative deadline; hard kill, TLS, production rate limiting not implemented |
| HTTP streaming / hard kernel timeouts | NOT_IMPLEMENTED | Proxy recommended for external serving |
| HF Transformers export | IMPLEMENTED / HOSTED_VERIFIED_V0.5 | Local custom-model safetensors export, fast tokenizer AutoTokenizer, roundtrip tests; does not upload |
| Distributed learning / FlashAttention integration | NOT_IMPLEMENTED | Phase 6 |
| Real 1B/3B training | NOT_RUN | No hardware or licensed corpus specified |

## v0.6 follow-up (2026-11-21)

- Added optional local SHA-256-pinned, purpose-restricted human dataset-review cards for data prepare/shard and benchmark scoring; auto-verification of actual legal rights remains impossible.
- MMLU-style raw-question overlap audit fixed; subject and macro scores and an interval added. This is **not** an official MMLU score.
- No official benchmark data imported, no real GPU or distributed training verified, no 1B/3B weights trained.

## v0.7 update

- Offline Hendrycks-style CSV evaluation (5-shot layout, conditional answer-label scoring): IMPLEMENTED/CPU TESTED.
- Human-reviewed, SHA-pinned local MMLU dev/test data inventory: IMPLEMENTED/CPU TESTED.
- Published/official MMLU test data evaluation and reproducible official benchmark score: **NOT_RUN**.
- Hardware GPU measurements, 1B/3B full model training: **NOT_RUN**.

See [MMLU_REFERENCE.md](MMLU_REFERENCE.md) for limits.