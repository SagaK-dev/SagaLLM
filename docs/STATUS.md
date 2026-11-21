# Implementation status

| Component | Status | Notes |
| --- | --- | --- |
| Transformer / RoPE / GQA / RMSNorm / SwiGLU | IMPLEMENTED | Native PyTorch, CPU test needed |
| Configs tiny / micro / pilot / 1b / 3b | IMPLEMENTED | 1B/3B never loaded in tests |
| Byte-level BPE training | IMPLEMENTED | Python educational backend + optional HF Tokenizers backend; hosted dependency verification pending |
| Single-device pretraining | IMPLEMENTED | CPU / optional CUDA/MPS unverified |
| Atomic checkpoint / resume | IMPLEMENTED | Fixed schedule and hyperparameter checks; bitwise CPU interrupted/uninterrupted test passes |
| CLI generation / KV cache | IMPLEMENTED | No production decoding server |
| Disk-backed dataset shards | PARTIAL | SQLite exact/normalized/heuristic near dedup, JSONL splits, uint32 shards; only basic privacy screening |
| Basic next-token held-out validation | IMPLEMENTED | Model loss, perplexity, accuracy; rejects training split by default |
| GitHub Actions CPU CI | IMPLEMENTED / HOSTED_VERIFIED_V0.3 | v0.3 CPU workflow passed; v0.4 requires its own hosted run |
| Standardized benchmarks | PARTIAL | Offline multiple-choice/MMLU-style adapter implemented; official datasets, splits and published scoring protocols NOT_RUN |
| Single-turn SFT | IMPLEMENTED | Assistant-response masking; local JSONL |
| DPO / scalable SFT | NOT_IMPLEMENTED | Phase 4 |
| Local inference HTTP API | PARTIAL | Three endpoints, optional bearer key |
| API rate/concurrency limits | IMPLEMENTED | In-process limiter and cooperative deadline; hard kill, TLS, production rate limiting not implemented |
| HTTP streaming / hard kernel timeouts | NOT_IMPLEMENTED | Proxy recommended for external serving |
| HF Transformers export | NOT_IMPLEMENTED | Phase 5 |
| Distributed learning / FlashAttention integration | NOT_IMPLEMENTED | Phase 6 |
| Real 1B/3B training | NOT_RUN | No hardware or licensed corpus specified |