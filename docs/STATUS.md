# Implementation status

| Component | Status | Notes |
| --- | --- | --- |
| Transformer / RoPE / GQA / RMSNorm / SwiGLU | IMPLEMENTED | Native PyTorch, CPU test needed |
| Configs tiny / micro / pilot / 1b / 3b | IMPLEMENTED | 1B/3B never loaded in tests |
| Byte-level BPE training | IMPLEMENTED | Small offline corpus only; no HF Tokenizers dependency |
| Single-device pretraining | IMPLEMENTED | CPU / optional CUDA/MPS unverified |
| Atomic checkpoint / resume | IMPLEMENTED | Data/tokenizer hash checks; schedule changes affect exact reproducibility |
| CLI generation / KV cache | IMPLEMENTED | No production decoding server |
| Basic data prep | PARTIAL | No near-dedup, streaming at training scale or comprehensive PII screening |
| Validation / standardized benchmarks | NOT_IMPLEMENTED | Phase 3/5 |
| SFT / DPO | NOT_IMPLEMENTED | Phase 4 |
| API / streaming HTTP | NOT_IMPLEMENTED | Phase 5 |
| HF Transformers export | NOT_IMPLEMENTED | Phase 5 |
| Distributed learning / FlashAttention integration | NOT_IMPLEMENTED | Phase 6 |
| Real 1B/3B training | NOT_RUN | No hardware or licensed corpus specified |