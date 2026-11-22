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
| GitHub Actions CPU CI | IMPLEMENTED / HOSTED_VERIFIED_V0.5 | v0.5 Python 3.11 CPU workflow passed: 53 tests after the final test update; v0.6: hosted CI passed 60 tests; v0.7: hosted CI passed 72 tests |
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

## v0.8 local hardware validation update

- Hardware inventory and local tiny CPU Forward/Backward, optimizer, checkpoint and generation: **IMPLEMENTED / LOCAL_CPU_PASS**.
- Two-rank local CPU/Gloo `all_reduce`: **IMPLEMENTED / LOCAL_CPU_PASS**; not DDP training.
- MPS, CUDA, BF16 real GPU tests: **NOT_RUN**, because no compatible GPU exists in this development container.
- Real physical machine operator attestation and multi-GPU training: **NOT_ATTESTED / NOT_RUN**.
- GitHub-hosted CPU CI for v0.8: **PENDING** until workflow runs.

## v0.9 Java natural-language starters

- `sagallm java new/chat/revise/templates/check/run`: IMPLEMENTED, local deterministic templates only.
- Nine Java tasks (hello, calculator, FizzBuzz, guessing game, TODO, prime, factorial, file reader, Swing): CPU/JDK COMPILE TESTED where JDK is installed.
- JDK compile and intentional Java run: LOCAL_TESTED; **not a sandbox**.
- Arbitrary prompt-to-Java by trained SagaLLM: NOT_IMPLEMENTED / NOT_TRAINED. Do not claim general vibe coding.
- IDE integration, autonomous file edits, automatic dependency management, automatic fixing of compile errors: NOT_IMPLEMENTED.

## v0.10 local model-backed Java experiment

- `java ai-revise` and `java ai-chat`: **IMPLEMENTED / LOCAL CPU TESTED** for
  using a trusted local SagaTransformer checkpoint to generate a candidate,
  compile with JDK, preview a diff and apply only after explicit approval.
- `java toy-sft`: **IMPLEMENTED / LOCAL CPU TESTED**, five original synthetic
  instruction/Java examples in existing SFT schema; one tiny SFT update tested.
- Learned Java coding quality, realistic coding benchmarks, reliable model
  self-repair, GPU testing, multi-file project agent, 1B/3B training: **NOT_RUN**.
- Java compilation does not provide a security sandbox; automated Java execution
  remains forbidden in the new model-generated workflow.

## v0.11 original Java instruction fixture and evaluation

- 10 original synthetic training examples / 1 validation / 2 test, separated by Java **task family**: IMPLEMENTED and CPU TESTED. Contains no third-party data and is not a model-quality corpus.
- `java dataset-build` / `dataset-verify`: SHA-256-pinned fixture with contamination and integrity guards: IMPLEMENTED and CPU TESTED. Human rights attestation is `NOT_ATTESTED`.
- `java evaluate`: real local checkpoint inference and held-out **javac syntax-only** rate; output never executed; semantic accuracy is `NOT_RUN`.
- `scripts/java_sft_smoke.py`: two optimizer steps on random 98,624-parameter model with train-only tokenizer training: CPU RUN; held-out behavior competence `NOT_ESTABLISHED`.
- Real-world Java generation benchmark, secure behavior test sandbox, 1B/3B training, GPU experiments: NOT_RUN.

## v0.12 — Java data generation and actual SFT experiment

- Original Japanese/English Java task data: **IMPLEMENTED**, 350 examples and
  26 algorithm families; train/validation/test disjoint by family.
- Synthetic dataset provenance/digest verification: **IMPLEMENTED**, no third-party corpus.
- Actual from-scratch CPU model SFT: **TRAINED**, 774,240 parameters, 1500 steps.
- Held-out compilation: **10/15 PASS_COMPILE**, exact reference match **0/15**.
- Java behavioral correctness: **NOT_RUN**; no independent useful coding capability established.
- Evidence: `docs/JAVA_ORIGINAL_TRAINING.md` and separately downloadable
  model checkpoint/tokenizer/dataset/report. **Do not upload weights to Git.**
- Real GPU, 1B/3B, broader programming datasets, safe sandbox execution: **NOT_RUN**.