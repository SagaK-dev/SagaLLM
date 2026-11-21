## v0.6 offline evaluation and dataset-attestation work (2026-11-21)

- `dataset_cards.py`: added a purpose-scoped (`train`, `evaluate`, `redistribute`), human-asserted dataset review record validated against the exact file bytes using SHA-256. It is not a legal-verification service. No automatic license approval or network fetch is performed.
- `benchmarks.py`: MMLU-style contamination audit now checks the original question rather than only the rendered zero-shot prompt; subject-level and macro accuracy, descriptive Wilson 95% interval, and explicit non-official protocol IDs are included in evaluation output.
- Optional `--dataset-card` for `data prepare`, `data shard`, `benchmark` and `sagallm data verify-license` CLI. Existing no-card workflows remain allowed but remain unverified.
- `tests/test_dataset_cards.py` and `tests/test_benchmark_protocol.py`: validate tamper rejection, purpose restriction, raw-question leakage checks, CLI and real tiny-model per-subject scoring.
- Development container Python 3.13 CPU: 56 PASS, 4 SKIP (optional dependencies unavailable). Hosted CI result must be checked separately.
- NOT_DONE: downloaded or officially evaluated MMLU data, legal rights verification, approved third-party training data, GPU hardware tests, 1B/3B training.

## v0.5 native Transformers export (2026-11-21)

- Added local-only, overwrite-refusing export of trusted SagaLLM checkpoints into safetensors and custom Transformers configuration/model loader files.
- Source model weights are checked against their native configuration; tokenizer SHA-256 must match when present. Checkpoint optimizer and RNG states are excluded from exported weights.
- FastBPE supports `AutoTokenizer`; legacy Python BPE is preserved as a native artifact without claiming HF AutoTokenizer compatibility.
- Added CPU roundtrip/CLI and optional Transformers 4.x integration tests. GitHub CPU run 44904787682 passed 52 tests; legacy Python BPE AutoModel regression was added after that run.
- Neither model weights nor training datasets are published; 1B/3B training and GPU tests remain NOT_RUN.

## v0.4 CPU validation (2026-11-21)

- Optional streaming-iterator HF Tokenizers integration and dispatch for both serialized tokenizer formats.
- Offline multiple-choice/MMLU-style question adapter, scoring, and leakage checks against verified train-document shards.
- No official benchmark dataset, internet API, pretrained weights, GPU training or pretrained 1B/3B model is included.
- Local `tokenizers` package was unavailable in the sandbox: 46 PASS, 2 SKIP. GitHub Actions installed `tokenizers` and confirmed **48 PASS**, including the native BPE training and shard-pretraining integration tests (run 44903131319). No GPU testing was performed.

# Implementation and verification report — follow-up v0.2

This report describes work in the execution container, **not** the repository owner's PC. No paid training or external dataset download was performed. No 1B/3B weights were initialized or trained.

## Implemented this iteration

- `src/sagallm/shards.py`: disk-backed SQLite exact dedup, stable train/validation split, JSONL shards, SHA-256 manifests, uint32-le token shards, on-disk random window sampling, path traversal checks.
- `src/sagallm/data.py`: TXT paragraphs are streamed rather than reading the whole file at once. A very large individual paragraph and JSON arrays still require significant memory.
- `src/sagallm/training.py`: optional periodic atomic checkpoint saving, fixed schedule horizon, parameter/data/hash validation on resume, RNG state handling, training on token shards, optional validation loss.
- `src/sagallm/evaluation.py`: shard validation and rejection of known training-data reuse unless explicitly allowed.
- `src/sagallm/model.py`: vocabulary limit during decoding and optional cooperative inference deadline.
- `src/sagallm/api.py`: bounded chat message lengths, request-window and inference-concurrency limits, cooperative deadline, EOS finish reason.
- `src/sagallm/cli.py`: `data shard`, `data tokenize`, new training flags and evaluation data split controls.
- Added tests covering data tampering, manifest traversal, sharded training, evaluation split checks, API limits, checkpoint/tokenizer hash, and CPU reproducibility.

## Tested

Run `OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest -q` from repository root and compare to the final reported run. `test_bitwise_resume_matches_uninterrupted_cpu` compares all model tensors for exact equality after a 4-step interruption and resume. This **does not** establish bitwise deterministic resume on every GPU or across different software versions.

## Limitations / next stages

- The custom Python BPE trainer and text file input are not yet scalable to multi-terabyte corpora. A production-grade streaming tokenizer trainer is still required.
- High-recall near-duplicate removal, comprehensive PII screening, trained language ID, true multi-node training, DPO, standardized benchmark batteries, hard HTTP timeouts and streaming API remain unimplemented.
- Checkpoint files are trusted input only; `weights_only=True` reduces pickle exposure but does not make arbitrary files safe.
- Dataset identifiers and license strings must be independently reviewed; they do not certify usage rights or prevent data contamination.
- Inference API controls apply to one process, not a production distributed deployment.
- 1B/3B configs remain inspect-only; no meaningful language quality is claimed.

## v0.3.0 local CPU quality pass (2026-11-21)

Implemented deterministic post-ingestion corpus curation (normalized and probable near duplicates), conservative sensitive-pattern screening, and an offline CPU CI verifier. Added a GitHub Actions workflow for public push / PR CPU testing with read-only repository permissions. Hosted Actions pass/fail must be verified separately from local pytest. This remains a research prototype, with no 1B/3B training or GPU verification.

## SagaLLM v0.7 reference-style benchmark milestone

- Added `src/sagallm/mmlu_reference.py` with six-column CSV parsing, SHA-256 reviewed inventory, prompt layout, 0–5-shot context fallback and explicit next-token/continuation scoring modes.
- Added CLI subcommands and synthetic CPU tests (`tests/test_mmlu_reference.py`), including round-robin subject sampling, review refusal, tamper detection and few-shot data leakage.
- Not implemented: official MMLU dataset ingestion and verified league/leaderboard comparability. No official benchmark data is bundled or fetched.

## SagaLLM v0.8 local hardware checks (2026-11-21)

- Added inspect-only `hardware inspect` (no weights allocated). The container reports Linux / CPU-only PyTorch, no CUDA or MPS. This is not the user's computer.
- `hardware smoke --device cpu`: **PASS_RUNTIME**, one tiny Forward/Backward/AdamW step, checkpoint save/reload, two-token generation. It does not measure language ability.
- `hardware distributed-smoke`: **PASS_RUNTIME**, two local CPU processes with Gloo all-reduce sum=3; it does not implement DDP or verify NCCL.
- Python 3.13 CPU tests: **78 passed, 4 skipped** (optional Transformers and tokenizer integrations not installed in the container). `scripts/check_cpu.py` passed.
- All GPU, MPS, BF16 GPU, multi-GPU and 1B/3B training: **NOT_RUN**. Automatic physical hardware attestation: **NOT_ATTESTED**.
- Exact reproducible commands and conditions are documented in `docs/PHYSICAL_VALIDATION.md`. Hosted GitHub Actions verification remains separate.

## SagaLLM v0.9 local Java starter coding (2026-11-21)

- Created `src/sagallm/java_vibe.py`, added `sagallm java` CLI, manual, and a regression suite.
- Nine deterministic Java standard-library templates, Japanese/English intent matching, preview-before-apply edits, source SHA-256 guards, optional javac 17-compatible compilation and opt-in Java execution.
- Current SagaLLM weights are **not** capable of reliable free-form Java vibe coding. Unsupported requests fail explicitly; no external pretrained models or paid API were used.
- Java compilation uses javac with annotation processors disabled and a short timeout. Explicit Java execution is a host process and **not isolated**; only trusted files should be run.
- The source LICENSE is still undecided. No third-party Java code dataset imported or trained.