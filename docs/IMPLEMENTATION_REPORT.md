## v0.13: validation-only checkpoint selection and optional guarded Java run

- Changed `scripts/train_java_original.py`: patience/min-delta-controlled
  early stop; `java-trained.pt` is now the BEST validation-selected model and
  `java-last.pt` stores the final optimizer state. No test data is evaluated
  before the selected checkpoint is frozen. Historical v0.12 metrics retained.
- Added `src/sagallm/java_behavior.py`, `scripts/evaluate_java_behavior.py`,
  `tests/test_java_behavior.py` and `tests/test_java_early_stopping.py`.
  Strict opt-in; no local OCI engine, so all model-generated Java execution
  remains NOT_RUN. All 15 original test reference programs have independently
  checked expected outputs.
- CPU v0.13: best step 150 after 300 total, test loss 3.7609, compile 8/15,
  exact match 0/15. No model trained at 1B/3B scale.

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

## SagaLLM v0.10 experimental Java model adapter (2026-11-21)

- Added `src/sagallm/java_ai.py`: local checkpoint-backed SagaTransformer
  inference, verified matching tokenizer, safe output parsing, bounded compiler
  diagnostics and repair iterations, preview/approve with source hashes.
- Added `java ai-revise`, `java ai-chat`, `java toy-sft` CLI commands and
  regression tests; model-generated Java is never automatically executed.
- One tiny checkpoint loading and one genuine 1-step CPU SFT optimization were
  tested with original synthetic Java pairs. This verifies plumbing only, NOT
  the ability to generate correct Java from natural language.
- Large checkpoints, long model outputs, unsupported context lengths, missing
  compilers and source mismatches are fail-closed. No 1B/3B model was allocated.
- Local Python 3.13 / PyTorch CPU testing must be distinguished from GitHub CI
  and from real GPU validation; official Java code generation benchmarks and
  production-grade self-repair remain NOT_RUN.

## SagaLLM v0.11 Java SFT / holdout compiler experiment (2026-11-22)

- Added `java_coding_eval.py` for generated-only Java tasks with hash-pinned task-family holdout and `java evaluate` compile-only protocol. The model is loaded only from a trusted local checkpoint with tokenization provenance.
- Added the real tiny `scripts/java_sft_smoke.py` two-step CPU SFT experiment. It uses only original fixture data, trains the tokenizer on the training split, and reports test loss before/after. This validates the SFT mechanism, not learned coding ability.
- Fixed benchmark accounting so a no-op generated response cannot count as a successful new Java compilation. All runtime semantic tests remain `NOT_RUN` since no secure sandbox exists.
- The generated dataset records `NOT_ATTESTED` for human legal review. No source code LICENSE has been selected and no external Java corpus was imported.

### Actual two-step CPU study (development container; not user's PC)

- Model: 98,624 random-initialized parameters; context 1024; train-only ByteBPE vocab 384; train/validation/test = 10/1/2 original synthetic tasks.
- Held-out **test** assistant-token loss: **5.953665 before** → **5.952992 after** two SFT optimizer updates; final train-step loss **5.901787**. Difference is too small and experiment too short to infer Java coding competence.
- Held-out compile-only generation: **0/2 compiled** (calculator/factorial). Runtime behavior: **NOT_RUN**. No model-generated Java executed.
- GPU training, real third-party coding corpus, official Java benchmark and 1B/3B training: **NOT_RUN**. Source LICENSE remains unselected.

## v0.12.0 original Java dataset and training (2026-11-22)

- Added `src/sagallm/java_generated_data.py`, original Java program variants,
  bilingual instructions, deterministic per-split manifests and strict
  SHA + regenerated-reference validation; 320 train, 15 validation, 15 test.
- Added `scripts/train_java_original.py` with bounded, random-initialized
  SagaTransformer CPU SFT, train-only tokenizer, periodic held-out validation,
  checkpoints with optimizer and training provenance, compile-only Java held-out
  tests and exact reference-code checks.
- Added `tests/test_java_generated_data.py` with 7 tests including Java 17
  compilation of all 26 original task families, tamper/forgery refusal and
  real 3-step tiny CPU training.
- Actual model: 774,240 parameters, 1500 optimizer steps, 512 vocab, 384
  context, seed=20261122, CPU. Train loss 6.2038→0.2634; validation loss
  6.1888→4.7872 (overfits after step 150); test loss 6.1769→6.0101.
  Compilation 10/15, exact reference match 0/15. Java execution NOT_RUN.
- Generated corpus and ~9 MB training checkpoint kept out of Git.
  Real GPU, behavioral safety harness, useful Java code competence,
  license attestation and larger model training still NOT_RUN.