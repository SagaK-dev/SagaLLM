## v0.5 native Transformers export (2026-11-21)

- Added local-only, overwrite-refusing export of trusted SagaLLM checkpoints into safetensors and custom Transformers configuration/model loader files.
- Source model weights are checked against their native configuration; tokenizer SHA-256 must match when present. Checkpoint optimizer and RNG states are excluded from exported weights.
- FastBPE supports `AutoTokenizer`; legacy Python BPE is preserved as a native artifact without claiming HF AutoTokenizer compatibility.
- Added CPU roundtrip/CLI and optional Transformers 4.x integration tests. GH-hosted status must be verified separately after upload.
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