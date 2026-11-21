# SagaLLM

**Prototype v0.4.0** — optional fast tokenizer, offline multiple-choice evaluation, and corpus overlap audit.

A local-first **from-scratch** decoder-only language model research project. The model uses native PyTorch implementations of RMSNorm, rotary embeddings, grouped-query attention, SwiGLU and a tied token embedding / language-modeling head. The repository includes a small educational byte-level BPE tokenizer, disk-backed JSONL / uint32 token shards, deterministic document splits, resumable single-device training, supervised fine-tuning, held-out token-level validation, an optional local inference API, and greedy/sampling generation.

> **Status: experimental foundation, not a trained 1B/3B LLM.** This project can initialize and train a small model with random weights. The 1B/3B configs are for inspection and future scaling; no large model is pretrained. Never interpret a falling toy loss as language proficiency.

## Quick start (CPU, offline after dependency installation)

Requires Python 3.10+ and an installed version of PyTorch and PyYAML. Installing dependencies may require network access. Nothing starts a paid cloud service.

```bash
python -m pip install -e '.[test]'
sagallm doctor
sagallm model inspect --config configs/model/1b.yaml
sagallm model inspect --config configs/model/3b.yaml
sagallm tokenizer train --data examples/toy_corpus.txt --output data/tokenizer.json --vocab-size 384
sagallm train pretrain --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data examples/toy_corpus.txt --checkpoint checkpoints/tiny.pt --steps 8 --batch-size 2 --block-size 32
sagallm generate --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json --prompt 'こんにちは' --max-new-tokens 16
# Prepare document-split token shards (for meaningful evaluation, use independent licensed data)
sagallm data shard --source examples/toy_corpus.txt --output data/prepared --license CC0-1.0 --max-docs-per-shard 3
sagallm data tokenize --prepared data/prepared --tokenizer data/tokenizer.json --output data/tokens --max-tokens-per-shard 256
# Train with a fixed LR horizon so resumption remains reproducible
sagallm train pretrain --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data data/tokens --checkpoint checkpoints/sharded.pt --steps 2 --schedule-steps 4 --block-size 8 --batch-size 1 --validate-every 1 --checkpoint-every 1
sagallm train resume --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data data/tokens --checkpoint checkpoints/sharded.pt --steps 4 --block-size 8 --batch-size 1 --validate-every 1
sagallm evaluate --checkpoint checkpoints/sharded.pt --tokenizer data/tokenizer.json --data data/tokens --block-size 8
# Optional supervised fine-tuning (toy instructions only)
sagallm train sft --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json --data examples/toy_sft.jsonl --output checkpoints/tiny-sft.pt --steps 2
# Optional local HTTP API: python -m pip install -e '.[serve]'
# sagallm serve --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json
python -m pytest -q
```

The `examples/toy_corpus.txt` is an artificial, repository-authored smoke-test corpus; it is **not** a licensed large-scale language training dataset. The `--license CC0-1.0` shown for this repository-authored demo does not certify permissions for third-party materials. `data/`, `checkpoints/` and binary model weights are gitignored. The default Python tokenizer is intentionally simple. The optional Rust-backed tokenizer can be selected with `--backend hf` (see `docs/TOKENIZERS.md`).

## Design constraints

- Pretraining starts from random model weights; no hosted LLM API is required.
- `configs/model/{tiny,micro,pilot,1b,3b}.yaml` specify architectures. `model inspect` calculates parameters **without allocating weights**.
- For safety, training over 200 million parameters is blocked unless `--allow-large` is explicitly supplied. This flag is **not** a guarantee that training fits memory.
- The tokenizer is byte-level and handles Japanese/English/code/Unicode without pre-trained vocabularies.
- Checkpoints should only be loaded from trusted locations. Resume verifies corpus, tokenizer SHA-256, schedule horizon and hyperparameters. When resuming, keep `--schedule-steps` unchanged or omit it to inherit the checkpoint value.
- Inference restricts generated token IDs to the tokenizer's vocabulary and checks checkpoint/tokenizer identity when provenance is available.
- The token-shard pipeline uses SQLite-backed exact and heuristic near-duplicate document deduplication, deterministic splits and disk-backed binary tokens. See `docs/DATA_PIPELINE.md`. It provides conservative near-duplicate removal and basic sensitive-pattern screening; neither can guarantee removal of paraphrases or all personal information.
- Data preparation requires a manually reviewed license identifier; pattern-based privacy screening is not a complete PII solution.
- CPU smoke tests must not be represented as CUDA/MPS tests.

## Scope and limitations

Implemented: native dense causal Transformer, small-scale BPE tokenizer, config validation, CLI diagnostics, TXT/JSON/JSONL/CSV/optional Parquet input, SQLite-backed exact and heuristic near-duplicate filtering, document-level train/validation JSONL shards, disk-backed uint32 token shards, small-device pretraining with integrated validation and checkpoint resume, SFT smoke training, simple held-out token loss/perplexity, localhost-first inference API with basic rate/concurrency controls, KV-cached generation, CPU tests.

Not yet implemented: official standardized benchmark protocol/results, preference training/DPO, production SFT batching and validation, FlashAttention-specific integration, distributed training, comprehensive privacy screening, guaranteed near-duplicate removal, guaranteed bounded-memory tokenizer training, Hugging Face export, streaming HTTP API, official standardized benchmark execution, model distribution, or training of a 1B/3B model. These are planned phases, not completed features.

## Optional fast BPE and offline benchmark harness

```bash
# Optional CPU-native BPE implementation (downloads the free package if missing):
python -m pip install -e '.[fast-tokenizer]'
sagallm tokenizer train --backend hf --data examples/toy_corpus.txt --output data/fast.json --vocab-size 512
# Existing python tokenizer format remains supported and is the default.

# Simple, offline benchmark smoke test. This is *not* an official MMLU score:
sagallm benchmark --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json \
  --data examples/toy_benchmark.jsonl --license CC0-1.0 \
  --output reports/benchmark-smoke.json
```

Without `--train-docs`, the report explicitly states that contamination has **not been checked**. For vetted document-shard training corpora, pass `--train-docs data/prepared` and refer to `docs/BENCHMARKS.md`. The adapter supports local MMLU-style four-choice JSONL but does not claim official leaderboard comparability. No external datasets are downloaded.

## Automatic lightweight CPU CI

`.github/workflows/cpu-ci.yml` runs fast offline tests on GitHub-hosted **CPU** for pushes to `main` and pull requests. It does **not** start large-scale model training or contact hosted LLM APIs. GitHub-hosted CI is not a local execution environment; to keep every test on your own machine instead, disable the workflow in GitHub and run:

```bash
python -m pip install -e ".[test,serve]"
python scripts/check_cpu.py
```

GitHub Actions v0.4 CPU CI passed (48 tests, run 44903131319), including the optional native tokenizer. A successful CPU CI run is not GPU validation.

## Licensing

**The SagaLLM source license is not yet selected by the repository owner.** Public visibility does not confer a license to reuse this repository. Third-party reference projects have their own independent licenses; see `docs/DEPENDENCIES.md`. Do not publish model weights or add a LICENSE file without the owner's decision.

## Project documentation

- `docs/ARCHITECTURE.md` — model, safety gates and roadmap.
- `docs/DATA_PIPELINE.md` — disk-backed data, validation and resume workflow.
- `docs/DEPENDENCIES.md` — upstream references and license metadata.
- `docs/HARDWARE.md` — local runtime limitations and resource estimation.
- `docs/STATUS.md` — implemented/unimplemented feature matrix.
- `docs/TOKENIZERS.md` — optional native BPE, format identity and limitations.
- `docs/BENCHMARKS.md` — offline multiple-choice scoring and contamination audit.

## API safety notes

The optional API exposes `/health`, `/v1/completions`, and `/v1/chat/completions`; it is an **experimental subset**, not a full OpenAI API implementation. The CLI defaults to `127.0.0.1`. Binding to a non-loopback address requires `--api-key`; additionally use TLS termination, reverse-proxy rate limiting, request timeouts and monitoring for any internet-facing deployment. Prompt length, rate (60 requests/minute per process by default), concurrent generations (1 by default), and token counts are bounded. Generation has a **cooperative**, not hard, time budget; a stalled native kernel cannot be forcibly interrupted. API streaming is not yet provided.

Data preparation requires a supplied license identifier, which is **not proof** that the input material is legally licensed. Operators must independently review permissions. The demonstration SFT data is toy material, not representative of useful instruction-following performance.