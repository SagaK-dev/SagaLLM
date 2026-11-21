# SagaLLM

A local-first **from-scratch** decoder-only language model research project. The model uses native PyTorch implementations of RMSNorm, rotary embeddings, grouped-query attention, SwiGLU and a tied token embedding / language-modeling head. The repository includes a small educational byte-level BPE tokenizer, basic data preparation, resumable single-device training, supervised fine-tuning, held-out token-level validation, an optional local inference API, and greedy/sampling generation.

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
# Optional: evaluate a held-out local corpus (sample corpus used only for demonstration)
sagallm evaluate --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json --data examples/toy_corpus.txt
# Optional supervised fine-tuning (toy instructions only)
sagallm train sft --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json --data examples/toy_sft.jsonl --output checkpoints/tiny-sft.pt --steps 2
# Optional local HTTP API: python -m pip install -e '.[serve]'
# sagallm serve --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json
python -m pytest -q
```

The `examples/toy_corpus.txt` is an artificial, repository-authored smoke-test corpus; it is **not** a licensed large-scale language training dataset. `data/`, `checkpoints/` and binary model weights are gitignored. The tokenizer is simple and intended for learning/testing, not scalable production data pipelines.

## Design constraints

- Pretraining starts from random model weights; no hosted LLM API is required.
- `configs/model/{tiny,micro,pilot,1b,3b}.yaml` specify architectures. `model inspect` calculates parameters **without allocating weights**.
- For safety, training over 200 million parameters is blocked unless `--allow-large` is explicitly supplied. This flag is **not** a guarantee that training fits memory.
- The tokenizer is byte-level and handles Japanese/English/code/Unicode without pre-trained vocabularies.
- Checkpoints should only be loaded from trusted locations. Resume verifies corpus and tokenizer SHA-256. A change in the total-step schedule can alter learning rates across interruptions.
- Data preparation requires an explicit verified license identifier; basic regex screening is not a complete PII solution.
- CPU smoke tests must not be represented as CUDA/MPS tests.

## Scope and limitations

Implemented: native dense causal Transformer, BPE tokenizer, config validation, CLI environment diagnostics, basic TXT/JSON/JSONL/CSV/optional Parquet input, exact deduplication, document-level split, small-device pretraining and checkpoint resume, SFT smoke training, simple held-out token loss/perplexity, localhost-first inference API, generation with KV caching, CPU tests.

Not yet implemented: preference training/DPO, production SFT batching and validation, FlashAttention-specific integration, distributed training, streaming training data shards, near-duplicate detection, comprehensive privacy screening, Hugging Face export, streaming HTTP API, standardized evaluation benchmark suite, model distribution, or training of a 1B/3B model. These are planned phases, not completed features.

## Licensing

**The SagaLLM source license is not yet selected by the repository owner.** Public visibility does not confer a license to reuse this repository. Third-party reference projects have their own independent licenses; see `docs/DEPENDENCIES.md`. Do not publish model weights or add a LICENSE file without the owner's decision.

## Project documentation

- `docs/ARCHITECTURE.md` — model, safety gates and roadmap.
- `docs/DEPENDENCIES.md` — upstream references and license metadata.
- `docs/HARDWARE.md` — local runtime limitations and resource estimation.
- `docs/STATUS.md` — implemented/unimplemented feature matrix.

## API safety notes

The optional API exposes `/health`, `/v1/completions`, and `/v1/chat/completions`; it is an **experimental subset**, not a full OpenAI API implementation. The CLI defaults to `127.0.0.1`. Binding to a non-loopback address requires `--api-key`; additionally use TLS termination, reverse-proxy rate limiting, request timeouts and monitoring for any internet-facing deployment. Prompt length and token counts are bounded but there is currently no built-in rate limiter or hard inference execution deadline. API streaming is not yet provided.

Data preparation requires a supplied license identifier, which is **not proof** that the input material is legally licensed. Operators must independently review permissions. The demonstration SFT data is toy material, not representative of useful instruction-following performance.