# SagaLLM

A local-first **from-scratch** decoder-only language model research project. The model uses native PyTorch implementations of RMSNorm, rotary embeddings, grouped-query attention, SwiGLU and a tied token embedding / language-modeling head. The repository includes a small educational byte-level BPE tokenizer, basic data preparation, resumable single-device training and greedy/sampling generation.

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

Implemented: native dense causal Transformer, BPE tokenizer, config validation, CLI environment diagnostics, basic TXT/JSON/JSONL/CSV/optional Parquet input, exact deduplication, document-level split, small-device pretraining and checkpoint resume, generation with KV caching, CPU tests.

Not yet implemented: instruction tuning/SFT, preference training, FlashAttention-specific integration, distributed training, streaming training data shards, near-duplicate detection, comprehensive privacy screening, Hugging Face export, OpenAI-compatible API, formal evaluation benchmark suite, model distribution, or training of a 1B/3B model. These are planned phases, not completed features.

## Licensing

**The SagaLLM source license is not yet selected by the repository owner.** Public visibility does not confer a license to reuse this repository. Third-party reference projects have their own independent licenses; see `docs/DEPENDENCIES.md`. Do not publish model weights or add a LICENSE file without the owner's decision.

## Project documentation

- `docs/ARCHITECTURE.md` — model, safety gates and roadmap.
- `docs/DEPENDENCIES.md` — upstream references and license metadata.
- `docs/HARDWARE.md` — local runtime limitations and resource estimation.
- `docs/STATUS.md` — implemented/unimplemented feature matrix.