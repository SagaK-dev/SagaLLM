# Offline, disk-backed training data pipeline (v0.3)

These commands **only** process local files. No datasets are downloaded. License identifiers are supplied by the operator and are **not proof of training permission**. Perform manual rights and privacy checks before using data.

## Workflow

For a local JSONL source containing one `{"text": "..."}` record per line:

```bash
sagallm tokenizer train --data examples/toy_corpus.txt --output data/tokenizer.json --vocab-size 384
sagallm data shard --source examples/toy_corpus.txt --output data/prepared --license CC0-1.0 --max-docs-per-shard 3
sagallm data tokenize --prepared data/prepared --tokenizer data/tokenizer.json --output data/tokens --max-tokens-per-shard 256
sagallm train pretrain --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data data/tokens --checkpoint checkpoints/toy.pt --steps 2 --schedule-steps 5 --block-size 8 --batch-size 1 --checkpoint-every 1 --validate-every 1
sagallm train resume --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data data/tokens --checkpoint checkpoints/toy.pt --steps 5 --block-size 8 --batch-size 1 --validate-every 1
sagallm evaluate --checkpoint checkpoints/toy.pt --tokenizer data/tokenizer.json --data data/tokens --block-size 8
```

> `--schedule-steps 5` fixes the total learning-rate horizon before a 2-step pause. `resume` reads the saved schedule when the flag is omitted. To extend a finished schedule, start a **new experiment**; changing the existing schedule silently is not reproducible.

**Important:** `examples/toy_corpus.txt` is a repository-authored example, not a real training dataset. Use a distinct corpus for meaningful held-out evaluations. The data-preparation command separates documents, but documents from the same source may still be semantically related.

## Format and memory behavior

- `data shard` reads TXT by paragraph, JSONL line-by-line, CSV row-by-row, and optional Parquet in batches. JSON array inputs are loaded as a single value; for large inputs use JSONL or Parquet instead.
- Disk-backed SQLite `PRIMARY KEY` indexes enable exact SHA-256 deduplication without retaining all documents in Python memory. A second deterministic curation pass removes Unicode/whitespace-normalized duplicates and **probable** near duplicates using 64-bit character-5-gram SimHash with four 16-bit LSH bands plus a sampled 5-gram Jaccard confirmation (threshold 0.84). This heuristic can miss paraphrases and similar material without a common LSH band.
- Split assignment occurs **after** curation; it sorts accepted SHA-256 document digests and reserves the requested fraction for validation. No document hash appears in both splits. This split is deterministic independent of input ordering, but can still be contaminated by paraphrases or related content.
- Documents matching email, obvious credential/private-key patterns, Japanese mobile numbers, and Luhn-valid card numbers are omitted. Screening emits per-category counts, not the matched values. These rules can reject benign data and miss sensitive content. The detector is incomplete and **not** a comprehensive privacy review. Large single-line or single-document input can still consume significant memory.
- Split files are `train-NNNNN.jsonl` and `validation-NNNNN.jsonl`, with manifest and per-file SHA-256 checks.
- `data tokenize` creates `uint32` little-endian `.bin` token shards with EOS between documents and a token-count/hash manifest. Shard targets are soft limits because a whole document is not split across files.
- `TokenShardDataset` validates shard hashes on startup, samples contiguous `(x,y)` windows by seeking directly into files, and does not load an entire token corpus into RAM. Sampling can cross document boundaries *within* a shard; EOS marks document breaks.
- A dataset with no shard containing `block_size + 1` tokens is rejected. Token samples are selected with replacement, and shorter files are weighted by their number of eligible starts.
- The built-in pure-Python BPE trainer is intended for small experiments and **is not** an out-of-core tokenizer trainer. Use a proven large-corpus tokenizer training pipeline before attempting 1B/3B pretraining.

## Data and checkpoint integrity

- Manifests include source hash, tokenizer hash, and individual shard hashes.
- Pretraining verifies source/tokenizer identity when resuming.
- Checkpoints include the optimizer, CPU/CUDA/MPS RNG state where available, data sampler RNG, schedule horizon, block size, batch size, accumulation and hashes.
- A resume with different training settings fails instead of silently changing experimental conditions.
- Evaluation uses the validation split by default and rejects the training split unless the operator explicitly supplies `--allow-training-data` for a smoke test.
- None of these checks independently establish that the corpus is legally permitted, private, free of benchmark contamination, or of high linguistic quality.

## Unimplemented production components

High-recall or semantic near-duplicate removal, robust PII detection/redaction, language-quality classifiers, distributed data loading, multi-terabyte preprocessing, efficient tokenizer training, scalable curriculum/mixture balancing, benchmark contamination checks, training-token accounting by source and legal review remain separate work items.

## v0.3 curation controls and privacy limitations

`data shard` enables conservative near-duplicate filtering by default. Pass `--keep-near-duplicates` to turn it off for controlled ablation experiments. Curation accepts candidates in SHA-256 order, not input order. The manifest records `rejected.normalized_duplicate`, `rejected.near_duplicate`, `rejected.low_quality` and `sensitive_categories` counts, and `near_dedup_enabled`. It contains no matching sensitive values, but **may contain the local source path**, so review manifests before sharing them.

A single 1 MiB source document can still require memory during parsing. JSON arrays are not streamed. This pipeline has **no semantic deduplication, entity-level PII detection, human privacy review, training-rights verification, or documented high-recall privacy guarantees**. Do not use unreviewed private/regulated corpora.