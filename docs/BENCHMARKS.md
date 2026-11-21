# Reproducible offline benchmark harness (v0.4)

SagaLLM provides **local multiple-choice likelihood scoring**. This is an evaluation harness, **not** a certified standard-benchmark result. It does not bundle, download, endorse or redistribute third-party datasets. Users must review benchmark licenses and data access rights themselves.

## JSONL input formats

`--format choices` (default), one record per line:

```json
{"id":"demo-1","prompt":"Question: What is 2 + 2?\nAnswer:","choices":[" 4"," 5"," 6"],"answer_index":0}
```

`--format mmlu` is a lightweight *MMLU-style adapter*, not official MMLU evaluation. It reads:

```json
{"id":"demo-mmlu-1","question":"What is 2 + 2?","choices":["1","2","4","8"],"answer":"C"}
```

The adapter formats a fixed English `Question: ...\nA. ...\nAnswer:` prompt, then scores the one-token-ish continuations ` A`, ` B`, ` C`, ` D`. It does **not** implement subject aggregation, standardized prompts, few-shot examples, official splits, or published scoring policies. Do not compare its scores with official leaderboards.

## Example commands

```bash
# Uses a local tiny checkpoint from the README demo. Reports without a
# --train-docs corpus clearly label contamination as NOT_CHECKED.
sagallm benchmark --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json \
  --data examples/toy_benchmark.jsonl --license CC0-1.0 \
  --output reports/toy-benchmark.json

# Provide the exact prepared document-shard directory used to create the
# checkpoint's training token shards (not just any clean-looking corpus):
sagallm benchmark --checkpoint checkpoints/sharded.pt --tokenizer data/tokenizer.json \
  --data examples/toy_benchmark.jsonl --license CC0-1.0 \
  --train-docs data/prepared --output reports/toy-benchmark-with-audit.json
```

The CLI refuses to overwrite a report file. Output includes checkpoint/tokenizer/benchmark-file SHA-256, selected format, scoring rule, per-question answer IDs and scores, total accuracy, and a contamination audit. Raw prompt text is not logged in results. `--scoring mean` (default) averages answer-token log probabilities to lessen length bias; `--scoring sum` uses total continuation log likelihood. Both can be biased and are not equivalent to official benchmark scoring.

## Contamination screening limitations

- If `--train-docs` is absent: `NOT_CHECKED`. Do not label the result held-out.
- If it is supplied, SagaLLM verifies the prepared-shard file hashes and looks for normalized benchmark prompts as **exact substrings** of *training documents*; matching examples cause refusal unless `--allow-contaminated` is explicitly passed.
- Prompts shorter than 24 normalized characters are `INCOMPLETE`, not declared clean.
- When the checkpoint recorded a token-shard source, the provided document manifest must match the one used during pretraining. If checkpoint provenance is missing, this is shown as `CHECKPOINT_CORPUS_NOT_RECORDED`.
- The algorithm does not detect paraphrases, partial question/answer leaks, semantic contamination, untracked prior training, or contamination of external models. `NO_EXACT_SUBSTRING_FOUND` is **not** a guarantee of data independence.

The `--license` argument is a user assertion, not automatic proof of usage rights. The toy examples are repository-authored evaluation smoke tests; any computed score is not a useful measure of actual model skill.