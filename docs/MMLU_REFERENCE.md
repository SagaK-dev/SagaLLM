# Local MMLU reference-protocol adapter (SagaLLM v0.7)

SagaLLM can read locally supplied MMLU-style **six-column CSV files** and run an offline, Hendrycks-style evaluation. It **does not** bundle or download the MMLU questions, verify rights automatically, or certify an official leaderboard score.

## Protocol reference and scope

Reference: [Hendrycks et al., original `evaluate.py`](https://github.com/hendrycks/test/blob/master/evaluate.py). The implementation follows that script's plain-text subject header, the first five examples from each subject's `dev` split, the format `question\nA. ...\n...\nAnswer: X`, the test prompt ending in `Answer:`, and next-token log-probability for answer labels ` A`, ` B`, ` C`, ` D`. The reference itself uses an OpenAI API; SagaLLM instead scores native local model logits and does not call any API.

**Important differences:** SagaLLM includes a BOS token, checks tokenization boundaries, reduces example shots to fit the configured model context, and may be used on partial local datasets. Therefore, prompt-layout matching does **not** imply identical evaluation conditions to a published model, and **every** result reports `official_score: false` and `standard_benchmark_certified: false`. The default safety cap is 100 test questions, drawn round-robin across the selected subjects; this is still a limited, non-representative diagnostic sample. For a complete locally reviewed data set, raise `--max-examples` explicitly (hard cap 20,000) and check the actual subject/test counts and effective shots.

`--scoring reference_next_token` is strict. Each of the four answer labels **must** be a stable single token at the boundary following the prompt, otherwise the command stops with `MMLUTokenizationError`. This prevents silently substituting an incompatible tokenization policy. For local diagnostics only, `--scoring continuation` uses full suffix likelihood, which **must not** be compared with next-token reference results.

## Local directory layout

Place licensed and independently reviewed CSV data under a folder with this structure:

```text
local-mmlu-data/
├── dev/
│   ├── abstract_algebra_dev.csv
│   └── ...
└── test/
    ├── abstract_algebra_test.csv
    └── ...
```

The `dev` and `test` folders must contain matching subject names. Files must be six-column, headerless UTF-8 CSV with the question, four options and an A/B/C/D answer. Symbolic links and unexpected files in these two folders are refused; the program caps file sizes and record counts. The repo's tests use **original synthetic questions**, not MMLU benchmark content.

## Usage (no network access or implicit approval)

```bash
# 1. Inspect your locally supplied dataset and create an UNAPPROVED file inventory.
sagallm mmlu inventory --data-dir local-mmlu-data --output reviews/mmlu-pending.json

# 2. Independently review source/version/license and usage rights; then update:
# source_uri, source_version, license_id, license_evidence, reviewed_by,
# reviewed_at (YYYY-MM-DD), review_status="approved", allowed_uses=["evaluate"].
# The generated SHA-256 list of all local files must not be edited.

# 3. Verify the approved metadata and that CSV bytes have not changed.
sagallm mmlu verify --data-dir local-mmlu-data --inventory reviews/mmlu-reviewed.json

# 4. Score a local model with the original-inspired 5-shot format.
sagallm mmlu evaluate \
  --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json \
  --data-dir local-mmlu-data --inventory reviews/mmlu-reviewed.json \
  --subjects abstract_algebra --shots 5 --max-examples 20 \
  --scoring reference_next_token --output reports/mmlu-subset.json
```

If the model vocabulary merges across answer-label boundaries, the reference-style evaluation will **reject** that configuration. `--scoring continuation` is an explicitly noncomparable diagnostic alternative. The `--output` report path must not already exist. No raw questions, answer options or prompts are included in the resulting JSON; IDs, scores, approved-review metadata and file hashes are retained.

## Contamination and provenance

`--train-docs` is an optional directory of **exact** prepared training-document shards. If present, SagaLLM verifies shard hashes and checks both the selected test questions **and all few-shot dev questions** against the training text. It also compares recorded training-dataset provenance in the checkpoint when available. Matches stop evaluation unless `--allow-contaminated` is explicitly requested. The overlap check only detects normalized, exact substrings of 24+ characters. With no corpus, the report says `NOT_CHECKED`; with no checkpoint corpus provenance, it explicitly says so. Paraphrases, benchmark answer leaks, unknown pretraining sources and semantic duplication are not ruled out. Passing this test does **not** establish that the model was trained on clean data.

All review cards are **human attestations bound to file hashes**; they cannot establish copyright ownership or grant permissions. Dataset review still needs to be performed by the user. Even a full 57-subject local run does not automatically become an official or comparable result. Reference alignment and scoring are stated separately in the JSON report.