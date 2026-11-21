# Dataset usage review cards (v0.6)

SagaLLM cannot infer copyright, consent, privacy permissions or dataset terms from a file name or URL. The `--license` argument **is only an assertion**. A dataset usage review card adds **a human-reviewed assertion bound to the exact local data bytes**, and rejects mismatches. Even a passing card is **not a legal clearance certificate**.

## Required fields

Prepare a local UTF-8 JSON card using [`examples/dataset-card.template.json`](../examples/dataset-card.template.json). Set:

- `format`: `sagallm-dataset-card-v1`
- `name`, `source_uri`, `source_version`: identify the actual source and revision
- `license_id`, `license_evidence`: exact license identifier and a human-readable evidence/reference (not verified by the tool)
- `review_status`: `approved` **only after real human review**
- `reviewed_by`, `reviewed_at` (YYYY-MM-DD): review attribution/date
- `allowed_uses`: subset of `train`, `evaluate`, `redistribute`, only where the rights analysis permits it
- `file_sha256`: lowercase SHA-256 of the exact local source file, not its pathname or rendered text

Review any supplementary terms, rights of third-party contributors, PII, and privacy requirements separately. Do not commit sensitive data or private review notes. A dataset's license is **not** the SagaLLM code's license.

## Verify (offline)

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('path/to/source.jsonl'); print(hashlib.sha256(p.read_bytes()).hexdigest())"

sagallm data verify-license --source path/to/source.jsonl \
  --card path/to/review-card.json --purpose train

sagallm data shard --source path/to/source.jsonl \
  --output data/prepared --license CC0-1.0 --dataset-card path/to/review-card.json

sagallm benchmark --checkpoint checkpoints/tiny.pt \
  --tokenizer data/tokenizer.json --data path/to/questions.jsonl \
  --format mmlu --license MIT --dataset-card path/to/review-card.json \
  --output reports/benchmark.json
```

The `CC0-1.0` and `MIT` above are **illustrative only**. Use the reviewed license for **your actual source**. The report includes source and card SHA-256, user reviewer identity and the result `REVIEW_ATTESTED_NOT_LEGALLY_VERIFIED`.

Without `--dataset-card`, legacy data and benchmark commands remain available for compatibility and continue to display license information as **user-asserted, unverified**. Use the card for new datasets. A card only permits its exact allowed purpose; `evaluate` cannot silently be substituted for `train`.

The tool performs no HTTP requests, does not fetch licenses or datasets, and cannot verify whether the card contents are truthful. External dataset selection and permission review are still operator tasks.

## Candidate dataset reference (not downloaded or used)

The [cais/mmlu dataset card](https://huggingface.co/datasets/cais/mmlu) currently describes the dataset under the MIT license, links to its source, and provides citation details. This observation is **not** an automated rights determination or a claim that any MMLU data has been imported into SagaLLM. Record an exact revision, files, SHA-256 and review evidence before running it. See [BENCHMARKS.md](BENCHMARKS.md) for why the SagaLLM adapter is **not** the official MMLU evaluation protocol.