# Java coding data and evaluation (v0.11)

## Scope

This is an **original SagaLLM template-based synthetic engineering fixture**,
not an authorized external Java dataset and not a coding-capable foundation
model. No third-party repositories or Java snippets are downloaded.
The source repository has no finalized open-source LICENSE; the generated
fixture therefore carries `dataset_license_review: NOT_ATTESTED` and must
not be presented as independently rights-cleared external training material.

## Data lifecycle

```bash
sagallm java dataset-build --output data/java-study
sagallm java dataset-verify --dataset data/java-study
```

The `manifest.json` records `sagallm-java-coding-fixture-v1`, family-separated
splits, file SHA-256 and counts. File tampering, unknown task IDs, family
leakage, edited reference code, unsafe symlinks and missing examples are
rejected. Building never overwrites existing content. `train.jsonl` contains
10 SFT records with the same `User: ...\nAssistant:` instruction framing
as the existing `post_training.format_sft_example`. `validation.jsonl` has
one **prime** task; `test.jsonl` has one **calculator** and one **factorial**
task. Baseline Java source scaffolding is shared across rows, so this is
not an independent statistical generalization benchmark.

Use the **train split only** for tokenizer training and SFT. Training on
`validation.jsonl` or `test.jsonl` invalidates future held-out scores.
The training script does not enforce an authorized license for third-party
corpora; never import external code without dataset-card review and license
verification.

## Compile-only evaluation

```bash
sagallm java evaluate --dataset data/java-study \
  --checkpoint checkpoints/your-java-sft.pt --tokenizer data/tokenizer.json \
  --split test --max-examples 2 --max-new-tokens 512 \
  --output reports/java-evaluation.json
```

This calls the **real local SagaTransformer** via `LocalCheckpointGenerator`.
The existing Java AI revision flow extracts a complete replacement `Main.java`,
compiles it in a temporary directory via `javac -proc:none` and deletes the
temporary files. It does not execute generated Java or apply any modification
to the user's project. The output records a model checkpoint hash, tokenizer
hash, dataset split hash, per-task compilation status and exact-match results.
`UNCHANGED` is **not** counted as a successful new compilation. The report
uses `runtime_behavior_test: NOT_RUN`, since compilation does not prove
semantic correctness or security. `javac` is not a security sandbox.

The fixture's reference answer may be functionally equivalent to some other
valid implementation; an exact byte mismatch does not establish incorrect
behavior. Conversely, even matching code has not been run by the evaluator.
A production Java benchmark requires independently rights-cleared tasks,
secure execution isolation, carefully defined test oracles, and an actual
trained coding model. None of these is supplied here.

## Small actual SFT experiment

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python scripts/java_sft_smoke.py \
  --output reports/java-tiny-smoke.json
```

This builds the fixture, trains a tokenizer **only on train rows**, randomly
initializes a 98,624-parameter Transformer, saves a starting checkpoint,
runs **two CPU SFT optimizer steps**, and compares masked answer-token loss on
the untouched test split. It then attempts to compile Java generated for both
holdout prompts. The temporary training artifacts are deleted. The default
experiment does **not** allocate a 1B/3B model or train a competitive coder.
Do not interpret loss changes in this small demonstration as Java competence.

## Outstanding work

- Select and license legally usable Java instruction/code sources; record all
  source versions, legal rights and training/evaluation usage per dataset card.
- Much larger curated training data, truly unseen benchmark tasks, compilation
  AND behavior test oracles in a secure isolated execution system.
- GPU memory/throughput measurement, scalable SFT and held-out checkpoints;
  complete Java coding benchmarks and model comparisons.