# Java diverse curriculum and CPU SFT — SagaLLM v0.15

## Scope and provenance

- **17,536** deterministic, project-original Java 17 instruction/source records: **17,080 train**, **228 validation**, **228 test**.
- **110** concept families. New material includes 32 training families and 8 held-out family types (40 additional families over v0.14).
- Task-family disjointness is enforced across train/validation/test. Both languages are used for instructions, and only the training split is used for tokenizer fitting/weight updates.
- SHA-256 checks and canonical regeneration detect modified examples, modified manifests, duplicate Java sources and unauthorized provenance changes. The review status is `NOT_LEGALLY_REVIEWED`; no third-party source or pretrained model has been imported.
- **These are structured variations of 110 original templates.** They do not constitute 17,536 independently implemented Java algorithms or a real-world code corpus.

## Reproduce the data

```bash
PYTHONPATH=src python -c "from sagallm.java_diverse_corpus import build_java_diverse_corpus; print(build_java_diverse_corpus('data/java-diverse'))"
PYTHONPATH=src python -c "from sagallm.java_diverse_corpus import verify_java_diverse_corpus; print(verify_java_diverse_corpus('data/java-diverse'))"
```

Build refuses to overwrite any existing output directory. The canonical verifier independently regenerates every expected program; a forged checksum in `manifest.json` is insufficient.

## From-scratch model training

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_diverse.py \
  --dataset data/java-diverse \
  --output checkpoints/java-diverse \
  --steps 17080 --max-seconds 1200 \
  --hidden-size 192 --layers 4 --lr 0.0003 \
  --progress-every 1000 --compile-limit 10
```

Only a small (~1.5M parameter) model is used in CPU experiments, not the 1B/3B config. Data from different task families is interleaved without replacement until every train sample has contributed one optimizer update. A fresh optimizer is used when continuing a prior local checkpoint.

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_diverse.py \
  --dataset data/java-diverse --output checkpoints/java-continued \
  --steps 17080 --max-seconds 1250 --lr 0.00015 \
  --init-checkpoint checkpoints/previous/java-last.pt \
  --init-tokenizer checkpoints/previous/tokenizer.json
```

Only use **your own verified local** trusted checkpoint. A different tokenizer cannot silently be swapped in. Training logs record each counted example separately from generated dataset size. Checkpoint selection is based exclusively on validation examples; test is tokenized/scored only after selection. Keep both best and last checkpoints: **best may see only part of the corpus; last may overfit**. If a run is stopped for its time budget, actual optimizer steps and unique examples seen are recorded and never rounded up.

## Model quality / security

The task-family-separated test checks response loss, compilation with `javac -proc:none`, and exact-source match. **No generated Java programs are run** in this experiment. Java compilation is not proof of semantic correctness or safe execution. For behavioral testing, use a disposable VM and the separate, explicit-opt-in OCI harness with a trusted, preinstalled image. OCI isolation alone is not a hard security boundary.

Known limitation: validation/test families are synthetic and not independent real-world benchmarks. Better corpus licensing and a high-quality independently authored code benchmark are necessary before claiming Java Vibe Coding proficiency.

## Optional next stage: local Java project inventory (NOT approved for training)

Real project Java is much more diverse than template expansions, but public
repositories are not automatically licensed for every ML training purpose.
This release provides a **local-only, no-network, fail-closed inventory**:

```bash
# First clone or otherwise obtain a project you are authorized to inspect.
# Revision must be a full 40-character pinned Git commit SHA.
PYTHONPATH=src python scripts/import_java_repository.py \
  --source /path/to/repository \
  --source-uri https://example.org/pinned-java-project \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --output data/java-repository-unreviewed
```

This exports a JSONL of UTF-8 Java documents with file hashes and a manifest
which always says `review_status=PENDING`, `training_authorized=false`,
`license_id=UNVERIFIED`. It excludes symlinks, build/vendor directories,
exact duplicate files and simple suspected emails/secrets/phone/payment cards.
The screening is heuristic and **not** a privacy guarantee. The declared commit
SHA is metadata, not cryptographic verification of a Git checkout.

**Do not feed this unreviewed JSONL to the trainer.** Before any real project
source is used, review its license, provenance, third-party contributions and
personal data; create a purpose-specific dataset card for the exact exported
JSONL with `sagallm.dataset_cards.verify_dataset_card` (see
`docs/DATASET_CARDS.md`). The importer does not automatically approve data,
download sources, or start training. No real third-party Java source was used
in the v0.15 training runs.

## Actual CPU experiment results

| Metric | Random initialization | Continued from v0.14 last |
| --- | ---: | ---: |
| Parameters | 1,476,288 | 1,427,136 |
| Optimizer updates | 17,080 | 17,080 |
| Distinct training rows seen | 17,080 | 17,080 |
| Task families visited | 90 / 90 | 90 / 90 |
| Best validation step | 1,000 | 1,000 |
| Best full-validation loss | 4.06096 | 3.56891 |
| Held-out test loss | 4.27249 | 4.49573 |
| Held-out javac compile | 4 / 10 | 6 / 10 |
| Exact Java reference match | 0 / 10 | 0 / 10 |

The two tokenizers are different (768 versus 512 token IDs); loss values are
**not strictly comparable**. These tests are tiny and synthetic. Neither run
established usable natural-language Java programming ability. Generated Java
was not executed, and Java behavior was **NOT_RUN**. The 10 evaluated test
cases were drawn from ten **unseen task families**, not from real-world code.
Each run also stores its last-step model (all 17,080 examples consumed), but
the best validation-selected checkpoint only incorporated 1,000 examples.
Training-only loss decreased strongly while validation loss worsened later:
the data volume increase did not solve generalization.

Full hashes, timing, configuration, selected checkpoints and metadata are in
`experiments/java_diverse_v015_results.json` and in each model archive's
`train-report.json`. The trained models and corpus are distributed as separate
user download artifacts, **not** checked into the public Git history.