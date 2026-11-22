# SagaLLM v0.14 — expanded original Java data and CPU learning

## Scope and provenance

The generator `src/sagallm/java_large_corpus.py` writes **5,436** original
Japanese/English natural-language instructions with complete Java 17 source
files. They are programmatically parameterized *variants of 70 project-authored
algorithm templates*, not 5,436 independently invented algorithms or 5,436
third-party open-source projects. No external Java repository, pretrained model,
paid API, or cloud GPU is used.

| Split | Rows | Fully held-out task families |
| --- | ---: | --- |
| Train | 5,220 | 58 training families |
| Validation | 108 | 6 different families |
| Test | 108 | 6 different families |

Topics include arithmetic, loops, strings, arrays, ArrayList, HashMap, HashSet,
queues, stacks, streams, exceptions and methods. Representative Java programs
for all 70 task families compile with JDK 21; this is a **reference dataset
syntax check**, not evidence of a successful neural generator.

The dataset manifest pins source JSONL bytes by SHA-256 and rejects manual
modifications even if the hash in the manifest is rewritten, by regenerating
canonical examples from the checked-in source generator. Original authored
programs do not contain a third-party dataset license; legal attestation is
**not performed**. The repository's own LICENSE remains owner-undecided.

## Build, verify and train

Requirements: Python >= 3.10, PyTorch, PyYAML and (for compile checks) JDK >= 17.
No network access is necessary.

```bash
PYTHONPATH=src python -c 'from sagallm.java_large_corpus import build_java_large_corpus; print(build_java_large_corpus("data/java-expanded"))'
PYTHONPATH=src python -c 'from sagallm.java_large_corpus import verify_java_large_corpus; print(verify_java_large_corpus("data/java-expanded"))'
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_expanded.py \
  --dataset data/java-expanded --output checkpoints/java-expanded-experiment \
  --steps 5220 --max-seconds 700 --hidden-size 192 --layers 4 \
  --progress-every 300 --compile-limit 18
```

Output directory must not already exist. This is a real from-scratch
random-initialized SagaTransformer SFT training run, **not a fine-tune of any
pretrained Java coding model**. The BPE tokenizer is trained from *train*
examples only. Its optimized Python heap implementation preserves token IDs
from the original prototype algorithm; the randomized Unicode test protects
this invariant.

Training processes the shuffled 5,220 train rows without replacement until
all have contributed one optimization update. The `train-report.json` contains
`distinct_training_examples_seen`, `unseen_training_examples`, actual steps,
wall clock, seed, SHA-256 hashes, full/sampled validation, and one final
held-out test loss. A wall-clock cap can interrupt before all rows were seen;
**never equate corpus size to the count used for gradients**.

Checkpoint selection is based exclusively on a fixed validation subsample and
the full validation set is scored afterward; this is not benchmark-independent
model development. Test split rows are audited for integrity prior to learning
but are **not used as training targets or for checkpoint selection**. After
selection, test loss is measured once and a small family-balanced test sample
is decoded and compiled with `javac -proc:none`. Generated Java is **never
executed**; semantic correctness remains NOT_ESTABLISHED.

## Limitations

A low training loss does not establish Java instruction following. The dataset
contains only 70 high-level algorithms and many numerical/text substitutions;
near-duplicate syntax and scaffolding can make training substantially easier
than real Java software development. Real coding-quality evaluation needs
separately written tasks and *isolated* runtime tests. Compiling generated
code or calling Java on the host is not a security sandbox. Use the generated
model only as a bounded local experiment, and review all code manually.

No model weights or full corpus are committed to GitHub. They are separate
conversation artifacts when a training run completes. Do not describe a
synthetic-data test as the model's performance on actual user projects.

## Actual CPU experiment (2026-11-22)

Two complete one-pass CPU learning runs were executed using the **same**
5,220 training examples, the same validation/test task-family split, and no
hosted API. They differ in initialization and tokenizer:

| Measurement | New random-initialized model | Continued v0.13 model |
| --- | ---: | ---: |
| Parameter count | 1,427,136 | 774,240 |
| Optimization steps executed | 5,220 | 5,220 |
| Unique train examples used by **last** checkpoint | 5,220 | 5,220 |
| Unique train examples used by **selected best** checkpoint | 600 | 600 |
| Validation-selected best step | 600 | 600 |
| Best checkpoint full validation loss | 2.6373 | 2.9376 |
| Best checkpoint held-out test loss | 3.7564 | 3.9300 |
| Compilable generated Java on 18 held-out tests | **2 / 18** | **1 / 18** |
| Exact match to reference Java | **0 / 18** | **0 / 18** |
| Behavioral execution of generated code | NOT_RUN | NOT_RUN |

The new model was trained entirely from random initialization. The second
started from the **locally trained** v0.13 checkpoint with a matching tokenizer,
then reset its optimizer to run on the expanded training set. Both models use
randomly initialized SagaLLM architecture; no pretrained third-party model.

**Crucial accounting:** the `java-last.pt` model has seen all 5,220 train
examples during its full one-pass training; the `java-best.pt` model is the
validation-selected checkpoint at step 600, and thus has only seen 600 of
those examples. The latter had the better held-out loss, but *neither* model
was proven to solve Java programming tasks. This is direct evidence that
maximizing exposure to templated examples need not improve generalization.

A completely separate learned Java coding benchmark and a true isolated
behavioral evaluator are still required. The small reference compile-only
check must not be treated as correctness on real-world Java projects.

### Continue the user's earlier v0.13 checkpoint

Use this only with a **trusted** local v0.13 checkpoint and its **exact**
original tokenizer (the code verifies the recorded tokenizer SHA):

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_expanded.py \
  --dataset data/java-expanded --output checkpoints/java-expanded-continued \
  --init-checkpoint path/to/v013/java-trained.pt \
  --init-tokenizer path/to/v013/tokenizer.json \
  --steps 5220 --max-seconds 700 --lr 0.00025 --progress-every 300
```

The local `experiments/java_expanded_v014_results.json` file contains the
measured evidence and checksum identifiers. Raw weights and generated corpus
are **not committed to GitHub**; they are packaged separately after training.