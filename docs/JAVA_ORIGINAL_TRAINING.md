# Original Java study data and real SagaLLM training (v0.12 / v0.13)

## What this is

SagaLLM v0.12 contains a **deterministic, newly authored** collection of small
Java 17 instruction/program pairs. There is no third-party Java source corpus,
no code downloaded from GitHub, no pretrained weights and no hosted LLM API.
It is an intentionally narrow, synthetic teaching set, **not** a production
instruction dataset, a public benchmark or a legal license certification.

`src/sagallm/java_generated_data.py` produces 350 examples:

| Split | Rows | Entirely disjoint task families |
| --- | ---: | --- |
| train | 320 | hello, add, subtract, multiply, divide, remainder, even_odd, count_up, count_down, sum_range, square, max_two, min_two, positive, array_sum, upper, lower, repeat, table, length |
| validation | 15 | factorial, reverse, gcd |
| test | 15 | fibonacci, prime, palindrome |

Each row contains a natural-language Japanese or English user request and one
complete `public class Main` program. The instruction format matches the
existing SagaLLM Java AI revision prompt. Every source is generated
independently of external model outputs. The SHA-256 manifest includes the
separate split files; verification regenerates the original examples and
rejects edits even when their manifest hashes are manually replaced.

## Run locally

Python 3.10+, PyTorch, and a JDK 17+ for compilation checks. No GPU is needed.
From the repository root:

```bash
python -m pip install -e .
PYTHONPATH=src python -c 'from sagallm.java_generated_data import build_java_dataset; print(build_java_dataset("data/java-original"))'
PYTHONPATH=src python -c 'from sagallm.java_generated_data import verify_java_dataset; print(verify_java_dataset("data/java-original"))'
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_original.py \
  --dataset data/java-original --output checkpoints/java-original-exp \
  --steps 1500 --max-seconds 155 --hidden-size 160 --layers 3 \
  --progress-every 50 --compile-examples 15 \
  --patience-evals 3 --min-delta 0.001
```

New output directories are required; the script refuses overwrites and applies
explicit step and wall-clock caps. It trains a **random-initialized** dense
Decoder-only SagaTransformer with assistant-response-only loss, standard AdamW,
a tokenizer learned **only on train**, and independent validation/test losses. Since v0.13 the default training
script uses **validation-only early stopping** and saves the best checkpoint
as `java-trained.pt`, with the last optimizer state separately in `java-last.pt`.
The test set is evaluated **once, after checkpoint selection**, and is never
used to select checkpoints. Old v0.12 experiments measured initial test loss;
those earlier results are unchanged and cannot be retroactively reclassified.
The `max-seconds` cap is cooperative and applies to the optimizer loop; offline
evaluation and compilation may add extra elapsed time. The training script
loads only the `train.jsonl` examples into optimization steps. Validation is
used for early-stop/selection; test loss is measured once after selecting the
best model. `test_loss_before` is now `null` to prevent test-split peeking.

All model and tokenizer files are written locally and ignored by Git. A training
checkpoint contains optimizer state and must be treated as **trusted input**;
only load checkpoint files that you created or verified. Java model output is
compiled with `javac -proc:none`; it is **never executed** by this evaluation.
Compiling Java source is not a secure sandbox and proves syntax only.

For the tested model, inference can use a matching local checkpoint and tokenizer:

```bash
sagallm java new --prompt 'Hello world' --output work/java-demo
sagallm java ai-revise \
  --project work/java-demo \
  --checkpoint checkpoints/java-original-exp/java-trained.pt \
  --tokenizer checkpoints/java-original-exp/tokenizer.json \
  --max-new-tokens 170 --attempts 1 \
  --prompt '1から10までの合計を表示して'
```

This integration is an **experiment**; it can generate compilable but incorrect
code. Diff review is required, and the workflow never executes Java output.

## Measured experiment — 2026-11-22, CPU only

CPU PyTorch 2.10; Python 3.13; JDK 21; 774,240 model parameters;
512-token vocabulary; 384 maximum context; 1,500 optimizer steps;
AdamW learning rate 0.0012, random seed 20261122. Measurements on the
execution container, **not** the repository owner's machine:

| Metric | Before training | After training |
| --- | ---: | ---: |
| train sample mean response-token loss | 6.2038 | 0.2634 |
| held-out validation loss | 6.1888 | 4.7872 |
| held-out test loss | 6.1769 | 6.0101 |
| test Java files compiled (`javac`) | NOT_RUN | 10 / 15 |
| exact reference-code matches | NOT_RUN | 0 / 15 |
| generated Java behavioral tests | NOT_RUN | NOT_RUN |

Validation loss decreased early, then **worsened after approximately step 150**
(3.0580 at step 150 vs 4.7872 at step 1500); there is clear overfitting.
10/15 compilation success does **not** mean 10 tasks were solved: generated
programs may perform the wrong task. The three held-out families are represented
by only five examples each, so confidence intervals/strong generalization
claims would be misleading. All behavioral runtime results remain `NOT_RUN`.

No model trained at 1B or 3B scale, no GPU experiment, and no genuine
open-ended Java programming competency has been demonstrated. The resulting
checkpoint is an **experimental teaching artifact**. The binary model weight
and generated corpus are **not pushed to GitHub**; they are separately
packaged for download where generated.

## Further work

For practical Java vibe coding, source and license-reviewed *diverse* Java
instruction/repair data, more capable base-model pretraining, substantially
larger training runs with early stopping, and behavior tests in an isolated
sandbox are needed. Do not use this tiny checkpoint as an autonomous coding
agent or execute its output without review.

## v0.13 re-training result (same original v0.12 dataset)

- CPU-only, random-initialized SagaTransformer, same model, seed, learning rate
  and 320/15/15 family-disjoint data as v0.12.
- Requested 1,500 optimizer steps; early stopping ended at step **300**, and
  selected the step **150** checkpoint using validation loss only.
- v0.12 step-1500 checkpoint: validation loss **4.7872**, test loss **6.0101**,
  compiled **10/15**, exact reference **0/15**.
- v0.13 step-150 checkpoint: validation loss **3.0580**, test loss **3.7609**,
  compiled **8/15**, exact reference **0/15**.
- Better held-out token loss has **not** produced reliable Java task completion;
  compile success actually decreased. No generated Java was run.
- Selection is based exclusively on validation. These two experiments are on
  the *same* synthetic benchmark, so further tuning on this holdout would
  constitute experimental overfitting. New independent tasks are needed.
- For opt-in guarded output-based evaluation, see
  [`JAVA_BEHAVIORAL_VALIDATION.md`](JAVA_BEHAVIORAL_VALIDATION.md).