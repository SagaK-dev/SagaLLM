# Java instruction SFT — v0.17.0

This is an **actual** supervised fine-tuning experiment, initialized from the
v0.16 checkpoint trained on pinned, permissively licensed Java source code.
It does **not** use an external pretrained model or commercial API.

## Inputs and provenance

- **Starting weights**: v0.16 GitHub Actions run `44981937649`, artifact
  `sagallm-v016-pinned-java-cpu`, best checkpoint SHA-256
  `96005da938d3317fe2ba5185b71cdbb6ed2fee6bc4b9c39e370006d5f2917d1b`.
- **Tokenizer**: exact v0.16 byte-level BPE tokenizer, SHA-256
  `186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe`.
- **SFT examples**: original project-generated Java v0.15 data with
  17,080 train / 228 validation / 228 test *before token-length filtering*.
  The examples are parameterized from 110 families, not independent real-world
  Java software development tasks. They are NOT a source of official benchmark
  scores or evidence of an effective coding assistant.
- Training examples are from the original **train** families only. Different
  whole families are held out for validation and test. Only the training
  split contributes gradient updates.
- The data loader regenerates the original dataset *canonically*, compares
  every record and SHA-256, and refuses modified/mislabeled data even if
  the manifest digest has been forged.

## Why short prompts?

The real-Java v0.16 model has a 192-token context window. The earlier long
`java ai-chat` prompt cannot be used as SFT input without dropping much of the
Java answer. SFT v0.17 uses:

```
User: Write a complete Java 17 public class Main. Task: {short request}
Assistant:
{full Java source}
```

The user prompt is masked out of the supervised cross-entropy loss. **No
truncation** of prompts or Java targets is permitted. Examples that exceed
192 token positions are excluded and counted separately for each split.
Both the accepted count and the actually-trained distinct count are reported.
The short format is *not* the existing `java ai-chat` prompt; results must
not be represented as compatibility with that full-context Java editor.

## Reproduction

1. Obtain the v0.16 artifact from the linked Actions run, unzip it into
   `java-base/` so that the files are `java-base/java-model/java-best.pt`
   and `java-base/java-model/tokenizer.json`.
2. Install `pip install -e '.[fast-tokenizer]'` and a JDK 17+ to evaluate
   compile-only Java source outputs.
3. Generate canonical project-owned original examples:

```bash
python -c 'from sagallm.java_diverse_corpus import build_java_diverse_corpus; build_java_diverse_corpus("java-sft-data")'
python scripts/train_java_instruction.py \
  --base java-base/java-model/java-best.pt \
  --tokenizer java-base/java-model/tokenizer.json \
  --dataset java-sft-data --output java-sft-model \
  --steps 650 --batch-size 4 --eval-every 100 --max-seconds 420 \
  --compile-cases 8
```

The `.github/workflows/java-instruction-sft-cpu.yml` GitHub Actions job runs
these steps and uploads best and last checkpoints, tokenizer, and JSON report.
The prior artifact may expire, in which case it must be renewed from its
pinned, reviewed source; never silently substitute a different model.

## Evaluation and interpretation

- Select checkpoint by **validation response-token loss**, recording the
  initial (step 0), best and last values.
- The **test** split is not loaded or tokenized until after checkpoint
  selection, and cannot influence the optimizer or selection.
- The same held-out test rows are then scored for loss on base, best and
  last. A small deterministic cross-family subset is greedy-generated for
  base and best, then sent to `javac` with `-proc:none` in a temporary dir.
- **Compile-only**, not behavioral correctness. No model-generated Java is
  executed. Java code can still invoke unsafe operations when executed;
  `javac` is not a security sandbox. Run this only on an appropriate trusted
  development/CI environment.
- A successful tiny SFT experiment does **not** establish robust instruction
  following, project editing, generalization to unseen real-world Java, or
  safety of generated code. Test results have small sample sizes and
  family/template distribution shift.

This stage does not alter the user's existing Java project, does not train
1B/3B models, and does not use a GPU. The source-code license of SagaLLM
remains an owner decision.

## Measured GitHub Actions SFT (2026-11-22)

GitHub Actions [run 44982940959](https://github.com/SagaK-dev/SagaLLM/actions/runs/44982940959)
completed the actual CPU fine-tune, using exactly the prior pinned v0.16
weights and tokenizer. See `experiments/java_instruction_v017_results.json`
for a machine-readable summary and the uploaded Actions artifact for the full
training report and checkpoint weights.

| Measure | Real result |
|---|---:|
| Optimizer steps completed | 650 |
| Distinct SFT examples used | 2,600 |
| Best validation-selected checkpoint | Step 100; 400 examples |
| Train rows fitting 192-token context | 16,518 / 17,080 |
| Validation response loss (base → best) | 3.6726 → 1.7723 |
| Held-out test response loss (base → best) | 4.2338 → 2.0855 |
| Greedy Java compile-only test (base → best) | 0 / 8 → 1 / 8 |
| Exact source matches (base → best) | 0 / 8 → 0 / 8 |
| Model-generated Java behavioral execution | NOT_RUN |

**Interpretation:** response prediction improved, but autonomous Java coding
ability did not. The best checkpoint is not the final-step checkpoint.
Only 400 new examples contributed to best-selected weights; 2,600 were
processed by the final-step weights. Do not equate those two totals.