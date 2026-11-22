# Java instruction SFT with bounded RoPE context extension (v0.18)

**Purpose:** extend the exact 192-token, 1,722,048-parameter Java SFT model
from v0.17 to **384 tokens**, without changing its learned parameter count,
then perform *actual CPU instruction fine-tuning*. This is a small experimental
step, **not a long-context-capable production Java model**.

## Input provenance and constraints

- The source is the **validation-selected v0.17 SFT** checkpoint, trained by
  GitHub Actions run [44982940959](https://github.com/SagaK-dev/SagaLLM/actions/runs/44982940959),
  SHA-256 `a17268d80ff443e4a0347f3a29df433556506f13537c78ce282836fa8fe4e5f9`.
  This contains 400 new SFT examples at its step-100 selection, **not** all
  2,600 examples seen by that experiment's final checkpoint.
- The tokenizer is unchanged, SHA-256
  `186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe`.
  A different checkpoint or tokenizer **must not** silently replace either.
- RoPE has **no learned absolute-position embeddings**. Expanding only
  `max_position_embeddings` to 384 preserves every weight tensor, with exact
  initial logits for original-length prompts in deterministic CPU tests.
  **Position extrapolation is unvalidated beyond 192 tokens**; fitting more
  tokens does not imply correctly understanding them.
- Only an extension strictly greater than the original and at most twice
  the original, and at most 512 positions, is supported. The source model
  must be small and the model configuration must differ only by the context
  limit; all weights are loaded strictly.
- The data remains the **17,536 template-derived examples** (17,080 train,
  228 validation, 228 test) from the project-authored v0.15 family-isolated
  curriculum. Longer examples fit without silently truncating prompt or
  answer. Newly admitted examples are counted and reported.

## Reproduction

Install dependencies with `pip install -e '.[fast-tokenizer]'` (plus JDK 21 for
compile-only diagnostics). Retrieve the original v0.17 artifact and extract
its paths below. It is SHA-checked by the training script.

```bash
python -c 'from sagallm.java_diverse_corpus import build_java_diverse_corpus; build_java_diverse_corpus("java-context-data")'
python scripts/train_java_instruction.py \
  --base java-sft-base/java-sft-model/java-sft-best.pt \
  --tokenizer java-sft-base/java-sft-model/tokenizer.json \
  --dataset java-context-data --output java-context-model \
  --context 384 --steps 400 --batch-size 4 --eval-every 50 \
  --lr 0.00004 --max-seconds 420 --compile-cases 8
```

The new `.github/workflows/java-context-sft-cpu.yml` performs this experiment
on GitHub-hosted CPU with the exact v0.17 artifact, then uploads **best and
last checkpoints**, tokenizer and full machine-readable report. The prior
artifact's retention is limited; expiry means the run must be deliberately
reproduced from the pinned source, not silently swapped for another weight.
The original v0.17 workflow remains usable without `--context`.

## Evaluation protocol and limitations

1. Canonically regenerate and verify all data splits for integrity. **Only
   training families** supply gradients; **validation families** select the
   best checkpoint. The test split is not tokenized or scored until after
   model selection, though integrity verification reads its bytes.
2. Measure validation response-token cross entropy on a fixed deterministic
   family-interleaved sample, separately from batch training loss.
3. Evaluate source, best and final model on the *same* family-heldout test
   examples at 384 tokens. This is a fair within-experiment comparison, but
   **not directly comparable** to the v0.17 192-token test-loss sample, which
   may contain different admitted examples.
4. Compile **only** eight fixed heldout generations via `javac -proc:none`
   as a narrow syntax signal. No generated Java is ever run. Output matching
   is separately counted; **behavioral correctness NOT_RUN**.
5. Record steps, distinct actually-trained examples, number of task families,
   baseline and selected checkpoint hashes, context lengths, and each split's
   new-versus-original-context inclusion counts. Best model might have seen
   far fewer examples than final model.

**Unverified:** effectiveness outside the original 192-token range, independent
real-world Java coding, code editing, runtime behavior, safe execution in a
real OCI environment, GPU training, 1B/3B weights and source-license selection.