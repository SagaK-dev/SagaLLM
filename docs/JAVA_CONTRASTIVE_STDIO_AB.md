# Prompt-conditioning collapse in SagaLLM Java: matched contrastive A/B

## Why this is the next step

In [PR #42](https://github.com/SagaK-dev/SagaLLM/pull/42), a 160-step
stdin/algorithm/stdout finetune of pinned v0.18 reduced synthetic response
validation loss from **2.26339381 to 0.36067117**, but all **18/18**
validation I/O cases still failed on BOTH baseline and candidate.

Reviewing the immutable structured evidence artifact (ID
`14813711672`) revealed an important problem: **different instructions
repeatedly yielded byte-identical generated Java**, as verified by SHA-256.

- Baseline v0.18: **3 unique Java source SHA-256 values among six distinct
  validation prompts**. The same output was used for `min` and one
  `count-odd` prompt.
- Candidate: **2 unique values among six distinct prompts**. Four `min`
  and `count-odd` prompts generated one identical program.
- This is a reproducible **prompt-conditioning warning**, not proof of
  the exact underlying cause. Failure could also involve limited model
  capacity, tokenization, insufficient context/output length, synthetic
  template memorization, or optimizer dynamics.

## Controlled intervention

The new experiment runs **two complete SFT arms** from the EXACT original
pinned v0.18 weights and tokenizer. Both use the same 160 training steps,
4-example batch, 2e-5 learning rate, the same two original-broad TRAIN rows
at each step, and the **same 320 distinct new stdin TRAIN rows**. The
single change is the order/pairing of those stdin examples:

1. **Control:** existing interleaved-by-family order from PR #42.
2. **Treatment:** group *two different algorithms* in a minibatch at
   the **same input-parameter variant** (e.g. add vs multiply, max vs
   absolute gap, uppercase vs reverse), with an otherwise unchanged
   training exposure budget.

All pairs are enumerated **before inference**, and the code proves that
the two arms see an equal multiset of all training examples. This
intervention tests whether contrasting operations within the same batch
improves instruction-specific generation, not whether a larger corpus
generally helps.

## Checkpoint selection and behavioral verification

- **Validation loss** is measured at steps **0, 80, 160** on 24 fixed
  examples from the project-authored disjoint-family validation corpus.
- Six *other* validation prompts (indices **2 and 3** for each of the
  min, odd-count and lowercase-line families) are excluded from this
  study's loss checkpoint selection, and reserved for **prospective
  18-case exact stdin/stdout behavior comparisons**.
- The previous study used behavior prompts **0 and 1**. Both slices
  still belong to the same previously used synthetic **validation
  families**, so this is **not** an independent unseen benchmark.
- The base model, control checkpoint and treatment checkpoint run the
  same six prompts and 18 test inputs. Each generated program is
  compiled and executed only inside **networkless, non-root,
  read-only, resource-limited OCI on a fresh GitHub-hosted VM**.
  A trusted Java standard-input smoke must succeed before any model
  generated Java is run.
- Observations include behavioral correct count, fully correct tasks,
  distinct source hashes (6 prompts), *cross-family* identical-source
  pairs, *within-family* identical-source pairs, loss trajectory and
  OCI status. Source text, raw input and stdout/stderr are never
  persisted.
- A *research validation signal* requires **at least +2 correct I/O
  cases** compared with BOTH base and interleaved control, no reduction
  in fully-correct tasks, and more unique generated sources than
  control. Even a positive result **does not automatically adopt or
  deploy weights**.
- Public Exercism external tasks, the original TEST split and any
  already used model-generated answers are **not training or
  validation inputs**.

## Running the experiment

The owner-branch/manual GitHub Actions workflow
`.github/workflows/java-contrastive-stdio-ab.yml` regenerates
the canonical and synthetic corpus, verifies the pinned model,
runs the equality/split unit tests, trains both arms and executes
the three-way comparison in a disposable VM. There is deliberately
**no main-branch push trigger for executing generated Java**, and
no fork-PR permission. OCI is best-effort isolation, not a formal
proof of container escape resistance.

The model remains at the **original pinned v0.18** in production and
the prior optional deterministic stdio generator remains a separate,
explicitly **non-LLM** feature. This is a bounded research iteration,
not evidence of general Java programming capability.
