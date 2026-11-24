# M1 — distinguish generation failures from task-conditioning failures

Development plan: PR #45 (hosted CPU CI 45368673638 succeeded on
83f62d332d90ae0e318b82371aaea1f17af12f40; review/merge remains pending).
This branch is independent of that draft and does not change model weights.

## Observed historical results (not a new execution)

From SHA-256-verified ab-report.json in run 45167198398 / artifact 14815096856:

| Arm | Nonzero exit OR javac error | Zero exit, incorrect stdout | Exact stdout |
| --- | ---: | ---: | ---: |
| base v0.18 | 18 | 0 | 0 |
| interleaved | 8 | 10 | 0 |
| contrastive | 8 | 10 | 0 |

Each has 18 OCI invocations on the same six synthetic validation prompts.
The combined nonzero/javac category does not distinguish compilation from
runtime errors. Zero exit with mismatched stdout is not correctness.
Cross-family identical source pairs were 2, 4, 12 respectively. The result
is consistent with a conditioning problem but does not identify its cause.

## Fixed diagnostic, no training and no generated-code execution

The new workflow loads only the pinned v0.18 and contrastive candidate
(checkpoints and tokenizer SHA-256 fixed in the script), and verifies the
historical report. The control checkpoint was not saved in the prior
artifact; only its historical counts are available. Never substitute weights.

- Tokenizer: verify six distinct prompt token sequences and round-trip,
  BOS, exact context admission, reference and prompt lengths.
- Generation: same 170-token cap, greedy decoding, same six IDs (variants
  002/003); record EOS vs limit, token counts, hashes, historical hash parity,
  and cross-family common token prefixes.
- Shape: fixed literal marker booleans, not a parser or proof of compilation.
  No source, identifiers, prompt text, token sequences or raw output saved.
- Conditioning: teacher-forced mean reference NLL under its own prompt
  and under the next family's same-variant prompt. Hold target tokens fixed.
  Positive swapped-minus-own NLL is a conditioning signal, not correctness.
  Prompt lengths may differ, and the absolute scores are not a causal test
  of attention, capacity, tokenizer quality or data sufficiency.
- New javac and behavior NOT_RUN; no old TEST, public benchmark or optimizer.

If historical source hashes differ, report the mismatch; do not interpret
the new generation as a reproduction of the prior I/O outcomes.

## Preregistered next single-factor experiment (not executed here)

H1: the generation cap is an important cause of incomplete Java.
Freeze original checkpoint, tokenizer, six existing validation prompts,
greedy decoding and 18 historical I/O cases. Compare 170-token cap vs
full available context, only if M1 detects a cap stop. If all six outputs
end with EOS, reject the cap explanation for this model/prompt set and
do not spend an OCI run on this intervention. If cap stops occur, test
both cap settings on the same 18 cases in the separately approved
disposable VM + restricted OCI. Report EOS/limit, compile status, exact
stdout and source hashes; require +2 exact cases, no loss of fully correct
tasks, no unavailable evaluations. No automatic promotion. This is
exploratory synthetic validation, not an independent benchmark.

H2: weak prompt conditioning. Compare own/swap NLL and repeated hashes.
If prompts tokenize distinctly and still produce identical code, literal
token-ID collision is not the explanation for these prompts. Further
controlled training experiment must be preregistered after inspecting
the diagnostic; do not tune on old TEST or the public candidate intake.

H3: malformed Java identifiers/syntax. The historical combined error
category cannot establish this; fixed shape markers are preliminary.
If needed, use separate approved compile-only diagnostics with existing
allowlisted javac categories; never execute generated Java on a host.

## Reproduction and blockers

Workflow: java-m1-conditioning.yml; artifact: sagallm-m1-conditioning-inference.
The workflow will fail if either pinned artifact is expired or missing.
Record run ID, head SHA, report SHA and environment before claiming
new measurements. Until then inference is NOT_RUN.
Issue #36 remains REVIEW_REQUIRED: follow JAVA_EXTERNAL_IO_REVIEW_GUIDE.md.
The existing public candidate intake has been previously exposed and must
not be used for model selection; recruit two real independent reviewers and
obtain new held-back human-authored tasks where necessary.


## Verified result — 2026-11-24

- Workflow [45369765716](https://github.com/SagaK-dev/SagaLLM/actions/runs/45369765716), job 138360683835, exact commit 6f36acc0cfb856e82788660acef39829f91586b4; succeeded, ten diagnostic unit tests passed.
- Artifact 14907598947. Downloaded archive SHA-256:
  14c827a9d2424978201c32328bc8a547e109b84c86a0d106409de64703687bfa.
  Extracted report SHA-256:
  191a4699aeabff3a5bd36143aa3379170b0f51c8fcb0fe23aa68f7223e583e07.
  Both were independently hashed after download; the report digest matches the job log.
- PyTorch 2.15.0+cpu. Six unique prompt token sequences; round-trip checks passed.
  Prompt lengths 58/62/69 by family; reference lengths 87/117/97.
- Base: all six EOS (78–79 generated tokens including EOS), three distinct
  source hashes, six historical hash matches, no literal System.in marker
  in any source, mean swapped-minus-own NLL +0.00618227 (4/6 positive).
- Contrastive: all six EOS (80 tokens each), one distinct source hash,
  six historical hash matches, six literal System.in markers, mean
  swapped-minus-own NLL +0.00853705 (6/6 positive).
- The literal token-ID collision and generation-cap explanations are
  ruled out for this exact probe (not for arbitrary other prompts/models).
  The cap A/B is **SKIPPED_NO_CAP_STOPS** per the preregistered rule.
  Weak conditioning remains a plausible research hypothesis, not a
  demonstrated cause. Small NLL margins are descriptive, not a statistical
  proof; the diagnostic averages the whole reference, including shared syntax.
  Literal markers do not prove stdin behavior or syntactic validity.
- No new javac or Java runtime evaluation. The historical 0/18 scores are
  unchanged. Control weights not available. Root cause NOT_ESTABLISHED;
  production adoption false; independent benchmark NOT_ESTABLISHED.

See [machine-readable summary](../experiments/java_m1_conditioning_results.json).

## M3 candidate preregistration — explicit conditioning objective (NOT_RUN)

Do not repeat only the failed same-variant minibatch reordering. Hypothesis:
a same-target, wrong-prompt penalty on algorithm-specific TRAIN response
tokens may increase task-dependent generation. Use a paired experiment from
the original SHA-pinned v0.18. Control: ordinary response-masked SFT; treatment:
the same SFT plus an explicit conditional ranking term. Define same-variant,
different-algorithm TRAIN pairs before optimization. Keep optimizer schedule,
learning rate, training examples, order, batch size, context, and seeds
(4201, 4202, 4203) equal; include the same wrong-prompt forward computation
in control with auxiliary coefficient zero. For each paired TRAIN target,
compute mean NLL on precomputed response token positions where the two
algorithm references differ, using a deterministic token-sequence alignment
fixed before training (exclude EOS and unaligned/ambiguous cases; log
admission counts). Use hinge penalty max(0, 0.05 + NLL_own - NLL_wrong)
with coefficient 1.0 in treatment. Never use historical TEST, public
Exercism, or validation targets for gradients. If alignment/admission fails
or equal exposure cannot be proved, stop before training.

Before running, freeze a new synthetic development holdback disjoint from
optimizer and checkpoint selection and document any semantic overlap.
The six existing validation prompts and 18 cases remain fixed regression
diagnostics, not model-selection criteria. Use fresh heldback prompts and
exact I/O on an explicitly approved disposable VM + restricted OCI;
freeze IDs, hashes and selection rules before inference. Validation-loss
checkpoint selection only on a separate validation slice. Record each
seed, conditional margin on algorithm-specific tokens, EOS/limit,
unique source hashes, compile status, exact I/O, full-task passes, and
unavailable cases. Proposed research gate: >=2 additional exact cases on
the fixed 18-case regression vs control and base, no decrease in fully
correct tasks, improved distinct-source count, plus improvement over
control on the new holdback in at least two of three seeds. No automatic
model promotion. If no genuinely new holdback is available, report
BLOCKED rather than reusing exposed evaluation as independent evidence.
This is a preregistered experiment proposal; no training or behavior
execution has been initiated by this document. Independent human review
in M2/#36 remains a separate required gate.
