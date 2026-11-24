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
