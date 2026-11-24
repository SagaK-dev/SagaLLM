# M3 conditional ranking: frozen training-only intervention

Follows the plan in PR #45 and the preregistration in PR #46. Both PRs remain
drafts. The latest M1 CPU and diagnostic checks succeeded. M2/#36 still
requires two genuine human reviewers; no independent benchmark approval.
This research branch is independent of both drafts.

## Frozen study design (before model inference)

- Original v0.18 SHA-pinned checkpoint/tokenizer, 384 context; three seeds
  4201, 4202, 4203; 160 AdamW updates per arm, batch four, LR 2e-5,
  weight decay 0.01 and norm clip 1.0. Each arm sees the same 320 distinct
  broad TRAIN rows and the same 320 stdio TRAIN rows in the same order.
- Both calculate the same response-only SFT loss and same auxiliary forward
  and backward graph; control coefficient zero, treatment coefficient one.
  Same-variant, different-algorithm pairs match the five existing pair types.
  Ranking loss is max(0, 0.05 + own NLL - wrong NLL). The target is identical;
  score only unambiguous substitutive tokens, excluding EOS and all positions
  with multiple possible partners/deletion in optimal unit-cost edit paths.
  A position is only a lexical algorithm-difference proxy, not a semantic label.
- Admission: all five pair types represented, at least 160/320 directional
  rows have a scoreable substitution; other rows still enter ordinary SFT.
  Freeze and hash alignment masks and per-seed exposure schedules before
  loading model weights. Abort before training if admission/parity fails.
- Canonical broad TRAIN-only regeneration matches pinned SHA
  181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d.
  Historical TEST records are not regenerated or read; family names only
  are inspected. Stdio manifest matches original pinned SHA.
- Selection: fixed 24 synthetic validation rows (variants 004–011);
  prior 000/001 and regression 002/003 excluded. Evaluate loss at 0/80/160,
  choose checkpoint with improvement >=0.005. Finish all six arms before
  any regression/fresh inference. No new suite inputs/targets in optimizer
  or checkpoint selection.
- Frozen six fresh project-authored synthetic development prompts / 18 I/O
  cases: examples/java_m3_development_holdback.json. SHA-256
  d641a53e6498a096fda15e92871e10db6cc2380fe1bb1be543c8f4a06aeb035a.
  Recompute each expected stdout in Python before training. Novel compositions
  (two smallest squares, adjacent increase threshold, digit/non-digit
  transitions) use previously seen concepts: minimum, square, loops, counting,
  comparisons, ASCII parsing. No exact prompt/family/ID overlap; broader
  semantic/pretraining overlap not ruled out. This is not an independent suite.
- Report selected checkpoints, per-seed loss, aligned validation margin,
  greedy generation EOS/limit and distinct hashes (both fixed and fresh
  prompts). Generated source is not saved. Save both arms' research weights
  with hashes, never auto-promote. Exact I/O and compilation are NOT_RUN.
- Behavior stage is separately blocked until explicit reviewed disposable
  VM/OCI approval. Future evaluator must pin both suites, both selected
  weights per seed, baseline, tokenizer and image digest. Required research
  gate follows PR #46: >=2 extra correct regression cases vs base/control,
  no fewer fully correct tasks, more distinct outputs, and improvement
  on fresh holdback vs control in at least two of three seeds; no unavailable
  results. This proposal cannot be called a behavior gain before actual run.
  Independent human review and production promotion remain separate.


## V1 measured admission failure and V2 protocol revision (before training)

Run [45372372592](https://github.com/SagaK-dev/SagaLLM/actions/runs/45372372592),
commit 6eb20c4d581c3aaaf5db4f91a9c4517d2c2bbaca, artifact 14909651235, ZIP digest
8f93dcdcf83ad6acc151d03950fab1ecbdf9193affb7d9fffd604919250df532,
report digest ed9e5b9fafa0b0c6c5c193ddb0ae852487346af1f145a680acc5896a5f5675d6
(from hosted log). V1 admitted 194/320 rows, 282 positions; pair coverage
64, 64, 2, 64, 0 in FAMILY_PAIRS order. The five-type gate failed and zero
optimizer steps completed. A successful reporting workflow was not a training pass.

V2 is a new preregistered lexical mask protocol, not a relaxation or a
successful rerun of V1. Keep V1 selectable and preserve all V1 evidence.
Use the uniquely identified operation expression or statement from each of
the ten canonical ASCII TRAIN templates (including structurally different
string methods). Match exact canonical task, source, variant and token IDs.
Reconstruct token boundaries by concatenating ASCII token decodings; require
exact equality to the source. Select all tokens intersecting the annotated
region, including boundary-crossing BPE tokens; no EOS or other special token.
This is a template-specific proxy, not a semantic parser; it may include
shared operands/punctuation. Unlike V1, it does not require the other
algorithm's source to have a corresponding substitution. The same target
and mask are used under its own and wrong prompt. Require all 320 directions
and all five pair types before loading weights; no dropped TRAIN row.
Keep all original seeds, schedules, optimizer, validation selection, hinge
margin, coefficients, holdback bytes and no-execution gate unchanged.
The old edit-v1 regression margin remains the fixed diagnostic metric,
not a model-selection criterion. No evaluation output was seen before this
revision: V1 never trained or inferred. V2 is not evidence of improvement
until its run completes and separately approved exact I/O is measured.
