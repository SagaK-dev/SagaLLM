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
