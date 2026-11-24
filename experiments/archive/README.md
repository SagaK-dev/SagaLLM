# SagaLLM experiment ledger — 2026-11-24

Main baseline 961130d684a459ee513e96d6be17511cb47315c8. This ledger is separate from unmerged research PRs
#45–#53; it does not merge their code or change production weights.
Run inventory: [actions-inventory.json](actions-inventory.json), 177
GitHub Actions runs before this PR, 154 successful,
9 failed, 14 cancelled,
0 other.
Latest attempt metadata only, not proof of training or behavior.
Six historical result files already exist in main's experiments directory.
The [manifest](manifest.json) pins 12 reviewed report members;
[reports/](reports/) contains original JSON bytes after the archive workflow
succeeds. No raw logs, generated Java, raw stdout/stderr, data dumps or model
weights are copied into Git. A ZIP hash is not a report hash.

| Stage | Run | Observed outcome | Decision |
| --- | --- | --- | --- |
| v0.18 context SFT | 44994208434 | 400 steps, 1600 examples, best at 50; validation 1.77098→1.76250; compile 2/8→0/8 | trained CPU; behavior NOT_RUN |
| paired compile | 45007278383 | 2/8→0/8 compiled; 16/16 EOS | regression, not Java behavior |
| stdin semantic SFT #42 | 45164032104 | 160 steps, val 2.2634→.3607; 0/18 exact before/after | no validated behavior gain |
| deterministic CLI #43 | 45165680758 | 13 operations, 39 exact tests of deterministic implementation | not an LLM result |
| matched contrastive #44 | 45167198398 | 0/18 exact for all 3 arms, unique hashes 3/2/1 | NO_VALIDATED_CONTRASTIVE_GAIN |
| M1 conditioning #46 | 45369765716, 45369987553 | 6 prompts round-trip; base/candidate unique 3/1; historical source hashes matched | root cause NOT_ESTABLISHED |
| M3 alignment V1 #47 | 45372165311, 45372372592 | initial failure; diagnostic 194/320 directions, 282 token positions, 0 optimizer steps | BLOCKED_ALIGNMENT_ADMISSION |
| M3 ranking V2 #47 | 45373241422 | 320/320, 3767 positions, 960 steps; NLL worse 3/3 seeds; diversity gain vs control 1/3 | no promotion; behavior NOT_RUN |
| M3 frozen replay #48 | 45374203447 | 84/84 hashes/length/EOS reproduced | inference only |
| parameter binding #49 | 45374793097 | 0/161 both-choice pairs across 7 models, 320/320 train directions admitted | parameter binding deficit on exposed probes |
| parameter CE #50 | 45377422360 | 960 updates, NLL worse 3/3, fresh both-choice 0/3 per model | NO_CONSISTENT_PARAMETER_SIGNAL |
| prompt signal #51 | 45378356573 | 126/126 generation replay; RMS lower 134/140, JS lower 0/140 | no simple signal-loss conclusion |
| activation swap #52 | 45386333879 | 133 model-pairs, 5852 conditions, 3192 checks | positive layer-1 mean, negative layer-2; no unique root cause |
| attention bias #53 | 45389874625 | 42/42 control generations replay, 1/7 models improve vs required 5/7; regression both-choice 0/3 all | NO_CONSISTENT_TRAIN_PROBE_SIGNAL |

Run links: https://github.com/SagaK-dev/SagaLLM/actions/runs/ followed by run ID.
Full exact report coverage is limited to the manifest; remaining historical
runs have metadata and existing documentation only. Older result archives
and all attempt/job logs have NOT been comprehensively inspected or
copied. Missing report is not a failed experiment. Artifact retention
is finite; hash evidence alone is not a permanent checkpoint backup.

## Required practice for subsequent experiments

1. Record question, frozen protocol, arms/seeds, data/tokenizer/checkpoint
   identities, acceptance rules before inference or optimization.
2. Record GitHub run/job/attempt, commit SHA, CI status, artifact ID,
   separately the ZIP digest and original report byte SHA, negative and
   failed/stopped outcomes, and unmeasured statuses.
3. Inspect report for source, raw stdout/stderr, secrets, private data,
   and rights restrictions; only approved report JSON members enter this
   manifest. Never copy arbitrary ZIP members or raw logs.
4. Add manifest row and a human-readable result summary in a reviewable
   evidence PR; verify exact bytes before committing. Run
   `python -m scripts.archive_experiment_reports --manifest experiments/archive/manifest.json --output experiments/archive/reports --verify-local`
   after storage. For future reports use an updated reviewed manifest
   and run archive command with Actions read access. No overwrite of
   existing evidence; corrections are new records, linked to prior run.
5. Preserve unresolved or expired evidence as REPORT_NOT_YET_ARCHIVED
   or BLOCKED_EVIDENCE_UNAVAILABLE, not invented figures. Checkpoints
   require a separate reviewed immutable artifact store and rights/cost
   decision; not silently committed as multi-MB Git blobs.
6. Keep workflow success, teacher-forced metrics, compile, exact I/O and
   independent behavior separate. No automatic model promotion.

## Next engineering decision after #53

Stop the failed fixed multiplier branch; do not sweep exposed probes.
The repeated matched negative results justify prioritizing M2 external
human task intake and M4 isolation readiness rather than another
unvalidated SFT or architecture sweep. Read the [handoff](../../docs/JAVA_M2_M4_HANDOFF_2026_11.md).
