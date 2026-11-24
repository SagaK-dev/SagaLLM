# M2/M4 handoff — next gated stage

The M3 attention pilot (PR #53, run 45389874625) failed its predeclared
TRAIN screen: 1/7 models improved both-choice counts, required 5/7.
Regression diagnostic remains 0/3 in every model. Do not tune multiplier
on these exposed examples. Root cause NOT_ESTABLISHED. M3 research can
continue only with a new hypothesis and preregistered evaluation, not a
promotion claim.

Issue #36 requires 5–12 truly independent human-authored Java 17 I/O tasks,
real provenance and redistribution rights, two distinct real reviewers per
task, semantic overlap review, frozen suite hash before inference, then
disposable Linux VM/OCI authorization and second audit. The current
six-task public Exercism intake is already exposed and is NOT an
independent human-reviewed benchmark. See
[JAVA_EXTERNAL_IO_REVIEW_GUIDE.md](JAVA_EXTERNAL_IO_REVIEW_GUIDE.md),
[JAVA_INDEPENDENT_IO_BENCHMARK.md](JAVA_INDEPENDENT_IO_BENCHMARK.md),
and [Issue #36](https://github.com/SagaK-dev/SagaLLM/issues/36).

## Actionable owner handoff (no invented tasks or signoffs)

- Recruit a human author/coordinator and two different real reviewers.
  Commission 5–12 new privately held-back Java 17 console tasks, not
  derived from the synthetic families or the exposed development sets.
  Keep actual task text, I/O and solutions outside the public research
  repository until freeze/permission decision; do not send to model
  training or selection. Ask for provenance/revision, license or
  explicit permission, Java 17 contract, boundary/locale assumptions and
  independently verified expected stdout.
- Each reviewer records dated, traceable per-task review of rights,
  family/semantic overlap, cases, and independence. Coordinator verifies
  identities and approval scope separately. Public tasks with uncertain
  pretraining exposure remain indeterminate.
- Freeze the reviewed suite, model checkpoint/tokenizer, generation
  parameters and evaluator/image digest before inference; preserve the
  exact bytes and SHA. No fabricated hashes of a suite not yet received.
- Before generated code execution, human operator supplies real reviewed
  disposable VM/OCI attestation: no secrets/persistent mounts, networkless,
  non-root, limits, reviewed image digest, operator/date and review URL;
  destroy VM afterward. Existing shared runner is not authorized.
- Run independent compile and exact stdout evaluation only after all
  gates, document failures/unavailable cases and second independent audit.
  No GPU/scale-up or release until plan gates and owner approval.

Status: M2 BLOCKED_ON_HUMAN_REVIEW; M4 NOT_RUN;
M5 NOT_AUTHORIZED; M6 NOT_READY. This is a worklist, not a review or a
benchmark. The ledger and archive infrastructure may be reviewed/merged
independently of experimental code. No changes to README capability
claims or production checkpoint.
