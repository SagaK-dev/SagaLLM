# Independent Java I/O evaluation — proposed protocol

The file `examples/java_independent_io_DRAFT.json` contains **three
assistant-authored, unreviewed proposal tasks** with nine I/O examples. It is
NOT an independently sourced, human-reviewed, licensed benchmark, and it does
not demonstrate model coding ability.

## Implemented

- Bounded schema checks and SHA-256 of the entire suite.
- A review gate requiring independently sourced tasks, rights/provenance,
  train-overlap review, and TWO distinct reviewer records.
- A separate explicit OCI execution gate; absent an opt-in, recognized
  review records, precommitted SHA-256, an acknowledgement of a disposable
  Linux VM, and a locally installed JDK image, execution is `NOT_RUN`.
- One fresh OCI container per I/O case, exact stdout comparison, read-only
  source and stdin input mount, no network, no privileges, resource limits,
  bounded output and timeout. Existing OCI restrictions are NOT a security
  guarantee; do not run untrusted code on a non-disposable host.
- No automatically executed generated Java in CI. Mocked tests exercise
  the protocol. No model checkpoint is loaded or retrained.

## How to obtain a genuinely independent benchmark

1. Source *new externally human-authored tasks* with documented permission
   and provenance. Do not rephrase SagaLLM's 110 synthetic families and call
   them independent.
2. Ask TWO distinct people to review the wording, expected I/O, corner
   cases, permission, and training/validation overlap. Record dates and
   traceable review links. The structural validator cannot independently
   authenticate reviewers or legal permissions.
3. Replace the draft with those reviewed tasks. Set
   `origin=EXTERNALLY_SOURCED_HUMAN_TASKS`,
   `review_status=APPROVED_FOR_INDEPENDENT_EVALUATION`,
   `train_overlap_review=ATTESTED_NO_KNOWN_OVERLAP`, and supply a valid
   `license_id`, `provenance_url` and reviewer records. Simply changing
   these strings without genuine reviews is insufficient.
4. Freeze and SHA-256 pin the exact benchmark JSON BEFORE any model
   inference, scoring, or test-result inspection.
5. On a dedicated disposable Linux VM with locally audited Docker/Podman
   and a preinstalled JDK image, place model-generated single-file
   `Main` classes at `<candidate-dir>/<task-id>.java`.
6. Run with separate, explicit approval flags and the exact frozen SHA.
   Never expose the OCI engine socket or sensitive host files to generated
   programs. Inspect the isolation environment before opt-in.

## Usage

Probe without execution:

```bash
python scripts/evaluate_java_independent_io.py \
  --suite examples/java_independent_io_DRAFT.json \
  --candidate-dir /nonexistent \
  --output benchmark-probe.json
```

Expected: `NOT_RUN` with `EXPLICIT_OPT_IN_REQUIRED`.

Only with independently reviewed tasks on the disposable VM, and a new
report destination:

```bash
sha256sum independently-reviewed-java-io.json
python scripts/evaluate_java_independent_io.py \
  --suite independently-reviewed-java-io.json \
  --candidate-dir model-generated-java \
  --output independently-reviewed-results.json \
  --allow-execution --acknowledge-disposable-vm \
  --expected-suite-sha256 <exact-64-character-sha256> \
  --backend docker --image <preinstalled-jdk-image>
```

The placeholders must be replaced with real, reviewed values. No Java is
executed without all gates. Results do not store Java sources, raw stdout,
stderr, or input/output test contents. Unavailable OCI and missing candidates
are reported separately and never counted as demonstrated failure or success.

## Limitations

`ELIGIBLE_SELF_ATTESTED` indicates only structurally complete supplied
review records, **not** independent validation. The protocol always labels
`benchmark_independence_proven=false` and bars automatic model selection.
Only stdin/stdout single-file tests are supported; the evaluation is not
comprehensive correctness verification. The proposed draft has not been
executed against SagaLLM, and no Java quality improvement is established.
