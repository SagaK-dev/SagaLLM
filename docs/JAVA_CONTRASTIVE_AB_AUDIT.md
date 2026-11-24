# Offline matched contrastive A/B evidence audit

The command below reads an already downloaded and extracted, immutable A/B
report. It performs **no network access**, model loading, training, inference,
Java compilation or generated-code execution.

    python -m scripts.audit_java_contrastive_ab --help

For a complete invocation with exact run ID, SHA and artifact details, see
[DEVELOPMENT_PLAN_2026_11.md](DEVELOPMENT_PLAN_2026_11.md).

## Checks

- Raw report bytes must match the **operator-supplied report SHA-256**.
- Input checkpoint/tokenizer hashes and pre-registered training exposures,
  heldback task IDs, seed and optimizer steps must be consistent.
- All three arms must report the **same six prompt IDs and 18 exact stdout
  test cases per model**. Counts of successes, unavailable cases and OCI
  invocations are independently recomputed from case categories.
- Unique generated-source hashes, within-family and cross-family duplicates,
  gate deltas and the non-deployment decision are recalculated.
- Unexpected raw Java source, raw stdout/stderr evidence, host execution,
  test contamination or falsely positive capability flags fail closed.
- Audit evidence is written to a **new** JSON path, without overwriting data.

## Scope and limitations

GitHub run ID, git commit SHA, artifact ID and ZIP digest are entered by the
operator. They are preserved explicitly as **operator-supplied, not remotely
authenticated** metadata: the tool does not fetch GitHub or verify the ZIP.
The GitHub artifact digest refers to the **archive**, not the inner
ab-report.json file. Verify both independently.

The tool's OFFLINE_REPORT_INVARIANTS_PASS result establishes **internal
consistency of that SHA-pinned report**, not independent human benchmark
validity, actual remote run provenance, Java correctness, OCI security, or
production eligibility. Synthetic validation remains exploratory. This tool
must not be used to promote a model.

Unit tests: python -m pytest -q tests/test_java_contrastive_ab_audit.py
