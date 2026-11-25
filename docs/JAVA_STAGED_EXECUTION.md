# M4.1 stage-specific outcomes (revision 7)

PR #60 final-head 619e96e01c975c0d3bdaf197f185b4342370e45b: matched report tests 45562297313,
archive verification 45562297289, CPU CI 45562297273 all successful.
PR #61 final-head fe31880f6cb9bde71e01b22942964cb6df327186:
legacy verifier 45562246397, archive verifier 45562246402 and CPU CI
45562246410 all successful. Still awaiting review/merge.

New independent evaluation path only requests staged execution with the
existing gated OCI launcher. Legacy exploratory runner defaults unchanged.
No real model or generated Java execution in these new tests.
The static shell maps compiler exit 1 to 40 (COMPILE_FAIL), other compiler
nonzero to 43 (COMPILE_UNAVAILABLE), setup to 42 (RUNTIME_ERROR),
Java command nonzero to 41 (JAVA_NONZERO_EXIT), and successful Java
command to 0 (PASS or OUTPUT_MISMATCH by exact stdout). Use javac
-proc:none --release 17, discard compiler stdout, do not treat any
generated stdout as control markers. Engine/unknown wrapper exits are
RUNTIME_ERROR (unavailable). Timeout and output cap are still distinct
but outer timeout cannot locate the phase. Counts are per I/O case
(compilation repeated per case), not unique compiled tasks.
For exit 41 we know the Java command did not return zero, not the precise
exception, signal, or cause. Compiler exit 1 is the compiler's error
category, not proof of why it failed. This assumes reviewed image and
trusted wrapper; process tampering and OCI are not formally secured.
No raw stdout, stderr or source in report. Existing VM gate remains
mandatory; no human approvals or actual M4 behavior claimed.

Comparison requires identical staged protocol in both arms, or both
legacy. Compile status totals are rederived from case statuses, and
not-run/unavailable are not successful or failed compilations.
Partial/unavailable means no full-suite behavior delta. A legacy
FAIL is not retrospectively resolved. Execution protocol and module
fingerprints distinguish historical results. New tests use only mocked
launcher and synthetic fixture reports, no real OCI/Java.

## Execution evidence

At b939cc2c1525869489448f2709171c0099864517, independent-protocol run 45573850033 passed 135 tests; matched-report run 45573849889 passed 35 tests; private-intake run 45573849921 passed 71 tests. Counts overlap; do not sum them as unique tests. The independent draft probe is NOT_RUN / REVIEW_REQUIRED. Artifact 15002852734 ZIP and member SHA-256 were checked after download. Exact bytes are preserved in experiments/archive/reports/45573850033-m4-staged-protocol-draft-probe.json. See experiments/java_staged_execution_results.json for CI snapshot. No actual independent execution or model promotion.
