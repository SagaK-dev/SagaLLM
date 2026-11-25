# M4.1 matched report comparison — offline only

Follows revision 5 and PR #59. Compare two byte-pinned evaluation reports,
not arbitrary summary scores. Same suite digest, ordered task IDs/case counts,
tokenizer, evaluator module fingerprint, generation settings, image and VM
review digest are mandatory; checkpoints must be different. Recompute case
and task totals, candidate pins and canonical freeze-report SHA. Unavailable
cases (including missing candidates) are separate from failed cases; partial
comparisons have no full-suite delta or fully-correct-task counts. NOT_RUN
returns NOT_COMPARABLE. All other unmatched or inconsistent inputs fail closed.

Old evaluator FAIL combines compile/nonzero execution/output mismatch; this
comparator does not infer which occurred. Separate compile and runtime stages
remain a future M4.1 item. TIMEOUT, OUTPUT_LIMIT, REJECTED, NOT_RUN and
RUNTIME_ERROR retain their separate reported categories. Report trust rests
on reviewed provenance; structure, exact bytes and matching declarations
do not prove freeze timing, inference provenance, independence, complete
training exposure or isolation. No automatic model selection or promotion.

Run locally with real reviewed reports (not fixtures):

    PYTHONPATH=src python -m scripts.compare_java_independent_reports \
      --left /private/baseline.json --left-sha256 BASE_REPORT_SHA \
      --right /private/candidate.json --right-sha256 CANDIDATE_REPORT_SHA \
      --output /private/matched-report.json

Tests use fabricated metadata and outcomes, never actual human approvals,
models, generated code or Java. Do not publish private benchmark contents.
M2.5 remains BLOCKED_ON_HUMAN_REVIEW; actual M4 remains NOT_RUN.

## Implementation evidence

Run 45562031183 / job 139027777403: 16 synthetic tests passed at commit 4654247546d965b0401614d1d8bf548db45dba74. Archive verifier run 45562031220 passed, including the exact original NOT_RUN probe from #59. CPU CI was still running at this evidence snapshot. No real reports compared or model evaluated. See experiments/java_matched_report_comparison_evidence.json.

## Staged execution protocol

New reports declare `sagallm-java-oci-staged-exit-v1` and count compile phases per case. The comparator rejects mixing legacy and staged reports; it validates the staged counts independently. See [JAVA_STAGED_EXECUTION.md](JAVA_STAGED_EXECUTION.md). Legacy `FAIL` is never reclassified. Outer timeout/output limit and launcher errors do not establish the compilation stage.
