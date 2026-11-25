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
