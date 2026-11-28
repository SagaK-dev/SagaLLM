# M2.4-i — legacy report reconciliation and hosted run discovery

Independent follow-up based on PR #75 to keep the hosted-run discovery review separate from the input reconstruction in PR #76. Read the exact PR #61 commit `fe31880f6cb9bde71e01b22942964cb6df327186`, whose CPU CI 45562246410, archive CI 45562246402 and three-original verification CI 45562246397 succeeded. Pin its manifest and 229-run snapshot by Git blob ID; recheck the three original report SHA-256 values and the twelve primary manifest report pins. Never execute code from the legacy commit. No modification to either archive is required.

## Legacy scope

- Paired compile-only diagnostic, run 45007278383: 16 generation metadata records, identical eight tasks; v0.17 2/8, v0.18 0/8. No Java behavior run, no root-cause claim.
- Exploratory public behavior, run 45161749481: 19 historical attempted cases (10 v0.17, 9 v0.18), all output mismatch; only previously compiled outputs, not a matched full suite. Human review/rights and contamination risk remain unresolved. OCI metadata is not a security proof.
- STDIO semantic SFT, run 45164032104: 36 historical case categories, 0/18 both models; 160 declared steps, candidate SHA `8d955d6c61fd677bda80934811f62ef5547090d610f41d4f06449a4a7d4ea3e2`. Add the conservative declared source-pool scope to the future integrated model map; no checkpoint bytes or optimizer trace are checked here. Historical NONZERO_EXIT_OR_JAVAC_ERROR is not a compile-only outcome.

## Hosted metadata discovery

Read current accessible Actions metadata with a UTC creation cutoff and bounded pagination (1000 runs maximum). Require consistent page counts, unique IDs and valid creation timestamps; fail closed on incomplete pages. Compare old and current latest-attempt metadata, list missing old IDs, newly visible IDs, identity conflicts and changes, reconcile all 15 original report/run identities against the pinned historical snapshot. Infrastructure classification is not a proof that no model-related operations took place. Unknown workflow paths are explicitly review-required, and successful unarchived research runs are not assumed to have trained a model. Failed and cancelled runs remain in the review inventory. Ancestor runs 44981937649 and 44982940959 are separately identified as previously referenced, not original reports in this archive scope.

The read-only CI uploads a metadata-only legacy-run-discovery.json. No raw API pages, sources, stdout/stderr, weights, secrets or execution logs are uploaded. The inventory is not atomic or complete across all prior attempts, deleted runs, private/local experiments or real input exposures. No training, inference, new Java execution, independent evaluation or promotion.

## Acceptance and next steps

CI must verify the staged exact originals, the historic snapshot and actual API page coverage, then upload the metadata. Integrate the STDIO candidate and source roles only after review of both PRs; reconstruct historical candidate inputs at its own revision, prioritize successful research runs lacking a pinned original report, and review partial/failed attempts. Independently recheck artifact availability, preserve originals and verify provenance/rights before any model-specific freeze. Issue #22 remains open.
