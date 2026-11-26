# Historical exposure ledger (M2.4-e)

PR #65 reconstruction passed (45770654618); PR #66 overlap workflow passed (45772294170). A successful overlap workflow is not evidence of zero matches.

This ledger verifies the 12 archived report bytes against the existing archive manifest and indexes explicitly named data digests and reported task/record IDs, including nested diagnostic results. IDs are report-local evidence, not globally unique task identities or proof of optimizer exposure. It does not recover raw prompts, infer missing IDs, or certify completeness. The activation-swap report was not readable through the connector during implementation; CI reads the actual repository bytes and must fail if unavailable or mismatched.

Known gaps: exact optimizer sample order per checkpoint, full raw development probes and selection suites, unarchived/local experiments, rights review and checkpoint-specific source inventory. Next reconstruct the declared stdio/pair/holdback datasets at historical revisions, compare their pins and map them to the existing exposure-shard schema. Independent evaluation NOT_RUN; no model selection or promotion. CI pending.
