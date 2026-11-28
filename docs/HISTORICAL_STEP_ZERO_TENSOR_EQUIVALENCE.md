# M2.4-q — selected step-zero tensor comparison

PR #82's semantic audit reconciled 34 checkpoint digests and eight step-zero output instances (six distinct serialized hashes). Historical reports and implementations indicate that both A/B arms selected the original model state. This does not itself establish tensor equality.

Pin the exact #82 metadata artifact, parent v0.18 ZIP and all four historical A/B artifact ZIPs. Verify each checkpoint member hash and embedded parent/arm/step metadata. Compare model-state tensor names, dtypes, shapes and raw-byte SHA-256 values, as well as configuration. Differences must be recorded, not silently ignored. The check does not infer optimizer-state equality or training lineage.

Unlike the earlier byte-only audits, this stage explicitly deserializes only fixed, previously verified repository checkpoints with modern CPU PyTorch weights-only loading, a static unsafe-global check, no custom allowlist, bounded archive/member/tensor sizes and a timeout. It does not instantiate a model, restore an optimizer, compile or execute Java, run inference or train. The token is removed from the audit process environment before loading; this is not a sandbox or security proof. The disposable hosted runner publishes metadata only, no model binaries or raw tensors.

Acceptance is actual comparison and reproducible metadata, not a predetermined equality claim. Independent evaluation, source rights and actual exposure remain unresolved. Issue #22 remains open.
