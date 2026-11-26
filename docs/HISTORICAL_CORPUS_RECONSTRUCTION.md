# Historical corpus reconstruction

PR #64 historical archive verification and CPU CI succeeded (45609067042, 45609067036). This follow-up reconstructs v0.16 public and v0.17 synthetic corpora using the exact historical SagaLLM commits and four pinned upstream commits. v0.18 declares the same synthetic manifest. The original builders and validators run before comparison against the exact archived manifests, all six split hashes and four license notice hashes. No Java execution, training, or raw corpus upload. Temporary files are removed. A mismatch or unavailable artifact fails the job. CI pending.

Even a successful byte match does not prove optimizer exposure, cross-corpus semantic independence, development-probe completeness or legal clearance. Keep M2.4 open; next inventory all development probes and compare cross-corpus overlap. Independent evaluation NOT_RUN; no promotion.
