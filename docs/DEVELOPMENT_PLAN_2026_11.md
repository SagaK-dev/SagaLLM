# SagaLLM Development Plan — 2026-11-24

Baseline main commit: 961130d684a459ee513e96d6be17511cb47315c8 (merged PR #44).
This plan is a controlled research roadmap; implementation and evaluation status must be recorded separately.

## Evidence and current decision

- Issue #22: the original v0.18 CPU SFT **was executed** in run 44994208434; 400 optimizer steps, 1,600 distinct examples, best selected at step 50, using the pinned v0.17 input. The recorded best v0.18 checkpoint SHA-256 is 9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc and tokenizer SHA-256 is 186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe. Source: experiments/java_context_v018_results.json.
- Held-out v0.18 compile-only: baseline 2/8, v0.18 0/8; exact reference 0/8. Compile-only is NOT behavioral correctness.
- PR #42: 160-step targeted stdin SFT lowered synthetic validation loss from 2.2634 to 0.3607 without improving exact behavior (0/18 before or after).
- PR #44: [Actions run 45167198398](https://github.com/SagaK-dev/SagaLLM/actions/runs/45167198398), commit 40dcce6fe2e4bd2ab83ea70cf5a9cd12368c0312, succeeded on 2026-11-23. Log-level summary: 0/18 exact cases for baseline, interleaved and contrastive; source diversity was 2/6 for interleaved and 1/6 for contrastive. Decision NO_VALIDATED_CONTRASTIVE_GAIN. NO MODEL PROMOTION.
- Artifact: sagallm-contrastive-stdio-matched-validation-ab, ID 14815096856; GitHub-reported **ZIP archive** digest sha256:f1716f3111bd8df1bf63282f1970b9b9aadf340aeb18f5133e66a5ea9d55955e. This is NOT the SHA-256 of enclosed ab-report.json.
- Independent human-reviewed Java benchmark: REVIEW_REQUIRED / NOT_RUN per [Issue #36](https://github.com/SagaK-dev/SagaLLM/issues/36). No general Java-coding ability is established.

Primary hypothesis: prompt-conditioning collapse (different instructions yield identical outputs). It is a hypothesis, NOT a root-cause finding.

## Milestones, in dependency order

| Priority / provisional window | Work and deliverable | Acceptance gate | Status |
| --- | --- | --- | --- |
| M0 / 11-24 to 11-27 | Reproducible evidence audit, pinned report-byte hash, integrity/tamper unit tests, run provenance ledger | Six matched prompt IDs, 18 cases/model, correct collision recount, SHA pin and non-promotion, fail closed on errors | IMPLEMENTED / LOCAL_AUDIT_PASS; GitHub CI and review pending |
| M1 / 11-27 to 12-04 | Root-cause diagnostic matrix for generated-code format, tokenization, compile errors, source-hash collisions and prompt encoding | Same fixed seed, inputs and candidates; a falsifiable single-variable hypothesis pre-registered | NOT_RUN |
| M2 / 12-01 to 12-15 | Collect 5–12 truly independent human-authored Java 17 I/O tasks; rights/license provenance and two distinct real reviews per task | Frozen task and I/O suite SHA, reviewer links/dates and family-overlap clearance before inference | BLOCKED_ON_HUMAN_REVIEW (#36) |
| M3 / 12-05 to 12-22 | One-variable-at-a-time interventions: instruction representation, generation termination, task conditioning and modest capacity/data changes | Equal exposure and compute, no test leakage; full exact-stdout, distinct source-hash and unavailable-case analysis | NOT_RUN |
| M4 / 12-15 to 01-05 | Run approved independent suite on disposable reviewed VM/OCI only | Opt-in, image digest, no network, non-root, resource limits; compile + exact stdout, corner cases and operator attestations | NOT_RUN |
| M5 / 01-05 to 01-20 | CPU/GPU cost and rights reviewed pilot; 1B/3B resource feasibility | Pilot independent metrics + rights gate first. No scale-up or claimed GPU training without approval and physical evidence | NOT_AUTHORIZED |
| M6 / 01-20 onward | Release hardening, inference safeguards, documented rollback, status/release evidence | Explicit independent evaluation, security review and artifact provenance for any production model promotion | NOT_READY |

Windows are planning targets, not promises of external reviews, GPU allocation or metric improvement.

## Rules for every experiment

1. Record workflow URL/run ID/commit SHA, model/tokenizer/data manifests and report hashes **separately**; report SHA and artifact archive digest are not interchangeable.
2. Verify train/validation/test family disjointness; no test examples in optimization or hyperparameter selection. Independent tasks are frozen before scoring.
3. Never execute generated Java on the GitHub Actions host or a local non-disposable workstation. Generated Java requires a separately reviewed disposable Linux VM and explicit gated, isolated OCI.
4. Report only hashes, bounded diagnostics, status categories and exact counts; do not publish generated source, raw stdout/stderr, or secrets.
5. Preserve TRAINED_CPU, NOT_RUN, NOT_ESTABLISHED, INCONCLUSIVE and REVIEW_REQUIRED labels. Workflow success and validation loss are NOT model-quality claims.
6. Keep production v0.18 weights unchanged until independent evidence and owner review authorize promotion. Do not automatically adopt any candidate.

## Immediate M0 checklist

- [x] Read Issue #22, v0.18 results and PR #44 implementation/workflow, inspect run 45167198398 log.
- [x] Specify an offline report consistency auditor and adversarial tests; no Java execution.
- [x] Retrieve artifact 14815096856; extracted ab-report.json SHA-256 = 1f59f98acafc5b2939f5dc043717beb3b76dd5ff923506064377ca8a7221c468; independently recount totals and collisions.
- [x] Verify extracted curriculum manifest SHA-256 = 0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13; candidate checkpoint SHA-256 = 379d57d6d8668995dc34fb310e0c5140041c5fd3e78b77932712785e96b01eae; both agree with the downloaded report. No unsupported environment versions inferred.
- [x] Run offline auditor against extracted report and exact SHA: OFFLINE_REPORT_INVARIANTS_PASS / NO_VALIDATED_CONTRASTIVE_GAIN. It still says NOT_VERIFIED_OFFLINE for operator-supplied run/archive metadata; GitHub run and artifact were inspected separately.
- [ ] Review CI and PR; update README/STATUS/changelog and close Issue #22 only after every independent acceptance item is traceable and actually satisfied.

Example (replace the report SHA placeholder only with actual extracted-file checksum):

    python -m scripts.audit_java_contrastive_ab --report /path/to/ab-report.json \
      --expected-report-sha256 REPORT_FILE_SHA256 --run-id 45167198398 \
      --run-sha 40dcce6fe2e4bd2ab83ea70cf5a9cd12368c0312 \
      --artifact-id 14815096856 \
      --artifact-digest sha256:f1716f3111bd8df1bf63282f1970b9b9aadf340aeb18f5133e66a5ea9d55955e \
      --output /path/to/new-audit.json

The auditor checks local report bytes and internal structure; it does not authenticate GitHub on the network. Report-file and downloaded archive SHA-256 have now been verified and the local invariant gate passed; see experiments/java_contrastive_v018_ab_evidence.json. This does not amount to independent training reproducibility. If artifacts expire or evidence is missing, record BLOCKED_EVIDENCE_UNAVAILABLE instead of estimating or fabricating metrics.

## External technology references (evaluation candidates, NOT integrated)

Tokenization: huggingface/tokenizers, llm-jp/llm-jp-tokenizer.
Data provenance and curation: huggingface/datatrove, NVIDIA-NeMo/Curator, huggingface/fineweb-2 (rights review mandatory).
Model/training architecture: karpathy/nanocha, allenai/OLMo-core, NVIDIA/Megatron-LM, pytorch/torchtitan, huggingface/transformers, huggingface/accelerate, allenai/open-instruct.
Testing design: UKGovernmentBEIS/inspect_ai. None are claimed to be installed, trained, or integrated by this plan.

## Primary references

- https://github.com/SagaK-dev/SagaLLM/issues/22
- https://github.com/SagaK-dev/SagaLLM/issues/27
- https://github.com/SagaK-dev/SagaLLM/issues/36
- https://github.com/SagaK-dev/SagaLLM/pull/42
- https://github.com/SagaK-dev/SagaLLM/pull/44
- https://github.com/SagaK-dev/SagaLLM/actions/runs/45167198398
