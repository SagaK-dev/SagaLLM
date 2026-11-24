# Model-specific independent evaluation freeze (revision 5)

Before inference, a real coordinator must freeze the exact suite, provenance,
inventory, shard manifest, model lineage and generation settings. The tool
checks bytes and declarations, not authorship, chronological authenticity or
whether an artifact was actually produced by a training run. Do not fabricate
a stage or mark the history verified. Legacy raw intake remains available
offline, but it cannot authorize evaluation.

Model manifest schema sagallm-java-model-freeze-v1 (exact keys):
schema, suite_sha256, provenance_sha256, inventory_sha256,
shard_manifest_sha256, tokenizer (path, sha256), stages, generation,
evaluator_sha256, oci_image. Paths are relative to the private artifact root.
Each stage: kind (pretraining first, then continued_pretraining or sft),
checkpoint_path, checkpoint_sha256, parent_sha256 (null only for first),
training_code_sha_declared (40-character commit), evidence_path,
evidence_sha256, source_indices (zero-based entries in pinned inventory).
All declared train/pretraining entries must be assigned; all stage
checkpoints and evidence bytes must exist. Evidence is opaque and not
authenticated. v1 requires a common tokenizer, a linear chain, max eight
stages, and no duplicate checkpoint hash. Missing real historical records
mean do not evaluate. Evidence files capped at 16 MiB.

Generation keys: seed, max_new_tokens (1..2048), decoding (greedy),
prompt_path (relative artifact path), prompt_sha256 (checked exact template
bytes), generation_code_sha_declared. Prompt template limit is 128 KB.
Get the current evaluator module fingerprint:

    PYTHONPATH=src python -c 'from sagallm.java_model_freeze import evaluator_sha256; print(evaluator_sha256())'

The fingerprint covers named evaluation modules, not the entire environment.
No model is loaded. The freeze does not prove generation used these settings.

After inference, pin a separate JSON with schema
sagallm-java-candidate-freeze-v1, model_freeze_sha256 and candidates (object
mapping every task ID to exact Java bytes SHA-256 or null for missing).
The evaluator checks all pins and reads the checked candidate bytes once
before engine probes. A missing declared candidate stays unavailable;
unexpected or changed source stops the entire evaluation.

Use the existing independent evaluator with the suite/provenance/inventory
pins and reviewed VM/image, plus:
    --model-freeze /private/model-freeze.json
    --expected-model-freeze-sha256 MODEL_FREEZE_SHA
    --shard-root /private/shards
    --artifact-root /private/artifacts
    --candidate-freeze /private/candidate-freeze.json
    --expected-candidate-freeze-sha256 CANDIDATE_FREEZE_SHA

All paths and SHA placeholders must refer to real private artifacts. The
model and all parent checkpoints are hashed as opaque bytes; the final
checkpoint and tokenizer are tied to the shard manifest. Do not mount any
private corpus or artifacts into generated code. If complete approved data
cannot be safely provided in the disposable VM, do not execute.

Results are machine preflight only and do not prove inventory completeness,
training lineage, inference provenance, pre-inference freeze timing, or
independence. No automatic selection or promotion. Shared CI uses mocks only.
