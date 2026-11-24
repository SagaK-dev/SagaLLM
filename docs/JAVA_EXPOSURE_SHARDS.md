# M2.4 — checkpoint-bound local exposure shards

The existing sagallm-doc-shards-v1 and uint32 token shards serve training.
This is a separate byte-preserving scan format for **declared** exposure
JSONL, including pretraining, train, validation, test and development.
It never tokenizes, deduplicates, modifies or drops input rows.

Use the private inventory and provenance formats in JAVA_PRIVATE_INTAKE.md.
Supply actual checkpoint and tokenizer files and their independent SHA-256
pins; these are hashed as opaque bytes and never deserialized. The supplied
training code SHA is a declaration only. The exact source inventory is pinned.
Keep private material and generated shards off shared CI and public Git.

Prepare (replace placeholders with real hashes and paths):

    PYTHONPATH=src python scripts/java_exposure_shards.py prepare \
      --inventory /private/inventory.json --expected-inventory-sha256 INVENTORY_SHA \
      --checkpoint /private/model.pt --expected-checkpoint-sha256 CHECKPOINT_SHA \
      --tokenizer /private/tokenizer.json --expected-tokenizer-sha256 TOKENIZER_SHA \
      --training-code-sha CODE_SHA --exposure-root /private/exposure \
      --output /private/shards

The output directory must not exist. Record the returned manifest SHA in
the private evidence ledger. Each source is partitioned in original byte
order, including newline convention and final line without newline.
A failed preparation removes its staging directory.

Audit (replace placeholders):

    PYTHONPATH=src python scripts/java_exposure_shards.py audit \
      --inventory /private/inventory.json --expected-inventory-sha256 INVENTORY_SHA \
      --checkpoint /private/model.pt --expected-checkpoint-sha256 CHECKPOINT_SHA \
      --tokenizer /private/tokenizer.json --expected-tokenizer-sha256 TOKENIZER_SHA \
      --suite /private/suite.json --expected-suite-sha256 SUITE_SHA \
      --provenance /private/provenance.json --expected-provenance-sha256 PROVENANCE_SHA \
      --shard-root /private/shards --expected-manifest-sha256 MANIFEST_SHA \
      --output /private/shard-audit.json --measure

The auditor validates inventory binding, exact file set, ordered names,
individual SHA/size/row counts and the reconstructed source SHA using the
same bytes consumed by the existing lexical/concept screen. Any missing,
extra, reordered, symlinked or tampered file rejects the audit. Findings are
unioned across shards; no text, paths or code are stored in the audit report.
Only a declared inventory can be verified. Missing actual exposures or
unrecorded parent checkpoints cannot be detected automatically.

Limits: up to 64 original inventory files; 8 GiB total; 2,048 shards;
16 MiB or 20,000 rows per shard; 256 KiB per row; 2 MiB manifest;
32 GiB per opaque artifact. The similarity budget applies per shard.
No silent sampling. Optional measurement reports elapsed time, bytes/s
and peak traced Python allocations (not total process RSS); tracing changes
runtime. Tests use small synthetic fixtures only, not real large corpora.

The independent evaluator now requires this shard audit through the model freeze gate; see JAVA_MODEL_FREEZE.md. No training-lineage verification, human rights approval, semantic
independence or model capability is established. Review remains required.
