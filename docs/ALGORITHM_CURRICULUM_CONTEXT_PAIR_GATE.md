# M2.6-g: Algorithm curriculum pair-aware context gate

## Evidence and decision

The final head of PR #109 (12c78986737f0787475e08dbeeb6fc0c43bbe38b) passed admission run 46129012984. The audit job 140929865340 uploaded metadata artifact 15256607793 (3358 bytes, ZIP SHA-256 ce28b14a82f1436a6c355cf81e2cf888884001d1114f7beb116ebed6add3d12b). The original report says 0/16 at 192 and 256, 6/16 at 384, 14/16 at 512, and 16/16 at 768. This is tokenizer admission, not model accuracy.

Do not silently select a language-skewed subset, truncate a response, expand the existing safety guard, or interpret tokenizer admission as runtime capacity. The original checkpoint context is 192; the current policy allows at most twice the origin and 512 absolute. 768 is blocked even for a hypothetical 384-context origin. This is not a claim that a 384 checkpoint has been promoted.

## Reproduction

The dedicated workflow downloads the exact three historical archives, verifies the admission ZIP and replays the entire prior reference/tokenizer/canonical overlap audit. It compares the complete report structurally, rechecks row membership, exact shifted token positions, paired source identities and static overlap. For each context it emits counts by language, complete and partial implementation pairs, complete two-implementation families, excluded IDs, and hashes of the complete-pair subset. No task or source text is emitted. The policy matrix invokes the existing guard with a synthetic small configuration, not a loaded checkpoint.

The output is algorithm-context-pair-gate.json in artifact algorithm-context-pair-gate-metadata. Record its run ID, job ID, final head SHA, artifact ID, ZIP hash and report member hash after successful CI. An expired or changed archive is a hard failure, not permission to substitute an unpinned input.

## Follow-up gates

1. Preferred next experiment: separately author compact fixed references and/or concise equivalent bilingual prompts targeting 384 positions, with explicit parent identity, no silent text transformations, renewed source compilation and oracle comparisons, exact tokenizer roundtrip, full bilingual pair admission and static overlap recheck. Keep all original constraints and algorithm requirements.
2. Alternative: separately review the 768 context policy, resource/attention budgets and prefix invariance, with a bounded synthetic architecture probe before any checkpoint experiment. Do not lift the 512 cap in this PR.
3. Before any new training, obtain rights review and pre-register a comparison on an unchanged internal validation set with paired seeds and equal update/exposure budgets, fixed starting checkpoint/tokenizer and exact trace replay. Do not access TEST for tokenization or scoring. Independent evaluation requires a genuinely independent, rights-reviewed benchmark.

No training, model weights loaded, new Java execution, semantic independence certification, benchmark evaluation, data publication or model adoption. Issue #22 remains open.
