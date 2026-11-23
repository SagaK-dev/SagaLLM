# Externally sourced Java I/O intake — NOT approved as independent benchmark

**Status:** upstream candidate acquisition and mechanical case mapping only.
**No human exercise review, rights certification, semantic overlap clearance,
or Java model testing has been performed.** Keep this outside train/validation.

## Frozen upstream source

- Repository: https://github.com/exercism/problem-specifications
- Git commit: `4812c213a2647e2034ad5ac78155ce9265cdd171`
- Repository license: [MIT license at pinned commit](https://github.com/exercism/problem-specifications/blob/4812c213a2647e2034ad5ac78155ce9265cdd171/LICENSE).
- License copy retained in `third_party_notices/exercism-problem-specifications-LICENSE.txt`.
- Source/metadata blob hashes, original source attributions and exact canonical
  test UUIDs: `examples/java_external_io_sources.json`.
- Draft Java 17 console I/O adaptations:
  `examples/java_external_io_INTAKE.json`.

| Candidate | Canonical test cases | Original exercise source attribution (from Exercism metadata) |
| --- | ---: | --- |
| Hamming | 4 | Rosalind |
| RNA Transcription | 4 | Hyperphysics |
| Luhn | 5 | Wikipedia |
| Roman Numerals | 4 | Roman Numeral Kata |
| Allergies | 5 | Turing / JumpstartLab |
| Atbash Cipher | 5 | Wikipedia |
| **Total** | **27** | **6 exercise specifications** |

Important distinctions:

- Exercism's **repository** has an MIT LICENSE, which has been preserved.
  Whether each upstream source's wording/tests can be redistributed under
  the desired project terms still requires human rights review; metadata
  explicitly cites antecedent sources.
- These are **human-maintained, externally published** practice tasks, not
  a newly collected set of independent human coding requests; benchmark
  contamination or adaptation similarity cannot be ruled out.
- The proposed Java `Main` console wrappers and stdin/stdout mapping were
  **made by the assistant**. They are not Exercism's canonical Java-track
  interface and may differ in edge behavior. A human must review each.
- Exactly zero **family label** collisions with SagaLLM's known 110
  synthetic curriculum labels is a shallow machine check, **not** a
  semantic similarity review or proof of independence.
- The intake is explicitly `PENDING_HUMAN_REVIEW`, not automatically
  eligible for OCI execution. No model inference is performed in CI.

## Integrity checks

The audit verifies every canonical test UUID and its original
`input`/`expected` values using one immutable Exercism Git revision.
Source and metadata files must match their pinned *Git blob SHA-1*, and
the unreviewed Java console adapter output must match the selected original
case after deterministic conversion.

A no-network structure-only check can be run at any time:

```bash
PYTHONPATH=src python scripts/audit_java_external_io_intake.py \
  --root . --output external-java-io-structure-report.json
```

This reports `INTAKE_STRUCTURE_ONLY` and is **not** an upstream data
integrity verification. The CI uses the stronger source-backed mode and
downloads the pinned files, validates Git blob hashes and fails closed:

```bash
PYTHONPATH=src python scripts/audit_java_external_io_intake.py \
  --root . --upstream-root pinned-upstream \
  --output external-java-io-upstream-audit.json
```

When the downloaded files match, the status is
`PINNED_UPSTREAM_CASES_VERIFIED`; **rights and review remain pending**.
No generated Java is executed or compiled by this audit.

## Human sign-off steps (Issue #36)

1. Review exercise-level rights including antecedent source attribution and
   whether the adapted exercises and selected test vectors may be reused.
2. Review all 6 assistant-created Java stdin/stdout contracts against the
   published Exercism description and Java 17 expectations, including
   special cases (empty input, Unicode, overflow, punctuation).
3. Inspect the complete SagaLLM training/validation/test curricula and
   third-party code datasets for semantic overlap, not just family names.
4. Obtain TWO truly independent human review records, with verified
   identities, dated review links, clear scope, and any objections recorded.
5. Decide whether this openly published set is an *independent* benchmark at
   all. Its public exposure means it must not be described as contamination
   proof. Acquire different hidden tasks if an unbiased external test is needed.
6. Freeze a final SHA-256, model checkpoint, tokenizer, evaluation and
   generation settings before any generated-code behavioral scoring.
7. Only on a dedicated disposable Linux VM with separately inspected
   isolation controls, consider the explicit OCI opt-in from
   `docs/JAVA_INDEPENDENT_IO_BENCHMARK.md`. Do not run untrusted model
   code in GitHub Actions. Never claim correctness from a compiler result.

The manifest/JSON fields must **not** be changed to approved status just
because the source blobs match. Upstream authorship, licensing, semantic
overlap and actual reviewer attestation remain unverified.
