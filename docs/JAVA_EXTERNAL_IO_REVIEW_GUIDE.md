# Java external I/O benchmark — two-human review procedure

**Current status: UNREVIEWED, NOT_RUN, no benchmark approval.** This guide
explains the real human work that remains under
[Issue #36](https://github.com/SagaK-dev/SagaLLM/issues/36).
It does not invite the assistant to self-certify or claim real-world ability.

## Reviewer materials

1. Read the approved-source *candidate intake*, not a final benchmark:
   [candidate suite](../examples/java_external_io_INTAKE.json) and
   [upstream attribution / test UUIDs](../examples/java_external_io_sources.json).
2. Read [six-task overlap leads and known gaps](JAVA_EXTERNAL_IO_INTAKE.md).
3. Download the **`java-io-HUMAN-REVIEW-REQUIRED.md`** artifact from the
   successful GitHub Actions run named
   `SagaLLM pinned external Java I/O intake audit (NO EXECUTION)`.
   This is a GENERATED WORKLIST, **not signed or approved evidence**.
4. The artifact contains source attribution links, exact pinned source SHA
   and metadata SHA, the original selected test UUIDs, task-level coverage
   risks, and empty separate Reviewer A/B signoff fields.

## Review assignment and independence

**Two different real people** must each examine all six tasks independently.
The repository owner must recruit/assign these people; no reviewer identity
is currently known or verified. The same person using two accounts does not
satisfy the requirement. Both reviewers should have access to source
specifications, licensing and the synthetic SagaLLM task-family inventory,
but should **not** receive SagaLLM-generated solutions, compilation results
or behavioral scores before the benchmark is frozen.

Each person posts their findings in the GitHub issue or a linked signed
review document. A coordinator then checks reviewer identity, agreement,
scope, independence and source permissions. The assistant cannot
authenticate these records or give final human approval.

## Template for EACH reviewer

Copy this section into a separately attributable review record; fill with
real evidence, not an automatically generated verdict.

~~~text
Review of SagaLLM candidate external Java I/O suite
Reviewer identity / verifiable account: [real person]
Date and timezone: [actual date]
Review URL / traceable record: [link]
Pinned suite SHA-256:
d1af64a934c304f04b2cfa822280c73cb7da7b64d09e83f62b071b945a244c74

For EACH of the SIX task IDs:
- Task ID: [ex_hamming / ex_rna_transcription / ex_luhn /
            ex_roman_numerals / ex_allergies / ex_atbash_cipher]
- Provenance and credited underlying sources: [links verified, uncertainty]
- Task statement and case-vector redistribution rights: [license/permission evidence]
- Java 17 stdin/stdout adapter and expected I/O: [pass / concerns + examples]
- Important edge cases missing: [specific uncovered cases]
- Semantic overlap with train / validation / test: [findings + citations]
- Possible prior exposure in base model pretraining: [findings or UNKNOWN]
- Suitability as truly independent benchmark: [YES / NO / INDETERMINATE]
- Recommendation: [ACCEPT FOR REVIEWED SUITE / REVISE / REJECT]
- Notes, conflicts of interest and objections: [specific details]

I examined the actual source files and selected case UUIDs independently:
[explicit signed statement if true]
~~~

Do **not** replace `[real person]` placeholders with AI identities or infer
approval from CI passes. If the reviewer cannot verify rights or pretraining
contamination, mark that issue *unknown*, not cleared.

## Coordinator's conditions before any Java behavior study

- Both real independent reviewers provide task-specific dated traceable
  assessments, with identity and scope checked out-of-band.
- Rights and antecedent author credits are resolved **for the actual prompt
  and selected vectors**, not merely the Exercism repository MIT license.
- Proposed Java console input/output transformations and missing cases
  are reviewed and corrected; a corrected suite has a **new** SHA.
- Semantic overlap with the FULL synthetic task family inventory and any
  accessible underlying pretraining sources is assessed and documented.
  Public Exercism exercises may not be unbiased independent questions.
  Acquire new held-back human-written tasks when the required claim
  cannot be supported.
- Independent signoff, study design and secure disposable VM/OCI environment
  are verified by responsible humans, separately from this worklist.
- Freeze the actual reviewed suite (and model SHA, tokenizer, decoding,
  scoring configuration and image digest) **before** seeing model answers.
- Do not use the candidate intake for model tuning, checkpoint selection or
  capability claims; the automated evidence status remains
  `REVIEW_REQUIRED`, `NOT_RUN`, and `model_selection_permitted=false`.

## Generate locally

The review packet is a deterministic presentation of the existing frozen
intake, not the act of reviewing:

~~~bash
PYTHONPATH=src python scripts/build_java_external_review_packet.py \
  --root . --output java-io-HUMAN-REVIEW-REQUIRED.md
~~~

It refuses to overwrite previous files and fails if either the frozen
6-task suite SHA or provenance manifest SHA changes. For a revised
suite, use a separately named reviewed version and a new independent
review; do not silently edit the original evidence or bypass this gate.
