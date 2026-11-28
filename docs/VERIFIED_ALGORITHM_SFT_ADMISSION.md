# Verified algorithm SFT admission preflight

Reopens the exact #107 reference artifact 15257146273 and checks the 16 row digests, eight reference implementations and 96 stored oracle comparisons against deterministic source/case reconstruction. Then uses the pinned v0.17 tokenizer and the existing SFT prompt path to measure untruncated admission at contexts 192, 256, 384, 512 and 768. Checks prompt/source tokenizer roundtrips and exact task/source/lexical-signature overlap with canonical train, validation and TEST. TEST is read statically only; no tokenization, generation, evaluation or use in training. Lexical overlap does not establish semantic independence. Output is metadata only, not a new dataset or weights. The reference report is prior execution evidence, not a fresh Java execution. Any length exclusion or roundtrip failure must be resolved before adding training rows. Rights review and independent evaluation remain pending.

## Bilingual development prompts at 384 positions

The pilot's 384-position context admits 6 of the 16 original instructions, all
in English. The longest original requires 551 positions. Ten compact task
replacements keep the same eight Java responses and the existing SFT envelope.
The other six task strings retain their original bytes. Each compact candidate
has a separate ID. Neither the canonical corpus nor the training loader includes
these candidates.

| Language | Original fit | Compact fit | Longest compact sequence |
| --- | ---: | ---: | ---: |
| Japanese | 0/8 | 8/8 | 383 |
| English | 6/8 | 8/8 | 382 |

The graph instructions use positional input notation: `(u v)*m` denotes m
undirected edge pairs, followed by query vertices a and b. An unreachable
destination returns -1. Reviewers should check the compact bilingual wording
for clarity, including these conventions, before admitting rows to training.

### Reproduce the comparison

Run from the repository root with the dependencies for the admission workflow.
Download each ZIP from its GitHub Actions artifact ID and run:

    PYTHONPATH=src python -m scripts.verify_compact_algorithm_admission \
      --parent-zip parent-14738368209.zip \
      --reference-zip reference-15257146273.zip \
      --admission-zip admission-15256607793.zip \
      --output compact-admission.json

The #109 admission artifact is from run 46129012984 at
`12c78986737f0787475e08dbeeb6fc0c43bbe38b`. Its archive SHA-256 is
`ce28b14a82f1436a6c355cf81e2cf888884001d1114f7beb116ebed6add3d12b`;
the JSON member's SHA-256 is
`a91edd752e57c6e5a4392b808c5b1178388571599494c84947921ca17814f367`.
The compact JSON SHA-256 is
`b0d7596dba88acaa07735897e7ee5fac4af0e9e779914073cc9ad5cdf9c0dba1`.

The verifier checks exact inputs and labels for each position, including
BOS, prompt masking, all response tokens and EOS. Each row records its
before/after sequence length and task/source hashes. Static checks cover exact
tasks, sources and lexical signatures across canonical train/validation/test.
There is no held-out scoring or new Java execution. Compile and oracle evidence
belongs to the eight byte-identical #107 reference sources. Bilingual semantic
review, training rights, model behavior and independent evaluation remain open.
