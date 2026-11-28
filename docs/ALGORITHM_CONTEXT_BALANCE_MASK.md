# Algorithm curriculum context balance and loss-mask audit (M2.6-g)

Based on PR #109 head `12c78986737f0787475e08dbeeb6fc0c43bbe38b`. Its dedicated
run 46129012984 and CPU CI run 46129012938 succeeded. The dedicated artifact
15256607793 has ZIP SHA-256
`ce28b14a82f1436a6c355cf81e2cf888884001d1114f7beb116ebed6add3d12b`.
The historical result is 0/0/6/14/16 of 16 admitted at 192/256/384/512/768.
A 384-position subset can bias family, language and implementation coverage.

## New gate

1. Reopen the exact archive, verify ZIP hash and sole member, and reproduce
   the entire admission report from pinned parent/reference inputs.
2. Bind every candidate ID and source to the prior report and exact membership.
3. Independently reconstruct BOS, shifted input/labels, assistant-only loss
   mask and terminal EOS; compare with the actual SFT encoder. Verify each
   exact context boundary and the absence of truncation.
4. Report counts by language, family, complete bilingual implementation and
   complete family at all five tested context lengths, including excluded IDs.
5. Report the exact maximum sequence length and first complete tested context.
   Preserve metadata only.

The new workflow must pass its unit tests and upload
`algorithm-context-balance-mask.json` before its output can be cited as
new measured evidence. The archived 768-position admission is not evidence
of model context-extension feasibility. No context config is changed.
No partial subset or complete curriculum is authorized for training by this
audit. Static overlap is not proof of semantic independence. All candidate
rows remain unassigned; canonical TEST is inspected statically by the
parent replay only. No test tokenization, scoring, generated Java execution,
model loading, training, weight publication, independent evaluation or
rights clearance. Issue #22 remains open.

Next gate: review semantic family overlap and rights, then establish a
separate context-extension preflight (position capacity, parameter count,
memory and bounded deterministic CPU smoke test). Pre-register validation-only
paired comparisons before any new training; no model adoption based on
reference-code tests or internal validation alone.
