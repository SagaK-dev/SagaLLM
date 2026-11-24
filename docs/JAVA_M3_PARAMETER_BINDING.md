# M3: parameter binding diagnostic and prospective intervention

Follows the frozen V2 replay in PR #48 and the evidence-first plan in #45.
The replay run 45374203447 succeeded with 84/84 source hash matches;
artifact 14910008423, report SHA-256
2676251393c7d589aabd50cc651229f4bc6576dec88be048a13dd295d6e2ef71.
The V2 all-criteria gain gate cannot pass on the frozen outputs because
only one seed improved regression diversity. Do not promote.

## Frozen diagnostic (before new scores)

- Load exactly the seven prior checkpoints and tokenizer; first reproduce
  all 84 original outputs, including EOS and lengths.
- Use 20 same-family TRAIN pairs (variants 000/001, 014/015, ten families)
  and three already-exposed regression pairs (002/003, three families).
  No original TEST or external human-review suite. Fresh development
  outputs are used only for prior collision counts, not new targets.
- For each pair, tokenize both canonical reference programs, find the first
  unequal non-special token after the same reference prefix. Under each
  instruction, score both candidate tokens with that reference prefix held
  fixed. The tokens can include adjacent punctuation due to BPE.
  Both-choice correctness means both prompts strictly prefer their own
  alternative token; ties fail. Interaction is the sum of the two signed
  logit margins, positive meaning an average shift in the expected direction.
  A positive interaction does not imply both choices correct.
- Compare every model, not a selected favorable seed. Report TRAIN and
  regression separately. This is a teacher-forced two-way diagnostic, not
  free generation accuracy, full literal copying, Java correctness,
  or an independent test. Prompt lengths can differ; no causal claim.
- Preflight 160 disjoint same-family TRAIN pairs (000/001 through 030/031),
  320 directions. Exact pair-token manifest SHA, no raw tokens/source in
  report. Fail if a prompt/target cannot be distinguished or does not fit.

## Prospective M3 parameter-only loss (NOT_RUN)

If the diagnostic supports investigating literal binding, compare ordinary
response SFT vs SFT plus symmetric two-way CE at the first differing
reference token. For each same-family pair (a,b), hold preceding response
tokens fixed and minimize -log softmax([logit_a, logit_b])[correct].
Both arms compute the same auxiliary graph, coefficient 0 vs 1, margin
ranking from V2 not added. Reuse original base, 3 seeds, 160 updates,
2 broad + 2 stdio TRAIN rows per step. Change only auxiliary coefficient
within this new paired comparison; schedule differs from V2 and must not
be described as a direct single-variable comparison with V2. Preserve
all canonical TRAIN exposure and pin schedule and masks before weights load.
Select only by the separate validation loss slice. Require a newly frozen
synthetic development holdback not used in previous diagnosis/selection
before training; original regression and previous fresh suite now exposed.
Predefine exact I/O evaluation on separately reviewed disposable VM/OCI;
no automatic promotion. Existing diagnostic metrics are not a release gate.
Do not optimize on the regression labels or reuse exposed suites as
independent evidence. Training/evaluation of this proposal is NOT_RUN.

M2 remains blocked by real human task provenance/review under #36.
No new generated Java compilation/execution, no GPU scale-up.
