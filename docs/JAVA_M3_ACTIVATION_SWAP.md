# M3 activation swap: fixed diagnostic protocol

Prior prompt-signal run [45378356573](https://github.com/SagaK-dev/SagaLLM/actions/runs/45378356573),
job 138390543325, commit 3b8dd807c39b83f0115cb0c2fc60c98467e20074, artifact 14911808494.
ZIP SHA-256 46b2df5cdb251e7abb3c11796c9cc0bc92402133474e1d56e9f0c0ee4253075f;
report SHA-256 535d10f4c9873f945040ddadfbadcd5041024878d59db6fda5203cda9038ab6e.
Both downloaded byte hashes independently verified. Six tests and full
CPU CI 45378356405 passed, 126/126 frozen generations reproduced in
that run. RMS smaller at end in 134/140 model-pairs; JS smaller in 0/140.
Equal-length stratum 19/20 per model. This does not establish signal loss.

## Protocol frozen before new scores

Use the same seven pinned checkpoints, the same 20 exposed TRAIN probes
and tokenizer, but exclude the single unequal-length pair from all swaps:
positions then align exactly. Verify previous diagnostics, CE artifact,
all weights, pair manifest and each clean margin (absolute tolerance 1e-4).
No fresh or independent evaluation input or selection. No new generation
replay in this study; the previous 126/126 remains historical evidence.

For the two prompts with an identical teacher-forced reference prefix,
run a batch of two uncached sequences. At each site swap the two rows'
activations at a specified region, leaving the rest unchanged. Both
directions measured in the same forward. The sites are token embedding,
four block residual outputs, four attention projected outputs and four
FFN outputs. Hook the existing modules without modifying model.py.
Regions: changed prompt-token positions, unchanged prompt-token positions
(contextual states there can legitimately differ), final common-response
position. Also all prompt positions at the five residual boundaries.
13 sites, 44 site-region combinations, 19 pairs x 7 models; record all,
not a favorable layer. Parameter differences may include BPE punctuation.

Report the raw two-choice logit shift in the donor's direction for each
recipient, mean of both directional shifts, clean and patched interaction,
and whether both recipients prefer the donor. No normalization by the
near-zero clean gap. Positive shift does not imply either choice is correct.
Full swap at residual boundary must reproduce opposite clean logits;
self-swap must be identity; changed embedding swap must reproduce donor,
unchanged embedding/final shared embedding must be identity, and final
block residual at prediction position must reproduce donor. These are
implementation controls, not evidence of localization. Check hooks
removed and clean logits unchanged after every pair. Abort on mismatch.

These interventional results apply only to the frozen models, probes and
chosen hybrid states; they may be off-manifold and cannot identify a unique
root cause, prove necessity/sufficiency of a layer in natural generation,
or establish coding ability. No new training, compile, behavior, independent
benchmark, GPU allocation or promotion. M2/#36 and reviewed disposable VM
still required. A later architecture/representation change needs its own
preregistered matched control, fresh synthetic holdback and behavioral gate.
