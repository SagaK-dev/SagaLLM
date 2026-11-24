# M3 parameter CE paired experiment (preregistered)

PR #49 diagnostic run 45374793097, SHA 8825779b00c96b247a3fef5d9a432ef270d77f90,
artifact 14911170944, ZIP SHA-256 ffc0c21fc07002303de1383fc8dd28da3d3ce6e85b6d2c8da4539b97bb92f2b3,
report SHA-256 b222a4d7ea8149b30f73c159459ee774a097902298f612ec2df4888d654fcc68.
Both downloaded hashes verified independently. Six unit tests and CPU CI
passed. All seven models were 0/20 TRAIN and 0/3 exposed regression
both-choice pairs; first divergence diagnostic, not Java accuracy.
All 160 disjoint TRAIN pairs (320 directions) admitted; manifest
4d9b2d6eb7ff216b653953eda1f355c4cae6def22d8a9af31016e51c7daa7d0b.

## Frozen design before training

Original v0.18 base/tokenizer and previous TRAIN corpora, 3 seeds
4201/4202/4203, 160 steps each arm, AdamW 2e-5, weight decay .01,
clip 1, batch 2 broad + 2 stdio. Same-family adjacent variant pairs
000/001 through 030/031; identical order and auxiliary graph in each
paired seed, coefficient 0 vs 1, no V2 ranking loss. The schedule differs
from V2; do not claim the cross-study difference isolates the objective.
Both directions predict the first unequal reference token using two-way
cross entropy, common reference prefix fixed. EOS excluded. Token
boundary can contain punctuation; not full literal copying.
Pin exact pair manifest, TRAIN bytes, prior and new suites before loading
weights. Selection by response NLL only at 0/80/160 on the existing
24 validation rows 004–011. Finish all arms before any diagnostic scoring.

New six synthetic development prompts / 18 cases plus authored
reference source: examples/java_m3_parameter_holdback.json, SHA-256
99c3de5b31128ecd7aa167014a3e736d883132a530ada2bad453e08f6d77317a.
New compositions of existing concepts (negative counts times a constant,
adjacent absolute differences plus offset, space replacement character).
Known semantic overlap with previous arithmetic/count/ASCII templates;
exact new IDs, task texts and family labels disjoint, not independent.
Python oracle recomputes all 18 cases. References are used ONLY after
checkpoint selection for two-way diagnostic (preflight tokenization
only, no gradients). Fresh exact I/O remains NOT_RUN until reviewed VM.

Report both-choice counts, signed interactions, source SHA diversity
and stops for baseline and all six arms; old regression and previous
development remain exposed diagnostics. Research screen, not a release
gate: at least 2/3 paired seeds improve both-choice count on the new
3 pairs vs control, without reducing fresh unique-source count.
If screen fails, report NO_CONSISTENT_PARAMETER_SIGNAL. If it passes,
report FORCED_CHOICE_SIGNAL_ONLY; require separate actual exact I/O
gates on reviewed disposable VM before any behavioral claim. No model
promotion, independent human review #36 remains pending.
