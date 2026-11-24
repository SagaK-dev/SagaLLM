# M3 first-layer prompt attention bias: fixed-weight pilot

Activation swap run 45386333879, job 138418094830, commit 0f15a7116315ea74089f2f1a29dd1e9899e89acc,
artifact 14915953140. Downloaded ZIP SHA-256
e428e95300195cfcde474592f721b9be45995789bbe51b31549cc780bd1c878d;
report SHA-256 3d8436e4cfee22aee4918a8718a5eb270d94f7a3af832adb724ff4691e47742c.
Both verified independently; 133 matched model-pairs, 5,852 site-region
conditions, 3,192 passing controls; six tests and CPU CI 45386333730 passed.
First-layer attention at the final response position moved two-choice
logit differences towards the donor on average in all seven models;
second-layer attention moved in the opposite direction. Not proof of a
unique causal bottleneck; hybrid states may be off-manifold.

## Frozen intervention before observing new scores

Research-only uncached forward reproduces existing first-layer GQA/RoPE.
Compare exactly the same forward with additive attention mask 0 versus
log(2) for non-BOS prompt keys, queries from final prompt position onward.
Only first layer; all heads; no oracle-selected token locations, no
changes to later layers, model weights, tokenizer, or model.py. This doubles
the unnormalized attention weights on eligible prompt keys, not the
normalized attention probability. Causal mask remains intact. No sweep.
Measure all seven original checkpoints, 20 exposed TRAIN same-family
pairs and 3 already-exposed regression pairs. Greedy generation of six
regression prompts per model/arm, same 170-token cap and vocabulary
limit. No new holdback or original TEST scoring. Before any treatment
runs, all zero-bias forced logits must agree with native within 1e-4,
prior margins must agree, and all 42 zero-bias uncached generations must
exactly reproduce the old cached outputs (hash, EOS, count). Abort on
any mismatch; no treatment score on partial replay. No source persisted.

Mechanism screen (TRAIN only, exploratory): at least 5/7 models improve
both-choice counts on fixed TRAIN pairs versus their zero-bias control.
Report TRAIN_PROBE_SIGNAL_ONLY or NO_CONSISTENT_TRAIN_PROBE_SIGNAL.
Models are not independent random samples, and TRAIN probes are exposed;
this is neither generalization nor a statistical significance claim.
Regression results descriptive only, never used to tune this screen.
No training or new Java execution. If promising, preregister a matched
training experiment with fresh synthetic holdback and separate behavior
gate before any weight change. If not, preserve the negative result
and do not sweep the multiplier on these probes. M2/#36 and reviewed
VM still block independent behavior and production promotion.
