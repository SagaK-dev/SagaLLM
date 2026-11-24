# M3 parameter CE outcome and prompt-signal diagnostic

Run [45377422360](https://github.com/SagaK-dev/SagaLLM/actions/runs/45377422360),
job 138387289566, commit fdf53625d8e64596748113b53fd6f4c63e2a1c74, artifact 14912381648.
Downloaded ZIP SHA-256 068d733b1a1059a1e904ed8931467a85aaf9f5a16968a6837ce2fb4b604bb447;
enclosed report SHA-256 26f143575842073c9b4119ecefcb3b50935891bba6c4c9fc60b0b0c8353a89b7.
All six checkpoint SHA-256s matched after download; four tests passed
and CPU CI run 45377422275 succeeded. 320/320 directional rows,
960 updates total, all selected step 160. All 126 original generations
stopped by EOS. No compile or behavior measured.

| seed | control NLL | parameter CE NLL | fresh interaction C/CE | fresh both correct C/CE | fresh unique C/CE |
|---|---:|---:|---:|---:|---:|
| 4201 | .417440 | .567531 | .003919/.007740 | 0/0 | 2/3 |
| 4202 | .403462 | .546280 | .003951/.007349 | 0/0 | 2/1 |
| 4203 | .401110 | .539905 | .002201/.010302 | 0/0 | 3/3 |

Frozen screen from #50: zero of three seeds passed;
NO_CONSISTENT_PARAMETER_SIGNAL. Do not retune on this fresh suite.
The interaction shifts in the expected direction but does not overcome
wrong-token preference. Regression/previous/new suite outputs are now
exposed. No model promotion.

## Next diagnostic (fixed before inference)

Instead of another objective sweep, test the descriptive hypothesis that
prompt-dependent differences in the next-token distribution become smaller
as an identical teacher-forced reference prefix grows. Use the previous
20 TRAIN probe pairs (10 families, variants 000/001 and 014/015) unchanged;
all seven parameter-CE study checkpoints and fixed tokenizer. Replay all
126 generation hashes, EOS and lengths first and verify six checkpoint
hashes, schedule, selection and counts, then check first-divergence
margins against the original report within 1e-4. No optimizer or Java.

Measure centered-logit RMS and Jensen-Shannon divergence (nats, full
vocabulary) at floor(0/25/50/75/100% of common reference prefix length).
Zero is the position just after the instruction, 100% is just before
the first divergent reference token. Compare equal-length prompts
(primary descriptive stratum) separately from unequal-length prompts
(position confounded), and report all pairs. Check causal gathering
against a truncated forward, and log per-pair/aggregate values and
lower-at-end counts. If equal-length stratum empty report NOT_AVAILABLE.
This is not a causal ablation or a test of generalization; changing
prediction contexts changes token distributions, so JS differences
can reflect entropy/context as well as instruction signal. Centered
logits remove additive shift only. No root-cause claim, no significance
claim, no generated source or raw token sequences persisted. All new
diagnostic inputs come from already-exposed TRAIN, no independent suite.

M2 real human review (#36) and reviewed disposable VM remain blockers.
No new scale-up or training until diagnostic reviewed; independent
benchmark and root cause NOT_ESTABLISHED, behavior NOT_RUN.
