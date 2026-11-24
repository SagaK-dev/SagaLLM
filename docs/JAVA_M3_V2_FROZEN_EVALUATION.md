# M3 V2: completed CPU experiment and next evaluation gate

Training run [45373241422](https://github.com/SagaK-dev/SagaLLM/actions/runs/45373241422),
job 138372591811, commit babcbe130905d4acc7e4617497f0ebcea68569a9, artifact 14909169047.
Downloaded archive SHA-256 2d4d6bf653c394ddf75016b26099efcf2fc5eb09ec1edd1086d7baedbedcc97f;
enclosed report SHA-256 90c8b9bb12e381c415e798d88875fc401143a0c703a3b0424cfab6c2283932cd.
All six checkpoint SHA-256s independently matched report after download.
11 unit tests passed; 320/320 rows, 3767 lexical operation positions,
five pair types, 960 total optimizer steps. All arms selected step 160.
All 84 original generations stopped by EOS. PyTorch 2.15.0+cpu.

| seed | control NLL | rank NLL | control regression margin | rank margin | regression unique C/R | fresh unique C/R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4201 | .399780 | .415776 | .046649 | .051732 | 2/2 | 3/2 |
| 4202 | .383798 | .403821 | .035289 | .044838 | 1/1 | 3/2 |
| 4203 | .382232 | .414112 | .043712 | .045315 | 1/3 | 3/4 |

Ranking worsened validation NLL in all three seeds and improved the old
edit-v1 regression margin in all three. Regression source diversity
improved in just one seed vs paired control, and never exceeded base (3).
The 4203 ranking regression outputs differ across families but all three
same-family variant pairs have identical hashes. No semantic correctness
can be inferred; root cause NOT_ESTABLISHED.

## Next: frozen replay then gated synthetic behavior

Use scripts/evaluate_java_m3_v2.py with the exact training archive, base
and tokenizer. The offline auditor validates archive layout and SHA,
report pins, exposure/selection counts, all six checkpoint hashes, and
recounts source collisions. Regenerate the canonical regression oracle
bytes (SHA pinned), verify the new holdback bytes, and replay all 84
generations; require every source SHA, token length and EOS to match before
any OCI execution. No generated source is saved. The default is inference
only, with compile and behavior NOT_RUN. The CI workflow never opts in.
Report deterministic upper bounds by grouping identical source hash plus
stdin and counting mutually incompatible expected stdout; only conditional
on deterministic output, not a performance measurement.

Operator-only optional stage: separately reviewed disposable isolated
Linux VM, with locally installed reviewed image identified by sha256 image
ID. No CI, no remote Docker, no host Java execution. Review record must
contain actual reviewer, review_date, review_url, vm_id, image_id,
report_sha256, archive_sha256, regression_oracles_sha256,
fresh_suite_sha256, and true values for disposable_linux_vm,
no_secrets_or_persistent_mounts, restricted_oci_reviewed,
approved_synthetic_java_execution. These are human attestations, not
automated certification. Review the actual checkout, host, OCI isolation,
image, absence of secrets and disposal plan. Hash the review record;
do not generate or fill it automatically. The operator supplies explicit
--allow-execution --acknowledge-disposable-vm --review PATH
--review-sha256 SHA --image sha256:IMAGE_ID only after that review.
Run in a dedicated VM that will be destroyed afterward.

The opt-in CLI performs trusted smoke, then separate restricted OCI
compile-only checks before behavior for each source; at most 84 source
compiles and 252 exact I/O cases (each behavior case recompiles in its
fresh OCI). The second compile can fail, recorded in the nonzero combined
category. Failure to compile counts as zero correct but is not execution.
No source/raw stdout/stderr saved. A conservative explicit interpretation
of the prior gate requires all regression gain, no full-task regression
relative to base/control, diversity gain relative to control, and fresh
gain vs control in at least two paired seeds. Any unavailable case makes
the study inconclusive. Since only one seed showed diversity gain,
this conjunction cannot pass on the frozen outputs. Do not revise the
gate to rescue the study. Actual behavior remains unmeasured.

Example default (no Java execution):

    python -m scripts.evaluate_java_m3_v2 --archive m3.zip \
      --base java-sft-best.pt --tokenizer tokenizer.json \
      --holdback examples/java_m3_development_holdback.json \
      --output frozen-replay.json

M2/#36 remains independently REVIEW_REQUIRED. This is synthetic M3,
not M4 or an independent benchmark. No GPU scale-up or production promotion.
