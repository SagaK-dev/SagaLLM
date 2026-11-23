# Java stdin, algorithm and stdout improvement experiment

The previous public (unreviewed) Java behavior study in
[PR #41](https://github.com/SagaK-dev/SagaLLM/pull/41)
showed that **four generated programs that passed `javac` all exited 0,
but matched zero out of 19 expected I/O outputs**. A compiler pass alone
was not evidence of correct Java solutions.

This stage implements **targeted additional training**, not another
benchmark-prompt patch, and evaluates only *new project-authored synthetic
validation data*. The six previously seen public Exercism tasks and the
existing historical Java test split are neither scored nor used to tune
the model.

## New executable-instruction curriculum

The immutable older corpus contained mostly literal values embedded in
the answer code. The new curriculum asks for code that instead reads
**runtime stdin**, computes an algorithm, and writes **exact newline-
terminated stdout** with `System.out.println`.

| Split | Families | Parameter variants | Purpose |
| --- | --- | --- | --- |
| TRAIN | 10 | 32 each; **320 total** | Optimizer input only |
| VALIDATION | 3 different families | 12 each; **36 total** | Loss selection and fixed behavioral check |
| TEST | None | 0 | Do not access or recreate existing heldout test set |

Train families: add, multiply, max, absolute difference, count greater
than threshold, sum of even values, digit sum, uppercase ASCII line,
reverse ASCII line, clamp integer, each using stdin and parameterized
offsets/bounds.

Validation families: **min of two integers**, **count odd values** and
**lowercase ASCII line**, each under its own disjoint family ID.
Exact Python oracles generate three stdin/stdout cases per validation
task, and the trusted project-authored Java reference is compiled with
`-proc:none --release 17` and executed to check all oracle examples
in unit tests.

These examples are **small, synthetic and partially related** in
algorithmic concepts. Disjoint family labels alone do not prove
independence or general Java competence.

## Fixed SFT protocol

- Start from the original SHA-verified v0.18 best model, 384 context
  and original pinned tokenizer; the existing model stays unchanged.
- Fixed seed 4201, 160 AdamW optimizer steps, batch 4,
  learning rate 2e-5, weight decay 0.01.
- Each batch uses **two** original TRAIN examples sampled across all
  original train families and **two** new stdin TRAIN examples.
  No training-example reuse within either source pool.
- Evaluate loss on 24 fixed, disjoint-family new VALIDATION examples at
  steps **0, 80 and 160**. Choose the lowest-loss step if improvement
  is at least 0.005, otherwise retain step 0.
- Only after checkpoint selection, compare base and selected weights
  using **six** fixed validation prompts (two per heldout family),
  **three** runtime inputs each — 18 exact stdout cases per model.
- Generated Java, if structurally plausible, compiles and runs
  exclusively inside a **new networkless, read-only, non-root Docker
  container per input** on a disposable GitHub-hosted Ubuntu VM.
  A trusted Java stdin/out smoke must pass first.
- Preserve only loss, compiler/runtime categories, case counts,
  model/checkpoint hashes and any candidate weights. Never preserve
  generated Java, raw stdout/stderr or private test sources.
- No automatic weight promotion, even if the candidate improves.
  A conservative *validation-only* signal requires loss improvement
  >=0.005, at least +2 correct I/O cases, no fewer wholly correct
  tasks and a functional OCI runtime.

OCI is best-effort isolation, not a formal sandbox security proof.
The targeted validation data is project-authored and may be easier than
unseen human tasks. Results are an engineering diagnostic, **not proof
of broad Java competence**.

## Execution

The workflow
`.github/workflows/java-stdio-semantic-sft.yml`
runs on the owner PR branch `experiment/java-stdio-semantic-curriculum-42`,
or through an explicit manual dispatch, **not on main push or fork PRs**.
It downloads the pinned v0.18 model, regenerates and verifies both
canonical TRAIN data and the new curriculum, runs unit tests and trusted
references, trains the model, then evaluates both weights using isolated
Java behavior. All test and public benchmark splits remain untouched.

Recreate only the new deterministic curriculum without running any
model-generated Java:

```bash
python - <<'PY'
from sagallm.java_stdio_curriculum import build_curriculum, verify_curriculum
print(build_curriculum('stdio-java-corpus'))
print(verify_curriculum('stdio-java-corpus'))
PY
```

Only evaluate generated Java with the workflow's opt-in flags and
disposable-VM confirmation. Do not run this on a personal workstation.
