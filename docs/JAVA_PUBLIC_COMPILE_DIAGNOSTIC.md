# Public Java compile-only diagnostic — unreviewed exploration

This experiment is intentionally **separate** from the independent, human
reviewed Java I/O benchmark tracked by
[Issue #36](https://github.com/SagaK-dev/SagaLLM/issues/36).

User requested progress without human review. Consequently, this experiment
**does not mark any reviews as completed** and does **not** relax
`src/sagallm/java_independent_benchmark.py` review/OCI execution gates.

It freezes 6 existing public Exercism-derived task IDs and checks the
ability of two *original pinned* model checkpoints (v0.17 / v0.18) to generate
a syntactically compilable Java 17 `Main` class. Only `javac --release 17
-proc:none` is used, never `java`. Every generated source file is
temporary and not retained. Error diagnostics are allowlisted categories;
source and compiler raw text are never stored.

Important limitations:

- The tasks were publicly available before model inference. They may be in
  training data. **This is not an independent unbiased benchmark.**
- To fit a small 384-token model context, we freeze **assistant-abridged
  prompts**, not the exact published problem statements. Different wording
  can change what a model generates.
- **Compilation is not functional correctness**. The 27 original input/output
  examples are *not scored*. `behavioral_status=NOT_RUN`.
- Six problems are too few for a reliable broad performance claim. Do not
  tune model weights or decoding parameters using these public results.
- The reviewer requirement still applies to any formal independent
  benchmark or rights/semantic contamination claims.
- The compiled Java programs are *never executed* in GitHub Actions. Javac
  compilation of generated source is not equivalent to a strong OS sandbox.

Run via GitHub Actions workflow
`java-public-compile-diagnostic.yml` using both pinned checkpoints and
tokenizer. The output includes only per-task source hashes, compilation
statuses and error categories, plus explicit flags `java_executed=false`,
`model_selection_permitted=false` and
`benchmark_independence_proven=false`.

**No new checkpoint is trained or promoted**, regardless of results.
