## v0.18 Java context extension (2026-11-22)

- IMPLEMENTED / LOCAL_CPU_TESTED: optional SHA-pinned v0.17 SFT model RoPE
  context limit expansion 192→384. Strict tensor loading, parameter-count and
  original-context logit-parity assertions, full-example admission accounting.
- HOSTED_TRAINING_PENDING: GitHub workflow for actual 384-position SFT using
  the known v0.17 best checkpoint. No success or metric claimed before the
  workflow's result is verified.
- NOT_ESTABLISHED: Java instruction compliance, real-world editing, long-range
  generalization, model-generated Java behavioral correctness, GPU/large models.

## v0.16 development — real open-source Java code learning

- Source pipeline checks four exact Git commit SHA-1 hashes and independently pinned license blobs, discards untracked/modified/unsafe code, retains license files and per-file content SHA-256. Actual corpus sizes are determined when the four pinned repositories are checked out.
- Small PyTorch causal code LM can train from random weights on train-only source chunks; report unique files/windows actually used and validation-only model selection. This is not the older 17,536 synthetic Java instruction-pair corpus.
- A separate PR-triggered GitHub Actions CPU job performs live pinned-source checkout and code training, preserving checkpoint artifacts. Until its run completes, public-source learning status is `NOT_RUN`.
- No 1B/3B, GPU, compiler/behavior scores, or demonstrated Java Vibe Coding competence from real-source pretraining. Upstream license evidence is SHA-pinned, not individualized legal clearance.

## Actual v0.15 CPU training evidence (2026-11-22)

- Both the new random-initialized (1,476,288-parameter) model and the v0.14 continuation (1,427,136 parameters) completed **17,080 optimizer steps** using **17,080 distinct records each**. This does NOT mean the continuation learned another 17,080 previously unseen programs: some v0.14 examples overlap.
- Family-interleaved sampling ensured **90/90 train families** contributed to both the final and the selected models. The best validation checkpoint occurred at **step 1,000**, so only 1,000 source examples contributed to that checkpoint; both last checkpoints processed every sample.
- Held-out compile-only results on a reproducible 10-question task-family-disjoint subset: **4/10** from-scratch; **6/10** continuation. **0/10 exact reference matches in both**, and no generated Java behavior run. Test loss **4.2725 / 4.4957** respectively, but the different tokenizers make loss comparison inappropriate.
- Local Python 3.13 + PyTorch 2.10 CPU + JDK 21: **194 tests passed, 4 optional-dependency tests skipped**. Real GPU and licensed external Java training are NOT_RUN.
- See `experiments/java_diverse_v015_results.json` and `docs/JAVA_DIVERSE_TRAINING.md` for full SHA-256 provenance and limitations.

## v0.15 Java curriculum and training (2026-11-22)

- Java 17 original synthetic corpus: 17,536 instruction/code examples and 110 task families; 17,080 train / 228 validation / 228 test; 0 exact source duplicates and whole-family split separation verified.
- Family-balanced training order, train-only tokenizer fitting, no held-out test optimization, best vs last checkpoint separation.
- Docker/Podman code-behavior checks are still unverified on this CPU development environment; model-generated Java behavior **NOT_RUN**.
- Local Java repository inventory added with explicit `PENDING` license status (never auto-trained).
- This is original synthetic data, **not** 17,536 separate human-authored algorithms, and not a trained 1B/3B general-purpose LLM.

## v0.14 Java expanded dataset and trained model

- Newly authored Java SFT material: **5,436 distinct source files as strings / 70 parameterized task families**, full schema- and SHA-verified regeneration. 5,220 training, 108 validation, 108 testing; disjoint families.
- Efficient dependency-free ByteBPE encoder: implemented with randomized parity regression tests against original algorithm.
- CPU Java training pipeline with exact distinct-example accounting, validation-only model selection, and test once after selection: implemented. Actual 2 runs each processed 5,220 examples; best validation checkpoint at 600. Random-init 1,427,136 params: held-out compile 2/18, exact 0/18. Prior-checkpoint continuation 774,240 params: compile 1/18, exact 0/18. Neither established useful Java ability.
- No claim that a large generated dataset implies independent algorithm variety or reliable task-following Java. No model-generated Java behavioral execution. No external Java corpus imported or license legally attested.
- User GPU, 1B/3B training, distributed training, DPO, source-code LICENSE: NOT_RUN / undecided.

## v0.13 local Java training selection and behavior checks (2026-11-22)

- Training pipeline verified on CPU: 1,500 steps requested, 300 performed,
  step 150 selected by validation; no test loss read before selection.
- Held-out token loss: 3.7609 (vs prior 6.0101); compile score 8/15 (vs
  prior 10/15); exact reference match 0/15; **coding ability NOT_ESTABLISHED**.
- Behavioral oracle verified against **15/15 trusted reference Java programs**
  on JDK 21; these are not generated code.
- Offline/local Docker/Podman runner implemented and argument-tested with
  a mocked engine, **real model-generated Java runtime NOT_RUN** because no
  engine is installed. OCI isolation is best effort and requires a disposable VM.
- Dataset and weights remain external artifacts, not committed into git.
- Independent benchmark, licensed external Java training corpus, 1B/3B,
  CUDA/MPS and multi-GPU training remain NOT_RUN.

# Implementation status

| Component | Status | Notes |
| --- | --- | --- |
| Transformer / RoPE / GQA / RMSNorm / SwiGLU | IMPLEMENTED | Native PyTorch, CPU test needed |
| Configs tiny / micro / pilot / 1b / 3b | IMPLEMENTED | 1B/3B never loaded in tests |
| Byte-level BPE training | IMPLEMENTED | Python educational backend + optional HF Tokenizers backend; HF backend passed GitHub-hosted CPU verification (v0.4) |
| Single-device pretraining | IMPLEMENTED | CPU / optional CUDA/MPS unverified |
| Atomic checkpoint / resume | IMPLEMENTED | Fixed schedule and hyperparameter checks; bitwise CPU interrupted/uninterrupted test passes |
| CLI generation / KV cache | IMPLEMENTED | No production decoding server |
| Disk-backed dataset shards | PARTIAL | SQLite exact/normalized/heuristic near dedup, JSONL splits, uint32 shards; only basic privacy screening |
| Basic next-token held-out validation | IMPLEMENTED | Model loss, perplexity, accuracy; rejects training split by default |
| GitHub Actions CPU CI | IMPLEMENTED / HOSTED_VERIFIED_V0.5 | v0.5 Python 3.11 CPU workflow passed: 53 tests after the final test update; v0.6: hosted CI passed 60 tests; v0.7: hosted CI passed 72 tests |
| Standardized benchmarks | PARTIAL | Offline multiple-choice/MMLU-style adapter implemented; official datasets, splits and published scoring protocols NOT_RUN |
| Single-turn SFT | IMPLEMENTED | Assistant-response masking; local JSONL |
| DPO / scalable SFT | NOT_IMPLEMENTED | Phase 4 |
| Local inference HTTP API | PARTIAL | Three endpoints, optional bearer key |
| API rate/concurrency limits | IMPLEMENTED | In-process limiter and cooperative deadline; hard kill, TLS, production rate limiting not implemented |
| HTTP streaming / hard kernel timeouts | NOT_IMPLEMENTED | Proxy recommended for external serving |
| HF Transformers export | IMPLEMENTED / HOSTED_VERIFIED_V0.5 | Local custom-model safetensors export, fast tokenizer AutoTokenizer, roundtrip tests; does not upload |
| Distributed learning / FlashAttention integration | NOT_IMPLEMENTED | Phase 6 |
| Real 1B/3B training | NOT_RUN | No hardware or licensed corpus specified |

## v0.6 follow-up (2026-11-21)

- Added optional local SHA-256-pinned, purpose-restricted human dataset-review cards for data prepare/shard and benchmark scoring; auto-verification of actual legal rights remains impossible.
- MMLU-style raw-question overlap audit fixed; subject and macro scores and an interval added. This is **not** an official MMLU score.
- No official benchmark data imported, no real GPU or distributed training verified, no 1B/3B weights trained.

## v0.7 update

- Offline Hendrycks-style CSV evaluation (5-shot layout, conditional answer-label scoring): IMPLEMENTED/CPU TESTED.
- Human-reviewed, SHA-pinned local MMLU dev/test data inventory: IMPLEMENTED/CPU TESTED.
- Published/official MMLU test data evaluation and reproducible official benchmark score: **NOT_RUN**.
- Hardware GPU measurements, 1B/3B full model training: **NOT_RUN**.

See [MMLU_REFERENCE.md](MMLU_REFERENCE.md) for limits.

## v0.8 local hardware validation update

- Hardware inventory and local tiny CPU Forward/Backward, optimizer, checkpoint and generation: **IMPLEMENTED / LOCAL_CPU_PASS**.
- Two-rank local CPU/Gloo `all_reduce`: **IMPLEMENTED / LOCAL_CPU_PASS**; not DDP training.
- MPS, CUDA, BF16 real GPU tests: **NOT_RUN**, because no compatible GPU exists in this development container.
- Real physical machine operator attestation and multi-GPU training: **NOT_ATTESTED / NOT_RUN**.
- GitHub-hosted CPU CI for v0.8: **PENDING** until workflow runs.

## v0.9 Java natural-language starters

- `sagallm java new/chat/revise/templates/check/run`: IMPLEMENTED, local deterministic templates only.
- Nine Java tasks (hello, calculator, FizzBuzz, guessing game, TODO, prime, factorial, file reader, Swing): CPU/JDK COMPILE TESTED where JDK is installed.
- JDK compile and intentional Java run: LOCAL_TESTED; **not a sandbox**.
- Arbitrary prompt-to-Java by trained SagaLLM: NOT_IMPLEMENTED / NOT_TRAINED. Do not claim general vibe coding.
- IDE integration, autonomous file edits, automatic dependency management, automatic fixing of compile errors: NOT_IMPLEMENTED.

## v0.10 local model-backed Java experiment

- `java ai-revise` and `java ai-chat`: **IMPLEMENTED / LOCAL CPU TESTED** for
  using a trusted local SagaTransformer checkpoint to generate a candidate,
  compile with JDK, preview a diff and apply only after explicit approval.
- `java toy-sft`: **IMPLEMENTED / LOCAL CPU TESTED**, five original synthetic
  instruction/Java examples in existing SFT schema; one tiny SFT update tested.
- Learned Java coding quality, realistic coding benchmarks, reliable model
  self-repair, GPU testing, multi-file project agent, 1B/3B training: **NOT_RUN**.
- Java compilation does not provide a security sandbox; automated Java execution
  remains forbidden in the new model-generated workflow.

## v0.11 original Java instruction fixture and evaluation

- 10 original synthetic training examples / 1 validation / 2 test, separated by Java **task family**: IMPLEMENTED and CPU TESTED. Contains no third-party data and is not a model-quality corpus.
- `java dataset-build` / `dataset-verify`: SHA-256-pinned fixture with contamination and integrity guards: IMPLEMENTED and CPU TESTED. Human rights attestation is `NOT_ATTESTED`.
- `java evaluate`: real local checkpoint inference and held-out **javac syntax-only** rate; output never executed; semantic accuracy is `NOT_RUN`.
- `scripts/java_sft_smoke.py`: two optimizer steps on random 98,624-parameter model with train-only tokenizer training: CPU RUN; held-out behavior competence `NOT_ESTABLISHED`.
- Real-world Java generation benchmark, secure behavior test sandbox, 1B/3B training, GPU experiments: NOT_RUN.

## v0.12 — Java data generation and actual SFT experiment

- Original Japanese/English Java task data: **IMPLEMENTED**, 350 examples and
  26 algorithm families; train/validation/test disjoint by family.
- Synthetic dataset provenance/digest verification: **IMPLEMENTED**, no third-party corpus.
- Actual from-scratch CPU model SFT: **TRAINED**, 774,240 parameters, 1500 steps.
- Held-out compilation: **10/15 PASS_COMPILE**, exact reference match **0/15**.
- Java behavioral correctness: **NOT_RUN**; no independent useful coding capability established.
- Evidence: `docs/JAVA_ORIGINAL_TRAINING.md` and separately downloadable
  model checkpoint/tokenizer/dataset/report. **Do not upload weights to Git.**
- Real GPU, 1B/3B, broader programming datasets, safe sandbox execution: **NOT_RUN**.

### v0.17.0 Java instruction SFT extension

- IMPLEMENTED: bounded, original family-separated Java instruction SFT on SHA-pinned v0.16 real-Java model; assistant-only loss, best/last checkpoints, test-only-after-model-selection, javac-only diagnostics.
- COMPLETED: real GitHub Actions CPU SFT, run 44982940959: 650 optimizer steps, 2,600 distinct SFT records processed. Validation-selected best at step 100 (400 records); 192-token context filter accepts 16,518/17,080 training examples. Validation loss 3.6726 → 1.7723; held-out test response loss 4.2338 → 2.0855. Compile-only 0/8 baseline, 1/8 tuned; exact matches 0/8; generated Java behavior NOT_RUN.
- NOT_ESTABLISHED: general-purpose Java coding accuracy, safe generated-code execution, full-context Java editor compatibility, real GPU, 1B/3B training.