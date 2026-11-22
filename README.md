# SagaLLM

**Prototype v0.16.0** — offline 5-shot MMLU-style evaluation, SHA-pinned dataset review, local hardware diagnostics, and CPU/Gloo readiness tests. The model itself remains experimental and untrained at 1B/3B scale.

A local-first **from-scratch** decoder-only language model research project. The model uses native PyTorch implementations of RMSNorm, rotary embeddings, grouped-query attention, SwiGLU and a tied token embedding / language-modeling head. The repository includes a small educational byte-level BPE tokenizer, disk-backed JSONL / uint32 token shards, deterministic document splits, resumable single-device training, supervised fine-tuning, held-out token-level validation, an optional local inference API, and greedy/sampling generation.

> **Status: experimental foundation, not a trained 1B/3B LLM.** This project can initialize and train a small model with random weights. The 1B/3B configs are for inspection and future scaling; no large model is pretrained. Never interpret a falling toy loss as language proficiency.

## Quick start (CPU, offline after dependency installation)

Requires Python 3.10+ and an installed version of PyTorch and PyYAML. Installing dependencies may require network access. Nothing starts a paid cloud service.

```bash
python -m pip install -e '.[test]'
sagallm doctor
sagallm model inspect --config configs/model/1b.yaml
sagallm model inspect --config configs/model/3b.yaml
sagallm tokenizer train --data examples/toy_corpus.txt --output data/tokenizer.json --vocab-size 384
sagallm train pretrain --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data examples/toy_corpus.txt --checkpoint checkpoints/tiny.pt --steps 8 --batch-size 2 --block-size 32
sagallm generate --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json --prompt 'こんにちは' --max-new-tokens 16
# Prepare document-split token shards (for meaningful evaluation, use independent licensed data)
sagallm data shard --source examples/toy_corpus.txt --output data/prepared --license CC0-1.0 --max-docs-per-shard 3
sagallm data tokenize --prepared data/prepared --tokenizer data/tokenizer.json --output data/tokens --max-tokens-per-shard 256
# Train with a fixed LR horizon so resumption remains reproducible
sagallm train pretrain --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data data/tokens --checkpoint checkpoints/sharded.pt --steps 2 --schedule-steps 4 --block-size 8 --batch-size 1 --validate-every 1 --checkpoint-every 1
sagallm train resume --config configs/model/tiny.yaml --tokenizer data/tokenizer.json --data data/tokens --checkpoint checkpoints/sharded.pt --steps 4 --block-size 8 --batch-size 1 --validate-every 1
sagallm evaluate --checkpoint checkpoints/sharded.pt --tokenizer data/tokenizer.json --data data/tokens --block-size 8
# Optional supervised fine-tuning (toy instructions only)
sagallm train sft --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json --data examples/toy_sft.jsonl --output checkpoints/tiny-sft.pt --steps 2
# Optional local HTTP API: python -m pip install -e '.[serve]'
# sagallm serve --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json
python -m pytest -q
```

The `examples/toy_corpus.txt` is an artificial, repository-authored smoke-test corpus; it is **not** a licensed large-scale language training dataset. The `--license CC0-1.0` shown for this repository-authored demo does not certify permissions for third-party materials. `data/`, `checkpoints/` and binary model weights are gitignored. The default Python tokenizer is intentionally simple. The optional Rust-backed tokenizer can be selected with `--backend hf` (see `docs/TOKENIZERS.md`).

## Java production-source learning (v0.16) — pinned, attributed open source

The new CPU pipeline can train the **native SagaLLM model from random weights**
on real Java from 4 specific Git revisions: TheAlgorithms/Java (MIT),
Google Gson and Guava (Apache-2.0), and Java Design Patterns (MIT).
Git revision and license blob IDs are pinned, Git-tracked production `.java`
files are deduplicated and screened, and source-package-disjoint splits are
used. Preserve license notices and source provenance. This is an engineering
license hash check, not a legal certification of all source contributions.

See `docs/JAVA_REAL_SOURCE_TRAINING.md` and the separate
`.github/workflows/java-public-cpu-train.yml` workflow. The training workflow
will report actual file counts, files that contributed gradients and held-out
code next-token loss. It uploads model artifacts only after a successful run.
**This is code language modeling, not Java instruction fine-tuning**; falling
loss does not demonstrate usable natural-language Java Vibe Coding. Generated
Java behavior/compilation in this step is `NOT_RUN`.

## Java diverse training v0.15.0: 17,536 synthetic examples, 110 families (2026-11-22)

The **original, deterministic** Java curriculum now covers 110 parameterized
Java 17 task families (90 training, 10 validation, 10 held-out test), with
**17,080 / 228 / 228** instruction-and-source examples. It includes 40 new
task types spanning strings, arrays, matrices, collections, number theory,
and the Java standard library. Text prompts use Japanese and English.

```bash
# All files are created locally; source examples are original template-derived code.
PYTHONPATH=src python -c "from sagallm.java_diverse_corpus import build_java_diverse_corpus; print(build_java_diverse_corpus('data/java-diverse'))"
# Train from random weights; 17,080 unique examples, one per optimizer update.
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_diverse.py \
  --dataset data/java-diverse --output checkpoints/java-diverse \
  --steps 17080 --max-seconds 1200 --compile-limit 8
# Optional: continue from your own trusted v0.14 checkpoint and tokenizer.
# Use --init-checkpoint ... --init-tokenizer ... (fresh optimizer).
```

The optimizer's **last checkpoint** can see all 17,080 examples; the
**validation-selected best checkpoint** may have seen far fewer. The report
records each quantity separately, including training families covered.
The tests do not execute model-generated Java; compilation alone is **not**
correct program behavior. The dataset consists of structured template
variations, **not 17,536 independently authored real-world applications**.
There is still no independent coding benchmark, external licensed Java corpus,
or verified general-purpose Java coding capability. See
`docs/JAVA_DIVERSE_TRAINING.md` and `experiments/java_diverse_v015_results.json`.

**Measured CPU runs:** both models completed 17,080 optimizer updates and saw every train row. On 10 family-disjoint tasks, the random-init model compiled 4 programs and the v0.14 continuation compiled 6; neither exactly matched its reference (0/10). **Both best validation checkpoints were at step 1,000**, and generated Java semantic correctness is NOT_ESTABLISHED. Local regression tests: 194 PASS / 4 SKIP (optional HF dependencies).

## Validation-selected Java training and guarded runtime checks (v0.13)

The original Java trainer now **stops early based on validation loss**, retains
best and last-step checkpoints, and accesses test labels only once after model
selection. On the same small, original 350-example dataset, an actual CPU run
stopped at 300 steps and selected step 150: test loss improved from **6.010**
to **3.761**, but compiler success decreased from **10/15 to 8/15**, and
reference equality remained **0/15**. This is not usable free-form Java coding.

A separate opt-in OCI behavioral evaluator is available under
`python scripts/evaluate_java_behavior.py`. It requires a preinstalled
Docker/Podman image, an explicit `--allow-execution`, and a dedicated
disposable VM. No generated Java is run on the host by this feature. The
current development environment has no OCI engine, so model-generated Java
**behavioral execution remains NOT_RUN**. OCI restrictions are not a security
guarantee. See `docs/JAVA_BEHAVIORAL_VALIDATION.md`.

## Original Java data and actual CPU training (v0.12)

SagaLLM now generates **350 newly authored bilingual Java 17 training/evaluation
examples** with strictly disjoint task families: 320 train, 15 validation and
15 test. A deterministic manifest and byte-by-byte verifier detect tampering,
including forged SHA-256 manifests. The code generator uses no external Java
corpus, no pretrained model and no paid API.

A **real** random-initialized 774,240-parameter SagaTransformer was trained
on the train split for **1,500 CPU optimizer steps**. Training loss fell from
6.2038 to 0.2634, but held-out test loss only fell 6.1769 to 6.0101.
On 15 unseen Java requests, **10 generated programs compiled, 0 matched the
reference code**, and **zero** generated Java programs were executed.
Validation loss worsened late in training, showing overfitting. **This is not
an independently useful Java coding LLM.** Model weights and the generated
corpus are not stored in Git; downloadable artifacts are supplied separately.

```bash
PYTHONPATH=src python -c 'from sagallm.java_generated_data import build_java_dataset; print(build_java_dataset("data/java-original"))'
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_original.py \
  --dataset data/java-original --output checkpoints/java-original-exp \
  --steps 1500 --max-seconds 155 --hidden-size 160 --layers 3
```

See [docs/JAVA_ORIGINAL_TRAINING.md](docs/JAVA_ORIGINAL_TRAINING.md) for
source, audit, precise settings, results and limitations.

## Java synthetic study kit and honest held-out checks (v0.11)

An **original, locally generated** Java SFT fixture now contains 10 training
examples, one validation example and two held-out test examples. Train,
validation and test split by **task family** (not by random instruction row),
which prevents literal answers from the held-out `prime` / `calculator` /
`factorial` families being used as supervised training examples. Every split
is SHA-256-pinned and verified against the known local reference templates.
This is only an engineering fixture: it is **not** enough data to teach useful
Java coding or a substitute for rights-reviewed real instruction/code data.

```bash
# Offline, no third-party data; refuses to overwrite existing directories.
sagallm java dataset-build --output data/java-study
sagallm java dataset-verify --dataset data/java-study
# Fine-tune an existing trusted checkpoint whose context fits these examples:
sagallm train sft --checkpoint checkpoints/base.pt --tokenizer data/tokenizer.json \
  --data data/java-study/train.jsonl --output checkpoints/java-study.pt \
  --steps 2 --device cpu
# Holdout evaluation: generated code is compiled, NEVER executed.
sagallm java evaluate --dataset data/java-study --checkpoint checkpoints/java-study.pt \
  --tokenizer data/tokenizer.json --split test --max-examples 2 \
  --max-new-tokens 512 --output reports/java-study.json
# Independent tiny random-base smoke (no production coding ability implied):
PYTHONPATH=src python scripts/java_sft_smoke.py --output reports/java-tiny-smoke.json
```

The compile pass rate measures **syntax only**. Exact-reference match is a
second, strict, diagnostic metric, not a behavioral test. Runtime correctness
is `NOT_RUN` because generated Java is untrusted and must not run on the host
without an isolated sandbox. The dataset reports license review `NOT_ATTESTED`;
it contains only original SagaLLM source templates, no imported Java corpus.
See [docs/JAVA_CODING_DATA.md](docs/JAVA_CODING_DATA.md).

## Model-backed Java revisions (v0.10, experimental)

SagaLLM can now **optionally use its own local, matching model checkpoint** to
propose edits to an existing Java starter project. Unlike the deterministic
starter workflow below, `java ai-revise` actually calls `SagaTransformer.generate`.
This does **not** make the tiny research checkpoints competent Java coders:
expect incomplete or invalid output and `REJECTED` responses.
No cloud LLM, pretrained model or paid API is used or silently substituted.

```bash
# Safe default: generate, compile with javac -proc:none, and show a diff.
sagallm java ai-revise --project work/my-java \
  --checkpoint checkpoints/your-trained-sft.pt --tokenizer data/tokenizer.json \
  --prompt 'Change the greeting to Hello Saga' --max-new-tokens 512

# Interactive: review the diff and type APPLY to install that exact candidate.
sagallm java ai-chat --project work/my-java \
  --checkpoint checkpoints/your-trained-sft.pt --tokenizer data/tokenizer.json

# Generate five original synthetic SFT examples (a test corpus, NOT enough for
# reliable code generation). Refuses to overwrite the output JSONL file.
sagallm java toy-sft --output data/java-toy-sft.jsonl

# Training requires a compatible pre-trained base checkpoint, tokenizer and a
# context window large enough for the full prompt and response. No training
# runs automatically and no 1B/3B model is loaded by this command.
sagallm train sft --checkpoint checkpoints/base.pt \
  --tokenizer data/tokenizer.json --data data/java-toy-sft.jsonl \
  --output checkpoints/java-toy-sft.pt --steps 1 --device cpu
```

A local **JDK 17+** must be on `PATH` for successful compilation. The model
must be trained separately and its checkpoint must contain tokenizer SHA-256
provenance. Candidate Java is **never run automatically**; `javac` confirms
syntax, not correctness or safety. The initial model path edits an *existing*
SagaLLM Java starter, keeping its class name and one source file; multiple-file
project generation remains NOT_IMPLEMENTED. The original template editing
command refuses to regenerate a model-edited file to avoid overwriting custom
work. See [docs/JAVA_VIBE.md](docs/JAVA_VIBE.md) for limitations.

## Java natural-language starter workflow (v0.9)

For **simple Java** tasks, SagaLLM now provides local, interactive starter generation,
reviewed revisions and optional JDK compile/run checks. This is **not** a trained
Java code-generating LLM: it recognizes nine explicit tasks, rejects unsupported
requests and runs without network access or pretrained model weights.

```bash
sagallm java templates
sagallm java chat --project work/my-java
# Or use noninteractive commands:
sagallm java new --prompt 'FizzBuzzを30まで表示' --output work/fizzbuzz
sagallm java check --project work/fizzbuzz
sagallm java revise --project work/fizzbuzz --prompt '50までに変更'  # preview only
sagallm java revise --project work/fizzbuzz --prompt '50までに変更' --apply
# Only on trusted generated code; NOT SANDBOXED:
sagallm java run --project work/fizzbuzz --trust-local-code
```

A local **JDK 17+** is needed to compile. Never run untrusted Java through the
experimental runner: Java can access the host filesystem and network.
See [docs/JAVA_VIBE.md](docs/JAVA_VIBE.md) (Japanese and English examples) for
supported project types, iteration flow and limitations.

## Design constraints

- Pretraining starts from random model weights; no hosted LLM API is required.
- `configs/model/{tiny,micro,pilot,1b,3b}.yaml` specify architectures. `model inspect` calculates parameters **without allocating weights**.
- For safety, training over 200 million parameters is blocked unless `--allow-large` is explicitly supplied. This flag is **not** a guarantee that training fits memory.
- The tokenizer is byte-level and handles Japanese/English/code/Unicode without pre-trained vocabularies.
- Checkpoints should only be loaded from trusted locations. Resume verifies corpus, tokenizer SHA-256, schedule horizon and hyperparameters. When resuming, keep `--schedule-steps` unchanged or omit it to inherit the checkpoint value.
- Inference restricts generated token IDs to the tokenizer's vocabulary and checks checkpoint/tokenizer identity when provenance is available.
- The token-shard pipeline uses SQLite-backed exact and heuristic near-duplicate document deduplication, deterministic splits and disk-backed binary tokens. See `docs/DATA_PIPELINE.md`. It provides conservative near-duplicate removal and basic sensitive-pattern screening; neither can guarantee removal of paraphrases or all personal information.
- Data preparation supports optional SHA-pinned human-review cards; a license identifier or card is not proof of rights. Pattern-based privacy screening is not a complete PII solution.
- CPU smoke tests must not be represented as CUDA/MPS tests.

## Scope and limitations

Implemented: native dense causal Transformer, small-scale BPE tokenizer, config validation, CLI diagnostics, TXT/JSON/JSONL/CSV/optional Parquet input, SQLite-backed exact and heuristic near-duplicate filtering, document-level train/validation JSONL shards, disk-backed uint32 token shards, small-device pretraining with integrated validation and checkpoint resume, SFT smoke training, simple held-out token loss/perplexity, localhost-first inference API with basic rate/concurrency controls, KV-cached generation, CPU tests.

Not yet implemented: official standardized benchmark protocol/results, preference training/DPO, production SFT batching and validation, FlashAttention-specific integration, distributed training, comprehensive privacy screening, guaranteed near-duplicate removal, guaranteed bounded-memory tokenizer training, fully standardized HF pipelines, streaming HTTP API, official standardized benchmark execution, model distribution, or training of a 1B/3B model. These are planned phases, not completed features.

## Optional fast BPE and offline benchmark harness

```bash
# Optional CPU-native BPE implementation (downloads the free package if missing):
python -m pip install -e '.[fast-tokenizer]'
sagallm tokenizer train --backend hf --data examples/toy_corpus.txt --output data/fast.json --vocab-size 512
# Existing python tokenizer format remains supported and is the default.

# Simple, offline benchmark smoke test. This is *not* an official MMLU score:
sagallm benchmark --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json \
  --data examples/toy_benchmark.jsonl --license CC0-1.0 \
  --output reports/benchmark-smoke.json
```

Without `--train-docs`, the report explicitly states that contamination has **not been checked**. For vetted document-shard training corpora, pass `--train-docs data/prepared` and refer to `docs/BENCHMARKS.md`. The adapter supports local MMLU-style four-choice JSONL but does not claim official leaderboard comparability. No external datasets are downloaded.

## Dataset review cards and benchmark protocol v0.6

Offline usage-review cards can bind dataset bytes to a human review, specified permitted use and SHA-256. This **does not prove license compliance**; it prevents a different or modified local file being used under a recorded review. Existing flags remain available but are unverified without a card.

```bash
# Start with examples/dataset-card.template.json (all placeholder values are unapproved).
# After independently verifying rights, fill it and calculate the source file SHA-256.
sagallm data verify-license --source /path/to/source.jsonl --card /path/to/card.json --purpose train
sagallm data shard --source /path/to/source.jsonl --output data/new-prepared \
  --license REVIEWED_LICENSE_ID --dataset-card /path/to/card.json
# Use a separate review card explicitly allowing 'evaluate' for benchmark data.
sagallm benchmark --checkpoint checkpoints/tiny.pt --tokenizer data/tokenizer.json \
  --data /path/to/mmlu-style.jsonl --format mmlu --license REVIEWED_LICENSE_ID \
  --dataset-card /path/to/benchmark-card.json --output reports/local-eval.json
```

The MMLU-style reader now checks normalized **raw questions** (not solely generated prompts) for train-corpus leakage, and reports subject-level and macro accuracy and a Wilson interval. It remains **zero-shot, local and not officially standardized**. These scores are not comparable to an official MMLU leaderboard; no benchmark dataset has been downloaded or scored. See `docs/DATASET_CARDS.md` and `docs/BENCHMARKS.md`.

## Local Transformers-compatible model export (v0.5)

`export` writes a **new local directory** with safetensors weights, custom architecture code,
configuration, generation settings, the original SagaLLM tokenizer and SHA-256 provenance.
It does not upload to the Hub or publish model weights. Inspect all generated Python files
before using `trust_remote_code=True` and do not load an untrusted training checkpoint.

```bash
python -m pip install -e '.[hf-export]'
# For an AutoTokenizer-ready export, train with --backend hf first:
sagallm tokenizer train --backend hf --data examples/toy_corpus.txt --output data/hf-fast.json --vocab-size 384
sagallm train pretrain --config configs/model/tiny.yaml --tokenizer data/hf-fast.json \
  --data examples/toy_corpus.txt --checkpoint checkpoints/hf-demo.pt --steps 2 --block-size 8
sagallm export --checkpoint checkpoints/hf-demo.pt --tokenizer data/hf-fast.json --output exports/hf-demo
```

Load and use **only after reviewing the exported custom code**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = 'exports/hf-demo'
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True, local_files_only=True)
ids = tok('こんにちは', return_tensors='pt', add_special_tokens=False).input_ids
out = model.generate(ids, max_new_tokens=8, do_sample=False)
print(tok.decode(out[0]))
```

The older Python BPE backend is saved as `sagallm_tokenizer.json` but **cannot** be
loaded with `AutoTokenizer`; model safetensors and custom code can still be exported.
Generation suppresses token IDs not in the exported tokenizer's vocabulary.
See `docs/HF_EXPORT.md` for limitations and validation.

## Automatic lightweight CPU CI

`.github/workflows/cpu-ci.yml` runs fast offline tests on GitHub-hosted **CPU** for pushes to `main` and pull requests. It does **not** start large-scale model training or contact hosted LLM APIs. GitHub-hosted CI is not a local execution environment; to keep every test on your own machine instead, disable the workflow in GitHub and run:

```bash
python -m pip install -e ".[test,serve]"
python scripts/check_cpu.py
```

GitHub Actions v0.5 CPU CI passed (53 tests, run 44905032184), including native tokenizer and Transformers model loading/generation. A successful CPU CI run is not GPU validation.

## Expanded Java training corpus (v0.14)

SagaLLM can deterministically generate **5,436 newly authored Japanese/English Java 17 instruction/code pairs** from **70 parameterized task families**. The data is split by **entire task family**: train 5,220 / validation 108 / test 108. These are variations of generated program templates, **not thousands of independent algorithms** and not real-world repository data. No third-party Java source, copyrighted corpus, or pretrained LLM is fetched.

```bash
PYTHONPATH=src python -c 'from sagallm.java_large_corpus import build_java_large_corpus; print(build_java_large_corpus("data/java-expanded"))'
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_expanded.py \
  --dataset data/java-expanded --output checkpoints/java-expanded-experiment \
  --steps 5220 --max-seconds 700 --hidden-size 192 --layers 4
```

Every model update uses exactly one train example; distinct examples **actually seen during optimizer updates** are counted separately from the number generated. The actual v0.14 CPU experiments each completed **5,220** optimizer steps, but the validation-selected best checkpoints were from **step 600** (thus each best checkpoint itself had seen only **600** training examples). Both models produced **0/18** exact Java reference matches, so reliable Java coding is not established. The tokenizer is trained on train examples only, checkpoint selection uses validation only, and held-out test loss and compilation are evaluated once after selection. Compilation does **not** establish semantic correctness. Model-generated Java is never executed in this experiment. For full usage and measured CPU results, see [docs/JAVA_EXPANDED_TRAINING.md](docs/JAVA_EXPANDED_TRAINING.md).

## Licensing

**The SagaLLM source license is not yet selected by the repository owner.** Public visibility does not confer a license to reuse this repository. Third-party reference projects have their own independent licenses; see `docs/DEPENDENCIES.md`. Do not publish model weights or add a LICENSE file without the owner's decision.

## Project documentation

- `docs/ARCHITECTURE.md` — model, safety gates and roadmap.
- `docs/DATA_PIPELINE.md` — disk-backed data, validation and resume workflow.
- `docs/DEPENDENCIES.md` — upstream references and license metadata.
- `docs/HARDWARE.md` — local runtime limitations and resource estimation.
- `docs/STATUS.md` — implemented/unimplemented feature matrix.
- `docs/JAVA_VIBE.md` — offline Java starter creation, conversational revision, compile/run safety and limits.
- `docs/TOKENIZERS.md` — optional native BPE, format identity and limitations.
- `docs/BENCHMARKS.md` — offline multiple-choice scoring and contamination audit.
- `docs/HF_EXPORT.md` — custom Transformers load and safetensors validation.
- `docs/DATASET_CARDS.md` — offline dataset rights-review attestations and SHA verification.

## API safety notes

The optional API exposes `/health`, `/v1/completions`, and `/v1/chat/completions`; it is an **experimental subset**, not a full OpenAI API implementation. The CLI defaults to `127.0.0.1`. Binding to a non-loopback address requires `--api-key`; additionally use TLS termination, reverse-proxy rate limiting, request timeouts and monitoring for any internet-facing deployment. Prompt length, rate (60 requests/minute per process by default), concurrent generations (1 by default), and token counts are bounded. Generation has a **cooperative**, not hard, time budget; a stalled native kernel cannot be forcibly interrupted. API streaming is not yet provided.

Data preparation requires a supplied license identifier, which is **not proof** that the input material is legally licensed. Operators must independently review permissions. The demonstration SFT data is toy material, not representative of useful instruction-following performance.

## MMLU reference-style offline CSV evaluation (v0.7)

Use `sagallm mmlu inventory`, `sagallm mmlu verify`, and `sagallm mmlu evaluate` with **user-supplied, manually approved local MMLU CSVs**. The strict single-token label scorer approximates the original Hendrycks 5-shot protocol but refuses incompatible tokenization. Outputs never claim to be certified MMLU scores. No third-party benchmark data is downloaded or checked in. See [docs/MMLU_REFERENCE.md](docs/MMLU_REFERENCE.md).

## Hardware validation and distributed readiness (v0.8)

No GPU is required for the built-in hardware evidence tools:

```bash
sagallm hardware inspect --output reports/hardware-inventory.json
sagallm hardware smoke --device cpu --output reports/cpu-smoke.json
sagallm hardware distributed-smoke --output reports/gloo-smoke.json
```

On the actual GPU-equipped computer, **explicitly** run `sagallm hardware smoke --device cuda --precision fp32` (or `--device mps` on supported Apple Silicon). BF16 is an opt-in CUDA-only check. All reports distinguish `PASS_RUNTIME` from `NOT_RUN` and are always `NOT_ATTESTED` for physical ownership. No paid cloud service, GPU provisioning, pretrained checkpoint or 1B/3B training is performed. See [docs/PHYSICAL_VALIDATION.md](docs/PHYSICAL_VALIDATION.md).

## Real-Java initialized Java instruction SFT (v0.17)

The optional `scripts/train_java_instruction.py` starts from the SHA-pinned v0.16
real-Java code LM, fine-tunes on the original v0.15 split-by-family instruction
examples, and evaluates compile-only generation on unseen template families.
See [Java instruction SFT](docs/JAVA_INSTRUCTION_SFT.md) and the
`java-instruction-sft-cpu` GitHub Actions workflow. This is an experimental
small-model SFT, **not** a reliable Java Vibe Coding assistant: code is **never
executed** and the old long-context Java editor cannot use the compact SFT
prompt with the v0.16 192-token context.

The v0.17 CPU SFT [run completed](https://github.com/SagaK-dev/SagaLLM/actions/runs/44982940959): 650 steps, 2,600 examples seen, validation-selected best at step 100. Held-out test loss 4.234 → 2.086, but only 1/8 generated Java programs compiled and 0/8 matched reference source. Full figures: `experiments/java_instruction_v017_results.json`.