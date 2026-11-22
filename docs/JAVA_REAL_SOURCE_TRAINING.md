# Real Java source training — SagaLLM v0.16.0

## Goal

Train a small **Java next-token language model**, from random weights, using
production Java source code from four repositories whose **specific revisions**
and top-level **MIT / Apache-2.0 license texts** have been inspected.

This is not Java Vibe Coding SFT, not an independently established right to use
other repositories, and not a claim that the model can follow Java instructions.

## Sources and retained license notices

The implementation pins Git SHA **and Git blob hash of each actual license file**
(see `src/sagallm/java_public_corpus.py`):

| Pinned repository | License | Source scope |
|---|---|---|
| [TheAlgorithms/Java](https://github.com/TheAlgorithms/Java) | MIT | `src/main/java` |
| [google/gson](https://github.com/google/gson) | Apache-2.0 | production `src/main/java` |
| [google/guava](https://github.com/google/guava) | Apache-2.0 | `guava/src/com/google/common` |
| [iluwatar/java-design-patterns](https://github.com/iluwatar/java-design-patterns) | MIT | pattern modules' `src/main/java` |

The exact revisions are in `UPSTREAMS`, not `master` or `main` floating references.
Any checkout at a different SHA, altered tracked file or different license
blob fails before dataset extraction. Only Git-tracked Java files are read.
The upstream code is never executed or compiled. Maven/Gradle dependencies,
submodules and build scripts are never installed or run.

The processed corpus retains copies of all license texts under `LICENSES/`,
plus source URLs, hashes and full relative paths. If distributing any source
copies, preserve applicable copyright notices and licenses. The manifest is a
technical provenance record, **not legal advice**; third-party ownership,
individual-file notices and suitability for training may require further review.

## Exact split and training method

- Source files filtered for source scope, UTF-8, <= 96,000 bytes and no suspected
  secret/PII. Screening is heuristic and can miss sensitive content.
- Exact-identical code is deduplicated across all four repositories.
- **Group-separated** train/validation/test, using source repository + source
  category/package, not randomly splitting similar files or adjacent token
  windows; tokenizer fitted on train files only.
- Code is transformed into next-token modeling windows, up to 4 per document.
  Grouped train examples are shuffled deterministically and interleaved so
  the first pass samples one window from as many different files as possible.
- Both best-validation and last checkpoint are saved separately; test loss is
  evaluated once, after selecting the best checkpoint using validation only.
- Reports distinguish Java files **available**, files actually contributing
  gradients, unique token windows actually processed, and best-checkpoint step.
- **No Java output from the model is executed or compiled**; reporting only
  held-out code next-token loss. Java instruction-following is `NOT_ESTABLISHED`.

## GitHub-hosted CPU training (no paid model service)

A separate workflow `.github/workflows/java-public-cpu-train.yml` downloads
only these four pinned repositories, verifies license Git blobs, prepares the
corpus, trains for at most 950 optimizer steps/480 seconds and uploads an
artifact with both checkpoints, tokenizer, JSON report, manifest and **license
notices**. It intentionally omits raw source-code JSONL from the artifact.
The workflow is run on a changed-files PR and can be explicitly dispatched.
Runner capacity, repository access and GitHub Actions limits apply.

## Offline reproduction with independently obtained checkouts

```bash
# Get the 4 specified revisions in four directories within java-upstream/:
# algorithms/, gson/, guava/, patterns/; do not run their build scripts.
PYTHONPATH=src python -m sagallm.java_public_corpus \
  --sources java-upstream --output data/java-public-reviewed
PYTHONPATH=src python -m sagallm.java_public_corpus --verify data/java-public-reviewed
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/train_java_public.py \
  --corpus data/java-public-reviewed --output checkpoints/java-public \
  --backend hf --steps 950 --max-seconds 480 \
  --block-size 192 --batch-size 6 --per-file 4
```

Do not report the number of repository files as the number of files the model
actually learned. Use the `distinct_training_files_seen` metric from the
training report. The model is tiny and will not be competitive with established
code language models or useful as a stand-alone Java coding assistant.