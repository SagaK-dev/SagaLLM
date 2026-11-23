# Public Java I/O experiment — disposable VM and networkless Docker

This experiment is **not** the approved, independently reviewed Java
benchmark. Its six tasks originate from a public Exercism-derived intake,
may appear in model pretraining, and use assistant-abridged prompts. The
user explicitly requested a development-only behavioral experiment without
the pending two-person review. That permission does **not** certify
independence, licensing, or production model quality.

## Isolation setup and scope

The `java-public-isolated-behavior.yml` workflow starts a **new
GitHub-hosted Ubuntu VM for the job**, then runs each generated Java program
inside a separate Docker container. GitHub documents that standard hosted
jobs run in fresh VMs, but the Docker daemon itself is privileged on the
VM. Docker is an additional best-effort isolation layer, NOT a proof of
secure containment. No generated Java is executed directly on the host.

The container has these enforced settings (defined in
`src/sagallm/java_behavior.py`):

- `--network=none`, `--pull=never`; image is pre-pulled before Java runs
- `--read-only`, a small private writable `/tmp`
- `--cap-drop=ALL`, `--security-opt=no-new-privileges`
- uid/gid `65534`, no privileged mode and no host Docker socket mount
- `--memory=384m`, `--memory-swap=384m`, `--pids-limit=64`, `--cpus=1.0`
- only generated `Main.java` and one test input are mounted **read-only**
- `javac -proc:none` compiles inside the container, then Java executes
  there with `-Xmx64m`; 20-second per-case timeout; bounded stdout
- output comparison uses stdout alone, exact match, without storing raw
  output or original generated Java.

The image is pulled before execution and then locked to its local
`sha256:<64-digit-id>` for all containers within the job. The exact image
ID is recorded in the result. This is a reproducible *per-run* identity,
not a claim that the mutable upstream JDK tag was pre-audited.

The workflow is manually dispatchable and also allows an initial run only
for the owner's branch `experiment/java-oci-behavior-41`; there is no
automatic `main` push trigger, no external-fork PR execution and no
long-lived self-hosted runner requirement.

## What is measured

From PR #40, four concrete model generations were already verified with
Javac and their **source SHA-256 fixed before this runtime study**:

| Model | Previously compiled public tasks | Input/output cases |
| --- | --- | ---: |
| v0.17 | Luhn, Atbash | 10 |
| v0.18 | Hamming, Luhn | 9 |
| Total | 4 model/task outputs | 19 |

Original v0.17/v0.18 checkpoints and tokenizer must match pinned SHA-256.
New generation uses the EXACT same short prompts, greedy decoding, 384
context and 170-token limit. Any changed source hash causes a **hard
failure before generated code runs**. An independent, known-good
`Main.java` smoke test must pass OCI compilation and stdout comparison
before model code can run.

Each of the 19 inputs is evaluated separately, with statuses
`PASS`, `FAIL`, `TIMEOUT`, `OUTPUT_LIMIT`, `NOT_RUN` or
`RUNTIME_ERROR`. Missing OCI capability makes an evaluation
**UNAVAILABLE**, never a false model failure/success.

The results report per-task passed case counts, task-level PASS iff
**all** its cases pass, the OCI image SHA, and safety metadata.
They do not store generated sources, input test contents, or raw output.

## Interpretation

The score measures **only these four previously compiling outputs**:
it is not a six-task behavioral success rate and cannot imply model
ability on the two uncompiled tasks or unseen tasks. Public task
contamination and prompt adaptation are significant limitations.
The study never updates or promotes the model. Independent benchmark
review in [Issue #36](https://github.com/SagaK-dev/SagaLLM/issues/36)
remains pending.

Do **not** run this workflow on a persistent personal workstation, with
access to secrets, or on a shared self-hosted runner. The GitHub-hosted VM
is disposable, but hardened OCI does not guarantee immunity from
container escape. Use a stronger separately controlled sandbox if
higher-assurance isolation is required.

To perform the same study on a **separately provisioned disposable Linux
VM**, pre-install Docker and a JDK image. Record the local image ID and use
explicit opt-in flags:

```bash
docker pull eclipse-temurin:21-jdk
docker image inspect eclipse-temurin:21-jdk --format '{{.Id}}'
python -m scripts.evaluate_java_public_behavior_oci \
  --v017 path/to/java-sft-best-v017.pt \
  --v018 path/to/java-sft-best-v018.pt \
  --tokenizer path/to/tokenizer.json \
  --suite examples/java_external_io_INTAKE.json \
  --oci-image-id sha256:<the-local-image-id> \
  --allow-unreviewed-public-execution \
  --acknowledge-disposable-vm \
  --output public-java-oci-result.json
```

Without both opt-in flags the script refuses to run any Java.
