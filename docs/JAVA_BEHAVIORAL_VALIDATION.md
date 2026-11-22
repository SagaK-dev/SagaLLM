# Java behavior checks with optional OCI isolation (v0.13)

## Purpose and evidence boundary

Previous synthetic Java evaluation measured `javac` compilation, **not program correctness**.
SagaLLM v0.13 introduces 15 independent expected-stdout oracles for the **fixed,
original v0.12 test families** (`fibonacci`, `prime`, `palindrome`). The oracles
are implemented in Python independently of Java reference source and checked
against all 15 **trusted, project-authored** Java reference programs using the
local JDK. These trusted references are not model outputs.

Model-generated Java is **never executed on the Python host by this feature**.
To evaluate generated code's output, you must explicitly opt in to Docker or
Podman **on a disposable, locally controlled Linux/WSL virtual machine**.
Containers are defense in depth, **not a guaranteed security sandbox**.

## Behavior test procedure

1. Install and verify a locally running Docker/Podman engine on the disposable VM.
2. Obtain an appropriate JDK 21 container image independently, inspect and
   approve it. The evaluator never downloads or pulls an image. The default
   name is `eclipse-temurin:21-jdk`. Pin your chosen local image to a known
   digest where possible.
3. Verify the original dataset (`build_java_dataset`/`verify_java_dataset`),
   and keep the checkpoint/tokenizer provenance files together.
4. Start with a **non-executing** probe:

```bash
PYTHONPATH=src python scripts/evaluate_java_behavior.py \
  --dataset data/java-original \
  --checkpoint checkpoints/java-exp/java-trained.pt \
  --tokenizer checkpoints/java-exp/tokenizer.json \
  --output reports/java-oci-probe.json
```

The result is `NOT_RUN` because execution was not explicitly authorized. If
Docker/Podman or the local image is unavailable, even `--allow-execution` must
return `NOT_RUN`, rather than a fabricated pass or fail.

5. **Only on the isolated testing VM**, explicitly authorize execution:

```bash
PYTHONPATH=src python scripts/evaluate_java_behavior.py \
  --dataset data/java-original \
  --checkpoint checkpoints/java-exp/java-trained.pt \
  --tokenizer checkpoints/java-exp/tokenizer.json \
  --backend docker --image eclipse-temurin:21-jdk \
  --allow-execution --max-examples 15 \
  --output reports/java-oci-behavior.json
```

Choose a **new** `--output` path. Never mount sensitive user directories or
Docker's control socket inside the container. Do not enable privileged mode.

## OCI restrictions in the implementation

- `--pull=never` (requires a locally preinstalled image)
- `--network=none`, `--read-only`, `--cap-drop=ALL`
- `--security-opt=no-new-privileges`, unprivileged UID/GID 65534
- one CPU, 384 MiB container memory and swap, PID limit 64
- read-only bind mount containing only `Main.java`
- small `noexec,nosuid,nodev` tmpfs for generated `.class` files
- `javac -proc:none`; bounded wall-clock time and bounded host stdout capture
- hard-stop/cleanup on output flood or timeout; model code never interpolated
  into the fixed container shell command
- no Java-generated code is executed if opt-in or runtime prerequisites are missing

These flags reduce exposure, but do not provide formal isolation, side-channel
protection, kernel-exploit immunity, complete resource isolation, or language
semantics guarantees. The Java CLI's separate `java run --trust-local-code`
remains **unisolated** and is not suitable for untrusted model output.

## Status of actual verification

- Python oracle versus **trusted** Java reference: 15/15 outputs match, local
  JDK 21 CPU smoke; this is an oracle-quality test, not a generated-code pass.
- Fake local OCI-launcher unit tests: verifies arguments, opt-in, failures,
  output ceiling and timeouts. This **does not verify Docker isolation**.
- The current execution environment does not have Docker/Podman: **model-
  generated Java OCI behavior tests NOT_RUN**.
- Neither compiler success nor exact match proves that arbitrary Java code is
  correct or safe. This benchmark remains a tiny synthetic test set.