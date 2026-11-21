# Local hardware validation and distributed readiness (v0.8)

SagaLLM does **not** have access to the repository owner's computer. The report
below describes the machine running the command only. No GPU, driver, USB
controller, AWS instance or third-party service is provisioned by these checks.
All commands are **opt-in** and run on the current computer.

## 1. Non-allocating inventory

```bash
sagallm hardware inspect --output reports/hardware-inventory.json
sagallm doctor
sagallm model inspect --config configs/model/1b.yaml
sagallm model inspect --config configs/model/3b.yaml
```

`hardware inspect` reads OS, Python/PyTorch, visible CUDA devices and total
VRAM, CUDA runtime and `nvidia-smi` driver **if available**, MPS availability,
free local disk, optional system RAM and PyTorch Gloo/NCCL backend availability.
It does **not** instantiate model weights. `model inspect` calculates parameter
state bytes only (activations, temporary memory, collective buffers omitted).

## 2. Tiny one-step runtime smoke

```bash
sagallm hardware smoke --device cpu --output reports/cpu-smoke.json
# Run ONLY on a computer with a local NVIDIA GPU:
sagallm hardware smoke --device cuda --precision fp32 --output reports/cuda-fp32.json
# Optional: supported NVIDIA CUDA cards only:
sagallm hardware smoke --device cuda --precision bf16 --output reports/cuda-bf16.json
# Optional: Apple Silicon machine with MPS enabled:
sagallm hardware smoke --device mps --precision fp32 --output reports/mps-smoke.json
```

A single **48k-parameter** random-weight toy model (not 1B or 3B) executes a
Forward/Backward pass, optimizer update, checkpoint save/reload, and 2-token
greedy generation. Tests fail on nonfinite loss or gradients and unsuccessful
weight/checkpoint/generation checks. BF16 mode is enabled only on compatible
CUDA devices. The report includes elapsed time and CUDA peak allocated memory
if applicable. It does **not** run real-data training or assert model quality.
The output is a local JSON file written with exclusive create; existing files
are never overwritten. `--device cuda` when CUDA is absent returns `NOT_RUN`,
not a fictitious pass; `--precision bf16` on unsupported hardware does likewise.

## 3. Two local CPU processes for distributed communication readiness

```bash
sagallm hardware distributed-smoke --timeout 40 --output reports/gloo-smoke.json
```

Explicitly launches two local `torchrun` processes on **CPU/Gloo**. They use
localhost rendezvous to perform one collective `all_reduce` of rank+1 and
verify total=3. There is a strict subprocess timeout and neither GPU nor
training data is used. This proves one local CPU collective worked; it does
**not** prove DDP/FSDP, NCCL, multi-GPU synchronization, network fabric,
performance at scale, or distributed checkpoint recovery. A failed smoke has
nonzero CLI exit status and a `FAIL` report with a short reason.

## 4. Evidence, acceptance, and what NOT to claim

Each report has `schema=sagallm-hardware-v1`, UTC timestamp, possible local
Git SHA, environment and accelerator metadata. `source_sha=null` means an
installed package/archive without Git metadata; it is not evidence of a
specific Git commit. `PASS_RUNTIME` verifies only the exact executed device
and checks. Every report sets `physical_evidence=NOT_ATTESTED` because a
runtime cannot independently prove its GPU is attached to the operator's
physical computer. Operator confirmation and a private lab log are separate
steps; do not add personal identity or serial numbers to public issues.

| Requirement | CPU (this environment) | CUDA | MPS | Multi-GPU |
| --- | --- | --- | --- | --- |
| Device inventory | PASS | NOT_RUN until device present | NOT_RUN until device present | NOT_RUN |
| Tiny FP32 train/checkpoint/generate | PASS | NOT_RUN | NOT_RUN | NOT_RUN |
| BF16 autocast | N/A | NOT_RUN | NOT_RUN | NOT_RUN |
| Two-rank CPU Gloo collective | PASS locally | N/A | N/A | NOT_RUN |
| CUDA/NCCL multi-GPU training | N/A | NOT_RUN | N/A | NOT_RUN |
| Physically operated real machine attestation | NOT_ATTESTED | NOT_ATTESTED | NOT_ATTESTED | NOT_ATTESTED |

Before claiming `PASS_PHYSICAL`, separately document OS version, GPU and driver,
operator and date, Git SHA, exact commands, unredacted local evidence location,
reviewed checks, and signed operator confirmation. No automatically generated
SagaLLM report currently uses the status `PASS_PHYSICAL`.

**Do not publish** reports containing operator identifiers, private hostnames,
or hardware serials. The `reports/` directory is Git-ignored. No GPU/driver
versions observed inside a container should be attributed to the user's PC.

## 5. Later work (not validated)

- Real CUDA/MPS smoke on the operator's machine and accompanying attestation.
- CUDA BF16 and FP16 gradient scaling performance validation.
- Multi-GPU NCCL DDP/FSDP and preemption/resume consistency.
- CUDA memory budget versus activations and optimizer states at 1B/3B.
- 1B/3B pretraining with a licensed multilingual corpus.