# Hardware and compute constraints

The runner available for development was Linux / x86_64, Python 3.13, PyTorch 2.10 CPU build, no CUDA or MPS. This **does not identify the owner's local PC**. Use `sagallm doctor` in the target environment.

Parameter-state memory alone is roughly `16 * parameters` bytes for FP32 AdamW (weights 4, gradients 4, first/second moments 8). Mixed precision or alternative optimizers vary. Runtime requires additional activation, communication and temporary memory. A 1B-class model therefore typically needs substantially more training memory than its weight file. Not enough available hardware is known to promise 1B/3B training within a useful time at zero cloud budget.

A `model inspect` action does not allocate model weights. `train pretrain` requires explicit invocation. CPU tiny smoke runs validate the pipeline, not model quality. The toy corpus is not a replacement for vetted multilingual data at scale.

## Local runtime validation (v0.8)

See [PHYSICAL_VALIDATION.md](PHYSICAL_VALIDATION.md) for inspect-only inventory, tiny runtime CPU/CUDA/MPS smoke, two-rank local CPU Gloo smoke, evidence schema, and strict NOT_RUN / NOT_ATTESTED semantics. This environment has CPU-only PyTorch; GPU evidence remains NOT_RUN.