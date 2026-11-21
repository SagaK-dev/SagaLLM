# External references and dependency notes

Research references consulted via GitHub on 2026-11-21. Repository license metadata is indicative; inspect each exact version and any per-file notices before copying code. We did **not** copy external source files into SagaLLM. `NVIDIA/Megatron-LM` reports NOASSERTION in GitHub license metadata and includes multiple upstream notices; **do not assume a single permissive license covers all files**.

| Repository | Role | GitHub license metadata | Used in current code? |
| --- | --- | --- | --- |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | End-to-end learning reference | MIT | design only |
| [allenai/Olmo-core](https://github.com/allenai/Olmo-core) | Training architecture | Apache-2.0 | design only |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Future multi-GPU systems | NOASSERTION; inspect LICENSE and per-file notices | no |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | Future distributed training | BSD-3-Clause | no |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Optional fast ByteLevel BPE | Apache-2.0 | optional external package |
| [llm-jp/llm-jp-tokenizer](https://github.com/llm-jp/llm-jp-tokenizer) | Japanese tokenizer comparison | Apache-2.0 | no |
| [huggingface/datatrove](https://github.com/huggingface/datatrove) | Future data processing | Apache-2.0 | no |
| [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator) | Future data curation | Apache-2.0 | no |
| [huggingface/fineweb-2](https://github.com/huggingface/fineweb-2) | Multilingual data pipeline research | Apache-2.0 | no data copied |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Custom loader (optional) | Apache-2.0 | optional external package |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) | Future device/distributed helpers | Apache-2.0 | no |
| [allenai/open-instruct](https://github.com/allenai/open-instruct) | Future SFT/preference training | Apache-2.0 | no |
| [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | Future evaluation | MIT | no |

Current runtime dependencies are PyTorch and PyYAML; optional pytest, pyarrow and psutil are declared in `pyproject.toml`. Versions in published environments should be locked after testing on the target host. Current upstream commit SHAs have not been pinned, because upstream code is **not vendored or imported**.

Optional HTTP serving uses FastAPI and Uvicorn (`sagallm[serve]`); see each package's published license when deploying. SFT/evaluation code in this repository is written as a small independent implementation and is not a copy of Open Instruct or Inspect AI.

Export additionally uses the optional `safetensors` package and `transformers>=4.46,<5`; both are separately installed via the `[hf-export]` extra. The generated custom loader copies SagaLLM-owned source files and does not vendor third-party model code.