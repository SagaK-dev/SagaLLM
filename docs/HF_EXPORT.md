# Local Hugging Face Transformers interoperability (v0.5)

The `sagallm export` command creates a **new local directory**. It neither uploads files nor grants public redistribution rights. It accepts only trusted SagaLLM `.pt` checkpoints (loaded with `torch.load(weights_only=True)`) and the exact tokenizer file used in training when that hash is present in the checkpoint.

## Commands

```bash
python -m pip install -e '.[hf-export]'
sagallm export --checkpoint checkpoints/model.pt --tokenizer data/fast.json --output exports/model
```

Source artifacts: `model.safetensors` (no optimizer/RNG state), `config.json`, `generation_config.json`, `configuration_sagallm.py`, `modeling_sagallm.py`, `model.py`, `config.py`, `sagallm_tokenizer.json`, `export_manifest.json`. For the native FastBPE backend, also `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json`. Export hashes all files except the manifest itself.

The custom `AutoModelForCausalLM` loader wraps the original SagaLLM PyTorch architecture. Native weights are prefixed with `transformer.` and tied `lm_head.weight` is left out of safetensors to avoid duplicating shared storage. The exported generation configuration suppresses token IDs outside the tokenizer vocabulary, which can be smaller than the model output size.

**Security:** `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` executes code from the supplied directory. Review the code and do not accept a directory from strangers. The exporter intentionally does not create a source LICENSE or push weights; licensing remains an owner decision.

## Verified / not verified

- CPU native export and exact forward-logit roundtrip are tested with a small model. The manifest marks whether checkpoint/tokenizer SHA identity was actually verified.
- GitHub CPU CI run `44904787682` passed 52 tests (including native fast-tokenizer AutoTokenizer load, custom AutoModel load, logits/labels parity, and no-cache greedy generation). Additional legacy Python BPE AutoModel parity coverage is being added.
- On CI, integration tests attempt `AutoConfig` / `AutoModelForCausalLM` / `AutoTokenizer` and greedy generation with Transformers 4.x and native FastBPE.
- The legacy educational ByteBPETokenizer **cannot** be loaded by HF `AutoTokenizer`. Its exact original file is still preserved for SagaLLM CLI.
- Standard `generate` uses a **limited no-padding / no-cache** pathway (`generation_config.json` sets `use_cache=false`); native SagaLLM generation retains its own KV cache. Attention masks with padding, `inputs_embeds`, custom position offsets, efficient HF batch generation, beam search across versions, quantization, Hub deployment, multi-GPU and third-party inference-server adapters are **not validated**.
- Export does not certify any tokenizer corpus rights, model quality or readiness to release weights.

## Troubleshooting

If AutoModel says `model_type=sagallm` is unknown, set `trust_remote_code=True` **after reviewing local export code**. If Transformers changes its custom module API, use the `[hf-export]` extra's tested version constraints and consult the CPU CI. A Python BPE tokenizer export intentionally omits HF `tokenizer.json`; retrain with `--backend hf` and use a checkpoint trained with that exact tokenizer for AutoTokenizer support.