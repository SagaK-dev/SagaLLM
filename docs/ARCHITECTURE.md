# SagaLLM architecture and development stages

The source implementation is a decoder-only dense autoregressive Transformer: embedding → repeated pre-norm [GQA attention with RoPE; SwiGLU FFN] → final RMSNorm → tied LM head. All projections are bias-free. Attention uses `torch.nn.functional.scaled_dot_product_attention` and repeats K/V heads for GQA; cache-aware attention constructs an explicit boolean lower-triangular mask with an offset. No pretrained weights are imported.

## Staged development

0. **Environment / design** — inspect CPU/CUDA/MPS, model configurations and relevant upstream references. Complete for the execution sandbox; **the user's local hardware remains unknown**.
1. **Minimal model** — tokenizer, Transformer, single-device train, checkpoint, CLI generate. Prototype implemented; test status must come from actual test runs.
2. **Data** — exact dedup, document-level split, source integrity and basic screening implemented. Near-duplicate detection, PII-quality guarantees and streaming remain pending.
3. **Advanced pretraining** — basic held-out loss/perplexity evaluation implemented; integrated validation loop, mixed precision, scalable sharding and optimized streaming pending.
4. **Post-training** — minimal single-turn SFT and response masking implemented; preference optimization, realistic datasets and large-scale batching pending.
5. **Serving and evaluation** — CLI and experimental local HTTP serving available; Inspect AI integration and standardized evaluations pending.
6. **Scale out** — DDP/FSDP/TorchTitan/Megatron-LM integration pending.

## Safety and reliability

No network access or cloud resources are needed at runtime. Large models are not initialized by `model inspect`. Actual training of >200M parameters requires explicit `--allow-large` and resource consideration. PyTorch checkpoint files must be trusted, and `weights_only=True` is used when loading; use safe format exports before untrusted distribution. No external dataset should be used unless rights are checked.

## Parameter count

For h=hidden_size, f=intermediate_size, L=layers, v=vocab size, k=head-count ratio `num_kv_heads / num_heads`, tied embeddings:

`N = L * (2*h*h + 2*h*h*k + 3*h*f + 2*h) + v*h + h`

This is accurate for the currently implemented bias-free architecture. The training-memory estimate shown by CLI is a heuristic based on 16 bytes/parameter for FP32 AdamW model weights, gradients and optimizer states, excluding activations, temporary buffers, and framework overhead.