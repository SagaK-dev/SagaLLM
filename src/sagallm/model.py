"""Native PyTorch decoder-only Transformer: GQA, RoPE, RMSNorm and SwiGLU."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rotary_cache(position: torch.Tensor, dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=position.device, dtype=torch.float32) / dim))
    phase = torch.outer(position.float(), inv_freq).repeat_interleave(2, dim=-1)
    return phase.cos().to(dtype)[None, None], phase.sin().to(dtype)[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_heads
        self.kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.max_context = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, past: KVCache | None = None, use_cache: bool = False) -> tuple[torch.Tensor, KVCache | None]:
        batch, length, _ = x.shape
        previous = 0 if past is None else past[0].size(-2)
        if previous + length > self.max_context:
            raise ValueError("Sequence length exceeds max_position_embeddings")
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(previous, previous + length, device=x.device)
        cos, sin = rotary_cache(positions, self.head_dim, self.rope_theta, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        updated = (k, v) if use_cache else None
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        # SDPA's is_causal=True is only correct for a square, uncached prefill.
        mask = None
        causal = previous == 0
        if previous:
            query_positions = torch.arange(previous, previous + length, device=x.device)[:, None]
            key_positions = torch.arange(previous + length, device=x.device)[None, :]
            mask = query_positions >= key_positions
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(batch, length, self.heads*self.head_dim)
        return self.o_proj(out), updated


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor, past: KVCache | None = None, use_cache: bool = False) -> tuple[torch.Tensor, KVCache | None]:
        a, new_cache = self.attention(self.attention_norm(x), past=past, use_cache=use_cache)
        x = x + a
        return x + self.feed_forward(self.ffn_norm(x)), new_cache


class SagaTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                past_key_values: list[KVCache] | None = None, use_cache: bool = False) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must have shape (batch, sequence) and torch.long dtype")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values length mismatch")
        x = self.embedding(input_ids)
        caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            previous = None if past_key_values is None else past_key_values[i]
            x, cache = block(x, previous, use_cache=use_cache)
            if use_cache:
                caches.append(cache)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must be same shape as input_ids (already shifted)")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
        return ModelOutput(logits=logits, loss=loss, past_key_values=caches)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None) -> torch.Tensor:
        if temperature < 0 or top_k < 0 or not (0 < top_p <= 1):
            raise ValueError("Invalid sampling parameters")
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("Provide a non-empty prompt with shape (batch, sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        self.eval()
        result = input_ids
        cache = None
        next_input = result[:, -self.config.max_position_embeddings:]
        finished = torch.zeros(result.size(0), device=result.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            if cache is not None and cache[0][0].size(-2) + next_input.size(1) > self.config.max_position_embeddings:
                cache = None
                next_input = result[:, -self.config.max_position_embeddings:]
            out = self.forward(next_input, past_key_values=cache, use_cache=True)
            scores = out.logits[:, -1, :].float()
            if temperature == 0:
                token = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k:
                    threshold = scores.topk(min(top_k, scores.size(-1))).values[..., -1, None]
                    scores = scores.masked_fill(scores < threshold, float("-inf"))
                if top_p < 1.0:
                    ordered, indices = torch.sort(scores, descending=True)
                    cumulative = ordered.softmax(-1).cumsum(-1)
                    exclude = cumulative - ordered.softmax(-1) >= top_p
                    ordered = ordered.masked_fill(exclude, float("-inf"))
                    scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, ordered)
                token = torch.multinomial(scores.softmax(-1), num_samples=1)
            if eos_token_id is not None:
                token = torch.where(finished[:, None], eos_token_id, token)
                finished |= token[:, 0] == eos_token_id
            result = torch.cat((result, token), dim=1)
            cache = out.past_key_values
            next_input = token
            if finished.all() and eos_token_id is not None:
                break
        return result