"""Transformers custom loader around SagaLLM's original PyTorch model.

This file is copied along with model.py, config.py, and
configuration_sagallm.py into each exported directory. Use
trust_remote_code=True *only* for code you have independently reviewed.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .configuration_sagallm import SagaLLMConfig
from .model import SagaTransformer


class SagaLLMForCausalLM(PreTrainedModel):
    config_class = SagaLLMConfig
    base_model_prefix = "transformer"
    main_input_name = "input_ids"
    _supports_sdpa = True
    _supports_cache_class = False
    _tied_weights_keys = ["transformer.lm_head.weight"]

    def __init__(self, config: SagaLLMConfig):
        super().__init__(config)
        self.transformer = SagaTransformer(ModelConfig(**config.native_config_kwargs()))
        # SagaTransformer initializes weights and ties embeddings itself. A
        # second post_init() would overwrite this initialization unnecessarily.

    def get_input_embeddings(self):
        return self.transformer.embedding

    def set_input_embeddings(self, embeddings):
        self.transformer.embedding = embeddings
        if self.config.tie_word_embeddings:
            self.transformer.lm_head.weight = embeddings.weight

    def get_output_embeddings(self):
        return self.transformer.lm_head

    def set_output_embeddings(self, lm_head):
        self.transformer.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("SagaLLM requires input_ids")
        if kwargs.get("position_ids") is not None or kwargs.get("inputs_embeds") is not None:
            raise ValueError("Explicit position_ids and inputs_embeds are not supported")
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise ValueError("SagaLLM currently supports only unpadded inputs")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            # HF causal-LM convention accepts unshifted labels. Native
            # SagaTransformer expects already shifted next-token targets.
            shifted = torch.full_like(labels, -100)
            shifted[:, :-1] = labels[:, 1:]
        else:
            shifted = None
        output = self.transformer(
            input_ids, labels=shifted, past_key_values=past_key_values,
            use_cache=bool(use_cache),
        )
        if return_dict is False:
            result = (output.logits, output.past_key_values)
            return (output.loss,) + result if output.loss is not None else result
        return CausalLMOutputWithPast(
            loss=output.loss, logits=output.logits,
            past_key_values=tuple(output.past_key_values) if output.past_key_values else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # Supports the legacy tuple-cache API used by SagaTransformer. No
        # beam reordering / left-padding / attention-mask offsets at this stage.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(tuple(state.index_select(0, beam_idx) for state in layer)
                     for layer in past_key_values)