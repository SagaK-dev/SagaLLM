"""Safety, weight parity, and real training regressions for v0.18 RoPE extension."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from sagallm.config import ModelConfig
from sagallm.java_diverse_corpus import build_java_diverse_corpus
from sagallm.java_instruction_sft import (
    _validate_base, encode_row, extend_rope_context, restore_verified_base_weights,
    train_instruction_sft,
)
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def tiny_config():
    return ModelConfig(vocab_size=260, hidden_size=32, num_layers=1,
                       num_heads=4, num_kv_heads=2, intermediate_size=64,
                       max_position_embeddings=192)


def source_row():
    return {
        'id': 'test-0', 'family': 'test',
        'messages': [
            {'role': 'user', 'content': 'User request (data):\nPrint 1\nCurrent source (data):\n// old'},
            {'role': 'assistant', 'content': 'public class Main { public static void main(String[] args) { System.out.println(1); } }'},
        ],
    }


def save_base(tmp_path, config):
    tokenizer_path = tmp_path / 'tokenizer.json'
    ByteBPETokenizer().save(tokenizer_path)
    model = SagaTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / 'base.pt'
    checkpoint_save(checkpoint_path, model, optimizer, 3, config,
                    {'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()})
    return checkpoint_path, tokenizer_path


def test_only_rope_limit_changes_and_bounds_are_enforced():
    old = tiny_config()
    new = extend_rope_context(old, 384)
    assert old.max_position_embeddings == 192
    assert new.max_position_embeddings == 384
    assert old.parameter_count() == new.parameter_count()
    assert extend_rope_context(old, None) == old
    for bad in (191, 192, 385, 512, 0, -1, True, 192.5):
        with pytest.raises(ValueError):
            extend_rope_context(old, bad)


def test_extended_model_preserves_short_prefix_logits_and_supports_long_prefill(tmp_path):
    torch.manual_seed(19)
    old_config = tiny_config()
    path, tok = save_base(tmp_path, old_config)
    state, original, *_ = _validate_base(path, tok, require_pinned=False)
    old = SagaTransformer(old_config)
    restore_verified_base_weights(old, state, original)
    long_config = extend_rope_context(original, 384)
    extended = SagaTransformer(long_config)
    restore_verified_base_weights(extended, state, original)
    old.eval(); extended.eval()
    x = torch.randint(4, old_config.vocab_size, (2, 81), dtype=torch.long)
    with torch.inference_mode():
        # These inputs have identical position indices and all learned weights.
        before = old(x).logits
        after = extended(x).logits
        long_logits = extended(torch.randint(4, 260, (1, 257))).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert long_logits.shape == (1, 257, 260)
    with pytest.raises(ValueError, match='max_position_embeddings'):
        old(torch.randint(4, 260, (1, 257)))


def test_restore_rejects_unrelated_architecture_change(tmp_path):
    old_config = tiny_config()
    path, tok = save_base(tmp_path, old_config)
    state, original, *_ = _validate_base(path, tok, require_pinned=False)
    wrong = replace(original, intermediate_size=96)
    with pytest.raises(ValueError, match='Only the RoPE context bound'):
        restore_verified_base_weights(SagaTransformer(wrong), state, original)
    corrupted = {**state, 'config': {'vocab_size': 1}}
    with pytest.raises(ValueError, match='Original checkpoint configuration'):
        restore_verified_base_weights(SagaTransformer(original), corrupted, original)


def test_context_extension_admits_longer_complete_java_without_truncating():
    tokenizer = ByteBPETokenizer()
    example = source_row()
    example['messages'][1]['content'] = ('public class Main { public static void main(String[] args) {'
                                        + 'System.out.println(1);'*6 + '} }')
    assert encode_row(example, tokenizer, 192) is None
    longer = encode_row(example, tokenizer, 384)
    assert longer is not None
    assert longer['sequence_positions'] > 192
    assert int((longer['y'] != -100).sum()) == longer['response_tokens']
    assert longer['sequence_positions'] == len(longer['x'])


def test_two_step_real_extended_context_sft_preserves_source_and_test_separation(tmp_path):
    data = tmp_path / 'data'
    build_java_diverse_corpus(data)
    base, tok = save_base(tmp_path, tiny_config())
    report = train_instruction_sft(
        base, tok, data, tmp_path / 'result', steps=2, batch_size=2,
        eval_every=1, eval_count=4, test_count=4, compile_cases=0,
        max_seconds=40, require_pinned=False, target_context=384, seed=17,
    )
    assert report['status'] == 'TRAINED_CPU'
    assert report['model_context_length'] == 384
    assert report['original_model_context_length'] == 192
    assert report['model_parameters'] == tiny_config().parameter_count()
    assert report['context_extension'] == 'ROPE_EXTRAPOLATION_UNVALIDATED'
    assert report['distinct_training_examples_seen'] == 4
    assert report['original_split_counts']['train']['newly_admitted_beyond_original_context'] > 0
    assert report['test_used_for_model_selection'] is False
    assert report['java_behavior_test'] == 'NOT_RUN'
    state = torch.load(tmp_path / 'result' / 'java-sft-best.pt', map_location='cpu', weights_only=True)
    assert state['config']['max_position_embeddings'] == 384
    assert state['train_state']['base_checkpoint_sha256'] == hashlib.sha256(base.read_bytes()).hexdigest()
    assert json.loads((tmp_path/'result'/'train-report.json').read_text())['best_checkpoint_sha256']