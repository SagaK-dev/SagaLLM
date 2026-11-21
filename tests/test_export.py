"""Local export is safe-by-default and preserves the native model weights."""
import hashlib
import json

import pytest
import torch

from sagallm.config import ModelConfig
from sagallm.export import export_transformers
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer, FastBPETokenizer
from sagallm.training import checkpoint_save


def _toy_checkpoint(tmp_path, *, fast=False):
    docs = ['こんにちは。Hello world! This is a tiny model.'] * 5
    if fast:
        pytest.importorskip('tokenizers')
        tok = FastBPETokenizer.train(iter(docs), vocab_size=270, min_frequency=1)
    else:
        tok = ByteBPETokenizer.train(docs, vocab_size=270)
    tokenizer_path = tmp_path / 'tokenizer.json'
    tok.save(tokenizer_path)
    cfg = ModelConfig(vocab_size=384, hidden_size=32, num_layers=1,
                      num_heads=4, num_kv_heads=2, intermediate_size=64,
                      max_position_embeddings=32)
    model = SagaTransformer(cfg).eval()
    opt = torch.optim.AdamW(model.parameters())
    cp = tmp_path / 'model.pt'
    checkpoint_save(cp, model, opt, 3, cfg, {'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()})
    return model, cp, tokenizer_path, tok


def test_local_export_roundtrip_legacy_tokenizer(tmp_path):
    safe = pytest.importorskip('safetensors.torch')
    model, cp, tokenizer, tok = _toy_checkpoint(tmp_path)
    dest = tmp_path / 'export'
    report = export_transformers(cp, tokenizer, dest)
    assert report['parameters'] == model.config.parameter_count()
    assert not report['hf_tokenizer_auto_load']
    assert not (dest / 'tokenizer.json').exists()
    assert (dest / 'sagallm_tokenizer.json').read_bytes() == tokenizer.read_bytes()
    data = safe.load_file(str(dest / 'model.safetensors'))
    assert 'transformer.embedding.weight' in data
    assert 'transformer.lm_head.weight' not in data  # tied, no double copy
    restored = SagaTransformer(model.config)
    # Prefix is used by the Transformers wrapper, not the native module.
    keys = {k.removeprefix('transformer.'): v for k, v in data.items()}
    missing, unexpected = restored.load_state_dict(keys, strict=False)
    assert missing == ['lm_head.weight']
    assert not unexpected
    assert restored.embedding.weight.data_ptr() == restored.lm_head.weight.data_ptr()
    x = torch.tensor([[1, 4, 6, 7]], dtype=torch.long)
    torch.testing.assert_close(restored.eval()(x).logits, model(x).logits, rtol=0, atol=0)
    manifest = json.loads((dest / 'export_manifest.json').read_text())
    for filename, digest in manifest['files_sha256'].items():
        assert hashlib.sha256((dest / filename).read_bytes()).hexdigest() == digest
    assert manifest['checkpoint_step'] == 3
    assert manifest['tokenizer_identity_verified']
    assert not manifest['uploaded']


def test_export_refuses_existing_target_and_tokenizer_mismatch(tmp_path):
    pytest.importorskip('safetensors.torch')
    _, cp, tp, _ = _toy_checkpoint(tmp_path)
    out = tmp_path / 'output'
    out.mkdir()
    with pytest.raises(FileExistsError):
        export_transformers(cp, tp, out)
    changed = tmp_path / 'tampered.json'
    changed.write_text(tp.read_text() + ' ', encoding='utf8')
    with pytest.raises(ValueError, match='SHA-256'):
        export_transformers(cp, changed, tmp_path / 'new-output')
    assert not (tmp_path / 'new-output').exists()


def test_export_rejects_oversized_config_before_model_allocation(tmp_path):
    cfg = ModelConfig.load('configs/model/1b.yaml')
    cp = tmp_path / 'fake-big.pt'
    torch.save({'config': cfg.to_dict(), 'model': {}}, cp)
    tok = ByteBPETokenizer()
    tp = tmp_path / 'tok.json'
    tok.save(tp)
    with pytest.raises(ValueError, match='requires --allow-large'):
        export_transformers(cp, tp, tmp_path / 'large-export')
    assert not (tmp_path / 'large-export').exists()


def test_hf_auto_model_and_tokenizer_roundtrip(tmp_path):
    pytest.importorskip('tokenizers')
    pytest.importorskip('transformers')
    pytest.importorskip('safetensors')
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, cp, tokenizer_path, tok = _toy_checkpoint(tmp_path, fast=True)
    dest = tmp_path / 'bundle'
    info = export_transformers(cp, tokenizer_path, dest)
    assert info['hf_tokenizer_auto_load']
    hf_tok = AutoTokenizer.from_pretrained(str(dest), local_files_only=True)
    sample = 'こんにちは Hello!'
    assert hf_tok.encode(sample, add_special_tokens=False) == tok.encode(sample)
    hf = AutoModelForCausalLM.from_pretrained(
        str(dest), trust_remote_code=True, local_files_only=True,
    ).eval()
    x = torch.tensor([tok.encode(sample, add_bos=True)], dtype=torch.long)
    with torch.inference_mode():
        native_logits = model(x).logits
        hf_logits = hf(x).logits
    torch.testing.assert_close(hf_logits, native_logits, atol=0, rtol=0)
    assert hf.get_input_embeddings().weight.data_ptr() == hf.get_output_embeddings().weight.data_ptr()
    labels = x.clone()
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    with torch.inference_mode():
        torch.testing.assert_close(hf(x, labels=labels).loss, model(x, labels=shifted).loss)
    # HF GenerationMixin compatibility: basic greedy decoding (no padding).
    generated = hf.generate(input_ids=x, do_sample=False, max_new_tokens=2, pad_token_id=0)
    expected = model.generate(x, temperature=0, max_new_tokens=2,
                              vocab_limit=tok.vocab_size)
    assert torch.equal(generated, expected)


def test_hf_custom_model_loads_legacy_python_tokenizer_weights(tmp_path):
    pytest.importorskip('transformers')
    pytest.importorskip('safetensors')
    from transformers import AutoModelForCausalLM
    model, cp, tp, _ = _toy_checkpoint(tmp_path, fast=False)
    dest = tmp_path / 'legacy-export'
    info = export_transformers(cp, tp, dest)
    assert not info['hf_tokenizer_auto_load']
    loaded = AutoModelForCausalLM.from_pretrained(
        str(dest), trust_remote_code=True, local_files_only=True
    ).eval()
    x = torch.tensor([[1, 5, 22, 9]], dtype=torch.long)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(input_ids=x).logits, model(x).logits, rtol=0, atol=0)