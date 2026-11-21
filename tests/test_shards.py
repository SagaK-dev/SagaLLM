"""End-to-end offline dataset shard tests, including tamper detection."""
import json
import torch
import pytest
from sagallm.config import ModelConfig
from sagallm.evaluation import evaluate_language_model
from sagallm.shards import TokenShardDataset, prepare_shards, tokenize_shards
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import pretrain


def _make(tmp_path):
    source = tmp_path / 'source.jsonl'
    docs = [
        f'こんにちは世界 {i}. Hello {i}. print({i}) ' * 10
        for i in range(12)
    ]
    docs.extend([docs[1], 'admin@example.org', 'password=topsecret12345'])
    source.write_text('\n'.join(json.dumps({'text': t}, ensure_ascii=False) for t in docs), encoding='utf-8')
    tokenizer = ByteBPETokenizer.train(docs[:12], vocab_size=300)
    tok_path = tmp_path / 'tokenizer.json'
    tokenizer.save(tok_path)
    prepared = tmp_path / 'prepared'
    manifest = prepare_shards(source, prepared, 'CC0-1.0', validation_fraction=0.25,
                              max_docs_per_shard=3)
    shards = tmp_path / 'tokens'
    token_manifest = tokenize_shards(prepared, tok_path, shards, max_tokens_per_shard=200)
    return manifest, token_manifest, tok_path, shards


def test_streaming_shards_do_not_leak_and_dedup(tmp_path):
    docs, tokens, tok_path, root = _make(tmp_path)
    assert docs['unique_documents'] == 12
    assert docs['rejected']['duplicate'] == 1
    assert docs['rejected']['sensitive'] == 2
    assert docs['splits']['validation']['documents'] == 3
    assert docs['splits']['train']['documents'] == 9
    assert tokens['splits']['train']['tokens'] > 0
    train = TokenShardDataset(root, split='train', block_size=8, tokenizer_path=tok_path)
    valid = TokenShardDataset(root, split='validation', block_size=8, tokenizer_path=tok_path)
    assert train.total_windows > valid.total_windows
    x, y = train.sample(torch.Generator().manual_seed(2), batch_size=3, device=torch.device('cpu'))
    assert x.shape == (3, 8) and torch.equal(x[:, 1:], y[:, :-1])
    assert max(x.flatten().tolist()) < ByteBPETokenizer.load(tok_path).vocab_size


def test_shards_reject_wrong_tokenizer_or_tampering(tmp_path):
    _, _, tp, root = _make(tmp_path)
    other = tmp_path / 'other.json'
    other.write_text(tp.read_text(encoding='utf-8')+' ', encoding='utf-8')
    with pytest.raises(ValueError, match='tokenizer'):
        TokenShardDataset(root, tokenizer_path=other)
    first = next(root.glob('train-*.bin'))
    first.write_bytes(first.read_bytes()[:-4])
    with pytest.raises(ValueError, match='size'):
        TokenShardDataset(root, tokenizer_path=tp)


def test_sharded_cpu_pretrain_and_heldout_validation(tmp_path):
    _, _, tp, root = _make(tmp_path)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    cp = tmp_path / 'checkpoint.pt'
    result = pretrain(cfg, tp, root, cp, steps=2, schedule_steps=2,
                      block_size=8, batch_size=1, validate_every=1)
    assert result['validation_loss'] is not None
    evaluation = evaluate_language_model(cp, tp, root, block_size=8, batches=2)
    assert evaluation['evaluation_type'] == 'next_token_heldout'
    with pytest.raises(ValueError, match='Training split'):
        evaluate_language_model(cp, tp, root, block_size=8, split='train')


def test_manifest_disallows_path_traversal(tmp_path):
    _, _, tp, root = _make(tmp_path)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    manifest['splits']['train']['files'][0]['path'] = '../sneaky.bin'
    manifest_path.write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(ValueError, match='simple filenames'):
        TokenShardDataset(root, tokenizer_path=tp)