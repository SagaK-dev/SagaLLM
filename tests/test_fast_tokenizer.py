"""Runs on hosted CI (fast-tokenizer extra); skipped if the Rust package is absent."""
import pytest

from sagallm.tokenization import ByteBPETokenizer, FastBPETokenizer, load_tokenizer


def test_python_backend_stays_compatible(tmp_path):
    source = ByteBPETokenizer.train(['hello world']*2, vocab_size=270)
    path = tmp_path / 'python.json'
    source.save(path)
    loaded = load_tokenizer(path)
    assert type(loaded) is ByteBPETokenizer
    assert loaded.decode(loaded.encode('hello')) == 'hello'


def test_native_bytelevel_roundtrip_and_format(tmp_path):
    pytest.importorskip('tokenizers')
    corpus = [
        'こんにちは。Hello, world! 🌏\n',
        'def f(x):\n    return x + 1\n',
        'C++: std::vector<int> xs; 🚀\t\n',
    ]
    tokenizer = FastBPETokenizer.train(iter(corpus*3), vocab_size=360, min_frequency=1)
    assert tokenizer.vocab_size <= 360
    assert tokenizer.SPECIAL['<bos>'] == 1
    path = tmp_path/'fast.json'
    tokenizer.save(path)
    loaded = load_tokenizer(path)
    assert isinstance(loaded, FastBPETokenizer)
    for sentence in corpus + ['', '\t\nUnicode: 漢字、ひらがな！💻']:
        assert loaded.decode(loaded.encode(sentence)) == sentence
        assert loaded.decode(loaded.encode(sentence, add_bos=True, add_eos=True)) == sentence
    assert loaded.encode('abc') == tokenizer.encode('abc')


def test_native_requires_optional_dependency(tmp_path, monkeypatch):
    import sagallm.tokenization as mod
    original_import = __import__
    def fake_import(name, *args, **kwargs):
        if name == 'tokenizers':
            raise ImportError('disabled in test')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr('builtins.__import__', fake_import)
    with pytest.raises(RuntimeError, match='fast-tokenizer'):
        mod.FastBPETokenizer._package()


def test_native_backend_works_with_sharded_pretraining(tmp_path):
    pytest.importorskip('tokenizers')
    import json
    from sagallm.config import ModelConfig
    from sagallm.shards import prepare_shards, tokenize_shards
    from sagallm.training import pretrain
    from sagallm.evaluation import evaluate_language_model

    docs = [f'This is original example {i}: 日本語 and Python code print({i}) ' * 8
            for i in range(6)]
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps({'text': item}, ensure_ascii=False) for item in docs), encoding='utf-8')
    tokenizer = FastBPETokenizer.train(iter(docs), vocab_size=360, min_frequency=1)
    tp = tmp_path / 'fast.json'
    tokenizer.save(tp)
    prepared = tmp_path / 'prepared'
    prepare_shards(source, prepared, 'CC0-1.0')
    tokens = tmp_path / 'tokens'
    tokenize_shards(prepared, tp, tokens, max_tokens_per_shard=256)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    cp = tmp_path / 'checkpoint.pt'
    result = pretrain(cfg, tp, tokens, cp, steps=1, batch_size=1, block_size=8)
    evaluation = evaluate_language_model(cp, tp, tokens, block_size=8, batches=1)
    assert result['steps'] == 1
    assert evaluation['evaluated_tokens'] == 16