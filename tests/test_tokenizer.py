import pytest
from sagallm.tokenization import ByteBPETokenizer


@pytest.mark.parametrize('text', [
    'こんにちは 世界 🌏', 'English text.\nNew line', 'if x < 4:\n    print("😀")',
    '日本語とEnglish mixed 100%', '', '\tdef f(x):\n\treturn x**2\n'])
def test_roundtrip(text, tmp_path):
    tok = ByteBPETokenizer.train([text, text], vocab_size=320)
    path = tmp_path / 'tokenizer.json'
    tok.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.decode(loaded.encode(text, add_bos=True, add_eos=True)) == text
    assert loaded.vocab_size <= 320


def test_invalid_vocab():
    with pytest.raises(ValueError):
        ByteBPETokenizer.train(['a'], 200)