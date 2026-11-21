"""Privacy and cross-split near-duplicate filtering tests (no actual personal data)."""
import json

import pytest

from sagallm.curation import low_quality, near_duplicate, normalized_text, sensitive_reason, simhash64
from sagallm.shards import prepare_shards


def _texts(folder, manifest):
    return {split: [json.loads(line)['text']
                    for file in manifest['splits'][split]['files']
                    for line in (folder / file['path']).read_text(encoding='utf-8').splitlines()]
            for split in ('train', 'validation')}


def test_sensitive_heuristics_detect_major_categories():
    samples = {
        'email': 'contact: fictional.user@example.test',
        'credential': '-----BEGIN RSA PRIVATE KEY-----\nFake testing material\n-----END RSA PRIVATE KEY-----',
        'phone': '問い合わせ先 090-1234-5678',
        'payment_card': 'test card 4111 1111 1111 1111',
    }
    for label, example in samples.items():
        assert sensitive_reason(example) == label
    assert sensitive_reason('card 4111 1111 1111 1112') is None
    assert sensitive_reason('Python function returns 12345') is None
    assert low_quality('\x00\x00\x00hello')
    assert not low_quality('print("hello")\nこんにちは')


def test_unicode_normalization_does_not_erase_code_operators():
    assert normalized_text('ＨＥＬＬＯ\t  World') == 'hello world'
    assert normalized_text('a == b') != normalized_text('a != b')


def test_near_duplicates_removed_before_train_validation_split(tmp_path):
    base = ('日本語 English Python code documentation with examples and descriptions. ' * 15)
    variant = base.replace('descriptions.', 'demonstrations.', 1)
    assert near_duplicate(base, variant)
    a = simhash64(base)
    b = simhash64(variant)
    assert any((a >> (16 * i)) & 0xffff == (b >> (16 * i)) & 0xffff for i in range(4))
    documents = [base, variant, 'HELLO   World', 'hello world',
                 'unrelated material: basic algebraic identities',
                 '土星の公転周期と月の満ち欠けについて',
                 'if __name__ == "__main__": print("safe")',
                 'Computers encode text with bytes and Unicode.']
    data = tmp_path / 'source.jsonl'
    data.write_text('\n'.join(json.dumps({'text': x}, ensure_ascii=False) for x in documents), encoding='utf-8')
    dest = tmp_path / 'curated'
    m = prepare_shards(data, dest, 'CC0-1.0', validation_fraction=0.25)
    assert m['near_dedup_enabled']
    assert m['rejected']['near_duplicate'] == 1
    assert m['rejected']['normalized_duplicate'] == 1
    splits = _texts(dest, m)
    retained = splits['train'] + splits['validation']
    assert not (base in retained and variant in retained)
    assert len(retained) == 6
    # Reverse source order; deterministic retained set and split membership.
    reversed_data = tmp_path / 'reversed.jsonl'
    reversed_data.write_text('\n'.join(json.dumps({'text': x}, ensure_ascii=False) for x in reversed(documents)), encoding='utf-8')
    dest2 = tmp_path / 'curated2'
    other = prepare_shards(reversed_data, dest2, 'CC0-1.0', validation_fraction=0.25)
    assert _texts(dest, m) == _texts(dest2, other)


def test_near_duplicate_filter_can_be_disabled(tmp_path):
    text = 'A longer description of technical material and repeated background. ' * 14
    source = tmp_path / 'docs.jsonl'
    source.write_text('\n'.join(json.dumps({'text': item}) for item in [
        text, text + ' end.', 'A wholly unrelated and distinct short document.'
    ]), encoding='utf-8')
    m = prepare_shards(source, tmp_path / 'prepared', 'CC0-1.0',
                       remove_near_duplicates=False)
    assert m['rejected']['near_duplicate'] == 0
    assert m['unique_documents'] == 3


def test_sensitive_values_not_written_to_shards_or_metadata(tmp_path):
    source = tmp_path / 'input.jsonl'
    docs = ['A harmless first text.', 'Second clean text.', 'contact fictional@sample.test',
            'AWS key AKIA' + 'A' * 16, 'call 080-1234-5678']
    source.write_text('\n'.join(json.dumps({'text': t}) for t in docs), encoding='utf-8')
    output = tmp_path / 'out'
    m = prepare_shards(source, output, 'CC0-1.0')
    assert m['rejected']['sensitive'] == 3
    assert m['sensitive_categories']['email'] == 1
    assert m['sensitive_categories']['credential'] == 1
    assert m['sensitive_categories']['phone'] == 1
    for path in output.iterdir():
        value = path.read_text(encoding='utf-8')
        assert 'fictional@sample.test' not in value
        assert 'AKIA' + 'A' * 16 not in value
        assert '080-1234-5678' not in value