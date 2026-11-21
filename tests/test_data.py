import json
import pytest
from sagallm.data import prepare_data


def test_dedup_and_split(tmp_path):
    path = tmp_path / 'in.jsonl'
    lines = [{'text': '青い空'}, {'text': 'Blue sky'}, {'text': 'Blue sky'},
             {'text': 'def f(): return 1'}, {'text': 'user@example.com'}, {'text': '海の音'}]
    path.write_text('\n'.join(json.dumps(line) for line in lines), encoding='utf-8')
    out = tmp_path / 'processed'
    with pytest.raises(ValueError):
        prepare_data(path, out, 'unknown')
    info = prepare_data(path, out, 'CC0-1.0')
    assert info['rejected']['duplicate'] == 1
    assert info['rejected']['sensitive'] == 1
    assert info['split_counts']['validation'] >= 1
    assert (out / 'train.txt').exists()