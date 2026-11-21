"""Review-card integrity, permission gates and CLI coverage (offline)."""
import hashlib
import json
import subprocess
import sys

import pytest

from sagallm.dataset_cards import DatasetCardError, verify_dataset_card
from sagallm.shards import prepare_shards


def _card(tmp_path, purpose='train'):
    file = tmp_path / 'source.jsonl'
    file.write_text('\n'.join(json.dumps({'text': x}) for x in [
        'An independently authored text about local language training and testing.',
        'Another original document discussing architecture for small neural networks.',
        'A third dataset record about evaluating local language models fairly.',
    ]), encoding='utf-8')
    card = tmp_path / 'card.json'
    payload = {
        'format': 'sagallm-dataset-card-v1',
        'name': 'Locally authored regression fixture',
        'source_uri': 'local://SagaLLM/test-fixture',
        'source_version': 'fixture-v1', 'license_id': 'CC0-1.0',
        'license_evidence': 'Locally authored test data, only',
        'review_status': 'approved', 'reviewed_by': 'test-maintainer',
        'reviewed_at': '2026-11-21', 'allowed_uses': [purpose],
        'file_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
    }
    card.write_text(json.dumps(payload), encoding='utf-8')
    return file, card, payload


def test_card_verifies_specific_purpose_and_sha(tmp_path):
    source, card, payload = _card(tmp_path)
    review = verify_dataset_card(source, card, purpose='train', expected_license='CC0-1.0')
    assert review['status'] == 'REVIEW_ATTESTED_NOT_LEGALLY_VERIFIED'
    assert review['source_sha256'] == payload['file_sha256']
    assert review['card_sha256'] and review['purpose'] == 'train'
    with pytest.raises(DatasetCardError, match='authorize'):
        verify_dataset_card(source, card, purpose='evaluate')
    with pytest.raises(DatasetCardError, match='license'):
        verify_dataset_card(source, card, purpose='train', expected_license='MIT')
    source.write_text(source.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(DatasetCardError, match='SHA-256'):
        verify_dataset_card(source, card, purpose='train')


def test_card_rejects_unreviewed_and_bad_metadata(tmp_path):
    source, card, payload = _card(tmp_path)
    payload['review_status'] = 'pending'
    card.write_text(json.dumps(payload))
    with pytest.raises(DatasetCardError, match='not been'):
        verify_dataset_card(source, card, purpose='train')
    payload['review_status'] = 'approved'
    payload['allowed_uses'] = ['train', 'train']
    card.write_text(json.dumps(payload))
    with pytest.raises(DatasetCardError, match='allowed_uses'):
        verify_dataset_card(source, card, purpose='train')
    payload['allowed_uses'] = ['train']
    payload['reviewed_at'] = 'November 21'
    card.write_text(json.dumps(payload))
    with pytest.raises(DatasetCardError, match='reviewed_at'):
        verify_dataset_card(source, card, purpose='train')


def test_prepared_manifest_includes_review_and_prevents_illegal_use(tmp_path):
    source, card, _ = _card(tmp_path)
    prepared = tmp_path / 'prepared'
    report = prepare_shards(source, prepared, 'CC0-1.0', dataset_card_path=card)
    assert report['dataset_card_review']['source_sha256']
    assert not report['license_review_required']
    with pytest.raises(DatasetCardError):
        prepare_shards(source, tmp_path / 'bad', 'MIT', dataset_card_path=card)
    assert not (tmp_path / 'bad').exists()


def test_cli_offline_review(tmp_path):
    source, card, _ = _card(tmp_path)
    result = subprocess.run([
        sys.executable, '-m', 'sagallm', 'data', 'verify-license',
        '--source', str(source), '--card', str(card), '--purpose', 'train',
    ], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['status'] == 'REVIEW_ATTESTED_NOT_LEGALLY_VERIFIED'