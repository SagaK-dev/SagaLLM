"""Offline MMLU reference protocol regression tests using ORIGINAL toy data only."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import pytest
import torch

from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.mmlu_reference import (
    MMLUDataError, MMLUTokenizationError, _label_scores, create_inventory_template,
    discover_csv_files, evaluate_mmlu, format_example, format_prompt, read_mmlu_csv,
    select_fitting_prompt, verify_inventory,
)
from sagallm.shards import sha256_file
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def _example(question: str, answer: str = 'C') -> list[str]:
    return [question, 'First option', 'Second option', 'Third option', 'Fourth option', answer]


def _dataset(tmp_path, subject='abstract_algebra'):
    root = tmp_path / 'mmlu_data'
    for split in ('dev', 'test'):
        (root / split).mkdir(parents=True)
    dev = [_example(f'Toy algebra training example number {i}?', 'A') for i in range(5)]
    tests = [_example('Which option is the toy correct choice?'),
             _example('Another independently authored practice question?', 'A')]
    for split, rows in (('dev', dev), ('test', tests)):
        with (root / split / f'{subject}_{split}.csv').open('w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerows(rows)
    return root


def _approved_inventory(root, tmp_path):
    dest = tmp_path / 'inventory.json'
    status = create_inventory_template(root, dest)
    assert status['status'] == 'PENDING_REVIEW'
    card = json.loads(dest.read_text())
    card.update({'source_uri': 'local://SagaLLM/synthetic-fixtures',
                 'source_version': '1', 'license_id': 'CC0-1.0',
                 'license_evidence': 'All sample questions authored for local tests.',
                 'review_status': 'approved', 'reviewed_by': 'fixture author',
                 'reviewed_at': '2026-11-21', 'allowed_uses': ['evaluate']})
    dest.write_text(json.dumps(card), encoding='utf-8')
    return dest


def _checkpoint(tmp_path, max_context=1024):
    tok = ByteBPETokenizer.train([' A', ' B', ' C', ' D', 'Simple tests.'], 280, min_frequency=1)
    tp = tmp_path / 'tokenizer.json'
    tok.save(tp)
    cfg = replace(ModelConfig.load('configs/model/tiny.yaml'), max_position_embeddings=max_context)
    model = SagaTransformer(cfg)
    cp = tmp_path / 'weights.pt'
    checkpoint_save(cp, model, torch.optim.AdamW(model.parameters()), 0, cfg, {})
    return cp, tp, tok


def test_prompt_exact_original_reference_layout():
    dev = [dict(question='Why do we test?', options=['one', 'two', 'three', 'four'], answer='B')]
    question = dict(question='What is selected?', options=['a', 'b', 'c', 'd'], answer='A')
    assert format_example(dev[0], include_answer=True) == (
        'Why do we test?\nA. one\nB. two\nC. three\nD. four\nAnswer: B\n\n')
    assert format_prompt('abstract_algebra', dev, question, 1) == (
        'The following are multiple choice questions (with answers) about  abstract algebra.\n\n'
        'Why do we test?\nA. one\nB. two\nC. three\nD. four\nAnswer: B\n\n'
        'What is selected?\nA. a\nB. b\nC. c\nD. d\nAnswer:')


def test_inventory_pending_manual_review_and_sha_mismatch(tmp_path):
    root = _dataset(tmp_path)
    dest = tmp_path / 'inventory.json'
    create_inventory_template(root, dest)
    with pytest.raises(MMLUDataError, match='pending'):
        verify_inventory(root, dest)
    dest.unlink()
    approved = _approved_inventory(root, tmp_path)
    result = verify_inventory(root, approved)
    assert result['status'] == 'HUMAN_ATTESTED_BYTES_VERIFIED'
    assert result['file_count'] == 2
    with (root / 'test/abstract_algebra_test.csv').open('a', encoding='utf-8') as stream:
        stream.write('a,b,c,d,e,A\n')
    with pytest.raises(MMLUDataError, match='SHA-256 mismatch'):
        verify_inventory(root, approved)


def test_inventory_rejects_extra_file_and_unsafe_symlink(tmp_path):
    root = _dataset(tmp_path)
    inv = _approved_inventory(root, tmp_path)
    (root / 'test/extra_test.csv').write_text('a,b,c,d,e,A', encoding='utf-8')
    with pytest.raises(MMLUDataError):
        verify_inventory(root, inv)
    (root / 'test/extra_test.csv').unlink()
    f = root / 'dev/abstract_algebra_dev.csv'
    original = tmp_path / 'original.csv'
    f.rename(original)
    try:
        f.symlink_to(original)
    except (NotImplementedError, OSError):
        pytest.skip('Cannot create local symlink')
    with pytest.raises(MMLUDataError, match='Unexpected'):
        discover_csv_files(root)


def test_csv_parser_rejects_invalid_rows(tmp_path):
    root = _dataset(tmp_path)
    p = root / 'test/abstract_algebra_test.csv'
    p.write_text('question,first,second,third,fourth,Z\n', encoding='utf-8')
    with pytest.raises(MMLUDataError, match='answer label'):
        read_mmlu_csv(p)
    p.write_text('missing,columns\n', encoding='utf-8')
    with pytest.raises(MMLUDataError, match='six nonempty'):
        read_mmlu_csv(p)


def test_fewshot_context_fallback(tmp_path):
    _, _, tok = _checkpoint(tmp_path)
    dev = [dict(question=f'Long exercise about symbolic reasoning number {i}?',
                options=['one','two','three','four'], answer='A') for i in range(5)]
    row = dict(question='A short toy test?', options=['a','b','c','d'], answer='B')
    one = format_prompt('abstract_algebra', dev, row, 1)
    ctx = len(tok.encode(one, add_bos=True)) + 1
    prompt, shots = select_fitting_prompt(tok, 'abstract_algebra', dev, row, 5, ctx,
                                          'reference_next_token')
    assert shots == 1 and prompt == one
    with pytest.raises(MMLUDataError, match='exceeds'):
        select_fitting_prompt(tok, 'abstract_algebra', dev, row, 5, 4,
                              'reference_next_token')


def test_reference_label_scoring_and_noncomparable_fallback(tmp_path):
    cp, tp, tok = _checkpoint(tmp_path)
    cfg = replace(ModelConfig.load('configs/model/tiny.yaml'), max_position_embeddings=1024)
    model = SagaTransformer(cfg).eval()
    # Use an unmerged last-token boundary: the tokenizer was trained only on
    # short isolated phrases, which should leave all ' A'...' D' stable.
    scores = _label_scores(model, tok, 'Ask:\nAnswer:', torch.device('cpu'),
                            'reference_next_token')
    assert len(scores) == 4
    class BadBoundary:
        def encode(self, value, add_bos=False):
            if value.endswith(' A') and value != ' A':
                return [1, 2]
            return [4] if value.startswith(' ') else [2]
    with pytest.raises(MMLUTokenizationError, match='stable single'):
        _label_scores(model, BadBoundary(), 'Ask:\nAnswer:', torch.device('cpu'),
                      'reference_next_token')


def test_full_local_toy_evaluation_and_partial_flags(tmp_path):
    root = _dataset(tmp_path)
    inventory = _approved_inventory(root, tmp_path)
    cp, tp, _ = _checkpoint(tmp_path, max_context=1024)
    result = evaluate_mmlu(cp, tp, root, inventory, requested_shots=5, max_examples=1,
                           scoring='continuation')
    assert result['questions'] == 1
    assert result['partial_evaluation'] is True
    assert result['official_score'] is False
    assert result['standard_benchmark_certified'] is False
    assert result['reference_compatible_scoring'] is False
    assert result['review']['status'] == 'HUMAN_ATTESTED_BYTES_VERIFIED'
    assert result['contamination']['status'] == 'NOT_CHECKED'
    assert all('question' not in r and 'prompt' not in r for r in result['results'])
    assert sum(result['effective_shots_histogram'].values()) == 1
    again = evaluate_mmlu(cp, tp, root, inventory, requested_shots=5, max_examples=1,
                          scoring='continuation')
    assert result == again


def test_missing_inventory_or_wrong_purpose_fails_before_loading_model(tmp_path):
    root = _dataset(tmp_path)
    inv = _approved_inventory(root, tmp_path)
    data = json.loads(inv.read_text())
    data['allowed_uses'] = ['train']
    inv.write_text(json.dumps(data))
    with pytest.raises(MMLUDataError, match='approved'):
        evaluate_mmlu(tmp_path / 'fake.pt', tmp_path / 'fake.json', root, inv)


def test_fewshot_dev_leakage_blocks_evaluation(tmp_path):
    from sagallm.benchmarks import BenchmarkContaminationError
    root = _dataset(tmp_path)
    inv = _approved_inventory(root, tmp_path)
    cp, tok, _ = _checkpoint(tmp_path)
    doc_root = tmp_path / 'train-docs'; doc_root.mkdir()
    dev = read_mmlu_csv(root / 'dev/abstract_algebra_dev.csv')
    # Use an exact normalized raw question in the recorded train split.
    f = doc_root / 'train-00000.jsonl'
    f.write_text(json.dumps({'text': 'Previous notes include: ' + dev[0]['question'] + ' and more.'}),
                 encoding='utf-8')
    (doc_root / 'manifest.json').write_text(json.dumps({
        'format':'sagallm-doc-shards-v1',
        'splits':{'train':{'files':[{'path':f.name,'sha256':sha256_file(f)}]}}
    }), encoding='utf-8')
    with pytest.raises(BenchmarkContaminationError):
        evaluate_mmlu(cp, tok, root, inv, training_docs=doc_root, requested_shots=5,
                      max_examples=1, scoring='continuation')


def test_cli_inventory_verify_and_no_overwrite(tmp_path, capsys):
    from sagallm.cli import main
    root = _dataset(tmp_path)
    inventory = tmp_path / 'review.json'
    main(['mmlu','inventory','--data-dir',str(root),'--output',str(inventory)])
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PENDING_REVIEW'
    with pytest.raises(FileExistsError):
        main(['mmlu','inventory','--data-dir',str(root),'--output',str(inventory)])
    data = json.loads(inventory.read_text())
    data.update({'source_uri':'local://fixture','source_version':'v1',
                 'license_id':'CC0-1.0', 'license_evidence':'Test-authored data',
                 'review_status':'approved', 'reviewed_by':'fixture author',
                 'reviewed_at':'2026-11-21','allowed_uses':['evaluate']})
    inventory.write_text(json.dumps(data), encoding='utf-8')
    main(['mmlu','verify','--data-dir',str(root),'--inventory',str(inventory)])
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'HUMAN_ATTESTED_BYTES_VERIFIED'


def test_actual_single_token_reference_mode_from_local_toy_weights(tmp_path):
    root = _dataset(tmp_path)
    inventory = _approved_inventory(root, tmp_path)
    cp, tp, _ = _checkpoint(tmp_path, max_context=1024)
    scored = evaluate_mmlu(cp, tp, root, inventory, requested_shots=5,
                           scoring='reference_next_token', max_examples=1)
    assert scored['reference_compatible_scoring'] is True
    assert scored['all_requested_shots_retained'] is True
    assert scored['effective_shots_histogram'] == {'5': 1}
    assert scored['official_score'] is False


def test_small_budget_samples_multiple_subjects_round_robin(tmp_path):
    root = _dataset(tmp_path)
    other = 'computer_science'
    for split in ('dev', 'test'):
        src = root / split / f'abstract_algebra_{split}.csv'
        dst = root / split / f'{other}_{split}.csv'
        dst.write_bytes(src.read_bytes())
    inv = _approved_inventory(root, tmp_path)
    cp, tp, _ = _checkpoint(tmp_path)
    report = evaluate_mmlu(cp, tp, root, inv, scoring='continuation',
                           requested_shots=0, max_examples=2)
    assert report['questions'] == 2
    assert report['subject_count'] == 2
    assert report['partial_evaluation'] is True
    assert [r['id'] for r in report['results']] == [
        'abstract_algebra/test/0', 'computer_science/test/0']