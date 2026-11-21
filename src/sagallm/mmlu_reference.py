"""Offline Hendrycks-style MMLU 5-shot evaluator for local, reviewed CSV data.

Matches the original reference *prompt construction* and next-answer-label
scoring where the SagaLLM tokenizer makes each label one stable token.
The output is never certified as an official MMLU leaderboard result.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import torch
from torch.nn import functional as F

from .benchmarks import audit_contamination, BenchmarkContaminationError, _choice_score
from .config import ModelConfig
from .model import SagaTransformer
from .shards import sha256_file
from .tokenization import load_tokenizer
from .training import checkpoint_load, resolve_device, verify_checkpoint_tokenizer


INVENTORY_FORMAT = 'sagallm-mmlu-inventory-v1'
PROTOCOL_ID = 'sagallm-hendrycks-csv-five-shot-v1'
_SUBJECT = re.compile(r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_ANSWERS = 'ABCD'
_MAX_CSV_BYTES = 32 * 1024 * 1024
_MAX_ROWS = 20_000


class MMLUDataError(ValueError):
    """Invalid data layout, review or benchmark record."""


class MMLUTokenizationError(ValueError):
    """The model tokenizer cannot perform reference one-token label scoring."""


def discover_csv_files(data_dir: str | Path) -> dict[str, Path]:
    """Find exact dev/test CSV pairs, disallow symlinks and unmatched subjects."""
    root = Path(data_dir)
    if not root.is_dir() or root.is_symlink():
        raise MMLUDataError('Data root must be a local directory, not a symlink')
    discovered: dict[str, Path] = {}
    split_subjects: dict[str, set[str]] = {}
    for split in ('dev', 'test'):
        directory = root / split
        if not directory.is_dir() or directory.is_symlink():
            raise MMLUDataError(f'Missing or unsafe {split} directory')
        subjects = set()
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.is_symlink() or path.suffix != '.csv':
                raise MMLUDataError(f'Unexpected file in {split} directory: {path.name}')
            suffix = f'_{split}.csv'
            if not path.name.endswith(suffix):
                raise MMLUDataError(f'Invalid {split} CSV filename: {path.name}')
            subject = path.name[:-len(suffix)]
            if not _SUBJECT.fullmatch(subject):
                raise MMLUDataError(f'Invalid MMLU subject: {subject}')
            if path.stat().st_size > _MAX_CSV_BYTES:
                raise MMLUDataError(f'CSV exceeds 32 MiB: {path.name}')
            subjects.add(subject)
            discovered[f'{split}/{path.name}'] = path
        if not subjects:
            raise MMLUDataError(f'No {split} CSV files')
        split_subjects[split] = subjects
    if split_subjects['dev'] != split_subjects['test']:
        raise MMLUDataError('MMLU dev/test subject sets are not identical')
    return discovered


def create_inventory_template(data_dir: str | Path, destination: str | Path) -> dict:
    """Write hashes and *pending* legal review. Never silently approve use."""
    discovered = discover_csv_files(data_dir)
    result = {
        'format': INVENTORY_FORMAT,
        'source_uri': 'REVIEW_REQUIRED', 'source_version': 'REVIEW_REQUIRED',
        'license_id': 'REVIEW_REQUIRED', 'license_evidence': 'REVIEW_REQUIRED',
        'review_status': 'pending', 'reviewed_by': '', 'reviewed_at': '',
        'allowed_uses': [],
        'files': {name: sha256_file(path) for name, path in sorted(discovered.items())},
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return {'inventory': str(path), 'status': 'PENDING_REVIEW',
            'file_count': len(discovered), 'subjects': len(discovered) // 2}


def verify_inventory(data_dir: str | Path, inventory_path: str | Path) -> dict:
    """Check exact local bytes against explicit human review metadata.

    Does not independently establish the author's rights or license grant.
    """
    path = Path(inventory_path)
    if path.stat().st_size > 256 * 1024:
        raise MMLUDataError('Inventory exceeds 256 KiB')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MMLUDataError('Invalid inventory JSON') from exc
    if not isinstance(data, dict) or data.get('format') != INVENTORY_FORMAT:
        raise MMLUDataError('Invalid MMLU inventory format')
    for key in ('source_uri', 'source_version', 'license_id', 'license_evidence',
                'reviewed_by', 'reviewed_at'):
        value = data.get(key)
        if (not isinstance(value, str) or not value.strip() or len(value) > 1024 or
            value.strip().lower() in ('review_required', 'unknown', 'pending', 'unspecified', 'none')):
            raise MMLUDataError(f'Invalid or pending inventory field: {key}')
    try:
        date.fromisoformat(data['reviewed_at'])
    except ValueError as exc:
        raise MMLUDataError('reviewed_at must be an ISO date') from exc
    if data.get('review_status') != 'approved' or data.get('allowed_uses') != ['evaluate']:
        raise MMLUDataError('MMLU data must be explicitly approved for evaluation only')
    discovered = discover_csv_files(data_dir)
    files = data.get('files')
    if not isinstance(files, dict) or set(files) != set(discovered):
        raise MMLUDataError('Inventory paths do not exactly match the dataset files')
    for name, csv_path in discovered.items():
        digest = files[name]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise MMLUDataError(f'Invalid SHA-256 for {name}')
        if sha256_file(csv_path) != digest:
            raise MMLUDataError(f'MMLU data SHA-256 mismatch: {name}')
    return {'status': 'HUMAN_ATTESTED_BYTES_VERIFIED',
            'reviewed_by': data['reviewed_by'], 'reviewed_at': data['reviewed_at'],
            'source_uri': data['source_uri'], 'source_version': data['source_version'],
            'license_id': data['license_id'], 'file_count': len(discovered),
            'inventory_sha256': sha256_file(path), 'files': dict(files)}


def read_mmlu_csv(csv_path: Path, *, max_rows: int = _MAX_ROWS) -> list[dict]:
    """Parse six-column MMLU CSV without pandas and without leaking text to logs."""
    if not 1 <= max_rows <= _MAX_ROWS:
        raise MMLUDataError('Invalid row limit')
    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, strict=True)
        for line_no, row in enumerate(reader, 1):
            if len(row) != 6 or any(not isinstance(v, str) or not v.strip() or len(v) > 8192 for v in row):
                raise MMLUDataError(f'Expected six nonempty columns at {csv_path.name}:{line_no}')
            if row[5].strip() not in _ANSWERS:
                raise MMLUDataError(f'Invalid answer label at {csv_path.name}:{line_no}')
            rows.append({'question': row[0], 'options': row[1:5], 'answer': row[5].strip()})
            if len(rows) > max_rows:
                raise MMLUDataError(f'CSV row limit exceeded: {csv_path.name}')
    if not rows:
        raise MMLUDataError(f'Empty MMLU CSV: {csv_path.name}')
    return rows


def format_example(row: dict, *, include_answer: bool) -> str:
    """Match hendrycks/test/evaluate.py example layout (no chat template)."""
    text = row['question'] + ''.join(f'\n{letter}. {option}' for letter, option in zip(_ANSWERS, row['options']))
    text += '\nAnswer:'
    if include_answer:
        text += f" {row['answer']}\n\n"
    return text


def format_prompt(subject: str, dev_rows: list[dict], row: dict, shots: int) -> str:
    if shots < 0 or shots > len(dev_rows):
        raise MMLUDataError('Not enough dev examples for requested shots')
    # Original format_subject() preserves the leading space, resulting in two spaces after 'about'.
    header = f"The following are multiple choice questions (with answers) about  {' '.join(subject.split('_'))}.\n\n"
    return header + ''.join(format_example(sample, include_answer=True)
                            for sample in dev_rows[:shots]) + format_example(row, include_answer=False)


def _fits_context(tokenizer, prompt: str, context_length: int, scoring: str) -> bool:
    prefix = tokenizer.encode(prompt, add_bos=True)
    if scoring == 'reference_next_token':
        return len(prefix) + 1 <= context_length
    return len(prefix) + max(len(tokenizer.encode(' ' + label)) for label in _ANSWERS) <= context_length


def select_fitting_prompt(tokenizer, subject: str, dev_rows: list[dict], row: dict,
                          requested_shots: int, context_length: int, scoring: str) -> tuple[str, int]:
    for shots in range(requested_shots, -1, -1):
        prompt = format_prompt(subject, dev_rows, row, shots)
        if _fits_context(tokenizer, prompt, context_length, scoring):
            return prompt, shots
    raise MMLUDataError('Test question alone exceeds model context; no question silently skipped')


def _label_scores(model, tokenizer, prompt: str, device: torch.device,
                  scoring: str) -> list[float]:
    if scoring == 'continuation':
        return [_choice_score(model, tokenizer, prompt, ' '+label, device, 'sum')
                for label in _ANSWERS]
    input_ids = tokenizer.encode(prompt, add_bos=True)
    suffix_ids = [tokenizer.encode(' '+label) for label in _ANSWERS]
    if any(len(ids) != 1 or tokenizer.encode(prompt+' '+label) != tokenizer.encode(prompt)+ids
           for label, ids in zip(_ANSWERS, suffix_ids)):
        raise MMLUTokenizationError(
            'Answer labels are not stable single tokens for this tokenizer. '
            'Reference next-token MMLU scoring is not available; use --scoring continuation '
            'only as a explicitly non-comparable diagnostic.')
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(tensor).logits[0, -1, :].float()
    probabilities = F.log_softmax(logits, dim=-1)
    return [float(probabilities[ids[0]].item()) for ids in suffix_ids]


def _checkpoint_corpus_status(state: dict, training_docs: str | Path | None,
                              tokenizer_path: str | Path) -> dict:
    if training_docs is None:
        return {'status': 'NOT_CHECKED', 'checkpoint_corpus_verified': False}
    expected = sha256_file(Path(training_docs) / 'manifest.json') + ':train:' + sha256_file(tokenizer_path)
    observed = state.get('train_state', {}).get('dataset_sha256')
    if observed is not None and observed != expected:
        raise MMLUDataError('Training document manifest does not match checkpoint corpus')
    return {'status': 'VERIFIED_SHARD_SOURCE' if observed == expected
            else 'CHECKPOINT_CORPUS_NOT_RECORDED', 'checkpoint_corpus_verified': observed == expected}


def evaluate_mmlu(checkpoint: str | Path, tokenizer_path: str | Path,
                  data_dir: str | Path, inventory_path: str | Path, *,
                  subjects: list[str] | None = None, requested_shots: int = 5,
                  max_examples: int = 100, scoring: str = 'reference_next_token',
                  device_name: str = 'cpu', training_docs: str | Path | None = None,
                  allow_contaminated: bool = False) -> dict:
    """Evaluate local reviewed CSV split with reference-like prompt, never downloading data."""
    if scoring not in ('reference_next_token', 'continuation'):
        raise ValueError('Unknown MMLU scoring mode')
    if not 0 <= requested_shots <= 5 or not 1 <= max_examples <= _MAX_ROWS:
        raise ValueError('Shots must be 0-5 and max_examples must be 1-20000')
    review = verify_inventory(data_dir, inventory_path)
    all_files = discover_csv_files(data_dir)
    available = sorted(n.removeprefix('test/').removesuffix('_test.csv')
                       for n in all_files if n.startswith('test/'))
    names = available if subjects is None else list(dict.fromkeys(subjects))
    if not names or any(not _SUBJECT.fullmatch(s) or s not in available for s in names):
        raise MMLUDataError('Requested subjects are not present in the reviewed data')

    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    verify_checkpoint_tokenizer(state, tokenizer_path)
    config = ModelConfig(**state['config'])
    if config.parameter_count() > 200_000_000:
        raise ValueError('CPU-first MMLU evaluator refuses to allocate models above 200M parameters')
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError('Tokenizer vocabulary exceeds model vocabulary')
    # Under a small local evaluation budget, sample subjects round-robin
    # rather than silently exhausting early alphabetical subjects first.
    subject_data = {}
    source_count = 0
    for subject in names:
        dev = read_mmlu_csv(all_files[f'dev/{subject}_dev.csv'])
        if len(dev) < requested_shots:
            raise MMLUDataError(f'Dev split for {subject} has fewer than {requested_shots} examples')
        test = read_mmlu_csv(all_files[f'test/{subject}_test.csv'])
        source_count += len(test)
        subject_data[subject] = (dev, test)
    records = []
    prompts_for_audit = []
    indexes = {subject: 0 for subject in names}
    while len(records) < max_examples:
        added = False
        for subject in names:
            dev, test = subject_data[subject]
            row_index = indexes[subject]
            if row_index >= len(test):
                continue
            row = test[row_index]
            rec_id = f'{subject}/test/{row_index}'
            records.append((subject, rec_id, dev, row))
            prompts_for_audit.append({'id': rec_id, 'source_question': row['question'],
                                      'prompt': row['question']})
            indexes[subject] += 1
            added = True
            if len(records) >= max_examples:
                break
        if not added:
            break
    for subject in names:
        dev, _ = subject_data[subject]
        for k, d in enumerate(dev[:requested_shots]):
            prompts_for_audit.append({'id': f'{subject}/dev/{k}',
                                      'source_question': d['question'], 'prompt': d['question']})
    if not records:
        raise MMLUDataError('No test examples')
    if training_docs is None:
        contamination = {'status': 'NOT_CHECKED', 'matched_ids': [], 'unchecked_short_ids': [],
                         'method': 'No prepared train-doc corpus supplied'}
    else:
        contamination = audit_contamination(prompts_for_audit, training_docs)
    if contamination['matched_ids'] and not allow_contaminated:
        raise BenchmarkContaminationError('Train documents overlap MMLU test or few-shot dev questions')
    contamination['checkpoint'] = _checkpoint_corpus_status(state, training_docs, tokenizer_path)

    device = resolve_device(device_name)
    model = SagaTransformer(config)
    checkpoint_load(checkpoint, model)
    model.to(device).eval()
    correct = 0
    counts = Counter()
    histogram = Counter()
    per_subject = defaultdict(lambda: {'questions': 0, 'correct': 0})
    results = []
    with torch.inference_mode():
        for subject, identifier, dev_rows, row in records:
            prompt, shots = select_fitting_prompt(tokenizer, subject, dev_rows, row,
                                                  requested_shots, config.max_position_embeddings,
                                                  scoring)
            scores = _label_scores(model, tokenizer, prompt, device, scoring)
            prediction = max(range(4), key=lambda i: scores[i])
            gold = _ANSWERS.index(row['answer'])
            passed = prediction == gold
            correct += passed
            histogram[str(shots)] += 1
            group = per_subject[subject]
            group['questions'] += 1
            group['correct'] += int(passed)
            results.append({'id': identifier, 'subject': subject,
                            'prediction': _ANSWERS[prediction], 'answer': row['answer'],
                            'correct': passed, 'answer_logprobs': scores,
                            'effective_shots': shots})
    metrics = {name: {'questions': v['questions'], 'correct': v['correct'],
                      'accuracy': v['correct'] / v['questions']}
               for name, v in sorted(per_subject.items()) if v['questions']}
    return {
        'evaluation_type': 'local_hendrycks_csv_mmlu', 'protocol_id': PROTOCOL_ID,
        'protocol_reference': 'https://github.com/hendrycks/test/blob/master/evaluate.py',
        'scoring': scoring, 'reference_compatible_scoring': scoring == 'reference_next_token',
        'standard_benchmark_certified': False,
        'official_score': False, 'partial_evaluation': (len(records) < source_count or
                                                       len(names) != len(available)),
        'subject_count': len(metrics), 'selection_policy': 'subject_round_robin_source_row_order',
        'questions': len(records), 'correct': int(correct),
        'accuracy': correct / len(records), 'subject_accuracy': metrics,
        'macro_subject_accuracy': sum(g['accuracy'] for g in metrics.values()) / len(metrics),
        'shots_requested': requested_shots, 'effective_shots_histogram': dict(histogram),
        'all_requested_shots_retained': histogram.get(str(requested_shots), 0) == len(records),
        'checkpoint_sha256': sha256_file(checkpoint),
        'tokenizer_sha256': sha256_file(tokenizer_path), 'review': review,
        'dataset_file_sha256': review['files'], 'device': str(device),
        'contamination': contamination,
        'limitations': ('Reference prompt layout and conditional label scoring only; '
                        'dataset completeness, source rights, model training provenance, '
                        'and official leaderboard comparability are not certified.'),
        'results': results,
    }