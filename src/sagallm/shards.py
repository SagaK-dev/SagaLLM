"""Offline, disk-backed document splits and memory-mapped-style uint32 token shards.

SQLite provides bounded-memory exact deduplication; binary shards are read with
seek/read rather than materializing the entire token corpus in RAM.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from typing import Iterator

import torch

from .data import EMAIL, SECRET, read_documents
from .tokenization import ByteBPETokenizer


def safe_shard_path(root: Path, filename: str) -> Path:
    """Keep manifest paths inside the specified shard directory."""
    if not isinstance(filename, str) or filename in {'', '.', '..'}:
        raise ValueError('Invalid shard filename')
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        raise ValueError('Manifest shard paths must be simple filenames')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Shard path escapes the dataset directory')
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_license(license_id: str) -> bool:
    return bool(license_id.strip()) and license_id.strip().lower() not in {
        'none', 'unknown', 'unverified', 'unspecified'
    }


def prepare_shards(source: str | Path, output: str | Path, license_id: str,
                   *, validation_fraction: float = 0.1, max_docs_per_shard: int = 1000,
                   max_document_bytes: int = 1024 * 1024) -> dict:
    """Stream records into a temporary SQLite db and produce deterministic JSONL shards.

    Every input document is deduplicated by its raw UTF-8 text after trimming;
    split membership is by a deterministic SHA-256 rank, independent of order.
    PII screening is heuristic and not a guarantee of privacy.
    """
    if not _valid_license(license_id):
        raise ValueError('A reviewed dataset license identifier is required')
    if not 0 < validation_fraction < 1 or max_docs_per_shard < 1 or max_document_bytes < 1:
        raise ValueError('Invalid split fraction or shard limits')
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {'empty': 0, 'sensitive': 0, 'duplicate': 0, 'oversize': 0}
    with tempfile.TemporaryDirectory(prefix='.sagallm-prepare-', dir=output.parent) as temporary:
        stage = Path(temporary)
        db = sqlite3.connect(stage / 'dedup.sqlite')
        try:
            db.execute('CREATE TABLE documents (hash TEXT PRIMARY KEY, text TEXT NOT NULL)')
            for text in read_documents(source):
                text = text.strip()
                if not text:
                    counts['empty'] += 1
                    continue
                if len(text.encode('utf-8')) > max_document_bytes:
                    counts['oversize'] += 1
                    continue
                if EMAIL.search(text) or SECRET.search(text):
                    counts['sensitive'] += 1
                    continue
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                cursor = db.execute('INSERT OR IGNORE INTO documents VALUES (?, ?)', (digest, text))
                if not cursor.rowcount:
                    counts['duplicate'] += 1
            db.commit()
            total = db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            if total < 2:
                raise ValueError('At least two eligible unique documents are required')
            val_count = min(total - 1, max(1, round(total * validation_fraction)))
            results = {}
            for split, skip, limit in [('validation', 0, val_count), ('train', val_count, total - val_count)]:
                files = []
                count = 0
                handle = None
                for _digest, text in db.execute('SELECT hash, text FROM documents ORDER BY hash LIMIT ? OFFSET ?', (limit, skip)):
                    if count % max_docs_per_shard == 0:
                        if handle is not None:
                            handle.close()
                        name = f'{split}-{len(files):05}.jsonl'
                        files.append(name)
                        handle = (stage / name).open('w', encoding='utf-8')
                    handle.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if handle is not None:
                    handle.close()
                results[split] = {'documents': count, 'files': [
                    {'path': name, 'sha256': sha256_file(stage / name)} for name in files
                ]}
            manifest = {
                'format': 'sagallm-doc-shards-v1', 'source': str(source),
                'source_sha256': sha256_file(source), 'license': license_id,
                'license_review_required': True,
                'validation_fraction': validation_fraction, 'unique_documents': total,
                'rejected': counts, 'splits': results,
                'limitations': ['exact dedup only', 'regex-based sensitive data screening only',
                                'no automatic license verification', 'no semantic quality scoring']
            }
        finally:
            db.close()
        (stage / 'dedup.sqlite').unlink(missing_ok=True)
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(stage, output)
    return manifest


def _documents_from_shard(path: Path) -> Iterator[str]:
    with path.open(encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            if not isinstance(record.get('text'), str):
                raise ValueError(f'Invalid document shard: {path}')
            yield record['text']


def tokenize_shards(prepared_dir: str | Path, tokenizer_path: str | Path, output: str | Path,
                    *, max_tokens_per_shard: int = 100_000) -> dict:
    """Convert text shards to uint32-le files; do not hold the full corpus in RAM."""
    if max_tokens_per_shard < 2:
        raise ValueError('max_tokens_per_shard must be at least 2')
    prepared_dir, tokenizer_path, output = Path(prepared_dir), Path(tokenizer_path), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing token dataset: {output}')
    source = json.loads((prepared_dir / 'manifest.json').read_text(encoding='utf-8'))
    if source.get('format') != 'sagallm-doc-shards-v1':
        raise ValueError('Incompatible input data manifest')
    tok = ByteBPETokenizer.load(tokenizer_path)
    if tok.vocab_size > 2**32:
        raise ValueError('Tokenizer exceeds uint32 vocabulary')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sagallm-tokenize-', dir=output.parent) as temp:
        stage = Path(temp)
        splits = {}
        for split in ('train', 'validation'):
            shards = []
            handle = None
            count = 0
            total_tokens = 0
            total_docs = 0

            def finalize():
                nonlocal handle, count
                if handle is not None:
                    handle.close()
                    filename = shards[-1]['path']
                    shards[-1]['sha256'] = sha256_file(stage / filename)
                    handle = None

            for source_item in source['splits'][split]['files']:
                input_file = safe_shard_path(prepared_dir, source_item['path'])
                if sha256_file(input_file) != source_item['sha256']:
                    raise ValueError(f'Prepared shard hash mismatch: {input_file}')
                for document in _documents_from_shard(input_file):
                    ids = tok.encode(document, add_eos=True)
                    if len(ids) < 1:
                        continue
                    if any(token < 0 or token >= tok.vocab_size for token in ids):
                        raise ValueError('Tokenizer produced invalid token ID')
                    if handle is None or (count and count + len(ids) > max_tokens_per_shard):
                        finalize()
                        filename = f'{split}-{len(shards):05}.bin'
                        handle = (stage / filename).open('wb')
                        count = 0
                        shards.append({'path': filename, 'tokens': 0})
                    # Length is bounded by prepare_shards max_document_bytes.
                    handle.write(struct.pack(f'<{len(ids)}I', *ids))
                    count += len(ids)
                    shards[-1]['tokens'] = count
                    total_tokens += len(ids)
                    total_docs += 1
            finalize()
            splits[split] = {'documents': total_docs, 'tokens': total_tokens, 'files': shards}
        manifest = {
            'format': 'sagallm-tokens-uint32le-v1',
            'tokenizer_sha256': sha256_file(tokenizer_path),
            'source_manifest_sha256': sha256_file(prepared_dir / 'manifest.json'),
            'source_sha256': source['source_sha256'],
            'license': source['license'], 'splits': splits,
            'note': 'Files contain little-endian uint32 IDs. Splits use document-level source separation.'
        }
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(stage, output)
    return manifest


class TokenShardDataset:
    """Read random contiguous token windows from immutable on-disk shards."""

    def __init__(self, directory: str | Path, *, split: str = 'train', block_size: int = 32,
                 tokenizer_path: str | Path | None = None, verify: bool = True):
        if split not in {'train', 'validation'} or block_size < 1:
            raise ValueError('Invalid split or block size')
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('format') != 'sagallm-tokens-uint32le-v1':
            raise ValueError('Unsupported token dataset format')
        if tokenizer_path and self.manifest['tokenizer_sha256'] != sha256_file(tokenizer_path):
            raise ValueError('Token dataset tokenizer does not match tokenizer file')
        self.block_size = block_size
        self.shards: list[tuple[Path, int]] = []
        self.cumulative: list[int] = []
        total = 0
        for entry in self.manifest['splits'][split]['files']:
            path = safe_shard_path(self.directory, entry['path'])
            tokens = int(entry['tokens'])
            if tokens < 0 or path.stat().st_size != tokens * 4:
                raise ValueError(f'Corrupt token shard size: {path}')
            if verify and sha256_file(path) != entry['sha256']:
                raise ValueError(f'Corrupt token shard hash: {path}')
            valid = max(0, tokens - block_size)
            if valid:
                total += valid
                self.shards.append((path, valid))
                self.cumulative.append(total)
        if not self.cumulative:
            raise ValueError(f'No token shard has at least {block_size + 1} tokens in split {split}')
        self.total_windows = total
        self.split = split
        self.data_sha = self.manifest['source_manifest_sha256'] + ':' + split

    def sample(self, generator: torch.Generator, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(self.total_windows, (batch_size,), generator=generator).tolist()
        inputs, labels = [], []
        for index in indices:
            shard = bisect_right(self.cumulative, index)
            offset = index - (self.cumulative[shard-1] if shard else 0)
            path, _ = self.shards[shard]
            with path.open('rb') as stream:
                stream.seek(offset * 4)
                data = stream.read((self.block_size + 1) * 4)
            values = struct.unpack(f'<{self.block_size+1}I', data)
            inputs.append(torch.tensor(values[:-1], dtype=torch.long))
            labels.append(torch.tensor(values[1:], dtype=torch.long))
        return torch.stack(inputs).to(device), torch.stack(labels).to(device)