"""Offline, fail-closed validation of *human-asserted* dataset usage reviews.

A card documents who reviewed a source and exactly which bytes were reviewed.
It cannot establish that the reviewer is correct or that the license grants rights.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re

from .shards import sha256_file


CARD_FORMAT = 'sagallm-dataset-card-v1'
USES = frozenset({'train', 'evaluate', 'redistribute'})
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_INVALID_LICENSE = frozenset({'', 'unknown', 'none', 'unverified', 'unspecified', 'tbd', 'pending'})


class DatasetCardError(ValueError):
    """Invalid review metadata, missing permission, or mismatched source data."""


def verify_dataset_card(source: str | Path, card: str | Path, *, purpose: str,
                        expected_license: str | None = None) -> dict:
    """Validate a local file against a manually reviewed, purpose-specific card.

    No network calls, automatic permission grants or source-data logging occur.
    The caller must independently assess the underlying rights and provenance.
    """
    if purpose not in USES:
        raise DatasetCardError('Unknown dataset use')
    card_path = Path(card)
    if card_path.stat().st_size > 32 * 1024:
        raise DatasetCardError('Dataset card exceeds 32 KiB')
    try:
        data = json.loads(card_path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetCardError('Invalid UTF-8 JSON dataset card') from exc
    if not isinstance(data, dict) or data.get('format') != CARD_FORMAT:
        raise DatasetCardError('Unsupported dataset card format')
    for key in ('name', 'source_uri', 'source_version', 'license_id',
                'license_evidence', 'reviewed_by', 'reviewed_at'):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 1024:
            raise DatasetCardError(f'Missing or invalid {key}')
    license_id = data['license_id'].strip()
    if license_id.lower() in _INVALID_LICENSE:
        raise DatasetCardError('Unreviewed or unspecified license')
    if expected_license is not None and license_id != expected_license:
        raise DatasetCardError('Dataset card license does not match requested license')
    allowed = data.get('allowed_uses')
    if not isinstance(allowed, list) or not allowed or any(
        not isinstance(use, str) or use not in USES for use in allowed
    ) or len(set(allowed)) != len(allowed):
        raise DatasetCardError('Invalid allowed_uses')
    if data.get('review_status') != 'approved':
        raise DatasetCardError('Dataset usage has not been explicitly reviewed and approved')
    try:
        reviewed = date.fromisoformat(data['reviewed_at'])
    except ValueError as exc:
        raise DatasetCardError('reviewed_at must be an ISO date') from exc
    if reviewed.isoformat() != data['reviewed_at']:
        raise DatasetCardError('reviewed_at must be YYYY-MM-DD')
    expected_hash = data.get('file_sha256')
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        raise DatasetCardError('file_sha256 must be a lowercase SHA-256 hex digest')
    actual_hash = sha256_file(source)
    if actual_hash != expected_hash:
        raise DatasetCardError('Dataset file SHA-256 differs from the reviewed source')
    if purpose not in allowed:
        raise DatasetCardError(f'Dataset card does not authorize the {purpose} use')
    return {
        'status': 'REVIEW_ATTESTED_NOT_LEGALLY_VERIFIED',
        'name': data['name'], 'source_uri': data['source_uri'],
        'source_version': data['source_version'], 'license_id': license_id,
        'reviewed_by': data['reviewed_by'], 'reviewed_at': reviewed.isoformat(),
        'allowed_uses': sorted(allowed), 'purpose': purpose,
        'source_sha256': actual_hash, 'card_sha256': sha256_file(card_path),
        'warning': 'Card is a human assertion, not automated proof of licensing rights.',
    }