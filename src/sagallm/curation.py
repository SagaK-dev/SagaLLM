"""Conservative local text-quality and privacy heuristics.

These rules are screening aids, not privacy guarantees. They never retain the
matched sensitive value in diagnostic metadata or logs.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

from .data import EMAIL, SECRET

_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE)
_CREDENTIAL = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
_JP_MOBILE = re.compile(r"(?<!\d)(?:070|080|090)[ -]?\d{4}[ -]?\d{4}(?!\d)")
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn(digits: str) -> bool:
    total = 0
    for offset, ch in enumerate(reversed(digits)):
        value = int(ch)
        if offset % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def sensitive_reason(text: str) -> str | None:
    """Return a category, not a matched secret; false positives are possible."""
    if EMAIL.search(text):
        return "email"
    if SECRET.search(text) or _CREDENTIAL.search(text) or _PRIVATE_KEY.search(text):
        return "credential"
    if _JP_MOBILE.search(text):
        return "phone"
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return "payment_card"
    return None


def normalized_text(text: str) -> str:
    """Normalize whitespace and Unicode without stripping meaningful code punctuation."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def low_quality(text: str) -> bool:
    """Only discard obviously non-textual documents; avoid language bias."""
    return bool(text) and sum(1 for char in text if not char.isprintable() and char not in "\n\r\t") * 5 > len(text)


def _shingles(text: str, *, stride: int = 1, width: int = 5):
    if len(text) < width:
        yield text
    else:
        for start in range(0, len(text) - width + 1, stride):
            yield text[start:start + width]


def simhash64(text: str) -> int:
    """Stable 64-bit character 5-gram SimHash, bounded at 4096 sampled grams."""
    text = normalized_text(text)
    count = max(1, len(text) - 4)
    stride = max(1, (count + 4095) // 4096)
    scores = [0] * 64
    for gram in _shingles(text, stride=stride):
        digest = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            scores[bit] += 1 if digest & (1 << bit) else -1
    return sum(1 << bit for bit, score in enumerate(scores) if score > 0)


def near_duplicate(a: str, b: str, *, min_similarity: float = 0.84) -> bool:
    """Confirm candidate similarity with sampled character 5-gram Jaccard."""
    if not 0 < min_similarity <= 1:
        raise ValueError("min_similarity must lie in (0, 1]")
    a, b = normalized_text(a), normalized_text(b)
    if a == b:
        return True
    if min(len(a), len(b)) < 80:
        return False
    sa = set(_shingles(a, stride=max(1, (len(a) + 8191) // 8192)))
    sb = set(_shingles(b, stride=max(1, (len(b) + 8191) // 8192)))
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= min_similarity