"""Deterministic near-duplicate and bilingual algorithm review leads.

No learned semantic model, Java parser, network, or inference is used. A hit
is a reason to review, not proof of shared semantics; a miss is not clearance.
"""
from __future__ import annotations

import re
import unicodedata

VERSION = "java-overlap-heuristic-v1"
GRAM_SIZE = 5
MIN_DISTINCT_GRAMS = 45
CONTAINMENT_PERCENT = 82
MAX_SCAN_CHARS = 128_000_000

# Deliberately small, explicit English/Japanese concept vocabulary. Generic
# words (input, output, integer, array) alone never trigger a semantic lead.
PATTERNS = {
    "weighted_graph": r"\bweighted (?:directed |undirected )?graphs?\b|重み(?:付|つ)き(?:有向|無向)?グラフ",
    "shortest_path": r"\bshortest (?:paths?|routes?|distance)\b|最短(?:経路|距離|路)",
    "deduplication": r"\b(?:deduplicat\w*|remove duplicates?|distinct elements?|unique elements?)\b|重複(?:を除|を取り除|排除|除去)|異なる要素",
    "descending": r"\bdescending(?: order)?\b|降順|大きい順",
    "prefix_sum": r"\b(?:prefix sums?|cumulative sums?)\b|累積和|接頭辞和",
    "range_query": r"\brange (?:sum )?quer(?:y|ies)\b|区間(?:和|クエリ|問い合わせ)",
    "half_open": r"\bhalf[- ]open\b|半開区間",
    "interval_overlap": r"\b(?:overlap(?:ping)? intervals?|interval intersection|intersection of (?:two )?intervals?)\b|区間の(?:重なり|共通部分|交差)",
    "matrix": r"\bmatri(?:x|ces)\b|行列",
    "transpose": r"\btranspos\w*\b|転置",
    "frequency": r"\b(?:frequenc(?:y|ies)|occurrence counts?)\b|出現回数|頻度",
    "tie_break": r"\b(?:tie[- ]break\w*|in (?:case of |the event of )?a tie|ties are broken)\b|同率|同数の場合|同点の場合",
    "lexicographic": r"\blexicograph\w*\b|辞書順",
    "binary_search": r"\bbinary search\b|二分探索|二分検索",
    "sorted_sequence": r"\b(?:sorted (?:array|sequence|list)|nondecreasing order)\b|整列済み|ソート済み|昇順に整列",
    "connected_components": r"\bconnected components?\b|連結成分",
    "union_find": r"\b(?:union[- ]find|disjoint[- ]set)\b|素集合データ構造|ユニオンファインド",
}
RULES = (
    ("weighted_graph", "shortest_path"),
    ("deduplication", "descending"),
    ("prefix_sum", "range_query"),
    ("half_open", "interval_overlap"),
    ("matrix", "transpose"),
    ("frequency", "tie_break", "lexicographic"),
    ("binary_search", "sorted_sequence"),
    ("connected_components", "union_find"),
)
COMPILED = {key: re.compile(value) for key, value in PATTERNS.items()}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _compact(text: str) -> str:
    # Keep Unicode letters/digits, ignoring formatting and punctuation.
    return re.sub(r"[\W_]+", "", text)


def _grams(text: str):
    for i in range(max(0, len(text) - GRAM_SIZE + 1)):
        yield text[i:i + GRAM_SIZE]


def _rules(text: str, relevant: set[str] | None = None) -> frozenset[int]:
    concepts = {key for key, pattern in COMPILED.items()
                if (relevant is None or key in relevant) and pattern.search(text)}
    return frozenset(i for i, rule in enumerate(RULES)
                     if all(key in concepts for key in rule))


class SimilarityScreen:
    """Bounded candidate-oriented scan with no retained exposure text.

    Only candidate-sized sets are retained; exposure text is walked once for
    character shingles, with no silent truncation at the end of a document.
    """
    def __init__(self, prompts: list[str]):
        normalized = [normalize(p) for p in prompts]
        self.grams = [frozenset(_grams(_compact(p))) for p in normalized]
        self.grams = [g if len(g) >= MIN_DISTINCT_GRAMS else frozenset()
                      for g in self.grams]
        self.target_grams = frozenset().union(*self.grams)
        self.rules = [_rules(p) for p in normalized]
        active = frozenset().union(*self.rules)
        self.relevant = {key for i in active for key in RULES[i]}
        self.scanned_chars = 0

    def compare(self, text: str) -> list[tuple[int, str]]:
        text = normalize(text)
        self.scanned_chars += len(text)
        if self.scanned_chars > MAX_SCAN_CHARS:
            raise ValueError("Similarity character budget exceeded")
        matched = set()
        if self.target_grams:
            for gram in _grams(_compact(text)):
                if gram in self.target_grams:
                    matched.add(gram)
        rules = _rules(text, self.relevant) if self.relevant else frozenset()
        results = []
        for i, grams in enumerate(self.grams):
            if grams and 100 * len(grams & matched) >= CONTAINMENT_PERCENT * len(grams):
                results.append((i, "CHAR_SHINGLE_CONTAINMENT"))
            if self.rules[i] & rules:
                results.append((i, "BILINGUAL_CONCEPT_RULE_MATCH"))
        return results
