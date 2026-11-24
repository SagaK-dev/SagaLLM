"""Synthetic calibration cases for fixed heuristic rules; no real benchmark."""
import pytest

from sagallm import java_overlap_screen as screen


@pytest.mark.parametrize("english,japanese", [
    ("Find the shortest path in a weighted graph.", "重み付きグラフで最短経路を求める。"),
    ("Remove duplicates and return the elements in descending order.", "重複を除去して降順に並べる。"),
    ("Use prefix sums to answer range queries.", "累積和で区間クエリに回答せよ。"),
    ("Find the intersection of two intervals, both half-open.", "半開区間について区間の共通部分を求める。"),
    ("Transpose the rectangular matrix.", "長方形の行列を転置せよ。"),
    ("Return the highest frequency word; in a tie use lexicographic order.", "出現回数の最大の語を求め、同数の場合は辞書順とする。"),
    ("Use binary search on the sorted array.", "整列済みの配列を二分探索する。"),
    ("Use union-find to count connected components.", "ユニオンファインドで連結成分の個数を数える。"),
])
def test_bilingual_positive_and_reverse(english, japanese):
    for query, exposure in [(english, japanese), (japanese, english)]:
        assert (0, "BILINGUAL_CONCEPT_RULE_MATCH") in screen.SimilarityScreen([query]).compare(exposure)


@pytest.mark.parametrize("query,exposure", [
    ("Find shortest path in a weighted graph.", "重み付きグラフの辺の総数を数える。"),
    ("Remove duplicates in descending order.", "数値を降順に並べる。"),
    ("Compute prefix sums for range queries.", "配列の累積和を出力する。"),
    ("Transpose a matrix.", "行列の対角成分を合計する。"),
    ("Use binary search on a sorted array.", "ソート済みの配列の最大値を出す。"),
    ("Read input and print an integer.", "入力を読み整数を出力する。"),
    ("Count the connected components with union-find.", "最短経路を求める。"),
])
def test_bilingual_negative_controls(query, exposure):
    assert screen.SimilarityScreen([query]).compare(exposure) == []


def test_near_duplicate_with_small_edit_and_long_document_tail():
    prompt = ("For every sensor packet, validate its header, recover the sequence "
              "identifier, and report the count of packets with corrupted checksums.")
    modified = prompt.replace("recover", "extract")
    result = screen.SimilarityScreen([prompt]).compare("background " * 3000 + modified)
    assert (0, "CHAR_SHINGLE_CONTAINMENT") in result
    # Two edits fall just below the fixed 82% threshold; no threshold tuning.
    below = modified.replace("report", "print")
    assert (0, "CHAR_SHINGLE_CONTAINMENT") not in screen.SimilarityScreen([prompt]).compare(below)


def test_unrelated_and_short_repetition_do_not_match():
    assert screen.SimilarityScreen(["abcde " * 100]).compare("abcde " * 100) == []
    query = "For each request, validate the header before decoding the timestamp and identifier."
    other = "A collection of birds migrates toward the coast when the season changes."
    assert screen.SimilarityScreen([query]).compare(other) == []


def test_character_budget_no_silent_truncation(monkeypatch):
    monkeypatch.setattr(screen, "MAX_SCAN_CHARS", 20)
    check = screen.SimilarityScreen(["A sufficiently long prompt to inspect later."])
    check.compare("0123456789")
    with pytest.raises(ValueError, match="budget"):
        check.compare("01234567890")
