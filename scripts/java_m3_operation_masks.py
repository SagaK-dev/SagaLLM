"""V2 masks for the frozen ASCII synthetic TRAIN templates.

The annotations are task-specific lexical regions, not a general Java parser.
No evaluation source or model output can choose the training mask.
"""
from __future__ import annotations

from sagallm.java_stdio_curriculum import TRAIN_FAMILIES, _task


def operation_text(family: str, variant: int) -> str:
    if family not in TRAIN_FAMILIES or type(variant) is not int or not 0 <= variant < 32:
        raise ValueError("Only fixed stdio TRAIN families and variants")
    lo = -12 + variant % 9
    hi = lo + 5 + variant % 7
    return {
        "stdin_add_offset": "a+b",
        "stdin_multiply_offset": "a*b",
        "stdin_max_offset": "Math.max(a,b)",
        "stdin_absolute_gap_offset": "Math.abs(a-b)",
        "stdin_count_greater": f"if(v>{variant-14})total++",
        "stdin_even_sum_offset": "if(v%2==0)total+=v",
        "stdin_digit_sum_offset": "total+=n%10;n/=10",
        "stdin_clamp": f"Math.max({lo},Math.min({hi},n))",
        "stdin_ascii_upper_suffix": "line.toUpperCase(java.util.Locale.ROOT)",
        "stdin_ascii_reverse_suffix": "new StringBuilder(line).reverse().toString()",
    }[family]


def span_positions(source: str, operation: str, ids: list[int], tokenizer) -> tuple[int, ...]:
    # The templates are ASCII; decoding one token at a time is lossless here.
    # Reconstruct every token boundary, and reject non-concatenative decoding.
    if not source.isascii() or not operation or source.count(operation)!=1:
        raise ValueError("Operation must identify exactly one ASCII region")
    if not ids or len(ids)>384 or set(ids)&set(tokenizer.SPECIAL.values()):
        raise ValueError("Unexpected special or excessive reference tokens")
    pieces=[tokenizer.decode([i]) for i in ids]
    if any(not p or not p.isascii() for p in pieces) or "".join(pieces)!=source:
        raise ValueError("Token boundary reconstruction mismatch")
    start=source.index(operation)
    end=start+len(operation)
    cursor=0
    result=[]
    for i,piece in enumerate(pieces):
        nxt=cursor+len(piece)
        # Include tokens crossing the operation boundary, explicitly.
        if cursor<end and nxt>start:
            result.append(i)
        cursor=nxt
    if not result:
        raise ValueError("Empty operation mask")
    return tuple(result)


def train_positions(row: dict, ids: list[int], tokenizer) -> tuple[int, ...]:
    family=row["family"]
    if family not in TRAIN_FAMILIES:
        raise ValueError("Non-TRAIN annotation")
    prefix=f"train-{family}-"
    key=row["id"]
    if not key.startswith(prefix) or len(key)!=len(prefix)+3 or not key[-3:].isascii() or not key[-3:].isdigit():
        raise ValueError("Invalid TRAIN row ID")
    variant=int(key[-3:])
    task, source=_task(family,variant)
    if row["source"]!=source or row["task"]!=task or tokenizer.encode(source)!=ids:
        raise ValueError("Noncanonical annotated reference")
    return span_positions(source,operation_text(family,variant),ids,tokenizer)
