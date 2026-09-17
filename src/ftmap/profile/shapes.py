"""Generalized value shapes."""

from __future__ import annotations

from collections import Counter

RUN_CAP = 8

_CYRILLIC = set(range(0x0400, 0x0500)) | {0x0490, 0x0491}


def _classify(ch: str) -> str:
    if ch.isdigit():
        return "d"
    if ch.isalpha():
        return "C" if ord(ch) in _CYRILLIC else "L"
    return ch


def shape(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip()
    if not text:
        return ""
    out: list[str] = []
    run_char = ""
    run_len = 0
    for ch in text:
        cls = _classify(ch)
        if cls == run_char:
            run_len += 1
            continue
        if run_char:
            out.append(run_char * run_len if run_len <= RUN_CAP else run_char + "+")
        run_char, run_len = cls, 1
    if run_char:
        out.append(run_char * run_len if run_len <= RUN_CAP else run_char + "+")
    return "".join(out)


def top_shapes(values: list[str | None], n: int) -> list[tuple[str, float]]:
    by_value = Counter(v for v in values if v is not None and str(v).strip() != "")
    total = sum(by_value.values())
    if not total:
        return []
    counts: Counter[str] = Counter()
    for value, held in by_value.items():
        counts[shape(value)] += held
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(s, c / total) for s, c in ranked[:n]]
