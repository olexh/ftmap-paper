"""Encoding and delimiter sniffing."""

from __future__ import annotations

import codecs
import csv
from itertools import islice
from typing import IO, TypeVar

ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252")
DELIMITERS = (",", ";", "\t", "|")

_T = TypeVar("_T")


def modal(values: list[_T]) -> _T:
    """The commonest member of `values` — in every caller, a row width."""
    return max(set(values), key=values.count)


def sniff_encoding(path: str) -> str:
    """The file's encoding, guessed from its first 200 000 bytes."""
    with open(path, "rb") as fh:
        raw = fh.read(200_000)
        final = fh.read(1) == b""
    for enc in ENCODINGS:
        try:
            codecs.getincrementaldecoder(enc)().decode(raw, final)
        except UnicodeDecodeError:
            continue
        return enc
    return "utf-8"


_SNIFF_ROWS = 50


_SNIFF_BYTES = 8 << 20


def read_head(fh: IO[str]) -> str:
    """As much of an OPEN delimited file as `sniff_delimiter` will look at."""
    head = fh.read(_SNIFF_BYTES + 1)
    if len(head) <= _SNIFF_BYTES:
        return head
    cut = head.rfind("\n")
    return head[:cut + 1] if cut >= 0 else ""


def _head_lines(text: str) -> list[str]:
    """The lines the sniffer may look at, sliced ONCE for all four delimiters.
    """
    parts = text.split("\n")
    return [line + "\n" for line in parts]


def _column_stability(head: list[str], delim: str) -> tuple[int, float]:
    """Modal column count over the head's rows, and how often it holds."""
    rows = [r for r in islice(csv.reader(head, delimiter=delim), _SNIFF_ROWS) if r]
    if not rows:
        return 0, 0.0
    widths = [len(r) for r in rows]
    width = modal(widths)
    return width, widths.count(width) / len(widths)


def sniff_delimiter(text: str) -> str:
    """Pick the delimiter that yields the most columns at the highest
    stability.
    """
    head = _head_lines(text)
    scored: list[tuple[float, int, str]] = []
    for d in DELIMITERS:
        width, stability = _column_stability(head, d)
        if width <= 1:
            continue
        scored.append((stability, width, d))
    if not scored:
        return ","
    scored.sort(reverse=True)
    return scored[0][2]
