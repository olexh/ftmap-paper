"""Readers. They produce a grid of strings and nothing else."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from ftmap.config import packaged_limits
from ftmap.fsutil import sha256_file
from ftmap.io.detect import (DELIMITERS, modal, read_head, sniff_delimiter,
                             sniff_encoding)

CSV_EXT = {".csv", ".tsv"}
XL_EXT = {".xlsx", ".xlsm", ".xls"}


@dataclass(frozen=True)
class Grid:
    rows: list[list[str | None]]
    sheet: str
    merges: list[tuple[int, int, int, int]]
    hidden: bool = False
    repairs: list[str] = field(default_factory=list)


def _cell(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    text = str(value).strip()
    return text or None


def _trim_row(row: list[str | None]) -> list[str | None]:
    """Drop a row's trailing empties as it is read, not after the grid is
    built.
    """
    end = len(row)
    while end and row[end - 1] is None:
        end -= 1
    return row[:end] if end != len(row) else row


def _trim_trailing_empty_columns(
    rows: list[list[str | None]],
    merges: list[tuple[int, int, int, int]],
) -> tuple[list[list[str | None]], list[tuple[int, int, int, int]]]:
    """Drop the padding both Excel readers report past the last real column."""
    width = 0
    for row in rows:
        for i in range(len(row) - 1, width - 1, -1):
            if row[i] is not None:
                width = i + 1
                break
    if all(len(row) == width for row in rows):
        return rows, merges
    squared = [row[:width] + [None] * (width - len(row)) if len(row) != width
               else row for row in rows]
    kept = [(r0, c0, r1, min(c1, width - 1)) for r0, c0, r1, c1 in merges
            if c0 < width]
    return squared, kept


MAX_DELIMITED_BYTES = packaged_limits()["delimited_bytes"]


class OversizeSource(ValueError):
    """A delimited source past the registered ceiling."""


def _read_csv(path: str) -> list[Grid]:
    size = os.path.getsize(path)
    if size > MAX_DELIMITED_BYTES:
        raise OversizeSource(
            f"{path}: {size:,} bytes, over the registered ceiling of "
            f"{MAX_DELIMITED_BYTES:,}. This reader holds the whole file and "
            "then its rows; a source this size needs a streaming reader, not "
            "a larger ceiling.")
    enc = sniff_encoding(path)
    with open(path, encoding=enc, newline="\n") as fh:
        delim = sniff_delimiter(read_head(fh))
        fh.seek(0)
        rows = [[_cell(c) for c in row] for row in csv.reader(fh, delimiter=delim)]
    rows, repairs = _split_header_on_its_own_delimiter(rows, delim)
    return [Grid(rows=rows, sheet="", merges=[], repairs=repairs)]


_OTHER_DELIMITERS = tuple(d for d in DELIMITERS if d != ",") + (",",)


def _split_header_on_its_own_delimiter(rows: list[list[str | None]],
                                       delim: str) -> tuple[list, list[str]]:
    """A header line delimited differently from its body is split on its own
    delimiter, and the repair is said.
    """
    if not rows or len([c for c in rows[0] if c is not None]) != 1:
        return rows, []
    body = [len(r) for r in rows[1:6] if any(c is not None for c in r)]
    if not body:
        return rows, []
    width = modal(body)
    if width <= 1:
        return rows, []
    text = next(c for c in rows[0] if c is not None)
    for other in _OTHER_DELIMITERS:
        if other == delim or other not in text:
            continue
        parts = [_cell(p) for p in text.split(other)]
        if len(parts) == width:
            shown = "tab" if other == "\t" else repr(other)
            return ([parts, *rows[1:]],
                    [f"the header line is delimited by {shown} while the body "
                     f"is delimited by {delim!r}; split into {width} columns "
                     f"on its own delimiter"])
    return rows, []


def _read_xlsx(path: str) -> list[Grid]:
    """Values by streaming iteration; merges and visibility by a second pass.
    """
    import openpyxl

    from ftmap.io.xlsx import preflight, sheet_metadata

    preflight(path)
    metadata = sheet_metadata(path, preflighted=True)

    with open(path, "rb") as fh:
        wb = openpyxl.load_workbook(fh, read_only=True, data_only=True)
        try:
            out: list[Grid] = []
            by_title = {m["title"]: m for m in metadata}
            for ws in wb.worksheets:
                ws.reset_dimensions = True
                rows = [_trim_row([_cell(v) for v in values])
                        for values in ws.iter_rows(values_only=True)]
                meta = by_title.get(ws.title, {})
                merges = list(meta.get("merges") or [])
                rows, merges = _trim_trailing_empty_columns(rows, merges)
                out.append(Grid(rows=rows,
                                sheet=meta.get("title", ws.title),
                                merges=merges,
                                hidden=bool(meta.get("hidden", False))))
        finally:
            wb.close()
    return out


def sheet_names(path: str) -> list[str]:
    """Every sheet's name, without parsing a single value."""
    from ftmap.io.xlsx import sheet_titles

    return sheet_titles(path)


def _read_xls(path: str) -> list[Grid]:
    import xlrd

    book = xlrd.open_workbook(path, formatting_info=True)
    out: list[Grid] = []
    for ws in book.sheets():
        rows = [_trim_row([_cell(v) for v in ws.row_values(r)])
                for r in range(ws.nrows)]
        merges = [
            (r0, c0, r1 - 1, c1 - 1) for (r0, r1, c0, c1) in getattr(ws, "merged_cells", [])
        ]
        merges.sort()
        rows, merges = _trim_trailing_empty_columns(rows, merges)
        out.append(Grid(rows=rows, sheet=ws.name, merges=merges,
                        hidden=getattr(ws, "visibility", 0) != 0))
    return out


def read_source(path: str) -> list[Grid]:
    ext = os.path.splitext(path)[1].lower()
    if ext in CSV_EXT:
        return _read_csv(path)
    if ext == ".xls":
        try:
            return _read_xls(path)
        except Exception:
            return _read_xlsx(path)
    if ext in XL_EXT:
        return _read_xlsx(path)
    raise ValueError(f"unsupported extension {ext!r} for {path}")
