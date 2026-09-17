"""Deciding what is a header, what is a record, and what is the table's own
furniture.
"""

from __future__ import annotations

import re

from ftmap.config import Config
from ftmap.io.detect import modal
from ftmap.io.frame import Column, Frame, FurnitureRow
from ftmap.io.tabular import Grid
from ftmap.profile.shapes import shape

SEARCH_ROWS = 10
_TOTALS = re.compile(r"^(разом|всього|усього|итого|всего|total)\b", re.I)
_CYRILLIC = re.compile(r"[а-яёіїєґА-ЯЁІЇЄҐ]")
_LATIN = re.compile(r"[a-zA-Z]")


def _only_digits_and_separators(text: str) -> bool:
    """Nothing in the cell but digits, `.`, `,` and spaces."""
    return text.replace(".", "").replace(",", "").replace(" ", "").isdigit()


def _pad(rows: list[list[str | None]]) -> tuple[list[list[str | None]], int]:
    grid = [list(r) for r in rows]
    width = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([None] * (width - len(r)))
    return grid, width


def _fill_merges(rows: list[list[str | None]], merges) -> tuple[list[list[str | None]], int]:
    grid, width = _pad(rows)
    filled = 0
    for r0, c0, r1, c1 in merges:
        if r0 >= len(grid) or c0 >= width:
            continue
        if c0 == 0 and c1 >= width - 1:
            continue
        value = grid[r0][c0]
        if value is None:
            continue
        for r in range(r0, min(r1, len(grid) - 1) + 1):
            for c in range(c0, min(c1, width - 1) + 1):
                if (r, c) != (r0, c0) and grid[r][c] is None:
                    grid[r][c] = value
                    filled += 1
    return grid, filled


def _non_null(row: list[str | None]) -> list[str]:
    return [c for c in row if c is not None and str(c).strip()]


def _below(grid: list[list[str | None]], i: int) -> list[list[str | None]]:
    """The next five rows under `i` that hold anything — what a candidate
    header is judged against."""
    return [r for r in grid[i + 1: i + 6] if _non_null(r)]


def _score_header(grid: list[list[str | None]], i: int) -> float:
    """Five signals, and `width_match` is the one that keeps a title row out.
    """
    row = grid[i]
    width = len(row)
    if width == 0:
        return 0.0
    cells = _non_null(row)
    if not cells:
        return 0.0
    fill = len(cells) / width
    distinct = len(set(cells)) / len(cells)
    stringy = sum(0.0 if _only_digits_and_separators(c) else 1.0
                  for c in cells) / len(cells)

    below = _below(grid, i)
    if below:
        differs = _differs(cells, below)
        counts = [len(_non_null(r)) for r in below]
        modal_width = modal(counts)
        width_match = (min(len(cells), modal_width)
                       / max(len(cells), modal_width, 1))
    else:
        differs = 0.0
        width_match = 1.0

    return (0.30 * fill + 0.20 * distinct + 0.15 * stringy
            + 0.15 * differs + 0.20 * width_match)


def _differs(cells: list[str], below: list[list[str | None]]) -> float:
    """How unlike the rows beneath it a row is, by value shape."""
    if not cells or not below:
        return 0.0
    row_shapes = {shape(c) for c in cells}
    below_shapes = {shape(c) for r in below for c in _non_null(r)}
    return 1.0 - len(row_shapes & below_shapes) / max(len(row_shapes), 1)


def _is_labels(row: list[str | None]) -> bool:
    """A second header row is a row of LABELS, and a label is a word."""
    cells = _non_null(row)
    if not cells:
        return False
    return not any(not any(ch.isalpha() for ch in str(c)) for c in cells)


def _script_mix(row: list[str | None]) -> tuple[int, int]:
    text = " ".join(_non_null(row))
    return len(_LATIN.findall(text)), len(_CYRILLIC.findall(text))


def _classify_furniture(
    grid: list[list[str | None]], i: int, header_cells: list[str], width: int
) -> FurnitureRow | None:
    row = grid[i]
    cells = _non_null(row)
    if not cells:
        return FurnitureRow(i, "blank", "no non-empty cell")
    if width >= 2 and [str(c) for c in row[: len(header_cells)]] == header_cells:
        return FurnitureRow(i, "repeat_header", "identical to the header row")
    filled = [j for j, c in enumerate(row) if c is not None and str(c).strip()]
    if filled and _TOTALS.match(str(row[filled[0]])):
        tail = [str(row[j]) for j in filled[1:]]
        if not tail or any(_only_digits_and_separators(t) for t in tail):
            return FurnitureRow(
                i, "totals", "totals label leads the row, tail is numeric or empty")
    if width >= 3 and len(cells) == 1:
        return FurnitureRow(i, "sparse", f"one filled cell of {width}")
    return None


def _group_bands(raw: list[list[str | None]], merges, header_row: int | None,
                 width: int) -> tuple[int | None, list[str | None]]:
    """The group header row and the group text per column, or (None, [None…]).
    """
    none = [None] * width
    if header_row in (None, 0) or header_row >= len(raw):
        return None, none
    i = header_row - 1
    row = list(raw[i]) + [None] * (width - len(raw[i]))
    if not _is_labels(row):
        return None, none
    vertical: set[int] = set()
    anchored: dict[int, int] = {}
    for r0, c0, r1, c1 in merges:
        if r0 <= i <= r1 and r1 >= header_row:
            vertical.update(range(c0, min(c1, width - 1) + 1))
        elif r0 == i == r1 and c1 > c0 and not (c0 == 0 and c1 >= width - 1):
            anchored[c0] = min(c1, width - 1)
    filled = [j for j, c in enumerate(row)
              if c is not None and str(c).strip() and j not in vertical]
    bands: list[tuple[int, int, bool]] = []
    for n, j in enumerate(filled):
        after = j + 1 < width and (row[j + 1] is None or not str(row[j + 1]).strip())
        if j in anchored:
            end, merged = anchored[j], True
        elif after and j + 1 in anchored:
            end, merged = anchored[j + 1], True
        elif n + 1 < len(filled):
            end, merged = filled[n + 1] - 1, False
        else:
            end, merged = j, False
        bands.append((j, end, merged))
    if not any(merged for _, _, merged in bands):
        return None, none
    groups: list[str | None] = list(none)
    for start, end, _ in bands:
        text = str(row[start]).strip()
        for c in range(start, end + 1):
            if c not in vertical:
                groups[c] = text
    return i, groups


def build_frame(grid_obj: Grid, path: str, sha256: str, source_id: str, cfg: Config) -> Frame:
    raw, _ = _pad(grid_obj.rows)
    grid, filled = _fill_merges(grid_obj.rows, grid_obj.merges)
    width = max((len(r) for r in grid), default=0)

    scores = [_score_header(raw, i) for i in range(min(SEARCH_ROWS, len(raw)))]
    header_row: int | None = None
    if scores:
        best = max(scores)
        if best >= cfg.header_score_floor:
            for i, sc in enumerate(scores):
                if sc >= cfg.header_score_floor and sc >= best - cfg.header_margin:
                    header_row = i
                    break

    label_row: int | None = None
    if header_row is not None and header_row + 1 < len(raw):
        nxt = header_row + 1
        if (_score_header(raw, nxt) >= cfg.header_score_floor
                and _differs(_non_null(raw[nxt]), _below(raw, nxt)) > 0.5):
            lat_h, cyr_h = _script_mix(grid[header_row])
            lat_l, cyr_l = _script_mix(grid[nxt])
            if ((lat_h > cyr_h and cyr_l > lat_l)
                    or (cyr_h > lat_h and lat_l > cyr_l)) and _is_labels(grid[nxt]):
                label_row = nxt

    head = grid[header_row] if header_row is not None else [None] * width
    labels = grid[label_row] if label_row is not None else [None] * width
    group_row, groups = _group_bands(raw, grid_obj.merges, header_row, width)
    columns = [
        Column(
            id=f"c{j}",
            index=j,
            header=head[j] if j < len(head) else None,
            label=labels[j] if j < len(labels) else None,
            group=groups[j],
        )
        for j in range(width)
    ]

    body_start = (label_row if label_row is not None else header_row)
    body_start = 0 if body_start is None else body_start + 1
    header_cells = [str(c) for c in head[:width]]

    furniture: list[FurnitureRow] = []
    if header_row is not None:
        for i in range(header_row):
            cells = _non_null(grid[i])
            if i == group_row:
                furniture.append(FurnitureRow(
                    i, "group", "names the bands of columns under it"))
                continue
            furniture.append(
                FurnitureRow(i, "blank" if not cells else "title",
                             "above the header row")
            )

    rows: list[list[str | None]] = []
    row_index: list[int] = []
    for i in range(body_start, len(grid)):
        verdict = _classify_furniture(grid, i, header_cells, width)
        if verdict is not None:
            furniture.append(verdict)
            if verdict.kind in ("blank", "repeat_header") or cfg.drop_furniture:
                continue
        rows.append(grid[i])
        row_index.append(i)

    furniture.sort(key=lambda f: f.index)
    return Frame(
        source_id=source_id, path=path, sheet=grid_obj.sheet, sha256=sha256,
        columns=columns, rows=rows, row_index=row_index,
        header_row=header_row, label_row=label_row, group_row=group_row,
        furniture=furniture, filled_cells=filled, hidden=grid_obj.hidden,
        repairs=list(grid_obj.repairs),
    )
