# tests/test_read_memory.py
"""What reading a workbook costs, against ceilings registered before the fix.

Every number here is in `docs/measurements/2026-08-25-scale-ceilings.md`, which
was written before `io/tabular.py` was touched. A threshold chosen after seeing
the optimized result is not a threshold — it is a description of whatever
happened — so the ceilings are imported rather than restated, and each one was
required to FAIL against the code as it stood.

The two fixtures fail differently and both are needed:

  * the PADDED workbook is formatting with almost no data in it. 16 384 columns
    over 40 rows, every cell declared in the XML and carrying a style, 240 of
    655 360 holding a value. Reading it cost 324 MB, and that is linear in
    rows: the same shape at 4 000 rows does not finish.
  * the LEGITIMATE workbook is 1.9 million real values. The grid genuinely
    holds them, so the ceiling asks for a factor of 2.5, not for the memory to
    vanish — what goes is openpyxl's per-cell object model, not the data.

Marked `slow` and deselected by default: the largest fixture reads 10 MB of
XLSX and the padded one allocates hundreds of megabytes, which is not something
every `pytest -q` should pay for. `scripts/verify.sh` runs them.
"""

import os

import pytest

from bench import measure

pytestmark = pytest.mark.slow

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "corpus-external")

# ---------------------------------------------------------------------------
# Registered 2026-08-25 in docs/measurements/2026-08-25-scale-ceilings.md §2.
# Each is stated beside what the code produced when the ceiling was written.
# ---------------------------------------------------------------------------

#: 16 384 declared columns, 40 rows, 240 real values. Was 324 MB.
PADDED_RSS_MB = 150

#: `ua_ship_register_2026-07-01.xlsx`: 1 902 383 cells. Was 1 781 MB / 11.1 s.
SHIP_RSS_MB = 700
SHIP_SECONDS = 20

#: `ua_transport_licences_2024-08-02.xlsx`: 312 976 cells. Was 185 MB / 1.5 s.
LICENCES_RSS_MB = 120


def _corpus(*parts) -> str:
    path = os.path.join(CORPUS, *parts)
    if not os.path.exists(path):
        pytest.skip(f"{path} is not on this machine; corpus-external is "
                    "restorable from its own SOURCES.md")
    return path


def test_a_padded_workbook_costs_what_its_data_costs(tmp_path):
    """R18: memory follows populated values, not declared formatting.

    The failure this bounds is not "a big file is slow". It is that a 1.7 MB
    file holding 240 values cost 324 MB, because every one of 655 360 declared
    but empty cells became an object. Nothing in the workbook says how much
    data it holds; the DIMENSION says how much space Excel reserved, and those
    are different numbers by five orders of magnitude here.
    """
    padded = str(tmp_path / "padded.xlsx")
    got = measure(f"""
from tests_padded import build
build({padded!r})
from ftmap.io.tabular import read_source
grids = read_source({padded!r})
result = {{"sheets": len(grids), "rows": len(grids[0].rows),
           "cols": max((len(r) for r in grids[0].rows), default=0)}}
""")
    # The data is read correctly — a reader that returned nothing would pass a
    # memory ceiling trivially.
    assert got.result == {"sheets": 1, "rows": 40, "cols": 6}, got.result
    assert got.peak_mb < PADDED_RSS_MB, (
        f"{got} — ceiling {PADDED_RSS_MB} MB, registered when this read "
        "cost 324 MB")


def test_the_largest_legitimate_workbook_stays_under_its_ceiling():
    """The ceiling asks for a factor of 2.5, not for the memory to vanish.

    1 902 383 populated cells are real values and the grid holds them. What the
    ceiling removes is openpyxl's per-cell object model — a `Cell` instance
    with a style reference for every one of them — not the data.
    """
    path = _corpus("shape-stress", "ua_ship_register_2026-07-01.xlsx")
    got = measure(f"""
from ftmap.io.tabular import read_source
grids = read_source({path!r})
result = {{"sheets": len(grids),
           "cells": sum(len(r) for g in grids for r in g.rows),
           "merges": sum(len(g.merges) for g in grids)}}
""")
    assert got.result["cells"] == 1902383, got.result
    # The merges are the metadata half of R18-R19: they have to survive a
    # reader that no longer builds cell objects to find them.
    assert got.result["merges"] == 7, got.result
    assert got.peak_mb < SHIP_RSS_MB, (
        f"{got} — ceiling {SHIP_RSS_MB} MB, registered when this read cost "
        "1 781 MB")
    assert got.seconds < SHIP_SECONDS, f"{got} — ceiling {SHIP_SECONDS}s"


def test_the_second_largest_workbook_stays_under_its_ceiling():
    path = _corpus("person-graph", "ua_transport_licences_2024-08-02.xlsx")
    got = measure(f"""
from ftmap.io.tabular import read_source
grids = read_source({path!r})
result = {{"cells": sum(len(r) for g in grids for r in g.rows)}}
""")
    assert got.result["cells"] == 312976, got.result
    assert got.peak_mb < LICENCES_RSS_MB, (
        f"{got} — ceiling {LICENCES_RSS_MB} MB, registered when this read "
        "cost 185 MB")
