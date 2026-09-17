# tests/test_corpus.py
"""The layout stage against real files, one per defect class named in the
design spec §4.

These fixtures are public data (`article-1/01-data/raw_835`), copied
unmodified into `tests/fixtures/`; see `tests/fixtures/README.md` for
provenance. Unlike the synthetic layout fixtures in `test_layout.py`, these
exercise the header search, the two-row-header rule, furniture detection and
merged-cell fill against material the pipeline was actually built to read.
"""

from __future__ import annotations

import os

import pytest

from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import read_source, sha256_file
from ftmap.profile.columns import profile_frame

CFG = Config.load(None)
FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _frame(name, sheet=0):
    path = os.path.join(FIX, name)
    grid = read_source(path)[sheet]
    return build_frame(grid, path, sha256_file(path), "fix/" + name, CFG)


def test_popolo_csv_headers_are_found_at_row_zero():
    f = _frame("deputies_popolo.csv")
    assert f.header_row == 0
    assert [c.header for c in f.columns][:4] == [
        "id", "votingIdentifier", "familyName", "name"]
    assert len(f.rows) > 5


def test_two_header_rows_are_both_kept():
    f = _frame("orgbook_two_header_rows.xlsx")
    assert f.header_row == 0 and f.label_row == 1
    headers = [c.header for c in f.columns]
    labels = [c.label for c in f.columns]
    assert "prefLabel" in headers
    assert "Повна назва" in labels


def test_title_rows_do_not_become_column_names():
    f = _frame("reception_title_rows.xlsx")
    headers = [h for h in (c.header for c in f.columns) if h]
    assert "ПІБ депутата" in headers
    assert not any(h.startswith("Дані про депутатів") for h in headers)


def test_semicolon_csv_splits_into_three_columns():
    f = _frame("reception_semicolon.csv")
    assert len(f.columns) == 3
    assert [c.header for c in f.columns] == ["Посада", "ПІБ", "Час та місце прийому"]


def test_merged_header_cells_are_filled():
    f = _frame("structure_merged.xlsx")
    assert f.header_row == 0
    assert [c.header for c in f.columns][:3] == [
        "№ з/п", "СТРУКТУРА", "Кількість штатних одиниць"]


@pytest.mark.parametrize("name", [
    "deputies_popolo.csv", "orgbook_two_header_rows.xlsx",
    "reception_title_rows.xlsx", "reception_semicolon.csv",
    "structure_merged.xlsx",
])
def test_every_fixture_profiles_without_raising(name):
    f = _frame(name)
    profiles = profile_frame(f, CFG)
    assert len(profiles) == len(f.columns)
    assert all(0.0 <= p.fill_rate <= 1.0 for p in profiles)
