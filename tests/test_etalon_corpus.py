# tests/test_etalon_corpus.py
"""The gold standard, checked against the bytes it was written from.

An etalon is the measuring instrument, and until now nothing verified that it
still described its source. `sha256` was recorded and only ever used to FIND a
run's summary; the headers, the key columns and — since 2026-08-29 — the
instance counts were assertions no test could contradict. A number mistyped
into a reference file scores every run against it for as long as it stands.

Marked `corpus` and deselected by default: these read the real files under
`corpus-external/`, which are hundreds of megabytes and are not in a clean
checkout. `scripts/verify.sh` runs them. A source the machine does not have is
skipped BY NAME, and `test_the_corpus_is_present` fails when none is there at
all, so a green run cannot mean "checked nothing".
"""

from __future__ import annotations

import hashlib
import glob
import os

import pytest

from ftmap.config import Config
from ftmap.etalon import load
from ftmap.io.frame import column_values
from ftmap.io.layout import build_frame
from ftmap.io.tabular import read_source
from ftmap.vocab.catalogue import Catalogue

pytestmark = pytest.mark.corpus

CFG = Config.load(None)
CAT = Catalogue.load()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `etalon/manual/` holds the etalons written for sources under `../data/`,
# kept apart from the public corpus's because their sources are not in any
# checkout but this machine's; the by-name skip below covers them the same way.
ETALONS = sorted(glob.glob(os.path.join(ROOT, "docs", "measurements", "etalon",
                                        "*.etalon.yaml"))
                 + glob.glob(os.path.join(ROOT, "docs", "measurements", "etalon",
                                          "manual", "*.etalon.yaml")))


def _doc(path):
    doc = load(path, CAT)
    source = os.path.join(ROOT, doc.path)
    if not os.path.exists(source):
        pytest.skip(f"{doc.path} is not on this machine")
    return doc, source


def _frame(doc, source):
    grids = read_source(source)
    grid = next((g for g in grids if (g.sheet or "") == (doc.sheet or "")), None)
    assert grid is not None, f"{doc.path} has no sheet {doc.sheet!r}"
    return build_frame(grid, source, doc.sha256, "etalon/", CFG)


def _instances(frame, entity) -> int:
    """What the source holds, computed the way the etalon means it.

    Distinct values of the key columns, over the rows the entity is built
    from — `rows` names those for a bloc of a polymorphic table, and every
    row otherwise.
    """
    picked = range(len(frame.rows))
    if entity.rows:
        selector = column_values(frame, entity.rows[0])
        picked = [i for i in picked if selector[i] == entity.rows[1]]
    columns = [column_values(frame, c) for c in entity.keys]
    seen = {tuple(col[i] or "" for col in columns) for i in picked}
    # Every part filled. A composite key with a part empty builds no entity
    # — followthemoney computes no id and skips the row — so a count that
    # admitted partial tuples was a count of something no run can emit.
    return len({s for s in seen if all(s)})


def test_the_corpus_is_present():
    """A file every other test here skips on is a suite that proves nothing."""
    present = [p for p in ETALONS
               if os.path.exists(os.path.join(ROOT, load(p, CAT).path))]
    assert present, ("no etalon's source is on this machine; these tests would "
                     "all skip and report green")


@pytest.mark.parametrize("path", ETALONS, ids=os.path.basename)
def test_the_source_is_the_bytes_the_etalon_was_written_from(path):
    """The one fact the etalon already recorded and nothing checked.

    A publisher reissuing a file under the same name is the case this exists
    for: every answer below it was written about different bytes.
    """
    doc, source = _doc(path)
    digest = hashlib.sha256()
    with open(source, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == doc.sha256, f"{doc.path} is not the file this etalon describes"


@pytest.mark.parametrize("path", ETALONS, ids=os.path.basename)
def test_every_declared_column_is_the_column_the_file_has(path):
    """Column ids are positional, so a shifted file silently re-targets every
    answer in the etalon onto its neighbour."""
    doc, source = _doc(path)
    frame = _frame(doc, source)
    # THE HEADER OR THE LABEL. A two-row header gives a column both — the ship
    # register's `№ з/п` is on the first line and its own header on the second
    # — and an etalon quotes whichever names the column for a reader. Either
    # satisfies this check; what it is really asking is that the etalon's `c7`
    # is still the file's `c7`.
    headers = {c.id: [h for h in (c.header, c.label) if h] for c in frame.columns}
    for column in doc.columns:
        assert column.id in headers, f"{column.id} is not a column of {doc.path}"
        if column.header is None:
            continue
        if not headers[column.id]:
            # THE READER RECOVERS NO HEADER FOR THIS COLUMN, so there is
            # nothing to compare the etalon's quotation against. One column
            # in each of the three shape-stress sources is like this — the
            # padded sheet's `Примітка`, the lease register's `№ з/п`, a
            # header that is a single space — and that they read as empty is
            # a fact about the file the etalon exists to record, not a drift
            # between the two.
            continue
        # THE ANNOTATOR ELIDES A LONG HEADER. Several etalons write
        # `Дата видачі… / термін дії` for a header of eighty characters,
        # because the file is quoted for a reader as well as for a machine.
        # An elision is checked on the parts either side of it.
        if "…" in column.header:
            head, _, tail = column.header.partition("…")
            matched = any(a.startswith(head.rstrip()) and a.endswith(tail.lstrip())
                          for a in headers[column.id])
        else:
            matched = column.header in headers[column.id]
        assert matched, (
            f"{column.id}: the etalon says {column.header!r}, the file says "
            f"{headers[column.id]}")


@pytest.mark.parametrize("path", ETALONS, ids=os.path.basename)
def test_every_instance_count_is_what_the_source_holds(path):
    """The counts written on 2026-08-29, recomputed.

    They were derived from the bytes and cross-checked against the
    annotator's prose; this is what keeps them true afterwards. An etalon
    whose count no longer matches its file is scoring every run against a
    number nobody can reach.
    """
    doc, source = _doc(path)
    counted = [e for e in doc.entities if e.instances is not None]
    if not counted:
        pytest.skip(f"{os.path.basename(path)} states no instance count")
    frame = _frame(doc, source)
    for entity in counted:
        assert _instances(frame, entity) == entity.instances, (
            f"{entity.key}: the etalon says {entity.instances} instances, the "
            f"file holds {_instances(frame, entity)}")


@pytest.mark.parametrize("path", ETALONS, ids=os.path.basename)
def test_a_row_selector_selects_some_rows_and_not_all_of_them(path):
    """A bloc that selects nothing is an entity no run can produce; a bloc
    that selects everything is not a bloc."""
    doc, source = _doc(path)
    selected = [e for e in doc.entities if e.rows]
    if not selected:
        pytest.skip(f"{os.path.basename(path)} declares no row selector")
    frame = _frame(doc, source)
    rows = len(frame.rows)
    for entity in selected:
        values = column_values(frame, entity.rows[0])
        hits = sum(1 for v in values if v == entity.rows[1])
        assert 0 < hits < rows, (
            f"{entity.key}: rows {entity.rows[0]}={entity.rows[1]!r} selects "
            f"{hits} of {rows}")
