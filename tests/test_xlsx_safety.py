# tests/test_xlsx_safety.py
"""Archives that are not workbooks, and the ceilings that stop them.

Every corpus in this project is downloaded from an open-data portal, which
means every workbook in it is attacker-supplied in the only sense that matters:
nobody in this project chose its bytes. The readers in `io/tabular.py` are
streaming, which bounds what an HONEST file costs and says nothing about a
hostile one — a member that expands 1 000:1 is decompressed before anything can
notice, and the process that would report it is the one being killed.

So the archive is checked before it is opened as a workbook at all, against
ceilings registered in `docs/measurements/2026-08-25-scale-ceilings.md` before
`io/xlsx.py` existed. The fixtures below are built rather than committed: a
2 GB expansion is not a thing to keep in git, and each is a dozen lines of
`zipfile`.

The ceilings sit an order of magnitude above the largest real workbook measured
(ratio 12.0×, 130 MB uncompressed, 124 members), because the thing they
separate is not size. `test_every_real_workbook_passes_the_preflight` is the
other half of that claim.
"""

import os
import zipfile

import openpyxl
import pytest

from ftmap.io.tabular import read_source
from ftmap.io.xlsx import (MAX_MEMBERS, MAX_MERGES_PER_SHEET, MAX_RATIO,
                           HostileWorkbook, preflight, sheet_metadata)

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "corpus-external")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _archive(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            z.writestr(name, data)
    return str(path)


def test_a_high_ratio_member_is_refused_before_it_is_decompressed(tmp_path):
    """The classic shape: a few kilobytes on disk, gigabytes in memory.

    Refused from the CENTRAL DIRECTORY — `file_size` and `compress_size` are
    recorded per member, so the ratio is available without decompressing a
    byte. A check that had to read the data first would be a check that ran
    after the damage.
    """
    # 64 MB of one repeated byte compresses to a few kilobytes.
    bomb = _archive(tmp_path / "bomb.xlsx",
                    [("xl/worksheets/sheet1.xml", b"\0" * (64 << 20))])
    assert os.path.getsize(bomb) < (1 << 20), "the fixture is not compressed"

    with pytest.raises(HostileWorkbook) as excinfo:
        preflight(bomb)
    assert "compression ratio" in str(excinfo.value)
    # And through the reader, which is where it actually matters.
    with pytest.raises(HostileWorkbook):
        read_source(bomb)


def test_too_many_members_is_refused(tmp_path):
    many = _archive(tmp_path / "many.xlsx",
                    [(f"xl/x{i}.xml", b"<a/>") for i in range(MAX_MEMBERS + 1)])
    with pytest.raises(HostileWorkbook) as excinfo:
        preflight(many)
    assert "archive members" in str(excinfo.value)


def test_a_declared_dtd_is_refused_rather_than_expanded(tmp_path):
    """The billion-laughs shape, on the metadata pass.

    THE FIRST VERSION OF THIS TEST ASSERTED THE WRONG THING. It claimed
    `ElementTree` "rejects a DTD outright", and the assertion below proves it
    does not: internal general entities expand, which is the entire mechanism.
    Only EXTERNAL entities are refused. So the guard is a check on the bytes —
    OOXML has no DTD, so anything carrying one is not a workbook this reader
    should be interpreting.
    """
    from xml.etree import ElementTree

    # The premise, asserted rather than assumed, because the guard exists only
    # because this is true.
    assert ElementTree.fromstring(
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaa">]><x>&a;</x>'
    ).text == "aaa", "ElementTree stopped expanding internal entities"

    lol = (b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
           b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
           b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
           b']><worksheet>&lol3;</worksheet>')
    path = _archive(tmp_path / "lol.xlsx", [
        ("xl/workbook.xml",
         b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
         b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
         b'<sheets><sheet name="s" sheetId="1" r:id="rId1"/></sheets></workbook>'),
        ("xl/_rels/workbook.xml.rels",
         b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         b'<Relationship Id="rId1" '
         b'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
         b'Target="worksheets/sheet1.xml"/></Relationships>'),
        ("xl/worksheets/sheet1.xml", lol),
    ])
    preflight(path)          # small and well-compressed; the ceilings pass it
    with pytest.raises(HostileWorkbook) as excinfo:
        sheet_metadata(path)
    assert "DTD" in str(excinfo.value)


def test_a_sheet_declaring_a_hostile_number_of_merges_is_refused(tmp_path):
    """Merge ranges are metadata, and metadata is not free.

    `layout` fills merged cells forward, so every range is retained and
    applied. The largest real sheet measured declares 366; a sheet declaring a
    million is describing something other than a table.
    """
    count = MAX_MERGES_PER_SHEET + 10
    merges = "".join(f'<mergeCell ref="A{i + 1}:B{i + 1}"/>'
                     for i in range(count))
    sheet = ('<?xml version="1.0"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetData/>'
             f'<mergeCells count="{count}">{merges}</mergeCells>'
             '</worksheet>').encode()
    path = _archive(tmp_path / "merges.xlsx", [
        ("xl/workbook.xml",
         b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
         b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
         b'<sheets><sheet name="s" sheetId="1" r:id="rId1"/></sheets></workbook>'),
        ("xl/_rels/workbook.xml.rels",
         b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         b'<Relationship Id="rId1" '
         b'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
         b'Target="worksheets/sheet1.xml"/></Relationships>'),
        ("xl/worksheets/sheet1.xml", sheet),
    ])
    with pytest.raises(HostileWorkbook) as excinfo:
        sheet_metadata(path)
    assert "merge ranges" in str(excinfo.value)


def test_a_hostile_workbook_commits_no_source_generation(tmp_path):
    """The refusal has to arrive before anything durable exists.

    A source that fails is recorded in the run report — one unreadable workbook
    must not cost the other 359 — and what must NOT happen is a committed
    generation for it.
    """
    from ftmap.config import Config
    from ftmap.pipeline import run_corpus
    from ftmap.plan.prompt import BINDING_MARKER
    from ftmap.vocab.catalogue import Catalogue
    from ftmap.workspace import open_workspace

    class Scripted:
        calls = cache_hits = 0

        def complete(self, system, user, schema, max_tokens=None):
            if BINDING_MARKER not in user:
                return {"subject": "Person", "entities": [
                    {"key": "person", "schema": "Person", "keys": ["c0"]}],
                    "edges": []}
            return {"bindings": []}

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _archive(corpus / "bomb.xlsx",
             [("xl/worksheets/sheet1.xml", b"\0" * (64 << 20))])
    (corpus / "fine.csv").write_text("ПІБ\nКоваленко Іван\n", encoding="utf-8")

    out = str(tmp_path / "work")
    report = run_corpus(str(corpus), Config.load(None), Catalogue.load(),
                        Scripted(), out)
    assert report["failed"] == 1
    assert report["sources"] == 1
    workspace = open_workspace(out)
    assert len(workspace.source_ids()) == 1
    assert not workspace.publication_grade()


def test_every_real_workbook_passes_the_preflight():
    """The other half of the claim: the ceilings refuse bombs, not workbooks.

    A limit that also rejects the corpus is not a limit, it is an outage. The
    three workbooks in `corpus-external/` are the largest and the most awkward
    available here — 67 MB uncompressed, 83 members, ratio 7.3.
    """
    found = 0
    for dirpath, _dirs, files in os.walk(CORPUS):
        for name in files:
            if not name.lower().endswith((".xlsx", ".xlsm")):
                continue
            found += 1
            stats = preflight(os.path.join(dirpath, name))
            assert stats["ratio"] < MAX_RATIO
            assert stats["members"] < MAX_MEMBERS
    if not found:
        pytest.skip("corpus-external is not on this machine")


def test_the_tracked_fixtures_pass_the_preflight():
    for name in sorted(os.listdir(FIXTURES)):
        if name.lower().endswith((".xlsx", ".xlsm")):
            assert preflight(os.path.join(FIXTURES, name))["members"] > 0


def test_a_workbook_openpyxl_cannot_read_fails_without_committing(tmp_path):
    """A well-formed archive that is not a workbook. Refused by the reader, not
    by the preflight — the ceilings are about amplification, not correctness."""
    path = _archive(tmp_path / "notxlsx.xlsx", [("hello.txt", b"not a workbook")])
    preflight(path)
    with pytest.raises(Exception):
        read_source(path)


def test_sheet_order_and_visibility_survive_the_metadata_pass(tmp_path):
    """`inventory` addresses a source by SHEET INDEX, and the index it means is
    workbook order — not the order the parts happen to sit in the archive."""
    path = tmp_path / "order.xlsx"
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title in ("перший", "другий", "третій"):
        wb.create_sheet(title=title).append(["ПІБ"])
    wb["другий"].sheet_state = "hidden"
    wb.save(path)

    meta = sheet_metadata(str(path))
    assert [s["title"] for s in meta] == ["перший", "другий", "третій"]
    assert [s["hidden"] for s in meta] == [False, True, False]
    # ...and the full reader agrees, which is what makes the two passes one
    # answer rather than two.
    grids = read_source(str(path))
    assert [g.sheet for g in grids] == ["перший", "другий", "третій"]
    assert [g.hidden for g in grids] == [False, True, False]


def test_a_chartsheet_does_not_shift_every_later_sheet(tmp_path):
    """`workbook.xml` lists chartsheets; `wb.worksheets` does not.

    The two were zipped together by position, so one chartsheet moved every
    later worksheet's title, merges and hidden flag onto the wrong grid — and
    `inventory`, which numbers sources from the first list, produced a
    `sheet_index` past the end of the second.
    """
    import openpyxl

    from ftmap.io.tabular import sheet_names
    from ftmap.io.xlsx import sheet_metadata

    path = str(tmp_path / "chart.xlsx")
    book = openpyxl.Workbook()
    sheet = book["Sheet"]
    sheet.title = "Дані"
    sheet["A1"], sheet["B1"] = "рік", "сума"
    sheet["A2"], sheet["B2"] = "2024", "10"
    sheet.merge_cells("A1:B1")
    book.create_chartsheet("Графік")
    book.move_sheet("Графік", offset=-1)          # in front of the worksheet
    book.save(path)

    # The chartsheet is not a source, so it is not named and not counted.
    assert sheet_names(path) == ["Дані"]
    meta = sheet_metadata(path)
    assert [m["title"] for m in meta] == ["Дані"]
    # And the worksheet keeps its OWN merge, which is what layout reads to
    # rebuild a header that spans columns.
    assert meta[0]["merges"] == [(0, 0, 0, 1)]


def test_the_metadata_pass_refuses_a_hostile_archive_too(tmp_path):
    """It is the first thing to touch an untrusted file.

    `inventory` calls `sheet_names` for every workbook under a corpus root
    before any run, and the ZIP ceilings used to apply only on the read path
    further in — so the pass that exists to be cheap was the one with no bound.
    """
    import zipfile

    from ftmap.io.tabular import sheet_names
    from ftmap.io.xlsx import HostileWorkbook

    path = str(tmp_path / "bomb.xlsx")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("xl/worksheets/sheet1.xml", "\0" * (64 << 20))

    with pytest.raises(HostileWorkbook):
        sheet_names(path)


def test_learning_sheet_names_does_not_stream_the_sheets(monkeypatch):
    """R19: a workbook is not fully parsed for inventory and again to run it.

    `sheet_names` used to go through `sheet_metadata`, which streams every
    `<row>` of every sheet to reach `<mergeCells>` at the end — so learning
    four strings from the ship register cost 61 MB of XML, and execution then
    streamed the same 61 MB again. The docstring said "without parsing a
    single value", which was true of openpyxl's `Cell` objects and false of
    the stream underneath them.
    """
    from ftmap.io import xlsx
    from ftmap.io.tabular import sheet_names

    streamed = []
    original = xlsx._merges
    monkeypatch.setattr(
        xlsx, "_merges",
        lambda archive, member: streamed.append(member) or original(archive, member))

    names = sheet_names(os.path.join(FIXTURES, "structure_merged.xlsx"))

    assert names, "no sheet was named"
    assert streamed == [], f"sheet XML was streamed to learn names: {streamed}"
