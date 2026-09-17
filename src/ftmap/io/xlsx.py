"""Reading an XLSX without paying for the space Excel reserved."""

from __future__ import annotations

import zipfile
from xml.etree import ElementTree

from ftmap.config import packaged_limits

_LIMITS = packaged_limits()
MAX_MEMBERS = _LIMITS["members"]
MAX_UNCOMPRESSED = _LIMITS["uncompressed_bytes"]
MAX_RATIO = _LIMITS["compression_ratio"]
MAX_MEMBER_BYTES = _LIMITS["member_bytes"]
MAX_SHARED_STRINGS = _LIMITS["shared_strings_bytes"]
MAX_MERGES_PER_SHEET = _LIMITS["merges_per_sheet"]

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = ("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
        "id")


_PROLOG_BYTES = 4096


def _reject_dtd(head: bytes, member: str) -> None:
    """Refuse any XML that declares a DTD, before it is parsed."""
    if b"<!DOCTYPE" in head:
        raise HostileWorkbook(
            f"{member} declares a DTD. OOXML has none, and `ElementTree` "
            "expands internal entities — which is how a few hundred bytes "
            "become gigabytes.")


class HostileWorkbook(ValueError):
    """An archive whose declared shape exceeds a registered ceiling."""


def preflight(path: str) -> dict:
    """Check the archive's shape against the registered ceilings."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise HostileWorkbook(f"{path}: not a readable archive ({exc})") from None

    total = sum(info.file_size for info in infos)
    compressed = sum(info.compress_size for info in infos) or 1
    ratio = total / compressed
    shared = sum(info.file_size for info in infos
                 if info.filename.endswith("sharedStrings.xml"))
    largest = max((info.file_size for info in infos), default=0)

    for value, ceiling, what in (
        (len(infos), MAX_MEMBERS, "archive members"),
        (total, MAX_UNCOMPRESSED, "cumulative uncompressed bytes"),
        (ratio, MAX_RATIO, "compression ratio"),
        (largest, MAX_MEMBER_BYTES, "largest member's uncompressed bytes"),
        (shared, MAX_SHARED_STRINGS, "sharedStrings.xml uncompressed bytes"),
    ):
        if value > ceiling:
            raise HostileWorkbook(
                f"{path}: {what} is {value:,.0f}, over the registered ceiling "
                f"of {ceiling:,.0f}. See "
                "docs/measurements/2026-08-25-scale-ceilings.md.")

    return {"members": len(infos), "uncompressed": total, "ratio": ratio,
            "shared_strings": shared, "largest_member": largest}


def sheet_metadata(path: str, *, preflighted: bool = False) -> list[dict]:
    """Sheet names, visibility and merge ranges, without reading a cell."""
    if not preflighted:
        preflight(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        order = _sheet_order(archive, names) if "xl/workbook.xml" in names else []
        out = []
        for sheet in order:
            member = sheet["part"]
            if not _is_worksheet(member) or member not in names:
                continue
            index = len(out)
            merges = _merges(archive, member)
            if len(merges) > MAX_MERGES_PER_SHEET:
                raise HostileWorkbook(
                    f"{path}: sheet {sheet['title']!r} declares "
                    f"{len(merges):,} merge ranges, over the registered "
                    f"ceiling of {MAX_MERGES_PER_SHEET:,}")
            out.append({"index": index, "title": sheet["title"],
                        "hidden": sheet["hidden"], "merges": merges})
    return out


def sheet_titles(path: str) -> list[str]:
    """Every worksheet's name, from `xl/workbook.xml` ALONE."""
    preflight(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names:
            return []
        return [sheet["title"] for sheet in _sheet_order(archive, names)
                if _is_worksheet(sheet["part"]) and sheet["part"] in names]


def _is_worksheet(member: str) -> bool:
    """Whether a workbook part is a worksheet, as opposed to a chartsheet."""
    return member.startswith("xl/worksheets/")


def _sheet_order(archive: zipfile.ZipFile, names: set[str]) -> list[dict]:
    """Sheets in workbook order, each resolved to its part name."""
    targets = {}
    if "xl/_rels/workbook.xml.rels" in names:
        raw = archive.read("xl/_rels/workbook.xml.rels")
        _reject_dtd(raw[:_PROLOG_BYTES], "xl/_rels/workbook.xml.rels")
        rels = ElementTree.fromstring(raw)
        for rel in rels:
            target = rel.get("Target", "")
            targets[rel.get("Id")] = (
                target[1:] if target.startswith("/")
                else "xl/" + target.removeprefix("/xl/").removeprefix("./"))

    raw = archive.read("xl/workbook.xml")
    _reject_dtd(raw[:_PROLOG_BYTES], "xl/workbook.xml")
    root = ElementTree.fromstring(raw)
    sheets = []
    for element in root.iter(f"{_MAIN}sheet"):
        rel_id = element.get(_REL)
        sheets.append({
            "title": element.get("name") or "",
            "hidden": element.get("state", "visible") != "visible",
            "part": targets.get(rel_id, ""),
        })
    return sheets


def _merges(archive: zipfile.ZipFile, member: str) -> list[tuple[int, int, int, int]]:
    """`(row0, col0, row1, col1)`, zero-based, as `Grid.merges` wants them."""
    merges: list[tuple[int, int, int, int]] = []
    sheet_data = None
    with archive.open(member) as prolog:
        _reject_dtd(prolog.read(_PROLOG_BYTES), member)
    with archive.open(member) as handle:
        for event, element in ElementTree.iterparse(handle, ("start", "end")):
            if event == "start":
                if element.tag == f"{_MAIN}sheetData":
                    sheet_data = element
                continue
            if element.tag == f"{_MAIN}mergeCell":
                bounds = _bounds(element.get("ref") or "")
                if bounds is not None:
                    merges.append(bounds)
                element.clear()
                if len(merges) > MAX_MERGES_PER_SHEET:
                    break
            elif element.tag == f"{_MAIN}row":
                element.clear()
                if sheet_data is not None:
                    sheet_data[:] = []
    merges.sort()
    return merges


def _bounds(ref: str) -> tuple[int, int, int, int] | None:
    """`"B2:D5"` to `(1, 1, 4, 3)`. Returns None for anything unparseable."""
    parts = ref.split(":")
    if len(parts) != 2:
        return None
    try:
        (r0, c0), (r1, c1) = (_cell_ref(parts[0]), _cell_ref(parts[1]))
    except ValueError:
        return None
    return (min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1))


def _cell_ref(ref: str) -> tuple[int, int]:
    """`"AB12"` to `(11, 27)` — zero-based row and column."""
    column = 0
    for index, char in enumerate(ref):
        if char.isdigit():
            if index == 0:
                raise ValueError(ref)
            return int(ref[index:]) - 1, column - 1
        column = column * 26 + (ord(char.upper()) - ord("A") + 1)
    raise ValueError(ref)
