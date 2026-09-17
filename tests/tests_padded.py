# tests/tests_padded.py
"""Build the padded-workbook fixture, by hand.

HAND-WRITTEN XML, NOT OPENPYXL, and that is the whole reason this file exists:
the fixture is a sheet whose DECLARED dimension is Excel's full grid while
almost nothing in it is populated, and openpyxl will not write one — it emits
what you put in it. Real workbooks with this shape are common (a template
formatted to the edge of the sheet, saved with a handful of rows filled in),
and one of them cost 324 MB to read.

Built rather than committed: 1.7 MB of XML that is 99.96 % padding is not a
thing to keep in git, and the shape is a dozen lines of generation.

Named `tests_padded` rather than `test_padded` so pytest does not collect it as
a test module.
"""
import os
import sys
import zipfile

ROWS, COLS = 40, 16384          # Excel's full column count
POPULATED_COLS, POPULATED_ROWS = 6, 30


def col_name(index: int) -> str:
    name = ""
    while index >= 0:
        name = chr(ord("A") + index % 26) + name
        index = index // 26 - 1
    return name


def build(path: str) -> None:
    last = col_name(COLS - 1)
    rows = []
    names = [col_name(c) for c in range(COLS)]
    for r in range(1, ROWS + 1):
        parts = []
        for c in range(COLS):
            if r <= POPULATED_ROWS and c < POPULATED_COLS:
                parts.append(f'<c r="{names[c]}{r}" t="inlineStr">'
                             f'<is><t>v{r}-{c}</t></is></c>')
            else:
                # DECLARED, STYLED, EMPTY. This is the padding that costs:
                # the cell exists in the XML, carries a style, and holds no
                # value — and a reader that materializes one object per
                # declared cell pays for all 655 360 of them.
                parts.append(f'<c r="{names[c]}{r}" s="1"/>')
        rows.append(f'<row r="{r}" spans="1:{COLS}">{"".join(parts)}</row>')
    # THE PADDING: a style run over every column of every row, declaring
    # formatting for 16 384 x 40 cells that hold nothing.
    cols = "".join(f'<col min="{c + 1}" max="{c + 1}" width="9" '
                   f'customWidth="1" style="1"/>' for c in range(COLS))
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last}{ROWS}"/>'
        f'<cols>{cols}</cols>'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '</worksheet>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>')
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   '<sheets><sheet name="padded" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                   '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   '</Relationships>')
        z.writestr("xl/styles.xml",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
                   '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
                   '<borders count="1"><border/></borders>'
                   '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
                   '<cellXfs count="2"><xf/><xf applyFont="1" fontId="0"/></cellXfs>'
                   '</styleSheet>')
        z.writestr("xl/worksheets/sheet1.xml", sheet)


if __name__ == "__main__":
    build(sys.argv[1])
    print(f"built {sys.argv[1]}: "
          f"{os.path.getsize(sys.argv[1]) / 1024:.0f} KB on disk")
