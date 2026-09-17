# Fixture provenance

Five real files, one per defect class named in the design spec
(design note §4), copied unmodified
from the public corpus at `article-1/01-data/raw_835/`. They are public
open-data exports (data.gov.ua), so they may live in git; nothing here comes
from `../data/`.

Each row below gives: the fixture file, the source file it was copied from
(path relative to `article-1/01-data/raw_835/`), the `resource_id` from that
corpus's `manifest.jsonl` (the resource-level id assigned by data.gov.ua, not
one we invented), the source dataset, and the defect class the fixture
exercises.

| fixture | source path | `resource_id` | dataset (organization) | defect class (spec §4) |
|---|---|---|---|---|
| `deputies_popolo.csv` | `deputies/0427f07d-5b47-4d7c-996f-5077ca4b56fe.csv` | `0427f07d-5b47-4d7c-996f-5077ca4b56fe` | «Дані про депутатів місцевих рад...» (buchanska-miska-rada) | clean control — already Popolo-shaped English headers (`id`, `votingIdentifier`, `familyName`, ...), header at row 0, no furniture. The "nothing wrong" case the other four are contrasted against. |
| `orgbook_two_header_rows.xlsx` | `orgbook/0402708c-3d69-4f32-9b48-71dbd97719fd.xlsx` | `0402708c-3d69-4f32-9b48-71dbd97719fd` | «Довідник підприємств, установ (закладів) та організацій Прилуцької міської ради» (vykonavchyi-komitet-prylutskoi-miskoi-rady) | two header rows — machine names (`prefLabel`, `headFn`, ...) at row 0, Ukrainian human labels (`Повна назва`, `Ім'я керівника`, ...) at row 1. |
| `reception_title_rows.xlsx` | `reception/033d6213-ea38-45d0-9328-8b3e52ab8011.xlsx` | `033d6213-ea38-45d0-9328-8b3e52ab8011` | «Дані про депутатів Довгинцівської районної в місті ради...» (vdrr) | title row (merged across all 3 columns) plus a blank spacer row above the real header, which sits at row 2. |
| `reception_semicolon.csv` | `reception/03d4926f-e0d0-4ed1-944d-3237cd41a4d0.csv` | `03d4926f-e0d0-4ed1-944d-3237cd41a4d0` | «Графік особистого прийому громадян керівництвом районної у місті ради» (vykonavchyi-komitet-podilskoyi-raionnoyi-u-misti-kropyvnytskomu-rady) | semicolon-delimited CSV; values themselves contain commas, so delimiter sniffing (not a fixed `,`) is required to get 3 columns instead of 1. |
| `structure_merged.xlsx` | `structure/0171603a-0baf-40f5-bd0b-dbd5270a3fa7.xlsx` | `0171603a-0baf-40f5-bd0b-dbd5270a3fa7` | «Структура та штатна чисельність апарату суду - 2020 рік» (hospodarskyi-sud-zakarpatskoyi-oblasti) | header row 0 continued across rows 1-3 by merged cells (`№ з/п` and `Кількість штатних одиниць` each merged rows 0-3); a second merged block lower in the sheet repeats the shape mid-file. |

## Note on `structure_merged.xlsx`

The merge in this file only covers column 0 (`№ з/п`) and column 2
(`Кількість штатних одиниць`) across rows 0-3; the middle cell, column 1, is
*not* merged there — each of rows 0-3 holds its own fragment of a wrapped
title ("СТРУКТУРА" / "і штатна чисельність апарату" / "Господарського суду
Закарпатської області станом на 04.02.2020" / "Назва структурного підрозділу
та посад"). After merge-fill, rows 1-3 read as
`["№ з/п", <title fragment>, "Кількість штатних одиниць"]` — the same header
text broadcast into columns 0 and 2, with row-specific prose in column 1 —
and none of `_classify_furniture`'s rules catch this shape: not
`repeat_header`, because column 1 differs; not `totals`, no totals label; not
`sparse`, all 3 cells are filled. They survive as three non-data body rows,
confirmed with a direct call to `build_frame`.
`tests/test_corpus.py::test_merged_header_cells_are_filled` only asserts the
header row itself is read correctly, which it is; this narrower gap is a
layout heuristic in `src/ftmap/io/layout.py`, not fixed here since it sits in
`src/` and this task works only in `tests/`.

## Regenerating

```bash
cd ftmap-paper
R=../article-1/01-data/raw_835
cp "$R/deputies/0427f07d-5b47-4d7c-996f-5077ca4b56fe.csv" tests/fixtures/deputies_popolo.csv
cp "$R/orgbook/0402708c-3d69-4f32-9b48-71dbd97719fd.xlsx" tests/fixtures/orgbook_two_header_rows.xlsx
cp "$R/reception/033d6213-ea38-45d0-9328-8b3e52ab8011.xlsx" tests/fixtures/reception_title_rows.xlsx
cp "$R/reception/03d4926f-e0d0-4ed1-944d-3237cd41a4d0.csv" tests/fixtures/reception_semicolon.csv
cp "$R/structure/0171603a-0baf-40f5-bd0b-dbd5270a3fa7.xlsx" tests/fixtures/structure_merged.xlsx
```
