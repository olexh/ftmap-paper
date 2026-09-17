# Fixtures

Five public open-data exports from data.gov.ua, copied unmodified. Each
exercises one layout defect.

| fixture | dataset (publisher) | resource id | defect |
|---|---|---|---|
| `deputies_popolo.csv` | «Дані про депутатів місцевих рад...» (buchanska-miska-rada) | `0427f07d-5b47-4d7c-996f-5077ca4b56fe` | none: Popolo-shaped English headers at row 0, the control case |
| `orgbook_two_header_rows.xlsx` | «Довідник підприємств, установ (закладів) та організацій Прилуцької міської ради» (vykonavchyi-komitet-prylutskoi-miskoi-rady) | `0402708c-3d69-4f32-9b48-71dbd97719fd` | two header rows: machine names at row 0, Ukrainian labels at row 1 |
| `reception_title_rows.xlsx` | «Дані про депутатів Довгинцівської районної в місті ради...» (vdrr) | `033d6213-ea38-45d0-9328-8b3e52ab8011` | merged title row and a blank spacer above the header at row 2 |
| `reception_semicolon.csv` | «Графік особистого прийому громадян керівництвом районної у місті ради» (vykonavchyi-komitet-podilskoyi-raionnoyi-u-misti-kropyvnytskomu-rady) | `03d4926f-e0d0-4ed1-944d-3237cd41a4d0` | semicolon-delimited, values contain commas |
| `structure_merged.xlsx` | «Структура та штатна чисельність апарату суду - 2020 рік» (hospodarskyi-sud-zakarpatskoyi-oblasti) | `0171603a-0baf-40f5-bd0b-dbd5270a3fa7` | header row continued across rows 1-3 by merged cells; a second merged block repeats mid-sheet |
