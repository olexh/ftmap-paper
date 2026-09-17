# tests/test_layout.py
from ftmap.config import Config
from ftmap.io.frame import column_values
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid

CFG = Config.load(None)


def _frame(rows, merges=()):
    return build_frame(
        Grid(rows=rows, sheet="s", merges=list(merges)),
        path="/x/y.xlsx", sha256="0" * 64, source_id="000000000000/s", cfg=CFG,
    )


def test_header_on_first_row():
    f = _frame([["ПІБ", "Дата народження"], ["Коваленко Іван", "17.09.1980"]])
    assert f.header_row == 0
    assert [c.header for c in f.columns] == ["ПІБ", "Дата народження"]
    assert len(f.rows) == 1


def test_title_rows_above_the_header_are_furniture():
    f = _frame([
        ["Дані про депутатів Довгинцівської ради", None, None],
        [None, None, None],
        ["№ з/п", "ПІБ депутата", "Політична партія"],
        ["1", "Авраменко Ганна Михайлівна", "Криворізька організація"],
        ["2", "Баландін Олексій Юрійович", "Дніпропетровська організація"],
    ])
    assert f.header_row == 2
    kinds = {(r.index, r.kind) for r in f.furniture}
    assert (0, "title") in kinds
    assert (1, "blank") in kinds
    assert len(f.rows) == 2


def test_two_header_rows_keep_machine_names_and_human_labels():
    f = _frame([
        ["a", "prefLabel", "headFn"],
        ["Ідентифікатор", "Повна назва", "Ім'я керівника"],
        ["22819278", "КП ТК ПРИЛУКИ", "Павлютіна Ірина Миколаївна"],
    ])
    assert f.header_row == 0
    assert f.label_row == 1
    assert [c.header for c in f.columns] == ["a", "prefLabel", "headFn"]
    assert [c.label for c in f.columns][1] == "Повна назва"
    assert len(f.rows) == 1


def test_a_latin_first_data_row_is_not_mistaken_for_a_label_row():
    """The script test alone would eat this record."""
    f = _frame([["ПІБ", "Посада"], ["Kovalenko Ivan", "deputy"],
                ["Shevchenko Olha", "secretary"]])
    assert f.label_row is None
    assert len(f.rows) == 2


def test_merged_cells_are_forward_filled_and_counted():
    f = _frame(
        [["Розділ", None, "Штат"], ["1", None, "2"]],
        merges=[(0, 0, 0, 1)],
    )
    assert [c.header for c in f.columns] == ["Розділ", "Розділ", "Штат"]
    assert f.filled_cells == 1


def test_a_full_width_merge_is_not_broadcast_across_the_columns():
    """orgbook/252ca367 merges a section label across all six columns. Filling
    it made «ЧАГОР» a value of the ЄДРПОУ, website, e-mail and phone columns at
    once, and hid the row from the sparse detector."""
    f = _frame(
        [["Назва", "ЄДРПОУ", "Телефон"],
         ["Комунальне підприємство", "22819278", "0501234567"],
         ["ЧАГОР", None, None]],
        merges=[(2, 0, 2, 2)],
    )
    assert column_values(f, "c1") == ["22819278", None]
    assert column_values(f, "c2") == ["0501234567", None]
    assert [(r.index, r.kind) for r in f.furniture] == [(2, "sparse")]
    assert f.filled_cells == 0


def test_a_title_merged_across_every_column_is_not_a_header():
    """Measured on reception/033d6213. Merge-filling gives the title a perfect
    fill rate and width match; scoring the unfilled grid is what saves it."""
    f = _frame(
        [["Дані про депутатів районної ради", None, None],
         [None, None, None],
         ["№ з/п", "ПІБ депутата", "Політична партія"],
         ["1", "Авраменко Ганна", "Криворізька організація"],
         ["2", "Баландін Олексій", "Дніпропетровська організація"]],
        merges=[(0, 0, 0, 2)],
    )
    assert f.header_row == 2
    assert [c.header for c in f.columns] == ["№ з/п", "ПІБ депутата", "Політична партія"]


def test_sparse_and_totals_are_reported_but_kept():
    f = _frame([
        ["ПІБ", "Посада", "Примітка"],
        ["Апарат ради", None, None],
        ["Коваленко Іван", "депутат", None],
        ["Разом", "12", None],
    ])
    kinds = {(r.index, r.kind) for r in f.furniture}
    assert (1, "sparse") in kinds
    assert (3, "totals") in kinds
    # Reported, not dropped: a sparse record looks exactly like a section title.
    assert len(f.rows) == 3


def test_a_lone_filled_cell_is_reported_as_a_fact_not_as_a_section_title():
    """orgbook/69389bda row 43 is a person's name alone in a wide row. The
    detector cannot tell that from a section title, so it must not claim to."""
    f = _frame([
        ["Посада", "ПІБ", "Телефон"],
        ["Начальник", "Коваленко Іван", "0501234567"],
        [None, "Світлана Миколаївна", None],
    ])
    sparse = [r for r in f.furniture if r.index == 2]
    assert sparse and sparse[0].kind == "sparse"
    assert "filled cell" in sparse[0].reason
    assert len(f.rows) == 2


def test_a_totals_label_leads_its_row_and_its_tail_is_figures():
    """A note that merely begins with «Всього» is not a totals row."""
    f = _frame([
        ["№", "ПІБ", "Примітка"],
        ["1", "Коваленко Іван", "Всього дітей у сім'ї — четверо"],
        [None, "Шевченко Ольга", "Всього дітей у сім'ї — двоє"],
        ["Разом", "12", None],
    ])
    kinds = {(r.index, r.kind) for r in f.furniture}
    assert (3, "totals") in kinds
    assert not any(r.kind == "totals" and r.index in (1, 2) for r in f.furniture)


def test_a_totals_label_behind_a_blank_ordinal_column_is_still_totals():
    """The dominant real shape: 439 rows in the corpus look like this."""
    f = _frame([
        ["№", "Підрозділ", "Штат"],
        ["1", "Керівництво", "2"],
        [None, "Разом", "6.0"],
    ])
    assert [(r.index, r.kind) for r in f.furniture] == [(2, "totals")]


def test_a_totals_label_with_no_figures_yet_is_still_totals():
    """structure/18a411a5 sheet dod1 row 115: the label alone, value not
    entered. It must not fall through to sparse."""
    f = _frame([
        ["№", "Підрозділ", "Штат"],
        ["1", "Керівництво", "2"],
        [None, "Всього у відділах, службах міської ради", None],
    ])
    assert [(r.index, r.kind) for r in f.furniture] == [(2, "totals")]


def test_a_prose_note_leading_its_row_is_not_totals():
    """Rejected on the tail: prose has no figures beside it."""
    f = _frame([
        ["ПІБ", "Примітка", "Джерело"],
        ["Коваленко Іван", "працює", "анкета"],
        [None, "Всього дітей у сім'ї — четверо", "зі слів"],
    ])
    assert not any(r.kind == "totals" for r in f.furniture)


def test_repeated_header_row_mid_file():
    f = _frame([
        ["ПІБ", "Посада"],
        ["Коваленко Іван", "депутат"],
        ["ПІБ", "Посада"],
        ["Шевченко Ольга", "секретар"],
    ])
    assert [(r.index, r.kind) for r in f.furniture] == [(2, "repeat_header")]


def test_the_earliest_good_enough_header_wins_not_the_best_scoring_one():
    """A repeated header mid-file scores higher than the real one, because the
    rows under it differ more. Taking the maximum would move the header down."""
    f = _frame([
        ["ПІБ", "Посада"],
        ["Коваленко Іван", "депутат"],
        ["ПІБ", "Посада"],
        ["Шевченко Ольга", "секретар"],
    ])
    assert f.header_row == 0


def test_clean_table_fires_no_detector_at_all():
    """Negative control. Over-firing inflates every downstream number.

    Three columns wide on purpose: the sparse detector requires a width of
    three, so a two-column control cannot exercise it at all and would give no
    signal on the detector most likely to over-fire.
    """
    rows = [["ПІБ", "Дата", "Посада"]] + [
        [f"Особа {i}", f"0{i}.01.1990", "депутат"] for i in range(1, 20)]
    f = _frame(rows)
    assert f.furniture == []
    assert len(f.rows) == 19


def test_column_values_reads_by_stable_id():
    f = _frame([["ПІБ", "Дата"], ["Коваленко", "17.09.1980"], ["Шевченко", "01.02.1990"]])
    assert column_values(f, "c0") == ["Коваленко", "Шевченко"]


def test_a_data_row_is_not_a_label_row_because_it_flipped_script():
    """The script-flip test is right about the aircraft register — a Ukrainian
    header above an English translation — and was wrong about the tax-debtor
    register, whose Latin header sits above ordinary data containing Cyrillic
    names. `_differs` should have caught it, except that file redacts 93 % of
    its identifier column, so the one row in seven carrying a real code looks
    unlike its neighbours by accident. `body_start` skips a label row, so a
    real record was being consumed as furniture."""
    rows = [["ondate", "tin_s", "name"],
            ["2022-02-22", "31648017", 'ТОВ "КАРНЕТ ПЛЮС"'],
            ["2022-02-22", "**********", "ЄГОРОВА ОЛЬГА ВАСИЛІВНА"],
            ["2022-02-22", "**********", "ПРУС ІГОР ІВАНОВИЧ"]]
    f = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/t.csv",
                    "0" * 64, "sid/s", CFG)
    assert f.header_row == 0
    assert f.label_row is None, "row 1 is data: 2022-02-22 and 31648017 are values"
    assert len(f.rows) == 3, "no record may be consumed as furniture"


def test_a_real_translation_row_is_still_a_label_row():
    """The case the rule exists for: labels are words, in a different script
    from the header above them."""
    rows = [["Тип/модель", "Державний знак"],
            ["Aircraft Type", "Nationality and Registration Mark"],
            ["АН-2", "UR-12345"],
            ["ЯК-52", "UR-54321"]]
    f = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/a.xls",
                    "0" * 64, "sid/s", CFG)
    assert f.label_row == 1
    assert [c.label for c in f.columns] == ["Aircraft Type",
                                            "Nationality and Registration Mark"]
    assert len(f.rows) == 2


def test_a_group_header_row_names_a_band_for_every_column_under_it():
    """The ship register's shape: the words sit in one cell and the merge
    covers the neighbours; the second band is merged on its own cell."""
    f = _frame([
        [None, None, "Власник, фрактувальник", None, None, "Характеристики", None],
        ["Реєстраційний №", "Назва судна", "ПІБ", "Назва (юр)",
         "Назва юр. (фрахтувальник)", "Довжина, м", "Ширина, м"],
        ["1", "Аврора", "Коваленко Іван", None, None, "7.81", "2.40"],
        ["2", "Нептун", None, "ТОВ Флот", "ПрАТ Пароплавство", "12.0", "3.1"],
        ["3", "Зоря", "Шевченко Ольга", None, None, "5.5", "1.9"],
    ], merges=[(0, 3, 0, 4), (0, 5, 0, 6)])
    assert f.header_row == 1 and f.group_row == 0
    assert [c.group for c in f.columns] == [
        None, None, "Власник, фрактувальник", "Власник, фрактувальник",
        "Власник, фрактувальник", "Характеристики", "Характеристики"]
    assert [c.header for c in f.columns][2:4] == ["ПІБ", "Назва (юр)"]
    assert (0, "group") in {(r.index, r.kind) for r in f.furniture}
    assert len(f.rows) == 3


def test_a_title_row_with_two_cells_and_no_merge_is_still_a_title():
    """Without a merge nothing says the cell is over a band rather than
    beside another title."""
    f = _frame([
        ["Реєстр суден", None, None, "станом на 1 липня", None],
        ["Реєстраційний №", "Назва судна", "ПІБ", "Назва (юр)", "Довжина, м"],
        ["1", "Аврора", "Коваленко Іван", None, "7.81"],
        ["2", "Нептун", None, "ТОВ Флот", "12.0"],
        ["3", "Зоря", "Шевченко Ольга", None, "5.5"],
    ])
    assert f.header_row == 1 and f.group_row is None
    assert all(c.group is None for c in f.columns)
    assert (0, "title") in {(r.index, r.kind) for r in f.furniture}


def test_a_banner_merged_across_every_column_is_not_a_group_row():
    f = _frame([
        ["Реєстр суден", None, None],
        ["Реєстраційний №", "ПІБ", "Назва (юр)"],
        ["1", "Коваленко Іван", None],
        ["2", None, "ТОВ Флот"],
    ], merges=[(0, 0, 0, 2)])
    assert f.header_row == 1 and f.group_row is None
    assert all(c.group is None for c in f.columns)


def test_a_header_merged_down_through_the_group_row_is_a_header_not_a_band():
    from ftmap.io.layout import _group_bands
    raw = [["Реєстраційний №", "Власник", None, "Судно"],
           [None, "ПІБ", "Назва (юр)", None],
           ["1", "Коваленко Іван", None, "Аврора"]]
    row, groups = _group_bands(raw, [(0, 0, 1, 0), (0, 1, 0, 2), (0, 3, 1, 3)], 1, 4)
    assert row == 0
    assert groups == [None, "Власник", "Власник", None]
