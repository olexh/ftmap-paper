# tests/test_columns.py
from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.profile.columns import profile_frame

CFG = Config.load(None)


def _profiles(rows):
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "000000000000/", CFG)
    return {p.id: p for p in profile_frame(frame, CFG)}


def test_rates_are_over_records_not_over_the_grid():
    p = _profiles([
        ["ПІБ", "Дата"],
        ["Коваленко Іван", "17.09.1980"],
        ["Шевченко Ольга", None],
    ])
    assert p["c1"].count == 2
    assert p["c1"].filled == 1
    assert p["c1"].fill_rate == 0.5
    assert p["c0"].distinct_ratio == 1.0


def test_detectors_and_shapes_are_attached():
    p = _profiles([["Дата"], ["17.09.1980"], ["01.02.1990"]])
    assert p["c0"].detectors["date"] == 1.0
    assert p["c0"].shapes[0] == ("dd.dd.dddd", 1.0)


def test_samples_are_bounded_and_distinct():
    rows = [["Місто"]] + [["Київ"]] * 30 + [["Львів"], ["Одеса"]]
    p = _profiles(rows)
    s = p["c0"].samples
    # Only 3 distinct values exist in this column, so that is exactly how
    # many samples there must be — "or CFG.sample_size" let the assertion
    # pass either way and never actually pinned down the count. Every part of
    # this column by length is `Київ`, so the three are reached only by the
    # rule's top-up; a sampler that stopped at what the parts gave would show
    # one value and call the column described.
    assert len(s) == 3
    assert len(set(s)) == len(s)


def test_samples_are_what_the_column_is_mostly_made_of():
    """The measured failure this rule replaced.

    `BRAND` in the МВС vehicle registry was shown to the model as CB, CZ, DL,
    DS, KV — five two-letter codes holding 0.03% of 39,607 rows — because they
    were the shortest distinct values. The samples must lead with what the
    column actually holds.
    """
    rows = ([["BRAND"]] + [["VOLKSWAGEN"]] * 40 + [["RENAULT"]] * 30
            + [["BMW"]] * 20 + [["CB"], ["CZ"], ["DL"], ["DS"], ["KV"]])
    s = _profiles(rows)["c0"].samples
    assert s[:3] == ["BMW", "RENAULT", "VOLKSWAGEN"]


def test_samples_span_the_length_distribution():
    """One kind of value is not a description of a column that holds two.

    The sanctions `name` column is a quarter vessel names and a quarter
    company names; shown only its short end it read as a column of codes and
    cost the binding outright.
    """
    rows = ([["name"]] + [[f"SHIP{i:02d}"] for i in range(20)]
            + [[f"JOINT-STOCK COMPANY NUMBER {i:02d}"] for i in range(20)])
    s = _profiles(rows)["c0"].samples
    assert any(len(v) < 10 for v in s)
    assert any(len(v) > 20 for v in s)


def test_a_near_unique_column_is_sampled_by_shape():
    """Where every value occurs once, the count says nothing and the shape
    tie-break is the whole rule: a VIN column must be sampled with VINs, not
    with the three short fragments that happen to sit in it."""
    rows = ([["VIN"]] + [[f"WVWZZZ1JZ3W{i:06d}"] for i in range(30)]
            + [["179"], ["0340"], ["2004"]])
    s = _profiles(rows)["c0"].samples
    assert len(s) == CFG.sample_size
    assert all(len(v) == 17 for v in s)


def test_to_dict_is_json_safe():
    import json
    p = _profiles([["ПІБ"], ["Коваленко"]])
    json.dumps(p["c0"].to_dict(), ensure_ascii=False)


def test_a_column_of_the_word_null_is_not_a_full_column():
    """`NULL` is not empty, so `fill_rate` counts it and a column that is
    nothing but those four letters reports as 100 % filled. The tax-debtor
    register has two, and the structure call declared an entity from one.
    Reported beside the fill rate, never subtracted from it."""
    frame = build_frame(Grid(rows=[["sub_name", "name"],
                                   ["NULL", "ТОВ КАРНЕТ"],
                                   ["NULL", "ЄГОРОВА ОЛЬГА"],
                                   ["NULL", "ПРУС ІГОР"]], sheet="s", merges=[]),
                        "/t.csv", "0" * 64, "sid/s", CFG)
    p = {c.id: c for c in profile_frame(frame, CFG)}
    assert p["c0"].fill_rate == 1.0, "fill_rate keeps meaning what it meant"
    assert p["c0"].sentinel_share == 1.0
    assert p["c1"].sentinel_share == 0.0


def test_a_zero_is_a_number_not_an_absence():
    """`0` is excluded from the sentinel list on purpose: a zero is a number
    far more often than it is a missing value, and a longer list starts
    swallowing real content."""
    frame = build_frame(Grid(rows=[["qty"], ["0"], ["0"], ["12"]],
                             sheet="s", merges=[]),
                        "/t.csv", "0" * 64, "sid/s", CFG)
    assert profile_frame(frame, CFG)[0].sentinel_share == 0.0


def test_a_masked_column_reports_what_is_left_of_it():
    """The tax-debtor register's `tin_s` in miniature: a real identifier on a
    minority of rows, a redaction constant on the rest.

    `distinct_ratio` answers "could this identify a row" and says no — for a
    reason that is entirely about the mask and not at all about the six real
    codes underneath it. On the full file that rejection cost the run its key
    and emitted 43 058 entities from 21 529 rows.
    """
    rows = [["tin_s"]] + [["**********"]] * 12 + [[f"3164801{i}"] for i in range(6)]
    p = _profiles(rows)["c0"]
    assert p.modal_share == 12 / 18
    assert round(p.distinct_ratio, 2) == round(7 / 18, 2)
    assert p.distinct_ex_modal == 1.0


def test_a_column_that_is_merely_coarse_stays_coarse():
    """The other half, and the reason this is not "drop the commonest value
    and hope": a managing body that genuinely appears on most rows does not
    become distinct when its own name is removed, so it stays under the floor
    and `reinstate_key` goes on handling it."""
    rows = [["manager"]] + [["ФОНД ДЕРЖАВНОГО МАЙНА"]] * 12 + [["МІНОБОРОНИ"]] * 6
    p = _profiles(rows)["c0"]
    assert p.modal_share == 12 / 18
    assert p.distinct_ex_modal == 1 / 6
    assert p.distinct_ex_modal < 0.5, "still under the default key floor"


def test_a_column_of_one_value_has_no_rest_to_report():
    p = _profiles([["ondate"]] + [["2022-02-22"]] * 9)["c0"]
    assert p.modal_share == 1.0
    assert p.distinct_ex_modal == 0.0


def test_the_group_header_is_carried_into_the_profile():
    from ftmap.io.frame import Column, Frame
    frame = Frame(source_id="s", path="/x.xlsx", sheet="s", sha256="0" * 64,
                  columns=[Column("c0", 0, "ПІБ", None, "Власник, фрахтувальник"),
                           Column("c1", 1, "Довжина, м", None, None)],
                  rows=[["Коваленко Іван", "7.81"]], row_index=[1],
                  header_row=0, label_row=None, group_row=None)
    profiles = profile_frame(frame, CFG)
    assert [p.group for p in profiles] == ["Власник, фрахтувальник", None]
    assert profiles[0].to_dict()["group"] == "Власник, фрахтувальник"
