# tests/test_shapes.py
from ftmap.profile.shapes import shape, top_shapes


def test_digits_and_separators():
    assert shape("17.09.1980") == "dd.dd.dddd"
    assert shape("+380501234567") == "+d+"


def test_scripts_are_distinguished():
    assert shape("Іван") == "CCCC"
    assert shape("Ivan") == "LLLL"
    # 9 letters is over RUN_CAP, so the run collapses.
    assert shape("Коваленко Іван") == "C+ CCCC"


def test_long_runs_collapse_so_free_text_does_not_explode():
    assert shape("а" * 40) == "C+"
    assert shape("1" * 12) == "d+"
    assert shape("x" * 8) == "LLLLLLLL"


def test_empty_and_none_are_their_own_shape():
    assert shape("") == ""
    assert shape("   ") == ""


def test_top_shapes_are_shares_over_non_null():
    vals = ["17.09.1980", "01.02.1990", "not a date", "", None]
    got = top_shapes(vals, 2)
    assert got[0] == ("dd.dd.dddd", 2 / 3)
    assert got[1][0] == "LLL L LLLL"
