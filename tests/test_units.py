# tests/test_units.py
"""The header's own unit, and what it rules out.

Every case here is a real column from `corpus-external/`, and the four that
must be refused are the four the etalon corpus caught being bound to a money or
tonnage property: a width in metres to `Vessel:amount`, a height in metres to
`Vessel:amountUsd`, an engine count and a deck count to `Vessel:amount` and
`Vessel:tonnage`.
"""

from __future__ import annotations

import pytest

from ftmap.plan.units import contradicts, declared_unit

REFUSED = [
    ("Ширина, м", "Vessel:amount"),
    ("Висота, м", "Vessel:amountUsd"),
    ("Кількість головних механізмів, од", "Vessel:amount"),
    ("Кількість палуб, од", "Vessel:tonnage"),
    ("Площа, кв.м", "RealEstate:amount"),
]

ALLOWED = [
    # A tonnage IS a mass, so the unit and the property agree.
    ("Вантажопідйомність, т", "Vessel:deadweightTonnage"),
    # REGISTER TONNAGE IS NOT A MASS. Gross and net tonnage are volume-derived
    # indices in register tons — dimensionless units — and every Ukrainian
    # ship register writes them «місткість, од». The etalon binds all three
    # of these; the rule refused them for a year as "a mass in units".
    ("Валова місткість, од", "Vessel:grossRegisteredTonnage"),
    ("Чиста місткість, од", "Vessel:tonnage"),
    ("Валова місткість, од", "Vessel:tonnage"),
    # Real money columns from this corpus. None declares a unit, so the rule
    # cannot fire on them — which is the point: it refuses a contradiction, it
    # does not decide what money looks like.
    ("Сума, грн.", "Contract:amount"),
    ("zn_all", "Debt:amount"),
    ("Пропозиція", "ContractAward:amount"),
    ("ecMaxContribution", "Project:amount"),
    # A unit that rules out money says nothing about a date or a name.
    ("Максимальна злітна маса, кг", "Airplane:buildDate"),
    ("Ширина, м", "Vessel:name"),
]


@pytest.mark.parametrize("header,qname", REFUSED)
def test_a_declared_unit_refuses_a_property_it_contradicts(header, qname):
    reason = contradicts(header, qname)
    assert reason, f"{header} -> {qname} should have been refused"
    # The reason is publishable: it quotes the header's unit, never a cell.
    assert declared_unit(header) in reason


@pytest.mark.parametrize("header,qname", ALLOWED)
def test_an_agreeing_or_unlabelled_header_is_left_alone(header, qname):
    assert contradicts(header, qname) is None


def test_a_header_that_counts_something_is_never_a_tonnage_or_money():
    """The deck count and the engine count were the cases the unit rule was
    written for, and `од` no longer rules a tonnage out — so what keeps a
    count out of `tonnage` is the header saying it COUNTS: «Кількість ...»
    is a count whatever unit follows it, and a count is neither a mass nor a
    price. Header only; no cell is read."""
    assert contradicts("Кількість палуб, од", "Vessel:tonnage")
    assert contradicts("Кількість палуб", "Vessel:tonnage")
    assert contradicts("Кількість головних механізмів, од", "Vessel:amount")
    assert contradicts("Number of decks", "Vessel:grossRegisteredTonnage")
    assert contradicts("Количество", "Vessel:amount")
    # A count is a fine `number`, and says nothing about a name or a date.
    assert contradicts("Кількість палуб, од", "Vessel:name") is None
    assert contradicts("Кількість палуб, од", "Vessel:buildDate") is None


def test_a_unit_word_loose_in_a_header_declares_nothing():
    """Only a trailing `, unit` or `(unit)` counts. A rule that matched a unit
    anywhere would read `м` out of half the Ukrainian language."""
    assert declared_unit("Модель судна") is None
    assert declared_unit("Найменування") is None
    assert declared_unit("Місцезнаходження") is None
    assert declared_unit("Ширина, м") == "м"
    assert declared_unit("Draught (m)") == "m"


def test_an_unheaded_column_is_not_second_guessed():
    assert declared_unit(None) is None
    assert contradicts(None, "Vessel:amount") is None
