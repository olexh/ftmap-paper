"""What a header says its own column is measured in, and what that rules out.
"""

from __future__ import annotations

import re

LENGTH = {"м", "cм", "мм", "км", "m", "cm", "mm", "km", "ft", "in"}
AREA = {"м2", "кв.м", "кв. м", "m2", "sq.m"}
VOLUME = {"м3", "куб.м", "куб. м", "m3", "l", "л"}
COUNT = {"од", "од.", "шт", "шт.", "pcs", "units"}
MASS = {"кг", "т", "kg", "t", "тонн", "tonnes"}
POWER = {"квт", "кв", "к.с", "к.с.", "kw", "hp"}
TIME = {"днів", "days", "год", "h"}

NON_MONETARY = LENGTH | AREA | VOLUME | COUNT | POWER | TIME | MASS
NON_TONNAGE = LENGTH | AREA | POWER | TIME

_COUNT_WORDS = re.compile(
    r"^\s*(кількість|к-сть|кіл-сть|число|количество|кол-во|number of|count of|"
    r"count|qty|quantity)\b", re.IGNORECASE)

MONEY = re.compile(r"^(amount|capital|netWorth|valueOfGoods)")
TONNAGE = re.compile(r"[Tt]onnage$")

_UNIT = re.compile(r"[,(]\s*([^,()]{1,8})\s*\)?\s*$")


def _folded(group: set[str]) -> frozenset[str]:
    """A unit group as the header's own units are read: case-folded, with a
    trailing full stop off, so `Од.` and `од` are one unit."""
    return frozenset(u.casefold().rstrip(".") for u in group)


_GROUPS_FOLDED = tuple(_folded(g)
                       for g in (LENGTH, AREA, VOLUME, COUNT, MASS, POWER, TIME))
_NON_MONETARY_FOLDED = _folded(NON_MONETARY)
_NON_TONNAGE_FOLDED = _folded(NON_TONNAGE)


def declared_unit(header: str | None) -> str | None:
    """The unit a header names for itself, or None."""
    if not header:
        return None
    m = _UNIT.search(header.strip())
    if not m:
        return None
    unit = m.group(1).strip().casefold().rstrip(".")
    for group in _GROUPS_FOLDED:
        if unit in group:
            return unit
    return None


def counts_something(header: str | None) -> bool:
    """Whether the header's first word says the column counts things."""
    return bool(header) and _COUNT_WORDS.match(header) is not None


def contradicts(header: str | None, qname: str) -> str | None:
    """A value-free reason when the header's own unit rules this property out.
    """
    name = qname.split(":", 1)[-1]
    unit = declared_unit(header)
    if counts_something(header) and (MONEY.match(name) or TONNAGE.search(name)):
        what = "a monetary value" if MONEY.match(name) else "a tonnage"
        return (f"the header says this column counts something"
                + (f", in {unit!r}" if unit else "")
                + f"; {name} is {what}, not a count")
    if unit is None:
        return None
    if MONEY.match(name) and unit in _NON_MONETARY_FOLDED:
        return (f"the header declares this column is measured in "
                f"{unit!r}; {name} is a monetary value")
    if TONNAGE.search(name) and unit in _NON_TONNAGE_FOLDED:
        return (f"the header declares this column is measured in "
                f"{unit!r}; {name} is a tonnage")
    return None
