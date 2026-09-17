"""Deterministic value detectors."""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from rigour.ids import INN
from rigour.territories import lookup_territory

from ftmap.normalize.canonical import (PHONE_RULES, VALUE_ALONE,
                                       ColumnEvidence, canonicalize,
                                       column_evidence, phone_rule_nsn, is_sentinel)

_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")
_URL = re.compile(r"^https?://\S+$", re.I)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def is_date(value: str, evidence: ColumnEvidence = VALUE_ALONE) -> bool:
    """Whatever `canonicalize(value, "date", evidence)` would accept — the
    written forms in DATE_PATTERNS always, and the two bare-number forms only
    where the column corroborates them — so the detector and the canonicalizer
    can never disagree about a date.
    """
    return canonicalize(value, "date", evidence).ok


def _phone_rule_detector(rule: dict) -> Callable[[str], bool]:
    """One PHONE_RULES row as a detector, via the SAME matching function
    `_canon_phone` uses per row, so `is_phone_<name>` and canonicalization
    agree about every value by construction."""
    def detect_one(value: str) -> bool:
        international = value.strip().startswith("+")
        return phone_rule_nsn(_digits(value), international, rule) is not None
    return detect_one


_PHONE_DETECTORS: dict[str, Callable[[str], bool]] = {
    f"phone_{rule['name']}": _phone_rule_detector(rule) for rule in PHONE_RULES
}
is_phone_ua = _PHONE_DETECTORS["phone_ua"]
is_phone_ru = _PHONE_DETECTORS["phone_ru"]


def is_email(value: str) -> bool:
    return bool(_EMAIL.match(value.strip()))


def is_url(value: str) -> bool:
    return bool(_URL.match(value.strip()))


def is_edrpou(value: str) -> bool:
    """Ukrainian legal-entity code: 8 digits with a weighted mod-11 check
    digit.
    """
    v = value.strip()
    if not (v.isdigit() and len(v) == 8):
        return False
    code = int(v)
    base = ((7, 1, 2, 3, 4, 5, 6) if 30_000_000 <= code <= 60_000_000
            else (1, 2, 3, 4, 5, 6, 7))
    total = sum(int(c) * w for c, w in zip(v[:7], base)) % 11
    if total == 10:
        total = sum(int(c) * (w + 2) for c, w in zip(v[:7], base)) % 11
        if total == 10:
            return False
    return total == int(v[7])


def is_rnokpp(value: str) -> bool:
    """Ukrainian taxpayer number: 10 digits with a weighted check digit."""
    v = value.strip()
    if not (v.isdigit() and len(v) == 10):
        return False
    weights = (-1, 5, 7, 9, 4, 6, 10, 5, 7)
    total = sum(int(c) * w for c, w in zip(v[:9], weights))
    return total % 11 % 10 == int(v[9])


def is_inn_ru(value: str) -> bool:
    return bool(INN.is_valid(value.strip()))


def is_snils_ru(value: str) -> bool:
    """Russian personal insurance number: 11 digits, last two a checksum."""
    v = _digits(value)
    if len(v) != 11:
        return False
    total = sum(int(c) * (9 - i) for i, c in enumerate(v[:9]))
    if total < 100:
        check = total
    elif total in (100, 101):
        check = 0
    else:
        check = total % 101
        if check in (100, 101):
            check = 0
    return check == int(v[9:])


def is_gender_code(value: str) -> bool:
    """Whatever GENDER_LEXICON's spellings accept, via the canonicalizer
    itself — the module docstring has the drift this replaced."""
    return canonicalize(value, "gender").ok


def is_country(value: str) -> bool:
    """A country NAME, or an explicit uppercase code."""
    v = value.strip()
    if len(v) < 2:
        return False
    if len(v) < 4 and not v.isupper():
        return False
    try:
        return lookup_territory(v) is not None
    except Exception:
        return False


_NAME_TOKENS = (2, 4)


def _name_text(value: str) -> str | None:
    """The value stripped, or None when it cannot be a name at all."""
    text = value.strip()
    if not text or any(ch.isdigit() for ch in text):
        return None
    return text


def is_proper_name(value: str) -> bool:
    """A personal or organisational name, written as a name."""
    text = _name_text(value)
    if text is None:
        return False
    tokens = text.split()
    if not _NAME_TOKENS[0] <= len(tokens) <= _NAME_TOKENS[1]:
        return False
    for token in tokens:
        if len(token) < 2 or not token[0].isupper():
            return False
        if not all(ch.isalpha() or ch in "-'’" for ch in token):
            return False
    return True


_LEGAL_FORMS = {
    "тов", "тзов", "пат", "прат", "ат", "зат", "ват", "тдв", "дп", "кп", "кнп",
    "ду", "днз", "дснз", "пп", "прп", "пмп", "фоп", "фг", "сфг", "сп", "гс",
    "го", "бо", "бф", "осбб", "жбк", "ск", "нвп", "нвф", "мп", "ап", "ко",
    "кт", "пф", "ррф", "ооо", "зао", "оао", "ао", "пао", "ип",
    "llc", "ltd", "inc", "corp", "co", "gmbh", "ag", "sa", "s.a", "srl",
    "s.r.l", "bv", "b.v", "nv", "plc", "oü", "sp. z o.o", "sp.z o.o",
    "limited", "corporation", "company",
}
_QUOTED = re.compile(r"[«\"„“”][^«»\"„“”]{2,}[»\"“”]")
_TRAILING_FORMS = ("limited liability company", "joint stock company",
                   "joint-stock company", "limited partnership",
                   "state enterprise", "communal enterprise")


def is_legal_name(value: str) -> bool:
    """An organisation's name, written as a register writes one."""
    text = _name_text(value)
    return text is not None and _legal_name(text)


def _legal_name(text: str) -> bool:
    """`is_legal_name` over a value already stripped and known digit-free."""
    tokens = text.split()
    if not 1 <= len(tokens) <= 12:
        return False
    if len(tokens) >= 2:
        edges = (tokens[0], tokens[-1])
        if any(t.strip(".,«»\"“”()").casefold() in _LEGAL_FORMS for t in edges):
            return True
        folded = text.casefold()
        if any(folded.endswith(form) for form in _TRAILING_FORMS):
            return True
    return _QUOTED.search(text) is not None


def is_numeric(value: str) -> bool:
    return bool(re.match(r"^-?\d+([.,]\d+)?$", value.strip()))


_INSTITUTION_WORDS = {
    "партія", "партії", "фракція", "фракції", "комісія", "комісії", "рада",
    "ради", "управління", "департамент", "відділ", "відділу", "служба",
    "служби", "суд", "поліція", "поліції", "університет", "інститут",
    "коледж", "академія", "ліцей", "школа", "гімназія", "лікарня", "центр",
    "фонд", "міністерство", "агентство", "інспекція", "адміністрація",
    "підприємство", "товариство", "організація", "об'єднання", "установа",
    "заклад", "кооператив", "асоціація", "спілка", "союз", "компанія",
    "корпорація", "банк", "госпіталь", "інтернат", "бюро", "комітет",
    "прокуратура", "митниця", "загін", "дирекція", "філія", "виконком",
    "облрада", "міськрада", "сільрада", "гу", "кз", "кнп", "нвк", "зош",
    "днз", "жек", "упп", "вдвс", "party", "faction", "commission",
    "council", "department", "ministry", "university", "hospital", "agency",
    "bureau", "committee", "association", "foundation", "institute",
}
_WORD = re.compile(r"[\w'’]+")


def is_org_name(value: str) -> bool:
    """An organisation's name, written without a legal form."""
    text = _name_text(value)
    if text is None:
        return False
    tokens = _WORD.findall(text)
    if not 1 <= len(tokens) <= 12:
        return False
    if any(t.casefold().replace("’", "'") in _INSTITUTION_WORDS for t in tokens):
        return True
    return _legal_name(text)


DETECTORS: dict[str, Callable[[str], bool]] = {
    "date": is_date,
    "phone_ua": is_phone_ua,
    "phone_ru": is_phone_ru,
    "email": is_email,
    "url": is_url,
    "edrpou": is_edrpou,
    "rnokpp": is_rnokpp,
    "inn_ru": is_inn_ru,
    "snils_ru": is_snils_ru,
    "gender_code": is_gender_code,
    "country": is_country,
    "numeric": is_numeric,
    "proper_name": is_proper_name,
    "legal_name": is_legal_name,
    "org_name": is_org_name,
}


def detect(values: list[str | None]) -> dict[str, float]:
    """Share of the column's REAL values each detector accepts. Zeroes are
    dropped.
    """
    non_null = [str(v).strip() for v in values if v is not None and str(v).strip()]
    non_null = [v for v in non_null if not is_sentinel(v)]
    if not non_null:
        return {}
    evidence = column_evidence(non_null)
    detectors = dict(DETECTORS, date=lambda v: is_date(v, evidence))
    counts = Counter(non_null)
    total = len(non_null)
    out: dict[str, float] = {}
    for name, fn in detectors.items():
        hits = 0
        for value, held in counts.items():
            try:
                if fn(value):
                    hits += held
            except Exception:
                continue
        if hits:
            out[name] = hits / total
    return out
