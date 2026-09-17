"""Canonicalization, before the mapping engine rather than inside it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, KeysView

FREE_TYPES = {"string", "text", "name", "address", "number", "entity", "html", "json", "checksum"}
FTM_SAFE_TYPES = {"email", "url", "language", "ip", "topic", "mimetype"}

_EXCEL_EPOCH = date(1899, 12, 30)
_CENTURY_PIVOT = 30


PHONE_RULES: tuple[dict, ...] = (
    {"name": "ua", "cc": "380", "nsn": 9, "trunk": "0", "nsn_prefixes": ()},
    {"name": "ru", "cc": "7", "nsn": 10, "trunk": "8", "nsn_prefixes": ("9",)},
)

GENDER_LEXICON: dict[str, tuple[str, ...]] = {
    "male": ("ч", "чол", "чоловік", "чоловіча", "м", "муж", "мужской",
             "1", "male", "m"),
    "female": ("ж", "жін", "жінка", "жіноча", "жен", "женский",
               "2", "female", "f"),
}

MULTIVALUE_JOIN = ";"

PURE_PACKERS = ";|"

EVERY_PART = "every-part"
SEPARATOR_ONLY = "separator-only"

SENTINELS = {"null", "n/a", "na", "none", "-", "--", "?", "", "xxx",
             "undetermined", "not applicable", "не вказано", "не визначено",
             "не відноситься", "відсутній", "відсутня", "відсутнє",
             "нет данных", "немає",
             "безпартійний", "безпартійна", "безпартійні", "позафракційний",
             "позафракційна"}


_REDACTED = re.compile(r"^\*{3,}$")
_ZERO_DATE = re.compile(r"^0{4}(?:[-./]0{2}){0,2}(?:[ T]0{2}:0{2}(?::0{2})?)?$"
                        r"|^0{2}[-./]0{2}[-./]0{4}$")


def _is_sentinel_line(text: str) -> bool:
    return (text.casefold() in SENTINELS or _ZERO_DATE.match(text) is not None
            or _REDACTED.match(text) is not None)


def is_sentinel(value: str) -> bool:
    """A placeholder, on one line or on several."""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return _is_sentinel_line(value.strip())
    return all(_is_sentinel_line(line) for line in lines)


MULTIVALUE_POLICY: dict[str, tuple["re.Pattern[str]", str]] = {
    "phone": (re.compile(r"\s*[;,/|]\s*|\s+"), EVERY_PART),
    "email": (re.compile(r"\s*[;,/|]\s*|\s+"), EVERY_PART),
    "url": (re.compile(r"\s*[;,|]\s*|\s+"), EVERY_PART),
    "identifier": (re.compile(r"\s*[;,|]\s*|\s+"), EVERY_PART),
    "country": (re.compile(r"\s*[;,|]\s*"), EVERY_PART),
    "date": (re.compile(r"\s*[;|]\s*"), EVERY_PART),
    "name": (re.compile(r"\s*;\s*"), SEPARATOR_ONLY),
    "address": (re.compile(r"\s*;\s*"), SEPARATOR_ONLY),
}

MULTIVALUE_TYPES: "KeysView[str]" = MULTIVALUE_POLICY.keys()

DATE_PATTERNS: tuple[str, ...] = (
    r"^(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})$",
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[T ](?P<hh>\d{2}):(?P<mi>\d{2})"
    r"(:(?P<ss>\d{2}))?(\.\d+)?(?P<tz>Z|[+-]\d{2}(:?\d{2})?)?$",
    r"^(?P<d>\d{1,2})[./-](?P<m>\d{1,2})[./-](?P<y>\d{2}|\d{4})$",
    r"^(?P<d>\d{1,2})-(?P<mon>[A-Za-z]{3,9})-(?P<y>\d{4})$",
)

_DATE_PATTERNS = tuple(re.compile(p) for p in DATE_PATTERNS)

MONTH_NAMES = {m: i + 1 for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}

BARE_YEAR_RANGE = (1850, 2100)
EXCEL_SERIAL_RANGE = (1, 80000)
NUMERIC_DATE_CORROBORATION = 0.95


@dataclass(frozen=True)
class CanonResult:
    """One cell's verdict, plus the refused parts of a partly accepted pack."""

    ok: bool
    value: str | None
    reason: str | None
    canonicalizer: str
    refused: tuple[tuple[str, str], ...] = ()


def _ok(value: str, who: str) -> CanonResult:
    return CanonResult(True, value, None, who)


def _no(reason: str, who: str) -> CanonResult:
    return CanonResult(False, None, reason, who)


@dataclass(frozen=True)
class ColumnEvidence:
    """What the rest of a column says about a value its own text cannot settle.
    """

    numeric_dates: bool = False


VALUE_ALONE = ColumnEvidence()

_BARE_NUMBER = re.compile(r"^\d+$")
_FOUR_DIGITS = re.compile(r"^\d{4}$")
_FIVE_DIGITS = re.compile(r"^\d{5}$")


def _bare_number_date(t: str) -> CanonResult | None:
    """The date a column of numbers like this one would make of `t`, or None.
    """
    if _FOUR_DIGITS.match(t):
        lo, hi = BARE_YEAR_RANGE
        return _ok(t, "date") if lo <= int(t) <= hi else None
    if _FIVE_DIGITS.match(t):
        lo, hi = EXCEL_SERIAL_RANGE
        if lo <= int(t) <= hi:
            return _ok((_EXCEL_EPOCH + timedelta(days=int(t))).isoformat(), "date")
    return None


def column_evidence(values: "list[str | None]") -> ColumnEvidence:
    """One column's values, read once, into the facts `canonicalize` cannot
    see.
    """
    bare = [t for t in (str(v).strip() for v in values if v is not None)
            if t and _BARE_NUMBER.match(t)]
    if not bare:
        return ColumnEvidence(numeric_dates=False)
    dated = sum(1 for t in bare if _bare_number_date(t) is not None)
    return ColumnEvidence(
        numeric_dates=dated / len(bare) >= NUMERIC_DATE_CORROBORATION)


def _utc_shift(tz: str | None) -> timedelta:
    """The offset to SUBTRACT from a wall clock to put it on UTC."""
    if not tz or tz == "Z":
        return timedelta(0)
    sign = -1 if tz[0] == "-" else 1
    digits = tz[1:].replace(":", "")
    hh, mm = digits[:2], digits[2:]
    return sign * timedelta(hours=int(hh), minutes=int(mm or 0))


def _canon_date(v: str, evidence: ColumnEvidence = VALUE_ALONE) -> CanonResult:
    t = v.strip()
    y = mo = d = None
    parts: dict[str, str | None] = {}
    for pattern in _DATE_PATTERNS:
        m = pattern.match(t)
        if m:
            parts = m.groupdict()
            y_text = m.group("y")
            month_text = parts.get("mon")
            if month_text is not None:
                key = month_text.upper()[:3]
                if key not in MONTH_NAMES:
                    continue
                mo = MONTH_NAMES[key]
            else:
                mo = int(m.group("m"))
            y, d = int(y_text), int(m.group("d"))
            if len(y_text) == 2:
                y += 1900 if y >= _CENTURY_PIVOT else 2000
            if mo > 12 and d <= 12 and parts.get("hh") is None:
                mo, d = d, mo
            break
    if y is None:
        if _BARE_NUMBER.match(t):
            if not evidence.numeric_dates:
                return _no("a bare number is a date only where the rest of the "
                           "column's numbers are dates too; these are not",
                           "date")
            got = _bare_number_date(t)
            if got is not None:
                return got
            return _no("a bare number outside every numeric date form in the "
                       "active date pack", "date")
        return _no("no pattern in the active date pack matches", "date")
    try:
        if parts.get("hh") is not None:
            stamp = datetime(y, mo, d, int(parts["hh"]), int(parts["mi"]),
                             int(parts["ss"] or 0)) - _utc_shift(parts.get("tz"))
            return _ok(stamp.isoformat(), "date")
        return _ok(date(y, mo, d).isoformat(), "date")
    except ValueError as exc:
        return _no(f"date validator: {exc}", "date")


_PHONE_PUNCTUATION = re.compile(r"[\s()+\-–—.]")
_PHONE_LABEL = re.compile(r"^[^\d+:]{1,40}:\s*(?=[+\d(])")


def phone_rule_nsn(digits: str, international: bool, rule: dict) -> str | None:
    """Digits-only `digits` matches `rule`, or None. Returns the national
    significant number, not a bool, because the caller builds the E.164 form.
    """
    cc, nsn_len, trunk = rule["cc"], rule["nsn"], rule["trunk"]
    if digits.startswith(cc) and len(digits) == len(cc) + nsn_len:
        nsn = digits[len(cc):]
    elif (not international and digits.startswith(trunk)
          and len(digits) == len(trunk) + nsn_len):
        nsn = digits[len(trunk):]
    else:
        return None
    prefixes = rule.get("nsn_prefixes") or ()
    if prefixes and not nsn.startswith(prefixes):
        return None
    return nsn


def _canon_phone(v: str) -> CanonResult:
    """E.164, driven entirely by PHONE_RULES. No country appears in this body.
    """
    v = _PHONE_LABEL.sub("", v.strip(), count=1)
    stripped = _PHONE_PUNCTUATION.sub("", v.strip())
    if not stripped.isdigit():
        return _no("contains characters that are not digits or phone punctuation",
                   "phone")
    international = v.strip().startswith("+")
    for rule in PHONE_RULES:
        nsn = phone_rule_nsn(stripped, international, rule)
        if nsn is not None:
            return _ok("+" + rule["cc"] + nsn, "phone")
    return _no("no rule in the active phone pack matches this number", "phone")


def _canon_gender(v: str) -> CanonResult:
    t = v.strip().lower().rstrip(".")
    for canonical, spellings in GENDER_LEXICON.items():
        if t in spellings:
            return _ok(canonical, "gender")
    return _no("no spelling in the active gender lexicon matches", "gender")


def _canon_country(v: str) -> CanonResult:
    """Two recognisers, because neither alone covers what the registers print.
    """
    from rigour.territories import lookup_territory

    t = v.strip()
    try:
        terr = lookup_territory(t)
    except Exception:
        terr = None
    if terr is not None and getattr(terr, "code", None):
        code = str(terr.code).lower()
        if "-" in code:
            return _no("resolves to a subdivision, not a country", "country")
        return _ok(code, "country")
    from followthemoney import registry

    got = registry.country.clean(t)
    if got:
        return _ok(got, "country")
    return _no("no territory matches this name", "country")


_ID_SEPARATORS = re.compile(r"[\-–—./\s]")
_ID_PUNCTUATION = re.compile(r"[-–—./]")
_CODE_SHAPED = re.compile(r"[^\W_]+(?:[-–—./][^\W_]+)+")


def _canon_identifier(v: str) -> CanonResult:
    """An identifier is a token, and it has a digit in it somewhere."""
    t = v.strip()
    if not t:
        return _no("empty", "identifier")
    tokens = t.split()
    if len(tokens) > 2:
        return _no("an identifier is one token; this has internal whitespace",
                   "identifier")
    if len(tokens) == 2:
        series_number = (
            (tokens[0].isalpha() and tokens[0].isupper() and tokens[1].isdigit())
            or (tokens[1].isalpha() and tokens[1].isupper()
                and tokens[0].isdigit()))
        if not series_number:
            return _no("two tokens that are not a series and a number are a "
                       "phrase, not an identifier", "identifier")
    d = _ID_SEPARATORS.sub("", t)
    if not d:
        return _no("empty after removing separators", "identifier")
    if not any(c.isdigit() for c in d):
        squeezed = t.replace(" ", "")
        code = _CODE_SHAPED.fullmatch(squeezed)
        name_shaped = all(seg.istitle()
                          for seg in _ID_PUNCTUATION.split(squeezed))
        if not code or name_shaped:
            return _no("no digit anywhere; identifiers in this ontology carry "
                       "one, and this is not a code either", "identifier")
    return _ok(d, "identifier")


def _canon_free(v: str) -> CanonResult:
    return _ok(" ".join(v.split()), "whitespace")


def _canon_ftm(v: str, type_name: str) -> CanonResult:
    from followthemoney import registry

    ftm_type = registry.get(type_name)
    if ftm_type is None:
        return _canon_free(v)
    got = ftm_type.clean(v.strip())
    if got:
        return _ok(str(got), f"ftm:{type_name}")
    return _no(f"ftm {type_name} validator rejected it", f"ftm:{type_name}")


CANONICALIZERS: dict[str, "Callable[[str], CanonResult]"] = {
    "date": _canon_date,
    "phone": _canon_phone,
    "gender": _canon_gender,
    "country": _canon_country,
    "identifier": _canon_identifier,
}


def register(type_name: str, fn: "Callable[[str], CanonResult]") -> None:
    """Add or replace the canonicalizer for an FtM value type."""
    CANONICALIZERS[type_name] = fn


def _part_verdict(got: CanonResult, guard: str) -> str | None:
    """Why one part of a candidate split is not a value of its type, or None if
    it is. Both rules in one place, so the caller asks the same question of
    every row of the policy table.
    """
    if not got.ok:
        return got.reason
    if guard == SEPARATOR_ONLY and not any(c.isalnum() for c in (got.value or "")):
        return ("no letter or digit in it, so it is separator debris rather "
                "than one of the values this cell packs")
    return None


def _packed_by_pure_packers(splitter: "re.Pattern[str]", text: str) -> bool:
    """Was every part of this candidate split produced by a character measured
    never to be content — `PURE_PACKERS` — rather than one a split may only
    guess from?
    """
    for m in splitter.finditer(text):
        sep = m.group(0).strip()
        if not sep or any(c not in PURE_PACKERS for c in sep):
            return False
    return True


MULTIVALUE_PACKS: dict[str, tuple["re.Pattern[str]",
                                  "Callable[[str, str], str | None] | None"]] = {}


_PHONE_NOT_DIALLABLE = re.compile(r"[^\d\s().\-–—+]")
_NON_DIGIT = re.compile(r"\D")


def _complete_phone(part: str, previous: str) -> str | None:
    """«3-00-07» after «+380414230830»: the same exchange, the local digits
    replaced. None when the part is not a bare local number — letters in it,
    fewer than five digits, or as long as a national number already."""
    if _PHONE_NOT_DIALLABLE.search(part):
        return None
    local = _NON_DIGIT.sub("", part)
    for rule in PHONE_RULES:
        cc = "+" + rule["cc"]
        if previous.startswith(cc):
            nsn = previous[len(cc):]
            if 5 <= len(local) < len(nsn):
                return cc + nsn[:len(nsn) - len(local)] + local
    return None


MULTIVALUE_PACKS["phone"] = (re.compile(r"\s*[;,/|]\s*|\s*\n\s*"), _complete_phone)


def _canon_pack(text: str, type_name: str, fn: "Callable[[str], CanonResult]",
                splitter: "re.Pattern[str]",
                complete: "Callable[[str, str], str | None] | None") -> CanonResult | None:
    """A cell split on the characters this type's publisher packs with, and
    nothing else. None when there is no such pack, or when no part of it is
    a value — the ordinary path then says why about the whole cell."""
    parts = [p.strip() for p in splitter.split(text) if p and p.strip()]
    if len(parts) < 2:
        return None
    results = [fn(p) for p in parts]
    if complete is not None:
        last_ok: str | None = None
        for i, got in enumerate(results):
            if got.ok:
                last_ok = got.value
            elif last_ok:
                retry = complete(parts[i], last_ok)
                if retry is not None:
                    again = fn(retry)
                    if again.ok:
                        results[i] = again
    verdicts = [_part_verdict(r, EVERY_PART) for r in results]
    if all(v is not None for v in verdicts):
        return None
    good = [r.value or "" for r, v in zip(results, verdicts) if v is None]
    if all(v is None for v in verdicts):
        return _ok(MULTIVALUE_JOIN.join(good), f"{type_name}x{len(good)}")
    refused = tuple((p, v) for p, v in zip(parts, verdicts) if v is not None)
    return CanonResult(True, MULTIVALUE_JOIN.join(good), None,
                       f"{type_name}x{len(good)}of{len(parts)}", refused)


def _canon_multivalue(text: str, type_name: str, fn: "Callable[[str], CanonResult]"
                       ) -> CanonResult | None:
    """One cell, several values. Returns None when this is not that."""
    policy = MULTIVALUE_POLICY.get(type_name)
    if policy is None:
        return None
    pack = MULTIVALUE_PACKS.get(type_name)
    if pack is not None:
        packed = _canon_pack(text, type_name, fn, *pack)
        if packed is not None:
            return packed
    splitter, guard = policy
    pieces = splitter.split(text)
    if len(pieces) < 2:
        return None
    parts = [p for p in pieces if p and p.strip()]
    if not parts:
        return _no("nothing but the separator this type packs values with, so "
                   "there is no value here to canonicalize", f"{type_name}x0")
    if len(parts) == 1:
        got = fn(parts[0])
        return got if got.ok else None
    results = [fn(p) for p in parts]
    verdicts = [_part_verdict(r, guard) for r in results]
    if all(v is None for v in verdicts):
        return _ok(MULTIVALUE_JOIN.join(r.value or "" for r in results),
                   f"{type_name}x{len(results)}")
    if not _packed_by_pure_packers(splitter, text):
        return None
    if all(v is not None for v in verdicts):
        return None
    good = [r.value or "" for r, v in zip(results, verdicts) if v is None]
    refused = tuple((p, v) for p, v in zip(parts, verdicts) if v is not None)
    return CanonResult(True, MULTIVALUE_JOIN.join(good), None,
                       f"{type_name}x{len(good)}of{len(parts)}", refused)


def canonicalize(value: str, type_name: str,
                 evidence: ColumnEvidence = VALUE_ALONE) -> CanonResult:
    text = "" if value is None else str(value).strip()
    if not text:
        return _no("empty", "empty")
    fn = CANONICALIZERS.get(type_name)
    if fn is _canon_date:
        fn = lambda v, _e=evidence: _canon_date(v, _e)
    if fn is None and type_name in FTM_SAFE_TYPES:
        fn = lambda v, _t=type_name: _canon_ftm(v, _t)
    if fn is None:
        fn = _canon_free
    many = _canon_multivalue(text, type_name, fn)
    if many is not None:
        return many
    got = fn(text)
    if (got.ok and type_name in MULTIVALUE_POLICY
            and MULTIVALUE_JOIN in (got.value or "")):
        return _no("packed with a separator this type could not take apart, "
                   "so it is neither one value nor a split every part of "
                   "which canonicalizes", got.canonicalizer)
    return got


def format_accepts(canonical: str, type_name: str, fmt: str) -> bool:
    """Whether followthemoney's own check for `fmt` would take this value."""
    from followthemoney import registry

    ftm_type = registry.get(type_name)
    if ftm_type is None:
        return True
    parts = (canonical.split(MULTIVALUE_JOIN)
             if type_name in MULTIVALUE_TYPES else [canonical])
    return all(ftm_type.clean_text(p, format=fmt) is not None
               for p in parts if p)


def acceptance(values: list[str], type_name: str,
               fmt: str | None = None) -> tuple[float, list[str]]:
    """Share of a column's real values that canonicalize, and up to five that
    did not. The only check in the pipeline that can contradict a plausible
    header, so both edge cases matter.
    """
    filled = [str(v).strip() for v in values if v is not None and str(v).strip()]
    vals = [s for s in filled if not is_sentinel(s)]
    if not vals:
        if filled:
            return 0.0, filled[:5]
        return 1.0, []
    if fmt is None and type_name in FREE_TYPES and type_name not in CANONICALIZERS:
        return 1.0, []
    evidence = column_evidence(vals)
    ok = 0
    rejected: list[str] = []
    for v in vals:
        got = canonicalize(v, type_name, evidence)
        if got.ok and (fmt is None
                       or format_accepts(got.value or "", type_name, fmt)):
            ok += 1
        elif len(rejected) < 5:
            rejected.append(v)
    return ok / len(vals), rejected
