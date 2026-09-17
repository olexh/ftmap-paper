"""Candidate retrieval. It decides nothing; it bounds what the model may say.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

from rapidfuzz import fuzz

from ftmap.config import Config
from ftmap.profile.columns import ColumnProfile
from ftmap.vocab.catalogue import Catalogue, PropInfo
from ftmap.vocab.labels import load_label_entries

TYPE_FOR_DETECTOR: dict[str, set[str]] = {
    "date": {"date"},
    "phone_ua": {"phone"},
    "phone_ru": {"phone"},
    "email": {"email"},
    "url": {"url"},
    "edrpou": {"identifier"},
    "rnokpp": {"identifier"},
    "inn_ru": {"identifier"},
    "snils_ru": {"identifier"},
    "gender_code": {"gender"},
    "country": {"country"},
    "numeric": {"number"},
    "proper_name": {"name"},
    "legal_name": {"name"},
    "org_name": {"name"},
}

DETECTOR_FLOOR = 0.5


def neighbours(cat: Catalogue, subject: str) -> list[str]:
    """The schemata a row about `subject` plausibly also describes."""
    out = {subject}
    for edge in cat.edges():
        if cat.is_descendant(subject, edge.source_range):
            out.update((edge.schema, edge.target_range))
        if cat.is_descendant(subject, edge.target_range):
            out.update((edge.schema, edge.source_range))
    out.update(cat.entity_property_ranges(subject))
    return sorted(out)


class Lexicon(dict):
    """The spellings, and the per-qname work derived from them."""

    __slots__ = ("resolved", "folded")

    def __init__(self, entries) -> None:
        super().__init__(entries)
        self.resolved: dict[str, list[str]] = {}
        self.folded: dict[str, list[str]] = {}


@lru_cache(maxsize=8)
def load_lexicon(path: str | None = None) -> Lexicon:
    """Observed header spellings, qname to spellings."""
    if path:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return Lexicon({k: v for k, v in raw.items() if not k.startswith("_")})

    text = resources.files("ftmap.vocab").joinpath("lexicon_uk.json").read_text("utf-8")
    out = {k: list(v) for k, v in json.loads(text).items() if not k.startswith("_")}

    for qname, entry in load_label_entries().items():
        for spelling in entry.get("spellings") or ():
            if spelling not in out.setdefault(qname, []):
                out[qname].append(spelling)
    return Lexicon(out)


def resolved_spellings(lexicon: dict[str, list[str]], cat: Catalogue,
                       qname: str) -> list[str]:
    """The spellings that apply to `qname`: its own, and those written under
    any schema it inherits the property from.
    """
    index = getattr(lexicon, "resolved", None)
    if index is not None and qname in index:
        return index[qname]
    schema, _, name = qname.partition(":")
    got: list[str] = []
    for key, spellings in lexicon.items():
        key_schema, _, key_name = key.partition(":")
        if key_name == name and cat.is_descendant(schema, key_schema):
            got.extend(sp for sp in spellings if sp not in got)
    if index is not None:
        index[qname] = got
    return got


def fold(text: str) -> str:
    """Case-, apostrophe- and underscore-insensitive. Ukrainian headers write
    the apostrophe five ways — `'`, `’`, `` ` ``, the prime `′`, the modifier
    letter `ʼ` — and the lexicon cannot carry every combination: measured on
    the ship register, «Прізвище Ім′я По-батькові» scored 0.64 against the
    lexicon's «прізвище ім'я по батькові» for the prime alone.
    """
    lowered = (text.lower().replace("’", "'").replace("`", "'")
               .replace("′", "'").replace("ʼ", "'").replace("_", " "))
    return " ".join(lowered.split())


_fold = fold


_CAMEL = re.compile(r"([a-zа-яіїєґ0-9])([A-ZА-ЯІЇЄҐ][a-zа-яіїєґ])")


def fold_header(text: str) -> str:
    """`_fold`, and an export's camelCase is a word separator too — on the
    HEADER side only.
    """
    return _fold(_CAMEL.sub(r"\1 \2", text))


_fold_header = fold_header


def _haystack(lexicon: dict[str, list[str]], cat: Catalogue,
              info: PropInfo) -> list[str]:
    """The folded strings a header is matched against for one property."""
    cache = getattr(lexicon, "folded", None)
    if cache is not None and info.qname in cache:
        return cache[info.qname]
    out = [_fold(info.name), _fold(info.label)]
    out += [_fold(s) for s in resolved_spellings(lexicon, cat, info.qname)]
    if cache is not None:
        cache[info.qname] = out
    return out


def _scope_props(cat: Catalogue, subject: str | None,
                 schemas: list[str] | None = None) -> list[PropInfo]:
    """The properties retrieval may draw from."""
    if schemas is not None:
        out: list[PropInfo] = []
        for schema in dict.fromkeys(schemas):
            out.extend(cat.properties_of(schema).values())
        return out
    if subject is None:
        return list(cat.all_props())
    out = []
    for schema in neighbours(cat, subject):
        out.extend(cat.properties_of(schema).values())
    return out


def _scope(cat: Catalogue, subject: str | None,
           schemas: list[str] | None = None) -> list[str]:
    """`_scope_props` as bare qnames, which is how the scope reads."""
    return [p.qname for p in _scope_props(cat, subject, schemas)]


def shortlist_scored(
    profile: ColumnProfile,
    cat: Catalogue,
    cfg: Config,
    subject: str | None = None,
    schemas: list[str] | None = None,
) -> list[tuple[str, float]]:
    lexicon = load_lexicon(cfg.lexicon_path or None)
    scope = _scope_props(cat, subject, schemas)
    signal = column_signal(profile)

    scored: list[tuple[float, str]] = []
    for info in scope:
        if info.type_name == "entity":
            continue
        _lexical, total = property_score(info, signal, lexicon, cat, cfg)
        if total > 0:
            scored.append((total, info.qname))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(q, s) for s, q in scored[: cfg.shortlist_size]]


def column_signal(profile: ColumnProfile) -> tuple[list[str], set[str]]:
    """What a column says about itself before any property is looked at: its
    folded header and label, and the FtM value types its detectors vouch for.
    Computed once per column and handed to `property_score` per property."""
    needles = [fold_header(t) for t in (profile.header, profile.label) if t]
    wanted_types: set[str] = set()
    for det, rate in profile.detectors.items():
        if rate >= DETECTOR_FLOOR:
            wanted_types |= TYPE_FOR_DETECTOR.get(det, set())
    return needles, wanted_types


def property_score(info: PropInfo, signal: tuple[list[str], set[str]],
                   lexicon: dict[str, list[str]], cat: Catalogue,
                   cfg: Config) -> tuple[float, float]:
    """Retrieval's score for one property against one column, as `(lexical,
    total)`.
    """
    needles, wanted_types = signal
    haystack = _haystack(lexicon, cat, info)
    base = 0.0
    for n in needles:
        for h in haystack:
            base = max(base, fuzz.token_set_ratio(n, h) / 100.0)
    if base < cfg.lexical_floor:
        base = 0.0
    bonus = 0.35 if info.type_name in wanted_types else 0.0
    penalty = 0.15 if info.type_name in ("string", "text") and not needles else 0.0
    return base, base + bonus - penalty


def shortlist(
    profile: ColumnProfile,
    cat: Catalogue,
    cfg: Config,
    subject: str | None = None,
    schemas: list[str] | None = None,
) -> list[str]:
    """Candidates, best first."""
    scoped = shortlist_scored(profile, cat, cfg, subject, schemas)
    if subject is None and schemas is None:
        return [q for q, _ in scoped]

    free = shortlist_scored(profile, cat, cfg, None)
    out: list[str] = []
    seen: set[str] = set()
    names: set[str] = set()
    for rank in range(max(len(scoped), len(free))):
        for source in (scoped, free):
            if rank >= len(source):
                continue
            qname = source[rank][0]
            name = qname.split(":", 1)[1]
            if qname in seen or name in names:
                continue
            seen.add(qname)
            names.add(name)
            out.append(qname)
        if len(out) >= cfg.shortlist_size:
            break
    return out[: cfg.shortlist_size]
