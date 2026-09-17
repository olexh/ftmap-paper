"""The binding-stage policies `[plan] binding_mode` switches between."""

from __future__ import annotations

from ftmap.plan.response_schema import NOT_OFFERED, UNMAPPED, valid_pairs
from ftmap.vocab.catalogue import Catalogue
from ftmap.vocab.shortlist import (DETECTOR_FLOOR, TYPE_FOR_DETECTOR,
                                   column_signal, load_lexicon, property_score)


def catalogue_pairs(declared: dict[str, str], cat: Catalogue) -> list[str]:
    """Every `"key|Schema:prop"` a column of this round could be bound to."""
    pairs: list[str] = []
    for key, schema in declared.items():
        for info in cat.properties_of(schema).values():
            if cat.bindable(schema, info.qname):
                pairs.append(f"{key}|{info.qname}")
    return pairs


def candidate_pairs(mode: str, shortlist: list[str], declared: dict[str, str],
                    cat: Catalogue) -> list[str]:
    """The pairs one column is offered under `mode`. See the module docstring."""
    if mode == "static":
        return catalogue_pairs(declared, cat)
    if mode == "property":
        return property_offers(valid_pairs(shortlist, declared, cat))
    return valid_pairs(shortlist, declared, cat)


def property_offers(pairs: list[str]) -> list[str]:
    """The `property` arm's offer: the properties of the pairs, without their
    entities, each once by its local name, in first-seen order. A property two
    entities could carry is offered once under the first spelling —
    `Person:name`, not `Person:name` and `PublicBody:name`, since the second
    spelling would name the participant the arm withholds — and which entity
    gets it is `assign_property`'s rule, not the model's choice.
    """
    out: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        qname = pair.rpartition("|")[2]
        local = qname.rpartition(":")[2]
        if qname and local not in seen:
            seen.add(local)
            out.append(qname)
    return out


def assign_property(raw: str | None, declared: dict[str, str],
                    cat: Catalogue) -> str | None:
    """The `property` arm's participant rule: a bare property answered by the
    model becomes the pair on the FIRST declared entity or edge, in declaration
    order, whose schema can carry it — the same re-spelling `valid_pairs`
    applies, the same `bindable` test. `None` when nothing can carry it, which
    `split_binding` reads as `unmapped`. A string that already names an entity
    (`"e|Schema:prop"`) passes through.
    """
    if not raw or raw == UNMAPPED:
        return None
    if "|" in raw:
        return raw
    local = raw.split(":", 1)[1] if ":" in raw else raw
    for entity_key, schema in declared.items():
        own = cat.properties_of(schema).get(local)
        offer = own.qname if own is not None else raw
        if cat.bindable(schema, offer):
            return f"{entity_key}|{offer}"
    return None


def _same_property(a: str, b: str, cat: Catalogue) -> bool:
    """One property under two spellings: the same local name on schemata that
    stand on one line of descent. The test `etalon/score._same_property`
    scores with, restated here so the tie rule and the scorer agree on what
    "distinct properties" means without the planner importing the scorer."""
    if a == b:
        return True
    (sa, _, na), (sb, _, nb) = a.partition(":"), b.partition(":")
    return na == nb and cat.related_to(sa, sb)


HEURISTIC_MODES = ("heuristic", "heuristic-first")


def heuristic_choice(profile, pairs: list[str], cat: Catalogue, cfg,
                     lexicon=None, first_on_tie: bool = False) -> tuple[str | None, str]:
    """The heuristic arm's answer for one column: `(pair, why)`, `pair` None
    for `unmapped`.
    """
    lexicon = lexicon if lexicon is not None else load_lexicon(cfg.lexicon_path or None)
    signal = column_signal(profile)
    scored: list[tuple[float, float, str]] = []
    for pair in pairs:
        qname = pair.partition("|")[2]
        info = cat.prop(qname)
        if info is None:
            continue
        lexical, total = property_score(info, signal, lexicon, cat, cfg)
        if total > 0:
            scored.append((round(total, 6), round(lexical, 6), pair))
    if not scored:
        return None, "heuristic: no offered candidate scores above zero"
    top = max(t for t, _, _ in scored)
    winners = [(lexical, pair) for t, lexical, pair in scored if t == top]
    distinct: list[str] = []
    for _, pair in winners:
        qname = pair.partition("|")[2]
        if not any(_same_property(qname, seen, cat) for seen in distinct):
            distinct.append(qname)
    if len(distinct) > 1 and not first_on_tie:
        return None, (f"heuristic: {len(distinct)} distinct properties tie at "
                      f"{top:.2f} ({', '.join(distinct)}); the header does not "
                      "choose between them")
    lexical, pair = winners[0]
    note = ""
    if len(distinct) > 1:
        note = (f"; first of {len(distinct)} distinct properties tying "
                f"({', '.join(distinct)})")
    elif len(winners) > 1:
        note = f"; first of {len(winners)} targets offering it"
    return pair, (f"heuristic: score {top:.2f} (lexical {lexical:.2f}, "
                  f"bonus {top - lexical:+.2f}){note}")


STRUCTURE_EXCLUDED = frozenset(NOT_OFFERED) | frozenset(
    {"Audio", "Video", "Image", "Folder", "Package"})

ROW_KEY = "row"


def subject_vote(profiles, cat: Catalogue, cfg, lexicon=None) -> dict[str, float]:
    """`vote(S)` for every eligible concrete schema, the plan's §2.1 formula.
    """
    lexicon = lexicon if lexicon is not None else load_lexicon(cfg.lexicon_path or None)
    signals = [(p, column_signal(p)) for p in profiles if p.filled > 0]
    memo: dict[tuple[str, str], float] = {}
    votes: dict[str, float] = {}
    for schema in cat.concrete_schemata():
        if schema in STRUCTURE_EXCLUDED:
            continue
        total = 0.0
        for profile, signal in signals:
            best = 0.0
            for info in cat.properties_of(schema).values():
                if info.type_name == "entity":
                    continue
                key = (profile.id, info.qname)
                score = memo.get(key)
                if score is None:
                    score = memo[key] = property_score(info, signal, lexicon, cat, cfg)[1]
                if score > best:
                    best = score
            total += best
        votes[schema] = round(total, 6)
    return votes


def identifier_keys(profiles, limit: int = 4) -> list[str]:
    """The live columns an identifier-type detector vouches for, in column
    order, at most `limit` — the grammar's own bound on `keys`. The detectors
    are the ones `TYPE_FOR_DETECTOR` maps to `identifier`; there is no VIN, IMO
    or registration-number detector, so such a column is not a key under this
    rule, and `keys.propose_key` may still propose one from the bindings later,
    as it does for the model's plans.
    """
    out: list[str] = []
    for p in profiles:
        if p.filled <= 0:
            continue
        if any(rate >= DETECTOR_FLOOR and TYPE_FOR_DETECTOR.get(det) == {"identifier"}
               for det, rate in p.detectors.items()):
            out.append(p.id)
            if len(out) == limit:
                break
    return out


def heuristic_structure(profiles, cat: Catalogue, cfg) -> tuple[dict, dict]:
    """The structure call's answer with no model: `({"subject", "entities"},
    note)`.
    """
    votes = subject_vote(profiles, cat, cfg)
    order = [s for s in cat.concrete_schemata() if s in votes]
    top = max(votes.values(), default=0.0)
    subject = next(s for s in order if votes[s] == top)
    keys = identifier_keys(profiles)
    answer = {"subject": subject,
              "entities": [{"key": ROW_KEY, "schema": subject, "keys": keys}]}
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    note = {"subject": subject, "vote": top, "no_signal": top <= 0,
            "tied": [s for s, v in votes.items() if v == top],
            "runners_up": ranked[1:4], "keys": keys}
    return answer, note
