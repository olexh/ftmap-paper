"""Relations a register states in its headers, read there instead of asked."""

from __future__ import annotations

import json
import re
from functools import lru_cache, partial
from importlib import resources

from ftmap.plan.response_schema import UNMAPPED
from ftmap.plan.groups import prefix_groups, selection, share_prefix_group
from ftmap.plan.keys import key_identification, one_to_one
from ftmap.vocab.catalogue import Catalogue
from ftmap.vocab.shortlist import fold as _fold

_GENERIC = {"name", "назва", "найменування", "код", "code", "id", "номер",
            "№", "number", "date", "дата", "type", "тип", "юр", "фіз", "legal",
            "full", "повна", "повне", "start", "end", "start_date", "end_date"}


@lru_cache(maxsize=1)
def load_roles() -> dict[str, dict[str, list[str]]]:
    text = resources.files("ftmap.vocab").joinpath("roles_uk.json").read_text("utf-8")
    raw = json.loads(text)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def load_levels() -> list[str]:
    """The level words, coarsest first — `_levels` in the role lexicon."""
    text = resources.files("ftmap.vocab").joinpath("roles_uk.json").read_text("utf-8")
    return list(json.loads(text).get("_levels") or ())


def _tokens(text: str) -> set[str]:
    """The tokens of a header that could tie it to another header: three
    letters or more, and not one every register writes («юр», «код»). Not
    `groups._header_tokens`, which reads an export's field name into its
    words in order."""
    return {t for t in re.split(r"[^\w']+", _fold(text))
            if len(t) >= 3 and t not in _GENERIC}


def key_tokens(by_id: dict, entity: dict) -> set[str]:
    """The tokens of an entity's key headers and label rows — what says a role
    column belongs to this entity and not the one beside it (`sti_chief_name`
    names the chief of `sti_name`). `by_id` is `{column id: profile}`."""
    out: set[str] = set()
    for k in entity.get("keys") or ():
        p = by_id.get(k)
        if p is not None:
            out |= _tokens(p.header or "") | _tokens(p.label or "")
    return out


Role = tuple[str, str, str]


def roles_in(text: str, roles: dict) -> list[Role]:
    """Every role word in a text, in text order, one per relation. Where two
    spellings overlap the longer wins, so «суб'єкта управління» is read
    before a shorter word it happens to contain."""
    folded = _fold(text)
    order = {relation: i for i, relation in enumerate(roles)}
    hits: list[tuple[int, int, int, str, str, str]] = []
    for relation, sides in roles.items():
        for side, spellings in sides.items():
            for spelling in spellings:
                sp = _fold(spelling)
                at = folded.find(sp)
                if at >= 0:
                    hits.append((at, -len(sp), order[relation], relation, side, spelling))
    out: list[Role] = []
    span = (-1, -1)
    seen: set[str] = set()
    for at, neg_len, _rank, relation, side, spelling in sorted(hits):
        end = at - neg_len
        if (at < span[1] and (at, end) != span) or relation in seen:
            continue
        span = (at, end)
        seen.add(relation)
        out.append((relation, side, spelling))
    return out


def roles_for(header: str | None, label: str | None,
              roles: dict) -> list[Role]:
    """Every (relation, side, spelling) a header or its label row names,
    longest spelling first and the lexicon's order within one spelling —
    the order `derive_relations` tries them in. One entry per relation."""
    found: list[tuple[int, int, Role]] = []
    seen: set[str] = set()
    for text in (header, label):
        for n, role in enumerate(roles_in(text or "", roles)):
            if role[0] in seen:
                continue
            seen.add(role[0])
            found.append((-len(_fold(role[2])), n, role))
    return [role for _len, _n, role in sorted(found, key=lambda t: t[:2])]


def role_in(header: str | None, label: str | None,
            roles: dict) -> Role | None:
    """(relation, side, spelling) named by a header or its label row, or
    None. The longest spelling wins across both texts; two relations on one
    word are settled by the lexicon's order (see `roles_in`)."""
    best: tuple[int, Role] | None = None
    for text in (header, label):
        for role in roles_in(text or "", roles):
            cand = (len(_fold(role[2])), role)
            if best is None or cand[0] > best[0]:
                best = cand
    return None if best is None else best[1]


_GROUP_ROLES: dict[tuple, tuple[dict, dict[str, tuple[Role, str]]]] = {}


def group_roles(profiles, roles: dict) -> dict[str, tuple[Role, str]]:
    """The role each column takes from its group header, keyed by column id,
    with the text of where it was read — for columns whose own header and
    label name none. See the module docstring for the walk."""
    signature = tuple((p.id, p.index, getattr(p, "group", None), p.header, p.label)
                      for p in profiles)
    cached = _GROUP_ROLES.get(signature)
    if cached is not None and cached[0] is roles:
        return cached[1]
    out = _group_roles(profiles, roles)
    if len(_GROUP_ROLES) > 16:
        _GROUP_ROLES.clear()
    _GROUP_ROLES[signature] = (roles, out)
    return out


def _group_roles(profiles, roles: dict) -> dict[str, tuple[Role, str]]:
    out: dict[str, tuple[Role, str]] = {}
    ordered = sorted(profiles, key=lambda p: p.index)
    n = 0
    while n < len(ordered):
        group = getattr(ordered[n], "group", None)
        if not group:
            n += 1
            continue
        band = [ordered[n]]
        while (n + len(band) < len(ordered)
               and getattr(ordered[n + len(band)], "group", None) == group
               and ordered[n + len(band)].index == band[-1].index + 1):
            band.append(ordered[n + len(band)])
        n += len(band)
        listed = roles_in(group, roles)
        if not listed:
            continue
        where = (f"«{group.strip()}» over {band[0].id}–{band[-1].id}"
                 if len(band) > 1 else f"«{group.strip()}» over {band[0].id}")
        current = 0
        for p in band:
            own = role_in(p.header, p.label, roles)
            if own is not None:
                later = [i for i, r in enumerate(listed) if r[0] == own[0]]
                if later and later[0] > current:
                    current = later[0]
                continue
            out[p.id] = (listed[current], where)
    return out


_PERSON_ROLES = frozenset({"Directorship", "CourtCaseParty"})
_POST_WORDS = ("post", "position", "посада", "роль", "role")
_NAME_FLOOR = 0.5
_PERSON_FLOOR = 0.8


def role_keyed(entity: dict, profiles, roles: dict | None = None) -> bool:
    """Whether a key column of this entity carries a role word — in its own
    header or label, or in the group header over it. Such an entity is a party
    the row NAMES (an owner, a director, a provider), never the entity the row
    IS: «Власник» over a column says whose the row's asset is, not what the row
    is about. Read by `validate`'s row-entity gate and by the counterpart
    choice here.
    """
    roles = load_roles() if roles is None else roles
    by_id = {p.id: p for p in profiles}
    from_group = group_roles(profiles, roles)
    for k in entity.get("keys") or ():
        p = by_id.get(k)
        if p is None:
            continue
        if role_in(p.header, p.label, roles) is not None or k in from_group:
            return True
    return False


def declare_parties(entities: list[dict], bindings: list[dict], profiles,
                    cat: Catalogue, roles: dict | None = None, values=None,
                    skip: set[str] | None = None,
                    ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """New party entities, their name bindings, and (column, key, reason)
    notes, for every column the plan neither keys nor binds whose header
    names a role and whose values are names. See the module docstring."""
    roles = load_roles() if roles is None else roles
    keyed: dict[str, list[dict]] = {}
    for e in entities:
        for k in e.get("keys") or ():
            keyed.setdefault(k, []).append(e)
    bound = {b["column"] for b in bindings
             if b.get("prop") and b["prop"] != UNMAPPED}
    taken = {e["key"] for e in entities}
    from_group = group_roles(profiles, roles)
    new_entities: list[dict] = []
    new_bindings: list[dict] = []
    notes: list[tuple[str, str, str]] = []
    level_headers: dict[str, str] = {}
    for e in entities:
        for k in e.get("keys") or ():
            pk = next((q for q in profiles if q.id == k), None)
            if pk is not None:
                r = role_in(pk.header, pk.label, roles)
                if r is not None and r[0] == "Organization:parent" and r[1] == "source":
                    level_headers.setdefault(_fold(pk.header or ""), k)
    skip = set() if skip is None else skip
    for p in sorted(profiles, key=lambda p: p.index):
        if p.id in bound or p.filled == 0 or p.id in skip:
            continue
        if any(w in _fold(p.header or "") for w in _POST_WORDS):
            continue
        found = role_in(p.header, p.label, roles)
        where = f"«{(p.header or p.label or '').strip()}»"
        if found is None and p.id in from_group:
            found, band = from_group[p.id]
            where = f"{band}, the group header of {where}"
        if found is None:
            continue
        relation, _side, spelling = found
        level = next((r for r in roles_for(p.header, p.label, roles)
                      if r[0] == "Organization:parent" and r[1] == "source"), None)
        if level is not None and values is not None and p.id not in keyed:
            relation, _side, spelling = level
        if (relation == "Organization:parent" and _side == "source"
                and values is not None and p.id not in keyed):
            folded = _fold(p.header or "")
            if folded in level_headers:
                notes.append((p.id, "", f"declined: {where} is a level word "
                              f"over a header already written at "
                              f"{level_headers[folded]}; a header written twice "
                              f"is a working copy"))
                continue
            level_headers[folded] = p.id
            if text_share(values(p.id)) < _NAME_TEXT_FLOOR:
                continue
            key = p.id
            n = 2
            while key in taken:
                key = f"{p.id}_{n}"
                n += 1
            taken.add(key)
            new_entities.append({"key": key, "schema": "Organization", "keys": [p.id]})
            new_bindings.append({"column": p.id, "prop": "Organization:name",
                                 "entity": key,
                                 "why": f"the header says «{spelling}», a level of "
                                        f"organisation, and the values are its "
                                        f"designations"})
            notes.append((p.id, key,
                          f"declared from {where}: «{spelling}» names a level of "
                          f"organisation, the values are words, and no declared "
                          f"entity was keyed on or bound to the column; an "
                          f"Organization keyed on {p.id} and named by it"))
            continue
        proper = p.detectors.get("proper_name", 0.0)
        legal = max(p.detectors.get("legal_name", 0.0),
                    p.detectors.get("org_name", 0.0))
        if max(proper, legal) < _NAME_FLOOR:
            continue
        looks = (f"{proper:.0%} of its values read as names" if proper >= legal
                 else f"{legal:.0%} of its values carry a legal form")
        holders = [e for e in keyed.get(p.id, ())
                   if cat.is_party(e["schema"])]
        if p.id in keyed:
            if len(holders) == 1:
                e = holders[0]
                new_bindings.append({"column": p.id, "prop": f"{e['schema']}:name",
                                     "entity": e["key"],
                                     "why": f"the header says «{spelling}», the "
                                            f"values are names, and this party "
                                            f"is keyed on the column"})
                notes.append((p.id, e["key"],
                              f"named from {where}: «{spelling}» names the party "
                              f"of a {relation}, {looks}, and {e['key']} "
                              f"({e['schema']}) is keyed on the column and bound "
                              f"to nothing; its name is this column"))
            continue
        person = (relation in _PERSON_ROLES and proper >= _PERSON_FLOOR
                  and legal < 1 - _PERSON_FLOOR)
        schema = "Person" if person else "LegalEntity"
        key = p.id
        n = 2
        while key in taken:
            key = f"{p.id}_{n}"
            n += 1
        taken.add(key)
        new_entities.append({"key": key, "schema": schema, "keys": [p.id]})
        new_bindings.append({"column": p.id, "prop": f"{schema}:name",
                             "entity": key,
                             "why": f"the header says «{spelling}» and the "
                                    f"values are names"})
        notes.append((p.id, key,
                      f"declared from {where}: «{spelling}» names the party "
                      f"of a {relation}, {looks}, and no declared entity was "
                      f"keyed on or bound to the column; a {schema} keyed on "
                      f"{p.id} and named by it"))
    return new_entities, new_bindings, notes


_MACHINERY_WORDS = ("впр",)
_MACHINERY_PHRASES = ("не трогать", "не протягивать", "ячейки для счета",
                      "ячейки для счёта", "не чіпати")


def machinery_header(header: str | None, label: str | None = None) -> str | None:
    """The machinery word a header carries, or None. A single word must be a
    whole token («впр л/н», never «впровадження»); a phrase may sit inside."""
    text = _fold(f"{header or ''} {label or ''}")
    tokens = set(re.split(r"[^\w']+", text))
    for w in _MACHINERY_WORDS:
        if w in tokens:
            return w
    for ph in _MACHINERY_PHRASES:
        if ph in text:
            return ph
    return None


_INTERCHANGEABLE = frozenset({"Membership", "Employment"})


_NAMED_BY_KEY = {"Position": "name", "Vehicle": "model"}
_NAME_TEXT_FLOOR = 0.9


_WORD = re.compile(r"[^\W\d_]{2,}")


def _is_words(value: str) -> bool:
    """A value that is words rather than a code: it holds a run of two or
    more letters and its letters outnumber its digits. «1 мсв» and «ТОВ
    Альфа» are; a VIN, a service number «АБ-123456» and a plate are not."""
    letters = sum(1 for ch in value if ch.isalpha())
    digits = sum(1 for ch in value if ch.isdigit())
    return bool(_WORD.search(value)) and letters > digits


def text_share(values) -> float:
    """The share of a column's filled values that are words rather than
    codes — the reading a rule may make from the values without keeping
    one. See `_is_words`."""
    filled = [str(v) for v in values if v is not None and str(v).strip()]
    if not filled:
        return 0.0
    return sum(1 for v in filled if _is_words(v)) / len(filled)


def name_from_key(entities: list[dict], bindings: list[dict], profiles, cat,
                  values, floor: float = _NAME_TEXT_FLOOR,
                  declined: set[str] | None = None,
                  rejected: set[str] | None = None,
                  relations: set[str] | None = None,
                  ) -> list[tuple[dict, str]]:
    """The name of a thing keyed on one column of words and bound to nothing:
    that column. (candidate binding, reason) pairs, to be checked like any
    other binding.
    """
    by_id = {p.id: p for p in profiles}
    declined = set() if declined is None else declined
    rejected = set() if rejected is None else rejected
    relations = set() if relations is None else relations
    live = [b for b in bindings if b.get("prop") and b["prop"] != UNMAPPED]
    bound_entities = {b.get("entity") for b in live}
    on_column: dict[str, list[dict]] = {}
    for b in live:
        on_column.setdefault(b["column"], []).append(b)
    out: list[tuple[dict, str]] = []
    for e in entities:
        keys = list(e.get("keys") or ())
        schema = e["schema"]
        if not keys:
            continue
        if not (cat.is_party(schema) or schema in _NAMED_BY_KEY):
            continue
        prop = _NAMED_BY_KEY.get(schema, "name")
        if not cat.bindable(schema, f"{schema}:{prop}"):
            continue
        if len(keys) == 1:
            col = keys[0]
        else:
            wordy = [k for k in keys if k in by_id and text_share(values(k)) >= floor]
            if len(wordy) != 1:
                continue
            col = wordy[0]
        p = by_id.get(col)
        if p is None or p.filled == 0:
            continue
        named_by_key = schema in _NAMED_BY_KEY
        taken = on_column.get(col, [])
        replaces: list[dict] = []
        on_relation = bool(taken) and cat.is_party(schema) and all(
            b.get("entity") in relations for b in taken)
        if e["key"] in bound_entities and not on_relation:
            continue
        if taken:
            if on_relation:
                replaces = list(taken)
            elif not (schema in _NAMED_BY_KEY
                      and all(b["prop"].endswith(":position") for b in taken)):
                continue
        elif col not in declined or col in rejected:
            continue
        if cat.is_party(schema) and any(
                w in _fold(p.header or "") for w in _POST_WORDS):
            continue
        if cat.is_descendant(schema, "Person") \
                and p.detectors.get("proper_name", 0.0) < _NAME_FLOOR:
            continue
        share = text_share(values(col))
        head = f"«{(p.header or '').strip()}» ({col})" if p.header else f"({col})"
        if share < floor and not named_by_key:
            continue
        cand = {"column": col, "prop": f"{schema}:{prop}", "entity": e["key"],
                "why": f"{head} identifies this {schema}; the column is its {prop}"}
        if replaces:
            cand["replaces"] = replaces
            what = ", ".join(f"{b['prop']} on {b['entity']}" for b in replaces)
            out.append((cand, f"named from its key: {head} is the one column "
                              f"identifying {e['key']} ({schema}), its values are "
                              f"names, and it was bound to a relation ({what}); a "
                              f"party's name belongs to the party, never to the "
                              f"link, so that binding gives way"))
            continue
        out.append((cand,
                    f"named from its key: {head} is the one column identifying "
                    f"{e['key']} ({schema}), {share:.0%} of its values are words "
                    f"rather than codes, and nothing was bound to the {schema}"
                    + ("; the person's position string on the same column is "
                       "the post's name said of the post"
                       if taken else "")))
    return out


def declare_places(entities: list[dict], bindings: list[dict], profiles, cat,
                   lexicon, values, declined: set[str] | None = None,
                   skip: set[str] | None = None,
                   ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """An Address from a column whose header spells `Address:city` exactly.
    (new entities, bindings, (column, key, note)).
    """
    from ftmap.vocab.shortlist import _fold_header, resolved_spellings
    declined = set() if declined is None else declined
    info = cat.prop("Address:city")
    if info is None:
        return [], [], []
    forms = {_fold_header(x) for x in (info.name, info.label,
                                       *resolved_spellings(lexicon, cat, "Address:city"))}
    keyed = {k for e in entities for k in (e.get("keys") or ())}
    bound = {b["column"] for b in bindings if b.get("prop") and b["prop"] != UNMAPPED}
    taken = {e["key"] for e in entities}
    skip = set() if skip is None else skip
    out_e: list[dict] = []
    out_b: list[dict] = []
    notes: list[tuple[str, str, str]] = []
    for p in sorted(profiles, key=lambda p: p.index):
        if p.id in keyed or p.filled == 0 or p.id in skip:
            continue
        if p.id in bound and p.id not in declined:
            continue
        text = (p.header or "").strip()
        if not text or _fold_header(text) not in forms:
            continue
        if text_share(values(p.id)) < _NAME_TEXT_FLOOR:
            continue
        key = f"place_{p.id}"
        while key in taken:
            key += "_"
        taken.add(key)
        out_e.append({"key": key, "schema": "Address", "keys": [p.id]})
        out_b.append({"column": p.id, "prop": "Address:city", "entity": key,
                      "why": f"the header «{text}» spells Address:city exactly"})
        notes.append((p.id, key,
                      f"declared by the place rule: «{text}» ({p.id}) spells "
                      f"Address:city exactly, the values are words, and nothing "
                      f"keyed or bound the column; an Address keyed on {p.id} "
                      f"with the column as its city"))
    return out_e, out_b, notes


def _level_keyed(entity: dict, by_id: dict, roles: dict) -> bool:
    """Whether any key column of the entity carries a level word."""
    levels = {"Organization:parent": roles.get("Organization:parent", {})}
    for k in entity.get("keys") or ():
        p = by_id.get(k)
        if p is not None and role_in(p.header, p.label, levels) is not None:
            return True
    return False


def derive_relations(entities: list[dict], edges: list[dict],
                     attachments: list[dict], profiles, subject: str | None,
                     cat: Catalogue, roles: dict | None = None,
                     bindings: list[dict] | None = None,
                     ) -> tuple[list[dict], list[dict], list[tuple[str, str]]]:
    """New edges, new attachments, and (key, reason) notes — one note per
    derivation and one per role word that found no single counterpart.
    """
    roles = load_roles() if roles is None else roles
    by_id = {p.id: p for p in profiles}
    from_group = group_roles(profiles, roles)
    edge_info = cat.edge_index()
    bound_to: dict[str, list[str]] = {}
    for b in bindings or ():
        if b.get("prop") and b.get("entity"):
            cols = bound_to.setdefault(b["entity"], [])
            if b["column"] not in cols:
                cols.append(b["column"])

    def row_entity(e: dict) -> bool:
        return (bool(subject) and cat.related_to(e["schema"], subject)
                and not role_keyed(e, profiles, roles))

    key_distinct = partial(key_identification, profiles=profiles)

    def the_row(cands: list[dict]) -> list[dict]:
        rows = [c for c in cands if row_entity(c)]
        if len(rows) <= 1:
            return rows
        ranked = sorted(rows, key=key_distinct, reverse=True)
        if key_distinct(ranked[0]) > key_distinct(ranked[1]):
            return ranked[:1]
        return rows

    groups = prefix_groups(profiles)
    same_group = partial(share_prefix_group, groups,
                         role_keyed=lambda e: role_keyed(e, profiles, roles))

    def columns_of(e: dict) -> list[str]:
        cols = list(e.get("keys") or ())
        return cols + [c for c in bound_to.get(e["key"], ()) if c not in cols]

    def choose(others: list[dict], header_tokens: set[str],
               in_range, party: dict | None = None,
               range_name: str | None = None) -> tuple[dict | None, str]:
        """The counterpart among ALL the other entities, then the range."""
        shared = [c for c in others if key_tokens(by_id, c) & header_tokens]
        if len(shared) == 1:
            chosen, why = shared[0], "its key header shares a token with the role column's"
        else:
            rows = the_row(others)
            if len(rows) == 1:
                chosen, why = rows[0], "it is the row's entity"
            elif len(others) == 1:
                chosen, why = others[0], "it is the only other declared entity on the row"
            else:
                return None, ("ambiguous" if others else "no other declared entity on the row")
        if not in_range(chosen):
            fitting = [c for c in others if in_range(c)
                       and not cat.is_party(c["schema"])]
            if len(fitting) == 1:
                return fitting[0], (f"it is the only {fitting[0]['schema']} on the row, "
                                    f"and {chosen['key']} ({chosen['schema']}) cannot "
                                    f"stand at the other end")
            if party is not None and range_name is not None \
                    and not cat.is_descendant(range_name, chosen["schema"]):
                near, why2 = nearest_in_range(party, others, profiles, in_range,
                                              columns_of=columns_of)
                if near is not None:
                    return near, (f"{chosen['key']} ({chosen['schema']}) cannot stand "
                                  f"at the other end, and {why2}")
            return None, (f"the entity it points at, {chosen['key']} "
                          f"({chosen['schema']}), cannot stand at the other end")
        return chosen, why

    new_edges: list[dict] = []
    new_attachments: list[dict] = []
    notes: list[tuple[str, str]] = []
    have_edges = {(e["schema"], e["source"], e["target"]) for e in edges}
    have_attachments = {(a["entity"], a["prop"], a["target"]) for a in attachments}

    def _try(party: dict, others: list[dict], where: str, header_tokens: set[str],
             relation: str, side: str, spelling: str) -> tuple[bool, tuple[str, str] | None]:
        """One relation a role word names, stated if it can be. (derived,
        note): a derived relation is appended and its note returned; a
        relation that cannot be stated returns why; a duplicate returns
        neither."""
        if relation in edge_info:
            info = edge_info[relation]
            own_range = info.source_range if side == "source" else info.target_range
            other_range = info.target_range if side == "source" else info.source_range
            if not cat.is_descendant(party["schema"], own_range):
                return False, (party["key"], f"{where} names the {side} of "
                               f"{relation}, but a {party['schema']} cannot "
                               f"stand there")
            target, why = choose(
                others, header_tokens,
                lambda e: cat.is_descendant(e["schema"], other_range), party,
                other_range)
            if target is None:
                return False, (party["key"], f"{where} names the {side} of "
                               f"{relation}; not derived: {why}")
            src, tgt = ((party["key"], target["key"]) if side == "source"
                        else (target["key"], party["key"]))
            if (relation, src, tgt) in have_edges:
                return True, None
            if relation in _INTERCHANGEABLE and any(
                    (other, src, tgt) in have_edges
                    for other in _INTERCHANGEABLE if other != relation):
                return True, None
            key = f"{party['key']}_{relation.lower()}"
            have_edges.add((relation, src, tgt))
            new_edges.append({"key": key, "schema": relation,
                              "source": src, "target": tgt})
            return True, (key, f"derived from {where}: «{spelling}» names "
                          f"the {side} of {relation}, and {target['key']} "
                          f"({target['schema']}) is the other end because "
                          f"{why}")

        info = cat.prop(relation)
        if info is None or info.type_name != "entity":
            return False, None
        holder_schema = relation.split(":", 1)[0]
        rng = info.range_schema or "Thing"
        if info.name == "parent":
            rng = holder_schema
        if side == "source":
            if not cat.is_descendant(party["schema"], holder_schema):
                return False, (party["key"], f"{where} names a holder of "
                               f"{relation}, but a {party['schema']} has no "
                               f"such property")
            layout = None if info.name == "parent" else party
            qname = f"{party['schema']}:{info.name}"
            if info.name == "parent" and any(
                    a["entity"] == party["key"] and a["prop"] == qname
                    for a in [*attachments, *new_attachments]):
                return False, None
            target, why = choose(others, header_tokens,
                                 lambda e: cat.is_descendant(e["schema"], rng),
                                 layout, rng if layout is not None else None)
            if target is None:
                return False, (party["key"], f"{where} names a holder of "
                               f"{relation}; not derived: {why}")
            if (party["key"], qname, target["key"]) in have_attachments:
                return True, None
            if any(a["entity"] == target["key"] and a["target"] == party["key"]
                   for a in [*attachments, *new_attachments]):
                return True, None
            have_attachments.add((party["key"], qname, target["key"]))
            new_attachments.append({"entity": party["key"], "prop": qname,
                                    "target": target["key"]})
            return True, (party["key"], f"{qname} attached to {target['key']}: "
                          f"{where} says «{spelling}», and {target['key']} "
                          f"({target['schema']}) is the other end because {why}")
        if not cat.is_descendant(party["schema"], rng):
            return False, (party["key"], f"{where} names the {relation} of "
                           f"the row, but a {party['schema']} is not one")
        holder, why = choose(
            others, header_tokens,
            lambda e: cat.is_descendant(e["schema"], holder_schema))
        if holder is None:
            return False, (party["key"], f"{where} names the {relation}; not "
                           f"derived: {why}")
        qname = f"{holder['schema']}:{info.name}"
        if (holder["key"], qname, party["key"]) in have_attachments:
            return True, None
        if any(a["entity"] == party["key"] and a["target"] == holder["key"]
               for a in [*attachments, *new_attachments]):
            return True, None
        have_attachments.add((holder["key"], qname, party["key"]))
        new_attachments.append({"entity": holder["key"], "prop": qname,
                                "target": party["key"]})
        return True, (holder["key"], f"{qname} attached to {party['key']}: "
                      f"{where} says «{spelling}», and {holder['key']} "
                      f"({holder['schema']}) holds it because {why}")

    for party in entities:
        if not cat.is_party(party["schema"]):
            continue
        columns = list(party.get("keys") or ())
        columns += [c for c in bound_to.get(party["key"], ()) if c not in columns]
        seen_relations: set[str] = set()
        for k in columns:
            p = by_id.get(k)
            if p is None:
                continue
            candidates = roles_for(p.header, p.label, roles)
            where = f"«{(p.header or p.label or '').strip()}» ({k})"
            if not candidates and k in from_group:
                found, band = from_group[k]
                candidates = [found]
                where = f"{band}, the group header of {where}"
            candidates = [c for c in candidates if c[0] not in seen_relations]
            if not candidates:
                continue
            others = [e for e in entities
                      if e is not party and selection(e) == selection(party)
                      and e["schema"] not in edge_info
                      and not same_group(e, party)]
            if _level_keyed(party, by_id, roles):
                others = [e for e in others if not _level_keyed(e, by_id, roles)]
            failed: list[tuple[str, str]] = []
            for relation, side, spelling in candidates:
                header_tokens = (_tokens(p.header or "") | _tokens(p.label or "")) \
                    - _tokens(spelling)
                ok, note = _try(party, others, where, header_tokens,
                                relation, side, spelling)
                if ok:
                    seen_relations.add(relation)
                    if note is not None:
                        notes.append(note)
                    break
                if note is not None:
                    failed.append(note)
            else:
                if len(candidates) == 1:
                    seen_relations.add(candidates[0][0])
                notes.extend(failed)
    return new_edges, new_attachments, notes


def nearest_in_range(party: dict, others: list[dict], profiles, in_range,
                     columns_of=None) -> tuple[dict | None, str]:
    """The counterpart the layout names when nothing stronger does."""
    by_id = {p.id: p for p in profiles}

    def indexes(e: dict) -> list[int]:
        cols = columns_of(e) if columns_of is not None else list(e.get("keys") or ())
        return [by_id[c].index for c in cols if c in by_id]

    mine = indexes(party)
    fitting = [c for c in others if in_range(c)]
    if not mine or not fitting:
        return None, ("no other declared entity in range" if not fitting
                      else "the party has no column to measure from")
    ranked = []
    for c in fitting:
        theirs = indexes(c)
        if theirs:
            ranked.append((min(abs(i - j) for i in mine for j in theirs), c))
    ranked.sort(key=lambda t: t[0])
    if not ranked:
        return None, "no candidate in range has a column of its own"
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None, (f"ambiguous: {ranked[0][1]['key']} and {ranked[1][1]['key']} "
                      f"stand {ranked[0][0]} column(s) away alike")
    d, near = ranked[0]
    return near, (f"it is the nearest entity in range by column position, "
                  f"{d} column(s) from the role column")


_DEBTOR_WORDS = ("боржник", "debtor")
_CREDITOR_WORDS = ("стягувач", "кредитор", "creditor")


def derive_debts(entities: list[dict], edges: list[dict], profiles,
                 subject: str | None, cat: Catalogue, roles: dict | None = None,
                 ) -> tuple[list[dict], list[tuple[str, str]]]:
    """A `Debt` from a party whose key header says debtor to the creditor the
    file names for it: the party under a creditor word when one is keyed on
    such a column, else the nearest body in range that is not a party of a
    stated relation of its own (`nearest_in_range`). Two at the same distance,
    or none, derive nothing — a Debt to nobody is not an edge this pipeline can
    write, and the note says so.
    """
    roles = load_roles() if roles is None else roles
    by_id = {p.id: p for p in profiles}
    groups = prefix_groups(profiles)
    edge_info = cat.edge_index()
    info = edge_info.get("Debt")
    if info is None:
        return [], []
    have = {(e["schema"], e["source"], e["target"]) for e in edges}
    new_edges: list[dict] = []
    notes: list[tuple[str, str]] = []

    def says(e: dict, words) -> str | None:
        for k in e.get("keys") or ():
            p = by_id.get(k)
            if p is None:
                continue
            for text in (p.header, p.label):
                folded = _fold(text or "")
                for w in words:
                    if w in folded:
                        return f"«{(text or '').strip()}» ({k})"
        return None

    in_group_of = partial(share_prefix_group, groups)
    key_distinct = partial(key_identification, profiles=profiles)

    debtors: list[dict] = []
    for party in sorted(entities, key=key_distinct, reverse=True):
        if not cat.is_descendant(party["schema"], info.source_range):
            continue
        where = says(party, _DEBTOR_WORDS)
        if where is None or any(s == "Debt" and src == party["key"] for s, src, _ in have):
            continue
        if any(in_group_of(party, d) for d in debtors):
            continue
        debtors.append(party)
        others = [e for e in entities
                  if e is not party and e["schema"] not in edge_info
                  and e.get("filter") == party.get("filter")
                  and not in_group_of(e, party)
                  and cat.is_descendant(e["schema"], info.target_range)]
        named = [e for e in others if says(e, _CREDITOR_WORDS)]
        if len(named) == 1:
            target, why = named[0], f"{says(named[0], _CREDITOR_WORDS)} names the creditor"
        else:
            free = [e for e in others if not role_keyed(e, profiles, roles)]
            target, why = nearest_in_range(party, free, profiles, lambda e: True)
        if target is None:
            notes.append((party["key"], f"{where} names a debtor; no Debt derived: {why}"))
            continue
        key = f"{party['key']}_debt"
        have.add(("Debt", party["key"], target["key"]))
        new_edges.append({"key": key, "schema": "Debt",
                          "source": party["key"], "target": target["key"]})
        notes.append((key, f"derived from {where}: the party owes, and "
                      f"{target['key']} ({target['schema']}) is the creditor because {why}"))
    return new_edges, notes


def bind_edge_properties(relations: list[dict], entities: list[dict],
                         bindings: list[dict], profiles, cat: Catalogue,
                         lexicon: dict, roles: dict | None = None,
                         ) -> list[tuple[dict, str]]:
    """Candidate bindings of the columns the plan left unbound onto the
    properties of the relations on the row, with a reason each.
    """
    from ftmap.vocab.shortlist import _fold_header, resolved_spellings
    roles = load_roles() if roles is None else roles
    by_id = {p.id: p for p in profiles}
    relation_schemata = cat.edge_schemata()
    keyed = {k for e in entities
             if e["schema"] not in relation_schemata or e.get("source") or e.get("target")
             for k in (e.get("keys") or ())}
    ent = {e["key"]: e for e in entities}

    def endpoint_tokens(edge: dict) -> set[str]:
        out: set[str] = set()
        for side in ("source", "target"):
            e = ent.get(edge.get(side))
            for k in (e or {}).get("keys") or ():
                p = by_id.get(k)
                if p is not None:
                    out |= set(_fold_header(p.header or "").split())
        return out

    spelled: dict[str, list[tuple[dict, str]]] = {}
    endpoint_words: dict[int, set[str]] = {}
    for edge in relations:
        endpoint_words[id(edge)] = endpoint_tokens(edge)
        for info in cat.properties_of(edge["schema"]).values():
            qname = info.qname
            if info.type_name == "entity" or not cat.bindable(edge["schema"], qname):
                continue
            forms = {_fold_header(x) for x in (info.name, info.label,
                                               *resolved_spellings(lexicon, cat, qname))}
            for f in forms:
                spelled.setdefault(f, []).append((edge, qname))
    by_column = {b["column"]: b for b in bindings if b.get("prop") and b["prop"] != UNMAPPED}
    groups = prefix_groups(profiles)
    same_group = partial(share_prefix_group, groups,
                         role_keyed=lambda e: role_keyed(e, profiles, roles))

    out: list[tuple[dict, str]] = []
    for p in sorted(profiles, key=lambda p: p.index):
        if p.id in keyed or p.filled == 0:
            continue
        model = by_column.get(p.id)
        for text in (p.header, p.label):
            if not text:
                continue
            words = _fold_header(text).split()
            hits: list[tuple[dict, str, str]] = []
            for edge in relations:
                rest = " ".join(w for w in words if w not in endpoint_words[id(edge)])
                for e2, qname in spelled.get(rest, ()):
                    if e2 is edge:
                        hits.append((edge, qname, rest))
            if len({(id(e), q) for e, q, _ in hits}) != 1:
                continue
            edge, qname, rest = hits[0]
            if model is not None:
                on_endpoint = model.get("entity") in (edge.get("source"), edge.get("target")) \
                    or any(same_group(ent.get(model.get("entity")), ent.get(edge.get(side)))
                           for side in ("source", "target"))
                if not on_endpoint or rest in _BARE or model.get("entity") == edge["key"]:
                    continue
                out.append(({"column": p.id, "prop": qname, "entity": edge["key"],
                             "replaces": model["prop"],
                             "why": f"the header «{text.strip()}» spells "
                                    f"{qname} once the relation's own words "
                                    f"are taken out"},
                            f"bound to {edge['key']} ({edge['schema']}) in place of "
                            f"{model['prop']} on {model.get('entity')}: «{text.strip()}» "
                            f"spells {qname} exactly once the words of the relation's "
                            f"endpoints are taken out, and a fact written under a "
                            f"party's own prefix is the relation's"))
                break
            out.append(({"column": p.id, "prop": qname, "entity": edge["key"],
                         "leftover": rest,
                         "why": f"the header «{text.strip()}» spells "
                                f"{qname} once the relation's own words "
                                f"are taken out"},
                        f"bound to {edge['key']} ({edge['schema']}): «{text.strip()}» "
                        f"spells {qname} exactly once the words of the "
                        f"relation's endpoints are taken out"))
            break
    return out


def split_folded_parties(entities: list[dict], bindings: list[dict], profiles,
                         frame, cat: Catalogue, roles: dict | None = None,
                         ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """A second party entity for a name column the model folded into another
    party's, when the two columns name different things on most rows.
    """
    from ftmap.io.frame import column_values
    roles = load_roles() if roles is None else roles
    by_id = {p.id: p for p in profiles}
    by_key = {e["key"]: e for e in entities}
    read: dict[str, list] = {}

    def values(col: str) -> list:
        got = read.get(col)
        if got is None:
            got = column_values(frame, col)
            read[col] = got
        return got

    taken = set(by_key)
    new_entities: list[dict] = []
    rehomed: list[dict] = []
    notes: list[tuple[str, str, str]] = []
    names: dict[str, list[dict]] = {}
    for b in bindings:
        info = cat.prop(b.get("prop") or "")
        if info is not None and info.type_name == "name" and b.get("entity") in by_key:
            names.setdefault(b["entity"], []).append(b)
    for key, bs in names.items():
        e = by_key[key]
        if not cat.is_party(e["schema"]):
            continue
        cols = [b["column"] for b in bs if b["column"] in by_id]
        worded = [c for c in cols
                  if role_in(by_id[c].header, by_id[c].label, roles) is not None
                  and not any(w in _fold(by_id[c].header or "") for w in _POST_WORDS)]
        if len(worded) < 2:
            continue
        ks = [k for k in (e.get("keys") or ()) if k in by_id]
        keyed = [c for c in worded if c in (e.get("keys") or ())]
        dependent = ([c for c in worded if one_to_one(values, ks, c) is not None]
                     if ks else [])
        anchor = keyed[0] if keyed else (dependent[0] if dependent else worded[0])
        a = values(anchor)
        for other in worded:
            if other == anchor:
                continue
            o = values(other)
            both = [(x, y) for x, y in zip(a, o)
                    if x not in (None, "") and y not in (None, "")]
            if not both:
                continue
            differ = sum(1 for x, y in both
                         if " ".join(str(x).casefold().split())
                         != " ".join(str(y).casefold().split()))
            if differ * 2 <= len(both):
                continue
            new_key = other
            n = 2
            while new_key in taken:
                new_key = f"{other}_{n}"
                n += 1
            taken.add(new_key)
            schema = e["schema"]
            new_entities.append({"key": new_key, "schema": schema, "keys": [other]})
            for b in bs:
                if b["column"] == other:
                    rehomed.append({**b, "entity": new_key,
                                    "prop": f"{schema}:{b['prop'].split(':', 1)[1]}"})
            notes.append((other, new_key,
                          f"split from {key}: «{(by_id[other].header or '').strip()}» "
                          f"and «{(by_id[anchor].header or '').strip()}» are both "
                          f"bound as {key}'s name and disagree on {differ} of "
                          f"{len(both)} rows that fill both; a name that "
                          f"disagrees with the entity's own on most rows is "
                          f"another thing's, so it is its own {schema}, keyed "
                          f"and named on {other}"))
    return new_entities, rehomed, notes


def dissolve_relation_things(entities: list[dict], edges: list[dict],
                             bindings: list[dict], cat: Catalogue,
                             ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """Entities declared with a relation's schema and no endpoints, folded into
    the one relation of that schema on the row.
    """
    edge_schemata = cat.edge_schemata()
    kept: list[dict] = []
    out_bindings = list(bindings)
    notes: list[tuple[str, str, str]] = []
    for e in entities:
        if e["schema"] not in edge_schemata or e.get("source") or e.get("target"):
            kept.append(e)
            continue
        same = [g for g in edges if g["schema"] == e["schema"]]
        if len(same) != 1:
            kept.append(e)
            continue
        target = same[0]
        moved = 0
        for b in out_bindings:
            if b.get("entity") == e["key"]:
                b["entity"] = target["key"]
                moved += 1
        notes.append((None, e["key"],
                      f"dissolved into {target['key']}: a {e['schema']} declared as a "
                      f"thing, keyed on its own {', '.join(e.get('keys') or ['row'])}, "
                      f"is the {e['schema']} relation on the row, and its {moved} "
                      f"binding(s) are that relation's"))
    return kept, out_bindings, notes


_BARE = {"date", "дата", "status", "статус", "type", "тип", "description", "опис",
         "summary", "amount", "сума", "number", "номер", "name", "назва", "id",
         "notes", "примітка", "примітки", "code", "код"}

_DETECTOR_PROPERTIES = {"phone_ua": "phone", "phone_ru": "phone", "email": "email"}
_DETECTOR_FLOOR_STRONG = 0.8


def _pack_rates(profile, frame) -> dict[str, float]:
    """The profile's detector rates, and for a phone column whose rate falls
    short of the floor, the share of its cells the CANONICALIZER accepts with
    packs counted. The profile's `phone_ua` reads one number per cell — the
    enforcement offices' «(04142) 3-08-30, 3-00-07» scores 0.44 — and stays so
    on the prompt, where a pack-aware rate re-rolled two files worse twice
    (`work-c12`, `work-p1`). The rate that licenses a binding is the engine's
    own: what it will keep. Computed here, from the values, and reported as a
    rate; no value leaves.
    """
    from ftmap.io.frame import column_values
    from ftmap.normalize.canonical import MULTIVALUE_JOIN, PHONE_RULES, canonicalize
    rates = dict(profile.detectors)
    if frame is None:
        return rates
    phones = [f"phone_{r['name']}" for r in PHONE_RULES]
    if not any(0.0 < rates.get(d, 0.0) < _DETECTOR_FLOOR_STRONG for d in phones):
        return rates
    values = [v for v in column_values(frame, profile.id) if v]
    if not values:
        return rates
    for rule in PHONE_RULES:
        cc = "+" + rule["cc"]
        ok = sum(1 for v in values
                 if (got := canonicalize(v, "phone")).ok
                 and any(part.startswith(cc) for part in (got.value or "").split(MULTIVALUE_JOIN)))
        rates[f"phone_{rule['name']}"] = max(rates.get(f"phone_{rule['name']}", 0.0),
                                             ok / len(values))
    return rates


def bind_by_detector(entities: list[dict], bindings: list[dict], profiles,
                     subject: str | None, cat: Catalogue, frame=None,
                     relations: set[str] | None = None,
                     skip: set[str] | None = None,
                     ) -> list[tuple[dict, str]]:
    """Candidate bindings for the columns the plan left unbound that a phone or
    e-mail detector reads on most of their values.
    """
    by_id = {p.id: p for p in profiles}
    relations = set() if relations is None else relations
    live = [b for b in bindings if b.get("prop")]
    on_column: dict[str, list[dict]] = {}
    for b in live:
        on_column.setdefault(b["column"], []).append(b)
    keyed = {k for e in entities if cat.is_party(e["schema"])
             for k in (e.get("keys") or ())}
    out: list[tuple[dict, str]] = []

    skip = set() if skip is None else skip
    for p in sorted(profiles, key=lambda p: p.index):
        if p.id in keyed or p.filled == 0 or p.id in skip:
            continue
        taken = on_column.get(p.id, [])
        replaces = [b for b in taken
                    if cat.prop(b["prop"]) is not None
                    and (b["prop"].endswith("Mentioned")
                         or (b.get("entity") in relations
                             and cat.prop(b["prop"]).type_name in ("string", "text")))]
        if taken and len(replaces) != len(taken):
            continue
        rates = _pack_rates(p, frame)
        hits = [(rate, det) for det, rate in rates.items()
                if det in _DETECTOR_PROPERTIES and rate >= _DETECTOR_FLOOR_STRONG]
        if not hits:
            continue
        rate, det = max(hits)
        name = _DETECTOR_PROPERTIES[det]
        header_tokens = _tokens(p.header or "") | _tokens(p.label or "")
        shared = [e for e in entities
                  if cat.is_party(e["schema"]) and p.id not in (e.get("keys") or ())
                  and key_tokens(by_id, e) & header_tokens]
        rows = [e for e in entities
                if subject and cat.related_to(e["schema"], subject)
                and not role_keyed(e, profiles)]
        chosen, why = None, ""
        if len(shared) == 1:
            chosen, why = shared[0], "its key header shares a token with the column's"
        elif len(rows) == 1:
            chosen, why = rows[0], "it is the row's entity"
        if chosen is None:
            continue
        qname = f"{chosen['schema']}:{name}"
        if cat.prop(qname) is None:
            continue
        cand = {"column": p.id, "prop": qname, "entity": chosen["key"],
                "why": f"{det} reads {rate:.0%} of the values"}
        if replaces:
            cand["replaces"] = replaces
            what = ", ".join(f"{b['prop']} on {b['entity']}" for b in replaces)
            out.append((cand, f"bound to {chosen['key']} ({chosen['schema']}): the "
                              f"{det} detector reads {rate:.0%} of "
                              f"«{(p.header or '').strip()}», which was bound as a "
                              f"relation's string or a thing's mention ({what}); a "
                              f"party's {name} belongs to the party, never to the "
                              f"link or to what a document mentions, so that "
                              f"binding gives way; {chosen['key']} holds it because "
                              f"{why}"))
            continue
        out.append((cand,
                    f"bound to {chosen['key']} ({chosen['schema']}): the {det} detector "
                    f"reads {rate:.0%} of «{(p.header or '').strip()}» and the model "
                    f"bound it to nothing; {chosen['key']} holds it because {why}"))
    return out


def id_from_key(entities: list[dict], bindings: list[dict], profiles, cat,
                values, declined: set[str], rejected: set[str],
                distinct_floor: float, floor: float = _NAME_TEXT_FLOOR,
                ) -> list[tuple[dict, str]]:
    """The identifier of a person keyed on one column of codes the binding call
    declined: that column, as `Person:idNumber`. (candidate, reason) pairs,
    checked like any binding.
    """
    by_id = {p.id: p for p in profiles}
    bound_cols = {b["column"] for b in bindings if b.get("prop") and b["prop"] != UNMAPPED}
    out: list[tuple[dict, str]] = []
    for e in entities:
        keys = list(e.get("keys") or ())
        if len(keys) != 1 or not cat.is_descendant(e["schema"], "Person"):
            continue
        col = keys[0]
        p = by_id.get(col)
        if p is None or p.filled == 0 or col in bound_cols:
            continue
        if col not in declined or col in rejected:
            continue
        if p.distinct_ratio < distinct_floor:
            continue
        qname = f"{e['schema']}:idNumber"
        if not cat.bindable(e["schema"], qname):
            continue
        share = text_share(values(col))
        if share >= floor:
            continue
        head = f"«{(p.header or '').strip()}» ({col})" if p.header else f"({col})"
        out.append(({"column": col, "prop": qname, "entity": e["key"],
                     "why": f"{head} identifies this person and holds codes, "
                            f"not words; the column is the person's identifier"},
                    f"identified from its key: {head} is the one column "
                    f"identifying {e['key']} ({e['schema']}), {p.distinct_ratio:.0%} "
                    f"distinct, {1 - share:.0%} of its values codes rather than "
                    f"words, and the binding call declined it"))
    return out


def rehome_varying(entities: list[dict], bindings: list[dict], profiles,
                   subject: str | None, cat, bound: int, per_key,
                   ) -> list[tuple[dict, dict, str]]:
    """A column that varies within the key of the thing it is bound to is the
    row's. (old binding, new binding, reason) triples.
    """
    by_id = {p.id: p for p in profiles}
    keyed = {e["key"]: e for e in entities if e.get("keys")}
    rows = [e for e in entities
            if subject and cat.related_to(e["schema"], subject)
            and not role_keyed(e, profiles)]
    if len(rows) != 1 or not cat.is_party(rows[0]["schema"]):
        return []
    row = rows[0]
    already = {b["column"] for b in bindings if b.get("entity") == row["key"]}
    carried = cat.properties_of(row["schema"])
    out: list[tuple[dict, dict, str]] = []
    for b in bindings:
        holder = keyed.get(b.get("entity"))
        if holder is None or holder is row or not b.get("prop") or b["prop"] == UNMAPPED:
            continue
        col = b["column"]
        p = by_id.get(col)
        if p is None or col in holder["keys"] or col in already:
            continue
        n = per_key(col, holder)
        if n is None or n <= bound:
            continue
        name = b["prop"].split(":", 1)[1]
        own = carried.get(name)
        if own is None or not cat.bindable(row["schema"], own.qname):
            continue
        new = {**b, "prop": own.qname, "entity": row["key"],
               "why": f"«{(p.header or '').strip()}» takes {n:.0f} values under one "
                      f"key of {holder['key']}; it is the row's"}
        out.append((b, new,
                    f"re-homed from {holder['key']} ({holder['schema']}) to "
                    f"{row['key']} ({row['schema']}): «{(p.header or '').strip()}» "
                    f"({col}) takes a median of {n:.0f} distinct values under one "
                    f"key of {holder['key']}, and a property of one thing takes at "
                    f"most {bound}; the column varies with the row and is the row's "
                    f"entity's"))
    return out


def spelled_property_beats_relation(entities: list[dict], bindings: list[dict],
                                    profiles, cat, lexicon, relations: set[str],
                                    attempted: list[dict] | None = None,
                                    ) -> list[tuple[dict, str]]:
    """A column whose header exactly spells ONE party's property, bound to a
    relation instead, is that party's. (candidate with `replaces`, reason).
    """
    from ftmap.vocab.shortlist import _fold_header, resolved_spellings
    by_id = {p.id: p for p in profiles}
    parties = [e for e in entities if cat.is_party(e["schema"])]
    if not parties:
        return []
    out: list[tuple[dict, str]] = []
    by_column: dict[str, list[dict]] = {}
    for b in bindings:
        if b.get("prop") and b["prop"] != UNMAPPED and b.get("entity") in relations:
            by_column.setdefault(b["column"], []).append(b)
    bound_cols = {b["column"] for b in bindings if b.get("prop") and b["prop"] != UNMAPPED}
    for b in attempted or ():
        if (b.get("prop") and b["prop"] != UNMAPPED and b.get("entity") in relations
                and b["column"] not in bound_cols):
            by_column.setdefault(b["column"], [])
    for col, taken in by_column.items():
        p = by_id.get(col)
        if p is None:
            continue
        text = (p.header or p.label or "").strip()
        folded = _fold_header(text)
        if not folded or folded in _BARE:
            continue
        claims: list[tuple[dict, str]] = []
        for party in parties:
            for qname, info in cat.properties_of(party["schema"]).items():
                if info.type_name == "entity" or not cat.bindable(party["schema"], info.qname):
                    continue
                forms = {_fold_header(x) for x in (info.name, info.label,
                                                   *resolved_spellings(lexicon, cat, info.qname))}
                if folded in forms:
                    claims.append((party, info.qname))
        if len(claims) != 1:
            continue
        party, qname = claims[0]
        what = (", ".join(f"{b['prop']} on {b['entity']}" for b in taken)
                or "a relation's property the value check refused")
        out.append(({"column": col, "prop": qname, "entity": party["key"],
                     "replaces": list(taken),
                     "why": f"the header «{text}» spells {qname} exactly, and the "
                            f"column was bound to a relation"},
                    f"the header «{text}» ({col}) spells {qname} exactly and no other "
                    f"party's property; it was bound to a relation ({what}), and a "
                    f"header that names the party's own property is not the link's, "
                    f"so that binding gives way to {party['key']} ({party['schema']})"))
    return out


def consecutive_integers(values: list[str], least: int = 1) -> bool:
    """Whether the values are 1, 2, 3 … in file order — a row number, not a
    record number, and not a fact about the row.
    """
    if len(values) < least:
        return False
    try:
        nums = [int(v.strip()) for v in values]
    except ValueError:
        return False
    return all(b == a + 1 for a, b in zip(nums, nums[1:]))


def key_relation_on_record(edges: list[dict], entities: list[dict], bindings: list[dict],
                           profiles, frame, subject: str | None, cat: Catalogue,
                           roles: dict | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """The row's one relation, keyed on the record number the row carries for
    it, with that number as its `recordId`.
    """
    from ftmap.io.frame import column_values
    roles = load_roles() if roles is None else roles
    edge_schemata = cat.edge_schemata()
    keyed = {k for e in (*entities, *edges) for k in (e.get("keys") or ())}
    bound = {b["column"] for b in bindings if b.get("prop") and b["prop"] != UNMAPPED}

    def related(e: dict) -> bool:
        schema = e["schema"]
        return (bool(subject) and schema not in edge_schemata
                and cat.related_to(schema, subject)
                and not role_keyed(e, profiles, roles))

    key_distinct = partial(key_identification, profiles=profiles)

    rows = sorted((e for e in entities if related(e)), key=key_distinct, reverse=True)
    if not rows or (len(rows) > 1 and key_distinct(rows[0]) == key_distinct(rows[1])):
        return [], []
    row = rows[0]
    if key_distinct(row) >= 0.99:
        return [], []
    own = [g for g in edges if row["key"] in (g["source"], g["target"]) and not g.get("keys")]
    if len(own) != 1:
        return [], []
    edge = own[0]
    qname = f"{edge['schema']}:recordId"
    if cat.prop(qname) is None:
        return [], []
    cands = [p for p in profiles
             if p.id not in keyed and p.id not in bound and p.fill_rate >= 0.9
             and p.distinct_ratio >= 0.99 and p.detectors.get("numeric", 0.0) >= 0.9
             and not consecutive_integers([v for v in column_values(frame, p.id) if v])]
    if len(cands) != 1:
        if len(cands) > 1:
            return [], [(edge["key"], "not keyed on a record number: "
                         + ", ".join(f"«{(p.header or '').strip()}» ({p.id})" for p in cands)
                         + " are each distinct on every row, and one relation has one number")]
        return [], []
    p = cands[0]
    edge["keys"] = [p.id]
    head = f"«{(p.header or '').strip()}» ({p.id})"
    return ([{"column": p.id, "prop": qname, "entity": edge["key"],
              "why": f"{head} is distinct on every row and keys nothing"}],
            [(edge["key"], f"keyed on {head}: a number distinct on every row that keys no "
                           f"entity is the record's own, {row['key']} ({row['schema']}) "
                           f"repeats across rows, and {edge['schema']} is the row's one "
                           f"relation; the number is its recordId")])
