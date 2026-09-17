"""Column blocks a layout fixes by regulation, read as the thing they spell."""

from __future__ import annotations

import re

_FIELDS: dict[str, str | None] = {
    "postcode": "Address:postalCode",
    "adminunitl1": "Address:country",
    "adminunitl2": "Address:region",
    "adminunitl3": None,
    "adminunitl4": None,
    "postname": "Address:city",
    "thoroughfare": "Address:street",
    "locatordesignator": "Address:full",
    "locatorbuilding": None,
    "locatorname": "Address:summary",
}
_KEY_FIELDS = ("thoroughfare", "locatordesignator")
_PREFIX = re.compile(r"^address[_ ]?", re.IGNORECASE)
ADDRESS_KEY = "address_block"


def _field(header: str | None) -> str | None:
    if not header:
        return None
    text = header.strip()
    if not _PREFIX.match(text):
        return None
    rest = _PREFIX.sub("", text).replace("_", "").casefold()
    return rest if rest in _FIELDS else None


def declare_address_block(entities: list[dict], profiles,
                          ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """(new entities, bindings, (column, key, note)) for the address block a
    layout spells, or ([], [], []) when the profiles hold none.
    """
    if any(e.get("key") == ADDRESS_KEY for e in entities):
        return [], [], []
    members: dict[str, object] = {}
    for p in profiles:
        f = _field(p.header)
        if f is not None and p.filled > 0 and f not in members:
            members[f] = p
    if len(members) < 3 or not ({"thoroughfare", "postname"} & set(members)):
        return [], [], []
    keys = [members[f].id for f in _KEY_FIELDS if f in members]
    if not keys:
        keys = [members["postname"].id]
    entity = {"key": ADDRESS_KEY, "schema": "Address", "keys": keys}
    bindings: list[dict] = []
    notes: list[tuple[str, str, str]] = []
    cols = ", ".join(p.id for p in members.values())
    for f, p in members.items():
        qname = _FIELDS[f]
        where = f"«{(p.header or '').strip()}» ({p.id})"
        if qname is None:
            notes.append((p.id, ADDRESS_KEY,
                          f"declined by the address-block rule: {where} is the "
                          f"format's {f} field, a level FollowTheMoney's Address "
                          f"does not hold"))
            continue
        bindings.append({"column": p.id, "prop": qname, "entity": ADDRESS_KEY,
                         "why": f"the format's {f} field of the address block"})
        notes.append((p.id, ADDRESS_KEY,
                      f"bound by the address-block rule: {where} is the format's "
                      f"{f} field, {qname} of the one Address the block "
                      f"({cols}) spells"))
    notes.insert(0, (keys[0], ADDRESS_KEY,
                     f"declared by the address-block rule: {len(members)} "
                     f"address fields of the ПКМУ-835 layout ({cols}) are one "
                     f"Address, keyed on {', '.join(keys)}"))
    return [entity], bindings, notes


def address_block_columns(profiles) -> set[str]:
    """The columns an address block claims — the model's bindings on them are
    overridden."""
    return {p.id for p in profiles if _field(p.header) is not None and p.filled > 0}


_PROCUREMENT_WORDS = ("закупівл", "тендер", "дк 021", "cpv", "лот", "процедур",
                      "пропозиц", "договір", "предмет", "prozorro", "tender",
                      "procurement")
CONTRACT_KEY = "contract_row"


def declare_contract(entities: list[dict], profiles, roles: dict | None = None,
                     ) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """A `Contract` for a row that is a purchase, when the plan declares none.
    """
    from ftmap.plan.roles import load_roles, role_in
    roles = load_roles() if roles is None else roles
    if any(e["schema"] == "Contract" or e["key"] == CONTRACT_KEY for e in entities):
        return [], []
    supplier_cols = []
    for p in profiles:
        found = role_in(p.header, p.label, roles)
        if found is not None and found[0] == "ContractAward":
            supplier_cols.append(p)
    if not supplier_cols:
        return [], []
    spoken = [p for p in profiles
              if any(w in (p.header or "").casefold() for w in _PROCUREMENT_WORDS)
              and p not in supplier_cols]
    if not spoken:
        return [], []
    link = next((p for p in profiles
                 if p.detectors.get("url", 0.0) >= 0.9 and p.distinct_ratio >= 0.9
                 and any(w in (p.header or "").casefold()
                         for w in ("посилання", "link", "url", "лінк"))), None)
    keys = [link.id] if link is not None else []
    entity = {"key": CONTRACT_KEY, "schema": "Contract", "keys": keys}
    heads = ", ".join(f"«{(p.header or '').strip()}»" for p in spoken[:3])
    note = (f"declared by the procurement rule: «{(supplier_cols[0].header or '').strip()}» "
            f"({supplier_cols[0].id}) names a supplier and {heads} speak "
            f"procurement, so the row is a Contract"
            + (f", keyed on the link column {link.id}" if link else
               ", keyed on the row — no column identifies the tender"))
    return [entity], [(supplier_cols[0].id, CONTRACT_KEY, note)]


def split_redacted_key(entities: list[dict], bindings: list[dict], profiles,
                       frame, cat) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """A second party for the rows whose identifier is redacted."""
    from collections import Counter
    from ftmap.io.frame import column_values
    from ftmap.normalize.canonical import is_sentinel
    by_id = {p.id: p for p in profiles}
    taken = {e["key"] for e in entities}
    new_entities: list[dict] = []
    new_bindings: list[dict] = []
    notes: list[tuple[str, str, str]] = []
    for e in list(entities):
        if e.get("filter") or not cat.is_party(e["schema"]):
            continue
        keys = [k for k in (e.get("keys") or ()) if k in by_id]
        if not keys:
            continue
        for k in keys:
            p = by_id[k]
            vals = [v for v in column_values(frame, k) if v not in (None, "")]
            if not vals:
                continue
            modal, count = Counter(str(v).strip() for v in vals).most_common(1)[0]
            if count * 2 < len(vals) or not is_sentinel(modal):
                continue
            if not any(p.detectors.get(d, 0.0) >= 0.9 for d in ("edrpou", "rnokpp", "inn_ru")):
                continue
            names = [b for b in bindings if b.get("entity") == e["key"]
                     and (cat.prop(b.get("prop") or "") or None) is not None
                     and cat.prop(b["prop"]).type_name == "name"
                     and b["column"] != k]
            if not names:
                continue
            name_col = names[0]["column"]
            from ftmap.profile.detectors import is_proper_name
            key_vals = column_values(frame, k)
            name_vals = column_values(frame, name_col)
            selected = [str(nv) for kv, nv in zip(key_vals, name_vals)
                        if kv not in (None, "") and str(kv).strip() == modal
                        and nv not in (None, "")]
            proper = (sum(1 for v in selected if is_proper_name(v)) / len(selected)
                      if selected else 0.0)
            schema = "Person" if proper >= 0.8 else "LegalEntity"
            new_key = f"{e['key']}_redacted"
            n = 2
            while new_key in taken:
                new_key = f"{e['key']}_redacted_{n}"
                n += 1
            taken.add(new_key)
            sibling = {"key": new_key, "schema": schema, "keys": [name_col],
                       "filter": {"column": k, "value": modal}}
            new_entities.append(sibling)
            for b in names:
                new_bindings.append({**b, "entity": new_key,
                                     "prop": f"{schema}:{b['prop'].split(':', 1)[1]}"})
            notes.append((k, new_key,
                          f"declared by the redaction rule: «{(p.header or '').strip()}» "
                          f"({k}) is {e['key']}'s identifier and reads {modal!r} on "
                          f"{count} of {len(vals)} rows — a redaction, a null; those rows "
                          f"are a {schema} keyed on {name_col} and selected on the "
                          f"redaction, since a null code identifies nothing"))
            break
    return new_entities, new_bindings, notes


def specialise_coded_party(entities: list[dict], profiles) -> list[tuple[str, str, str]]:
    """A party keyed on a ЄДРПОУ column is a legal person: the code is issued
    to organisations only (a sole trader has a РНОКПП). In place; returns
    (column, key, note) for each entity moved from LegalEntity to
    Organization."""
    by_id = {p.id: p for p in profiles}
    notes: list[tuple[str, str, str]] = []
    for e in entities:
        if e["schema"] != "LegalEntity" or e.get("filter"):
            continue
        for k in e.get("keys") or ():
            p = by_id.get(k)
            if p is not None and p.detectors.get("edrpou", 0.0) >= 0.9 \
                    and p.fill_rate >= 0.5:
                e["schema"] = "Organization"
                notes.append((k, e["key"],
                              f"specialised to Organization: keyed on «{(p.header or '').strip()}» "
                              f"({k}), a ЄДРПОУ code on {p.detectors['edrpou']:.0%} of its "
                              f"values, and a ЄДРПОУ is issued to legal persons only"))
                break
    return notes


_CERTIFICATE_STEMS = ("сертифікат", "посвідчен", "свідоцтв", "ліценз", "дозв",
                      "certif", "cert", "licen", "permit", "qualif")
_NUMBER_WORDS = ("№", "номер", "number", "num", "no")
IDENTIFICATION_KEY = "identification_row"


def certificate_column(profiles):
    """The one column whose header names a certificate and whose values are
    its numbers: a number word in the header, or values distinct on nearly
    every row that read as neither names nor dates. None when there is no
    such column or more than one."""
    from ftmap.vocab.shortlist import _fold_header
    hits = []
    for p in profiles:
        toks = set()
        for text in (p.header, p.label):
            toks |= set(_fold_header(text or "").split())
        if not any(stem in t for stem in _CERTIFICATE_STEMS for t in toks):
            continue
        numbered = any(t in _NUMBER_WORDS for t in toks)
        identifying = (p.distinct_ratio >= 0.9 and p.fill_rate >= 0.9
                       and p.detectors.get("proper_name", 0.0) < 0.2
                       and p.detectors.get("date", 0.0) < 0.5)
        if numbered or identifying:
            hits.append(p)
    return hits[0] if len(hits) == 1 else None


def declare_identification(entities: list[dict], profiles, cat,
                           ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]]]:
    """A certificate number beside a party is an Identification: FtM's document
    «issued to a person for identification purposes, or as a licence or
    permit», with a number, an authority, a start and an end date and an
    entity-typed holder.
    """
    col = certificate_column(profiles)
    if col is None:
        return [], [], []
    head = f"«{(col.header or '').strip()}» ({col.id})"
    binding = {"column": col.id, "prop": "Identification:number",
               "entity": IDENTIFICATION_KEY,
               "why": f"{head} is the certificate's number"}
    keyed = [e for e in entities if col.id in (e.get("keys") or ())]
    if keyed:
        e = keyed[0]
        if cat.is_descendant(e["schema"], "Identification"):
            return [], [], []
        document_kind = (e["schema"] == "Thing"
                         or cat.is_descendant(e["schema"], "Contract")
                         or cat.is_descendant(e["schema"], "Document"))
        if not document_kind:
            return [], [], [(col.id, e["key"],
                             f"declined: {head} names a certificate, but {e['key']} "
                             f"({e['schema']}) is keyed on it and is not a document; "
                             f"a certificate number may be an asset's or a party's key")]
        was = e["schema"]
        e["schema"] = "Identification"
        binding["entity"] = e["key"]
        return [], [binding], [(col.id, e["key"],
                                f"re-schemed from {was} to Identification: {head} names a "
                                f"certificate, a document issued to its holder, and a "
                                f"{was} has no holder")]
    entity = {"key": IDENTIFICATION_KEY, "schema": "Identification", "keys": [col.id]}
    return [entity], [binding], [(col.id, IDENTIFICATION_KEY,
                                  f"declared by the certificate rule: {head} names a "
                                  f"certificate and holds its number, so the row carries "
                                  f"an Identification keyed on it")]


def attach_identification_holder(entities: list[dict], edges: list[dict],
                                 attachments: list[dict], profiles, subject: str | None,
                                 cat) -> tuple[list[dict], list[tuple[str, str]]]:
    """Whose the certificate is: the row's party when the row is a party, the
    only party otherwise, and for a document of the row's asset — a vehicle's
    registration certificate — the asset's owner, since a registration document
    is held by the owner. Two candidates, or none, attach nothing and say so.
    """
    from ftmap.plan.roles import role_keyed
    by_key = {e["key"]: e for e in entities}

    def related(e: dict) -> bool:
        return bool(subject) and cat.related_to(e["schema"], subject)

    out: list[dict] = []
    notes: list[tuple[str, str]] = []
    for ident in entities:
        if not cat.is_descendant(ident["schema"], "Identification"):
            continue
        qname = f"{ident['schema']}:holder"
        if any(a["entity"] == ident["key"] and a["prop"] == qname
               for a in [*attachments, *out]):
            continue
        parties = [e for e in entities if cat.is_party(e["schema"])
                   and e.get("filter") == ident.get("filter")]
        rows = [e for e in parties if related(e) and not role_keyed(e, profiles)]
        holder, why = None, ""
        if len(rows) == 1:
            holder, why = rows[0], "it is the row's entity"
        elif len(parties) == 1:
            holder, why = parties[0], "it is the only party on the row"
        else:
            assets = {e["key"] for e in entities
                      if related(e) and cat.is_descendant(e["schema"], "Asset")}
            owners = {g["source"] for g in edges
                      if g["schema"] == "Ownership" and g["target"] in assets}
            if len(owners) == 1:
                holder = by_key.get(next(iter(owners)))
                why = "it owns the row's asset, and a registration document is the owner's"
        if holder is None:
            notes.append((ident["key"], f"{qname} not attached: "
                          + ("no party on the row" if not parties else
                             f"{len(parties)} parties on the row and nothing says whose "
                             f"the certificate is")))
            continue
        out.append({"entity": ident["key"], "prop": qname, "target": holder["key"]})
        notes.append((ident["key"], f"{qname} attached to {holder['key']}: {why}"))
    return out, notes
