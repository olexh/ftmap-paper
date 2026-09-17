"""Prefix groups: the columns an export names as one thing's fields."""
from __future__ import annotations

from ftmap.vocab.shortlist import _fold_header


def _header_tokens(header: str | None) -> list[str]:
    """The words a one-word header is built from, in order — `DEBTOR_NAME` is
    «debtor», «name». Empty for a header with whitespace in it, which is prose
    and not an export's field name. Not `roles._tokens`, which answers a
    different question (the tokens that could TIE two headers together) and
    returns a set.
    """
    text = (header or "").strip()
    if not text or any(ch.isspace() for ch in text):
        return []
    return _fold_header(text.replace("-", " ").replace(".", " ")).split()


def prefix_groups(profiles) -> dict[str, list[str]]:
    """`{prefix: [column ids in column order]}` for every prefix at least two
    one-word headers share, the prefix spelled as its folded tokens joined
    by a space."""
    toks = {p.id: _header_tokens(p.header) for p in profiles}
    by_first: dict[str, list] = {}
    for p in sorted(profiles, key=lambda p: p.index):
        t = toks[p.id]
        if len(t) >= 2:
            by_first.setdefault(t[0], []).append(p)
    out: dict[str, list[str]] = {}
    for members in by_first.values():
        if len(members) < 2:
            continue
        common = list(toks[members[0].id])
        for m in members[1:]:
            t = toks[m.id]
            n = 0
            while n < len(common) and n < len(t) and common[n] == t[n]:
                n += 1
            common = common[:n]
        while common and any(len(toks[m.id]) <= len(common) for m in members):
            common = common[:-1]
        if common:
            out[" ".join(common)] = [m.id for m in members]
    return out


def group_of(groups: dict[str, list[str]], column: str) -> str | None:
    """The prefix of the group a column belongs to, or None."""
    for prefix, cols in groups.items():
        if column in cols:
            return prefix
    return None


def remainder(profile, prefix: str) -> str:
    """The field name a header spells once its group prefix is taken out:
    `CORRUPTIONER_LAST_NAME` under `corruptioner` is «last name»."""
    t = _header_tokens(profile.header)
    n = len(prefix.split())
    return " ".join(t[n:]) if t[:n] == prefix.split() else ""


def key_groups(groups: dict[str, list[str]], entity: dict) -> set[str]:
    """The prefixes of the groups an entity's key columns fall in."""
    return {group_of(groups, k) for k in (entity.get("keys") or ())} - {None}


def share_prefix_group(groups: dict[str, list[str]], a: dict | None,
                       b: dict | None, *, role_keyed=None) -> bool:
    """Whether two entities are keyed inside one prefix group — `DEBTOR_NAME`
    and `DEBTOR_CODE` — so that the file's own naming says they are one thing
    and neither is the other's counterpart.
    """
    if a is None or b is None or a is b:
        return False
    if role_keyed is not None and (role_keyed(a) or role_keyed(b)):
        return False
    return bool(key_groups(groups, a) & key_groups(groups, b))


def selection(entity: dict) -> tuple[str, str] | None:
    """The row selection an entity is built from — `(column, value)` — or None
    when it is built from every row. Two entities in different selections live
    in different FtM queries and cannot see one another; three rules ask that
    and this is the one answer."""
    f = entity.get("filter")
    return (f["column"], f["value"]) if f else None


def group_entity(prefix: str, cols: list[str], entities: list[dict],
                 bindings: list[dict], profiles, role_keyed) -> dict | None:
    """The one entity a prefix group describes: keyed on a member column,
    or — when none is — bound to one; never one whose key carries a role
    word, which is a party the group RELATES (`sti_chief_name` under
    `sti_name`). None when there is not exactly one."""
    keyed = [e for e in entities
             if any(k in cols for k in (e.get("keys") or ()))
             and not role_keyed(e, profiles)]
    if len(keyed) == 1:
        return keyed[0]
    if keyed:
        return None
    by_key = {e["key"]: e for e in entities}
    bound = {b["entity"] for b in bindings
             if b.get("prop") and b.get("entity") in by_key and b["column"] in cols
             and not role_keyed(by_key[b["entity"]], profiles)}
    return by_key[next(iter(bound))] if len(bound) == 1 else None


def bind_group_members(entities: list[dict], bindings: list[dict], profiles, cat,
                       lexicon: dict, role_keyed, unmapped: str = "UNMAPPED",
                       ) -> list[tuple[dict, str]]:
    """Candidate bindings of a prefix group's unbound columns onto the group's
    entity, by the exact spelling of the field name left once the prefix is
    taken out.
    """
    from ftmap.vocab.shortlist import _fold_header, resolved_spellings
    groups = prefix_groups(profiles)
    by_id = {p.id: p for p in profiles}
    bound = {b["column"] for b in bindings if b.get("prop") and b["prop"] != unmapped}
    out: list[tuple[dict, str]] = []
    for prefix, cols in groups.items():
        owner = group_entity(prefix, cols, entities, bindings, profiles, role_keyed)
        if owner is None:
            continue
        others_keys = {k for e in entities if e is not owner for k in (e.get("keys") or ())}
        spelled: dict[str, list[str]] = {}
        for info in cat.properties_of(owner["schema"]).values():
            if info.type_name == "entity" or not cat.bindable(owner["schema"], info.qname):
                continue
            forms = {_fold_header(x) for x in (info.name, info.label,
                                               *resolved_spellings(lexicon, cat, info.qname))}
            for f in forms:
                spelled.setdefault(f, []).append(info.qname)
        for cid in cols:
            p = by_id.get(cid)
            if p is None or cid in bound or cid in others_keys or p.filled == 0:
                continue
            rest = remainder(p, prefix)
            hits = sorted(set(spelled.get(rest, ())))
            if len(hits) != 1:
                continue
            qname = hits[0]
            head = (p.header or "").strip()
            out.append(({"column": cid, "prop": qname, "entity": owner["key"],
                         "why": f"«{head}» spells {qname} once the group prefix "
                                f"«{prefix}» is taken out"},
                        f"bound to {owner['key']} ({owner['schema']}): «{head}» is a "
                        f"column of the «{prefix}» group, whose entity {owner['key']} "
                        f"is, and «{rest}» spells {qname} exactly"))
    return out


def merge_group_twins(entities: list[dict], bindings: list[dict], edges: list[dict],
                      profiles, role_keyed, key_identification, prefer: set | None = None,
                      ) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Two entities of one schema keyed inside one prefix group, sharing a key
    column, are one thing declared twice.
    """
    groups = prefix_groups(profiles)
    dropped: dict[str, str] = {}
    notes: list[tuple[str, str, str]] = []
    seen_groups: dict[tuple, set[str]] = {}

    def group_keys(e: dict) -> set[str]:
        ks = tuple(e.get("keys") or ())
        got = seen_groups.get(ks)
        if got is None:
            got = key_groups(groups, e)
            seen_groups[ks] = got
        return got

    survivors = list(entities)
    changed = True
    while changed:
        changed = False
        for a in survivors:
            for b in survivors:
                if a is b or a["schema"] != b["schema"] or a.get("filter") != b.get("filter"):
                    continue
                if not (group_keys(a) & group_keys(b)):
                    continue
                shared = [k for k in (a.get("keys") or ()) if k in (b.get("keys") or ())]
                if not shared or role_keyed(a, profiles) or role_keyed(b, profiles):
                    continue
                preferred = [e for e in (a, b) if e["key"] in (prefer or ())]
                if len(preferred) == 1:
                    keep = preferred[0]
                    go = b if keep is a else a
                    how = "keyed as the rule that read the layout keyed it"
                else:
                    keep, go = sorted((a, b), key=key_identification, reverse=True)
                    was = list(keep.get("keys") or ())
                    keep["keys"] = [k for k in was if k in shared] or was
                    how = "the part both readings agree on"
                dropped[go["key"]] = keep["key"]
                survivors.remove(go)
                notes.append((None, keep["key"],
                              f"merged {go['key']} into it: two {keep['schema']} keyed inside "
                              f"one prefix group on a shared column ({', '.join(shared)}) are "
                              f"one thing declared twice; keyed on "
                              f"{', '.join(keep['keys'])}, {how}"))
                changed = True
                break
            if changed:
                break
    if dropped:
        for b in bindings:
            if b.get("entity") in dropped:
                b["entity"] = dropped[b["entity"]]
        for g in edges:
            for side in ("source", "target"):
                if g.get(side) in dropped:
                    g[side] = dropped[g[side]]
    return survivors, notes
