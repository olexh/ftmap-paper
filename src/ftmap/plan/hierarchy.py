"""Organisations a sheet nests, read from their columns instead of asked."""

from __future__ import annotations

from collections import Counter

from ftmap.plan.roles import load_levels, load_roles, role_in, roles_in
from ftmap.vocab.shortlist import fold as _fold


def level_column(profile, roles: dict | None = None) -> bool:
    """Whether a column's header or label carries a level word — one the
    role lexicon lists as naming a subunit."""
    roles = load_roles() if roles is None else roles
    levels = {"Organization:parent": roles.get("Organization:parent", {})}
    return role_in(profile.header, profile.label, levels) is not None


def level_rank(profile, roles: dict | None = None) -> int | None:
    """How fine a level the column's header names, by the lexicon's order
    of level words (`_levels`, coarsest first): the finest word in the
    header — «До роты/взвода» names the platoon level. None where the
    header carries no level word, or one the order does not list."""
    roles = load_roles() if roles is None else roles
    order = load_levels()
    levels = {"Organization:parent": roles.get("Organization:parent", {})}
    text = _fold(f"{profile.header or ''} {profile.label or ''}")
    ranks = [order.index(sp) for sp in levels["Organization:parent"].get("source", ())
             if sp in order and _fold(sp) in text]
    return max(ranks) if ranks else None


def nested(values, inner: str, outer: str, floor: float) -> tuple[int, int] | None:
    """`(consistent, rows)` when `inner` nests in `outer`: over the rows
    filling both, `consistent` of `rows` carry, for their inner value, the
    outer value that inner value most often carries, and the share clears
    `floor`; and the outer takes strictly fewer distinct values. None
    otherwise. `values` reads a column id to its values in row order.
    """
    pairs: dict[str, Counter] = {}
    rows = 0
    for a, b in zip(values(inner), values(outer)):
        a = str(a).strip() if a is not None else ""
        b = str(b).strip() if b is not None else ""
        if not a or not b:
            continue
        rows += 1
        pairs.setdefault(a, Counter())[b] += 1
    if not rows:
        return None
    outer_distinct = len({b for c in pairs.values() for b in c})
    if outer_distinct >= len(pairs):
        return None
    consistent = sum(c.most_common(1)[0][1] for c in pairs.values())
    if consistent / rows < floor:
        return None
    return consistent, rows


def organisation_tree(entities: list[dict], values, cat, floor: float,
                      profiles=None, roles: dict | None = None,
                      ) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """`{child key: parent key}` over the organisations keyed on one column
    each, that column headed by a level word, built from the whole table;
    and a note per pair read. With no `profiles` no header can be read and
    nothing nests. `organisation_ancestors` is the same reading kept whole."""
    parents, _ancestors, notes = _tree(entities, values, cat, floor, profiles, roles)
    return parents, notes


def organisation_ancestors(entities: list[dict], values, cat, floor: float,
                           profiles=None, roles: dict | None = None,
                           ) -> dict[str, set[str]]:
    """`{key: every organisation it nests in}`. The parent is the nearest of
    these; the rest matter to what a relation into the tree implies — two
    constants over the whole sheet, the unit and the company, nest neither in
    the other and are both above every platoon, so a person's relation to
    either is implied by the one to the section.
    """
    _parents, ancestors, _notes = _tree(entities, values, cat, floor, profiles, roles)
    return ancestors


def _tree(entities, values, cat, floor, profiles, roles, rekey: bool = False):
    """See `organisation_tree`. With `rekey`, a level organisation whose
    designations are RELATIVE — «1 рота» in every battalion, «1 взвод» in every
    company, so its column names one value of no level above it — is keyed on
    its path: the keys of the nearest level column to its left, then itself. A
    sheet writes its levels in reading order, and a relative designation is
    identified by what it is written under. Done in place on the entity, once,
    left to right, and recorded in the notes; a path-keyed level nests in the
    level its keys extend, with no value test — a prefix IS containment.
    """
    by_id = {p.id: p for p in (profiles or ())}
    orgs = [e for e in entities
            if cat.is_descendant(e["schema"], "Organization")
            and e.get("keys") and not e.get("filter")
            and all(k in by_id and level_column(by_id[k], roles) for k in e["keys"])]
    orgs.sort(key=lambda e: by_id[e["keys"][-1]].index)
    distinct: dict[str, int] = {}
    for e in orgs:
        distinct[e["key"]] = len({str(v).strip() for v in values(e["keys"][-1])
                                  if v is not None and str(v).strip()})
    notes: list[tuple[str, str]] = []

    def value_ancestors(child) -> list[tuple[int, int, int, str, int]]:
        """The levels a single-keyed level nests in by value — read on the
        other's LAST key column, so a path-keyed level counts too."""
        found = []
        for other in orgs:
            if other is child or len(child["keys"]) != 1:
                continue
            got = nested(values, child["keys"][0], other["keys"][-1], floor)
            if got is None:
                continue
            consistent, rows = got
            found.append((distinct.get(other["key"], 0), consistent,
                          by_id[other["keys"][-1]].index, other["key"], rows))
        return found

    if rekey:
        ranked = [(level_rank(by_id[e["keys"][-1]], roles), e) for e in orgs]
        ranked = [(r, e) for r, e in ranked if r is not None]
        ranked.sort(key=lambda t: (t[0], by_id[t[1]["keys"][-1]].index))
        for rank, child in ranked:
            if len(child["keys"]) != 1 or distinct.get(child["key"], 0) < 2:
                continue
            coarser = [(r, e) for r, e in ranked if r < rank and e is not child]
            if not coarser:
                continue
            nearest = max(coarser, key=lambda t: (t[0], -by_id[t[1]["keys"][-1]].index))[1]
            if nested(values, child["keys"][0], nearest["keys"][-1], floor) is not None:
                continue
            path = [*nearest["keys"], child["keys"][0]]
            child["keys"] = path
            notes.append((child["key"],
                          f"keyed on its path {path}: «{(by_id[path[-1]].header or '').strip()}» "
                          f"names a finer level than «{(by_id[nearest['keys'][-1]].header or '').strip()}» "
                          f"and its designations recur under it — relative, «1 …» "
                          f"under every {nearest['key']} — so it is identified by "
                          f"what it is written under"))

    parents: dict[str, str] = {}
    ancestors: dict[str, set[str]] = {}
    by_keys = {tuple(e["keys"]): e for e in orgs}
    for child in orgs:
        if len(child["keys"]) > 1:
            above: set[str] = set()
            parent = None
            for n in range(len(child["keys"]) - 1, 0, -1):
                other = by_keys.get(tuple(child["keys"][:n]))
                if other is None:
                    continue
                above.add(other["key"])
                if parent is None:
                    parent = other["key"]
            if parent is None:
                continue
            root = by_keys.get((child["keys"][0],))
            if root is not None:
                above |= ancestors.get(root["key"], set())
            parents[child["key"]] = parent
            ancestors[child["key"]] = above
            notes.append((child["key"], f"nested in {parent}: its key extends "
                          f"{parent}'s, and a prefix is containment"))
            continue
        found = value_ancestors(child)
        if not found:
            continue
        ancestors[child["key"]] = {t[3] for t in found}
        found.sort(key=lambda t: (-t[0], -t[1], t[2]))
        d, consistent, _index, parent, rows = found[0]
        parents[child["key"]] = parent
        notes.append((child["key"],
                      f"nested in {parent}: its column names one value of "
                      f"{parent}'s on {consistent} of the {rows} rows filling "
                      f"both, and {parent} takes {d} distinct values to its "
                      f"{distinct[child['key']]}"))
    return parents, ancestors, notes


def nest_organisations(entities: list[dict], edges: list[dict],
                       attachments: list[dict], values, cat, floor: float,
                       parents: dict[str, str] | None = None, profiles=None,
                       ) -> tuple[list[dict], list[dict], list[tuple[str, str, str]],
                                  dict[str, str]]:
    """The edges that survive the tree, the parent attachments it adds, `(key,
    verdict, reason)` notes, and the tree itself. See the module docstring.
    """
    if parents is None:
        parents, ancestors, tree_notes = _tree(entities, values, cat, floor, profiles,
                                               None, rekey=True)
        notes: list[tuple[str, str, str]] = [(k, "accepted", n) for k, n in tree_notes]
        derive = True
    else:
        _p, ancestors, _n = _tree(entities, values, cat, floor, profiles, None)
        notes = []
        derive = False
    if not parents:
        return list(edges), [], notes, {}
    schema_of = {e["key"]: e["schema"] for e in entities}
    members = set(parents) | set(parents.values()) \
        | {a for above_all in ancestors.values() for a in above_all}

    def above(key: str) -> set[str]:
        return ancestors.get(key, set())
    new_attachments: list[dict] = []
    have = {(a["entity"], a["prop"], a["target"]) for a in attachments}
    for child, parent in (parents.items() if derive else ()):
        qname = f"{schema_of[child]}:parent"
        if cat.prop(qname) is None or (child, qname, parent) in have:
            continue
        if any(a["entity"] == parent and a["target"] == child
               for a in [*attachments, *new_attachments]):
            continue
        have.add((child, qname, parent))
        new_attachments.append({"entity": child, "prop": qname, "target": parent})
        notes.append((child, "accepted",
                      f"{qname} attached to {parent}: a level word heads each "
                      f"column and the values nest the one in the other, and a "
                      f"parent is how containment is said"))

    kept: list[dict] = []
    by_source: dict[str, list[dict]] = {}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src in members and tgt in members and (
                tgt in above(src) or src in above(tgt)):
            notes.append((e.get("key", "?"), "rejected",
                          f"{e['schema']} between {src} and {tgt}, one nested in "
                          f"the other: containment said as a relation, and the "
                          f"parent already says it"))
            continue
        if src not in members and tgt in members:
            by_source.setdefault(src, []).append(e)
            continue
        kept.append(e)
    for src, group in by_source.items():
        targets = {e["target"] for e in group}
        implied = {t for t in targets if any(t in above(o)
                                             for o in targets if o != t)}
        for e in group:
            if e["target"] in implied:
                leaf = next(o for o in targets if e["target"] in above(o))
                notes.append((e.get("key", "?"), "rejected",
                              f"{e['schema']} from {src} to {e['target']} is implied: "
                              f"{src} holds it to {leaf}, which is nested in "
                              f"{e['target']}, and the parent says the rest"))
                continue
            kept.append(e)
    order = {id(e): i for i, e in enumerate(edges)}
    kept.sort(key=lambda e: order[id(e)])
    return kept, new_attachments, notes, parents
