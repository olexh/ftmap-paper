"""One namespace, resolved once."""

from __future__ import annotations


def rename_reason(rename: dict) -> str:
    """The one wording, so `decisions.jsonl` reads the same whichever stage
    resolved the collision."""
    return (f"two entities declared the key {rename['key']!r}; renamed to "
            f"{rename['renamed_to']!r} so both survive instead of one "
            "silently overwriting the other")


def resolve_entity_keys(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return the entities with every key distinct, plus what was renamed."""
    resolved: list[dict] = []
    renames: list[dict] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.get("key")
        if key in seen:
            base, n = key, 2
            candidate = f"{base}_{n}"
            while candidate in seen:
                n += 1
                candidate = f"{base}_{n}"
            renames.append({"key": key, "renamed_to": candidate,
                            "schema": entity.get("schema")})
            entity = {**entity, "key": candidate}
            key = candidate
        seen.add(key)
        resolved.append(entity)
    return resolved, renames


def key_column_reason(resolution: dict) -> str:
    """The one wording, so `decisions.jsonl` reads the same whichever stage
    resolved the reference."""
    return (f"the key column was given as {resolution['given']!r}, which is "
            f"not a column id but is the header of exactly one column; read "
            f"as {resolution['column']!r}")


def resolve_key_columns(entities: list[dict], profiles) -> tuple[list[dict], list[dict]]:
    """Return the entities with every key that names a HEADER rewritten to that
    header's column id, plus what was rewritten.
    """
    known = {p.id for p in profiles}
    by_header: dict[str, list[str]] = {}
    for p in profiles:
        header = (p.header or "").strip().casefold()
        if header:
            by_header.setdefault(header, []).append(p.id)

    resolved: list[dict] = []
    resolutions: list[dict] = []
    for entity in entities:
        keys: list[str] = []
        rewritten = False
        for key in entity.get("keys") or []:
            if key in known:
                keys.append(key)
                continue
            match = by_header.get(str(key).strip().casefold(), [])
            if len(match) == 1:
                resolutions.append({"key": entity.get("key"), "given": key,
                                    "column": match[0]})
                keys.append(match[0])
                rewritten = True
            else:
                keys.append(key)
        resolved.append({**entity, "keys": keys} if rewritten else entity)
    return resolved, resolutions


KEYABLE_TYPES = ("identifier", "name")


def one_to_one(values, columns: list[str], other: str) -> int | None:
    """How many rows fill `other` and every column of `columns` at once, when
    the two sides stand in a one-to-one correspondence over exactly those rows;
    None when they do not.
    """
    side = [values(c) for c in columns]
    a2b: dict[tuple, set] = {}
    b2a: dict[str, set] = {}
    pairs = 0
    for i, v in enumerate(values(other)):
        key = tuple(x[i] for x in side)
        if v in (None, "") or any(x in (None, "") for x in key):
            continue
        pairs += 1
        a2b.setdefault(key, set()).add(str(v))
        b2a.setdefault(str(v), set()).add(key)
    if pairs and all(len(x) == 1 for x in a2b.values()) \
            and all(len(x) == 1 for x in b2a.values()):
        return pairs
    return None


def propose_key(entity: dict, bindings: list[dict], profiles,
                fill_floor: float, by_id: dict | None = None) -> tuple[str, str] | None:
    """A key column for an entity that declared none, or None."""
    mine: set[str] = set()
    first: dict[str, dict] = {}
    for b in bindings:
        if b.get("entity") != entity.get("key"):
            continue
        first.setdefault(b["column"], b)
        if b.get("prop"):
            mine.add(b["column"])
    if by_id is None:
        by_id = {p.id: p for p in profiles}
    best: tuple[int, float, str] | None = None
    for col in sorted(mine):
        p = by_id.get(col)
        if p is None or p.fill_rate < fill_floor:
            continue
        type_name = first[col].get("type_name")
        if type_name not in KEYABLE_TYPES:
            continue
        rank = KEYABLE_TYPES.index(type_name)
        cand = (-rank, p.distinct_ratio, col)
        if best is None or cand > best:
            best = cand
    if best is None:
        return None
    col = best[2]
    p = by_id[col]
    return col, (f"no key was declared, so this entity would be one per row; "
                 f"keyed on the {KEYABLE_TYPES[-best[0]]} column it is bound "
                 f"to with fill {p.fill_rate:.2f} and distinct "
                 f"{p.distinct_ratio:.2f}")


def reinstate_key(entity: dict, soft_rejected: list, bindings: list[dict]):
    """The best key the distinct floor refused, when refusing left none."""
    types = {b["column"]: b.get("type_name") for b in bindings
             if b.get("entity") == entity.get("key")}
    eligible = [(p.distinct_ratio, k, p) for k, p in soft_rejected
                if types.get(k) in KEYABLE_TYPES]
    if not eligible:
        return None
    ratio, col, p = max(eligible)
    return col, (f"reinstated: every declared key was below the distinct "
                 f"floor, and the alternative is one entity per row. This "
                 f"column is bound as a {types[col]} with fill {p.fill_rate:.2f} "
                 f"and distinct {ratio:.2f}, so it identifies something that "
                 f"recurs")


def reinstate_by_dependence(entity: dict, soft_rejected: list,
                            bindings: list[dict], frame):
    """A declared key the distinct floor refused, reinstated because it is
    one-to-one with a name bound on the same entity.
    """
    names = [b["column"] for b in bindings
             if b.get("entity") == entity.get("key")
             and b.get("type_name") == "name"]
    if not names:
        return None
    from ftmap.io.frame import column_values
    read: dict[str, list] = {}

    def values(col: str) -> list:
        got = read.get(col)
        if got is None:
            got = column_values(frame, col)
            read[col] = got
        return got

    for col, p in soft_rejected:
        for name_col in names:
            pairs = one_to_one(values, [col], name_col)
            if pairs is not None:
                return col, (f"reinstated: below the distinct floor "
                             f"({p.distinct_ratio:.2f}), but one-to-one with "
                             f"{name_col}, the name bound on this entity, over "
                             f"{pairs} rows — the register's own identifier for "
                             f"what the name names")
    return None


def key_identification(entity: dict, profiles=None, frame=None) -> float:
    """How well an entity's key identifies rows: the weakest of its key
    columns, each read past its commonest value the way the distinct floor
    reads it, scaled by fill — a code filled on 8 % of the rows identifies 8 %
    of them however distinct its values are. An entity keyed on nothing scores
    0. One reading for four readers: the row entity a relation's counterpart is
    chosen against, the debtor that owes, the relation a record number keys,
    and the subject `validate` and `emit` fall back to when the declared one
    names no Thing — the enforcement register's plan declared the row a `Debt`
    (`work-p2`), and a subject read from binding counts went to the office
    while one read from keys went to the debtor. Read from profiles when given,
    else from the frame's own values.
    """
    keys = list(entity.get("keys") or ())
    if not keys:
        return 0.0
    scores = []
    if profiles is not None:
        by_id = {p.id: p for p in profiles}
        for k in keys:
            p = by_id.get(k)
            if p is not None:
                scores.append(max(p.distinct_ratio, p.distinct_ex_modal)
                              * p.fill_rate)
    elif frame is not None:
        from collections import Counter

        from ftmap.io.frame import column_values
        for k in keys:
            values = list(column_values(frame, k))
            filled = [v for v in values if v]
            if not values:
                continue
            if not filled:
                scores.append(0.0)
                continue
            counts = Counter(filled)
            distinct = len(counts) / len(filled)
            modal = counts.most_common(1)[0][1]
            ex_modal = ((len(counts) - 1) / (len(filled) - modal)
                        if len(filled) > modal else 0.0)
            scores.append(max(distinct, ex_modal) * len(filled) / len(values))
    return min(scores) if scores else 0.0
