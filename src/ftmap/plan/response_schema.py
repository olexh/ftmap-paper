"""The grammar the model answers under."""

from __future__ import annotations

from functools import lru_cache

from ftmap.vocab.catalogue import Catalogue

UNMAPPED = "unmapped"
NO_ENTITY = "none"


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


NO_FILTER = "none"


def split_schema(entity_keys: list[str], filters: list[str]) -> dict:
    """Which rows each of these entities is built from."""
    if not entity_keys:
        raise ValueError(
            "split_schema requires at least one entity; a plan with no "
            "polymorphic siblings should never reach the model"
        )
    branches = [_obj({
        "entity": {"const": key},
        "filter": {"type": "string", "enum": [NO_FILTER] + list(filters)},
        "why": {"type": "string"},
    }) for key in entity_keys]
    return _obj({
        "selections": {
            "type": "array",
            "minItems": len(entity_keys),
            "maxItems": len(entity_keys),
            "prefixItems": branches,
        },
    })


def structure_schema(cat: Catalogue, column_ids: list[str]) -> dict:
    """`keys` names columns, so it is a closed enum of the column ids this
    frame has — the reasoning that closed the edge endpoints below, applied to
    the other field that references something by name.
    """
    if not column_ids:
        raise ValueError(
            "structure_schema requires at least one column; a frame with no "
            "live columns should never reach the model"
        )
    schemata = cat.concrete_schemata()
    return _obj({
        "subject": {
            "type": "string",
            "enum": schemata,
            "description": "The FtM schema that one row of this table is mainly about.",
        },
        "entities": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": _obj({
                "key": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,23}$"},
                "schema": {"type": "string", "enum": schemata},
                "keys": {"type": "array", "minItems": 0, "maxItems": 4,
                         "items": {"type": "string", "enum": column_ids}},
            }),
        },
    })


NOT_OFFERED = {"UnknownLink"}


@lru_cache(maxsize=None)
def specialise_for(cat: Catalogue, declared: str, needed: str) -> str | None:
    """The one concrete subschema of `declared` that satisfies `needed`, or
    None.
    """
    if cat.is_descendant(declared, needed):
        return declared
    candidates = [s for s in cat.concrete_schemata()
                  if cat.is_descendant(s, declared) and cat.is_descendant(s, needed)]
    return candidates[0] if len(candidates) == 1 else None


def valid_edges(entities: list[dict], cat: Catalogue) -> list[str]:
    """Every `"Schema|source|target"` the grammar may offer for this plan."""
    schema_of = {e["key"]: e["schema"] for e in entities}
    out: list[str] = []
    for info in cat.edges():
        if info.schema in NOT_OFFERED:
            continue
        for src, src_schema in schema_of.items():
            if specialise_for(cat, src_schema, info.source_range) is None:
                continue
            for tgt, tgt_schema in schema_of.items():
                if src == tgt:
                    continue
                if specialise_for(cat, tgt_schema, info.target_range) is not None:
                    out.append(f"{info.schema}|{src}|{tgt}")
    return out


def required_specialisations(cat: Catalogue, entities: list[dict], schema: str,
                             src: str, tgt: str) -> dict[str, str]:
    """What each endpoint must become for this edge to be legal, if anything.
    """
    info = cat.edge_index().get(schema)
    if info is None:
        return {}
    schema_of = {e["key"]: e["schema"] for e in entities}
    out: dict[str, str] = {}
    for key, needed in ((src, info.source_range), (tgt, info.target_range)):
        declared = schema_of.get(key)
        if declared is None:
            continue
        want = specialise_for(cat, declared, needed)
        if want and want != declared:
            out[key] = want
    return out


def split_edge(raw: str | None) -> tuple[str, str, str] | None:
    """The inverse of `valid_edges`' encoding. `None` for anything malformed,
    so a hand-edited plan or a cached prompt log from before this change reads
    as no edge rather than crashing the run."""
    parts = (raw or "").split("|")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 and all(parts) else None


def edge_schema(cat: Catalogue, entities: list[dict],
                combinations: list[str] | None = None) -> dict:
    """Edges, as one closed choice per edge rather than three open ones."""
    combos = valid_edges(entities, cat) if combinations is None else list(combinations)
    if not combos:
        raise ValueError(
            "edge_schema requires at least one valid combination; a plan whose "
            "entities no FtM edge can relate should be asked for no edges"
        )
    return _obj({
        "edges": {
            "type": "array",
            "maxItems": 6,
            "items": _obj({
                "key": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,23}$"},
                "edge": {"type": "string", "enum": combos,
                         "description": "Schema|source|target, from the list."},
            }),
        },
    })


def valid_pairs(qnames: list[str], declared: dict[str, str], cat: Catalogue) -> list[str]:
    """Every `"entity|prop"` string the grammar may offer for one column."""
    pairs: list[str] = []
    for entity_key, schema in declared.items():
        carried = cat.properties_of(schema)
        seen: set[str] = set()
        for qname in qnames:
            own = carried.get(qname.split(":", 1)[1])
            offer = own.qname if own is not None else qname
            if offer in seen or not cat.bindable(schema, offer):
                continue
            seen.add(offer)
            pairs.append(f"{entity_key}|{offer}")
    return pairs


def split_binding(raw: str | None) -> tuple[str, str]:
    """The inverse of `valid_pairs`' encoding: `"entity|prop"` back to
    `(entity, prop)`.
    """
    if not raw or raw == UNMAPPED or "|" not in raw:
        return NO_ENTITY, UNMAPPED
    entity, _, prop = raw.partition("|")
    return entity, prop


def binding_schema(
    column_ids: list[str],
    shortlists: dict[str, list[str]],
    declared: dict[str, str],
    cat: Catalogue,
    pairs: dict[str, list[str]] | None = None,
) -> dict:
    """One branch per column, positionally, so each column's enum is its own
    candidate list — a closed enum of (entity, property) PAIRS rather than two
    independent enums, and one answer per column rather than an array the model
    may cut short or fill with repeats.
    """
    if not column_ids:
        raise ValueError(
            "binding_schema requires at least one column; a frame with no "
            "columns should never reach the model"
        )
    branches = []
    for cid in column_ids:
        known = pairs.get(cid) if pairs is not None else None
        if known is None:
            known = valid_pairs(shortlists.get(cid, []), declared, cat)
        allowed = list(known) + [UNMAPPED]
        branches.append(_obj({
            "column": {"type": "string", "const": cid},
            "binding": {"type": "string", "enum": allowed},
            "why": {"type": "string", "maxLength": 160},
        }))
    return _obj({
        "bindings": {
            "type": "array",
            "minItems": len(column_ids),
            "maxItems": len(column_ids),
            "prefixItems": branches,
        }
    })
