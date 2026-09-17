"""Compile a validated plan into an FtM mapping, and normalize the records it
will read.
"""

from __future__ import annotations

import copy
import csv
import os

import yaml

from ftmap.fsutil import open_private
from ftmap.io.frame import Frame, column_values
from ftmap.normalize.canonical import (MULTIVALUE_JOIN, MULTIVALUE_TYPES,
                                       CanonResult, canonicalize,
                                       column_evidence)
from ftmap.plan.claims import column_order, mergeable
from ftmap.vocab.catalogue import Catalogue

_KEY_LITERAL_DELIM = "\x00"


def _key_literal(plan_key: str) -> str:
    return plan_key + _KEY_LITERAL_DELIM


_KEY_FIELD_SUFFIX = "__key"


def key_field(column: str) -> str:
    """The normalized-CSV field the KEY path reads for `column`."""
    return column + _KEY_FIELD_SUFFIX


UNSAFE_URL_CHARS = "#?$\t\r\n"


class UnsafeWorkspacePath(ValueError):
    """A workspace root the pinned FtM loader cannot be trusted to read."""


def filter_field(column: str) -> str:
    """The field a row filter matches on, and why it is not the value field."""
    return f"{column}__is"


def ensure_mappable_root(path: str) -> str:
    """Refuse a workspace root before inventory, and before any model call."""
    absolute = os.path.abspath(path)
    bad = sorted({c for c in absolute if c in UNSAFE_URL_CHARS})
    if bad:
        raise UnsafeWorkspacePath(
            f"workspace path {absolute!r} contains "
            + ", ".join(repr(c) for c in bad)
            + "; followthemoney's mapping loader truncates or expands these, "
              "so `ftm map` would read a different file than the mapping names")
    return absolute


_COMPOSITE_PREFIX = "__ck_"


def composite_field(plan_key: str) -> str:
    """The normalized-CSV field carrying `plan_key`'s whole composite key."""
    return _COMPOSITE_PREFIX + plan_key


def composite_value(parts: list[str]) -> str:
    """`<len>:<value>|` per part, in order, or empty when every part is."""
    if not any(parts):
        return ""
    return "".join(f"{len(part)}:{part}|" for part in parts)


def key_value(raw: str) -> str:
    """The key form of a raw cell: its own text, with whitespace collapsed."""
    return " ".join(str(raw).split())


def _entity_fields(entity: dict) -> list[str]:
    """The normalized-CSV fields one declared entity's identity reads."""
    return [key_field(c) for c in (entity.get("keys") or [])] or ["_row"]


def key_layout(vplan) -> dict[str, list[str]]:
    """Every declared key — entity AND edge — mapped to its component fields.
    """
    fields = {e["key"]: _entity_fields(e) for e in vplan.entities}
    layout = dict(fields)
    for edge in vplan.edges:
        source = fields.get(edge["source"], ["_row"])
        target = fields.get(edge["target"], ["_row"])
        own = [key_field(c) for c in (edge.get("keys") or [])]
        layout[edge["key"]] = sorted(set(source) | set(target) | set(own))
    return layout


def mapping_keys(fields: list[str], plan_key: str) -> list[str]:
    """What the mapping document's `keys:` says, given the component fields."""
    return fields if len(fields) == 1 else [composite_field(plan_key)]


_column_order = column_order


def _cell(row: list, index: dict, col: str) -> str | None:
    """One source cell as text, or None when it holds nothing."""
    j = index[col]
    raw = row[j] if j < len(row) else None
    if raw is None or not str(raw).strip():
        return None
    return str(raw)


CANON_MEMO_MAX = 4096


def normalize_frame(frame: Frame, vplan, out_csv: str,
                    evidence: dict | None = None) -> list[dict]:
    """Write the record the engine reads, and report what was refused."""
    bound = {b["column"]: b for b in vplan.bindings}
    key_cols: set[str] = set()
    for e in (*vplan.entities, *vplan.edges):
        key_cols.update(e.get("keys", []))
    value_cols = sorted(bound, key=_column_order)
    composites = {plan_key: fields
                  for plan_key, fields in key_layout(vplan).items()
                  if len(fields) > 1}
    keyed_cols = sorted(key_cols, key=_column_order)
    filter_cols = sorted({e["filter"]["column"] for e in vplan.entities
                          if e.get("filter")}, key=_column_order)
    index = {c.id: c.index for c in frame.columns}
    if evidence is None:
        evidence = {col: column_evidence(column_values(frame, col))
                    for col in value_cols}

    rejects: list[dict] = []
    canon_memo: dict[str, dict[str, CanonResult]] = {col: {}
                                                    for col in value_cols}
    with open_private(out_csv) as fh:
        fields = (["_row"] + value_cols + [key_field(c) for c in keyed_cols]
                  + [filter_field(c) for c in filter_cols]
                  + [composite_field(k) for k in sorted(composites)])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(frame.rows):
            out = {"_row": str(frame.row_index[i])}
            for col in value_cols:
                raw = _cell(row, index, col)
                if raw is None:
                    out[col] = ""
                    continue
                binding = bound[col]
                memo = canon_memo[col]
                result = memo.get(raw)
                if result is None:
                    result = canonicalize(raw, binding["type_name"],
                                          evidence[col])
                    if len(memo) < CANON_MEMO_MAX:
                        memo[raw] = result
                if result.ok:
                    out[col] = result.value or ""
                for part, reason in result.refused:
                    rejects.append({
                        "row": frame.row_index[i], "column": col,
                        "prop": binding["prop"],
                        "value": part, "reason": reason,
                        "canonicalizer": result.canonicalizer,
                        "stage": "normalize",
                    })
                if not result.ok:
                    out[col] = ""
                    rejects.append({
                        "row": frame.row_index[i], "column": col,
                        "prop": binding["prop"],
                        "value": raw, "reason": result.reason,
                        "canonicalizer": result.canonicalizer,
                        "stage": "normalize",
                    })
            for col in keyed_cols:
                raw = _cell(row, index, col)
                out[key_field(col)] = "" if raw is None else key_value(raw)
            for col in filter_cols:
                raw = _cell(row, index, col)
                out[filter_field(col)] = "" if raw is None else raw
            for plan_key, parts in composites.items():
                out[composite_field(plan_key)] = composite_value(
                    [out.get(field, "") for field in parts])
            writer.writerow(out)
    return rejects


def _add_source(props: dict[str, dict], binding: dict) -> None:
    """One more column for one property of one entity."""
    name = binding["prop"].split(":", 1)[1]
    source = _binding_source(binding)
    prior = props.get(name)
    info = Catalogue.load().prop(binding["prop"])
    type_name = info.type_name if info else binding.get("type_name")
    if prior is None or not mergeable(type_name) or "entity" in prior:
        props[name] = source
        return
    columns = list(prior.get("columns") or [prior["column"]])
    if source["column"] not in columns:
        columns.append(source["column"])
    merged = {"columns": columns}
    if "split" in prior or "split" in source:
        merged["split"] = MULTIVALUE_JOIN
    props[name] = merged


def _binding_source(binding: dict) -> dict:
    """How the engine should read one column for one property."""
    source = {"column": binding["column"]}
    if binding["type_name"] in MULTIVALUE_TYPES:
        source["split"] = MULTIVALUE_JOIN
    return source


NO_ENTITY_REASON = (
    "no declared entity survived validation, so the plan says nothing about "
    "this source; a mapping with no entity is one followthemoney refuses, and "
    "nothing is invented to stand in for the reading the evidence did not "
    "support")


def unmappable_reason(vplan) -> str | None:
    """Why this plan cannot be compiled, or `None` when it can."""
    if vplan.entities or vplan.edges:
        return None
    return NO_ENTITY_REASON


def compile_mapping(vplan, frame: Frame, csv_path: str) -> dict:
    layout = key_layout(vplan)
    by_entity: dict[str, list[dict]] = {}
    for b in vplan.bindings:
        by_entity.setdefault(b["entity"], []).append(b)

    edges = {e["key"]: e for e in vplan.edges}
    entities: dict[str, dict] = {}

    for e in vplan.entities:
        props: dict[str, dict] = {}
        for b in by_entity.get(e["key"], []):
            _add_source(props, b)
        entities[e["key"]] = {
            "schema": e["schema"],
            "keys": mapping_keys(layout[e["key"]], e["key"]),
            "key_literal": _key_literal(e["key"]),
            "properties": props,
        }

    for a in vplan.attachments:
        entities[a["entity"]]["properties"][a["prop"].split(":", 1)[1]] = \
            {"entity": a["target"]}

    for key, edge in edges.items():
        info = Catalogue.load().edge_index()[edge["schema"]]
        props: dict[str, dict] = {
            info.source_prop: {"entity": edge["source"]},
            info.target_prop: {"entity": edge["target"]},
        }
        for b in by_entity.get(key, []):
            _add_source(props, b)
        entities[key] = {
            "schema": edge["schema"],
            "keys": mapping_keys(layout[key], key),
            "key_literal": _key_literal(key),
            "properties": props,
        }

    return {
        "csv_url": "file://" + os.path.abspath(csv_path),
        "entities": entities,
    }


def compile_queries(vplan, frame: Frame, csv_path: str) -> list[dict]:
    """The mapping's queries: one for the rows every entity sees, one per
    filtered kind.
    """
    base = compile_mapping(vplan, frame, csv_path)
    groups: dict[tuple | None, dict] = {}
    for e in vplan.entities:
        f = e.get("filter")
        groups.setdefault((f["column"], f["value"]) if f else None, {})
        groups[(f["column"], f["value"]) if f else None][e["key"]] = \
            base["entities"][e["key"]]
    for edge in vplan.edges:
        where = [g for g, block in groups.items()
                 if edge["source"] in block and edge["target"] in block]
        if where:
            groups[where[0]][edge["key"]] = base["entities"][edge["key"]]

    filtered = [g for g in groups if g is not None]
    if filtered:
        catalogue = Catalogue.load()

        attachments = vplan.attachments
        by_key = {e["key"]: e for e in vplan.entities}
        base_has_party = any(not e.get("filter") and catalogue.is_party(e["schema"])
                             for e in vplan.entities)
        for e in vplan.entities:
            if e.get("filter") or catalogue.is_party(e["schema"]):
                continue
            if base_has_party:
                props = (groups.get(None) or {}).get(e["key"], {}).get("properties", {})
                for a in attachments:
                    if a["entity"] != e["key"]:
                        continue
                    target = by_key.get(a["target"]) or {}
                    name = a["prop"].split(":", 1)[1]
                    if target.get("filter") and props.get(name) == {"entity": a["target"]}:
                        del props[name]
                continue
            block = (groups.get(None) or {}).pop(e["key"], None)
            if block is None:
                continue
            for g in filtered:
                copy_ = copy.deepcopy(block)
                props = copy_["properties"]
                for a in attachments:
                    if a["entity"] != e["key"]:
                        continue
                    name = a["prop"].split(":", 1)[1]
                    target = by_key.get(a["target"]) or {}
                    tf = target.get("filter")
                    tg = (tf["column"], tf["value"]) if tf else None
                    if tg == g:
                        props[name] = {"entity": a["target"]}
                    elif props.get(name) == {"entity": a["target"]}:
                        del props[name]
                groups[g][e["key"]] = copy_
        if not groups.get(None):
            groups.pop(None, None)

    queries = []
    for group in sorted(groups, key=lambda g: ("", "") if g is None else g):
        query = {"csv_url": base["csv_url"], "entities": groups[group]}
        if group is not None:
            query["filters"] = {filter_field(group[0]): group[1]}
        queries.append(query)
    return queries


def write_mapping(mapping: dict, path: str, dataset: str) -> None:
    """Write the standard FollowTheMoney mapping document."""
    document = {dataset: {"queries": (list(mapping)
                                      if isinstance(mapping, list)
                                      else [mapping])}}
    with open_private(path) as fh:
        yaml.safe_dump(document, fh, allow_unicode=True, sort_keys=True)
