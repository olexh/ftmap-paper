"""Up to four calls per source, plus validate.py's own repair round."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from ftmap.plan.candidates import (HEURISTIC_MODES, assign_property, candidate_pairs,
                                   heuristic_choice, heuristic_structure)
from ftmap.plan.client import ModelError
from ftmap.plan.keys import resolve_entity_keys, resolve_key_columns
from ftmap.plan.prompt import (BINDING_SYSTEM, EDGE_SYSTEM, STRUCTURE_SYSTEM,
                               SPLIT_SYSTEM, binding_prompt, edge_prompt,
                               kind_columns, split_prompt, structure_prompt)
from ftmap.plan.response_schema import (NO_ENTITY, UNMAPPED, binding_schema,
                                        edge_schema, required_specialisations,
                                        split_binding, split_edge, valid_edges,
                                        NO_FILTER, split_schema,
                                        structure_schema, valid_pairs)
from ftmap.plan.subject import resolve_subject_contradiction
from ftmap.vocab.shortlist import shortlist


@dataclass
class Plan:
    subject: str | None
    entities: list[dict]
    edges: list[dict]
    bindings: list[dict]
    shortlists: dict[str, list[str]] = field(default_factory=dict)
    key_renames: list[dict] = field(default_factory=list)
    key_columns_resolved: list[dict] = field(default_factory=list)
    subject_replaced: dict | None = None
    subject_contradictions: list[dict] = field(default_factory=list)
    specialisations: list[dict] = field(default_factory=list)
    absorbed: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    structure_rule: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _bloc_profiles(frame, cfg, filt: dict, live: list) -> list:
    """The live columns re-profiled over ONLY the rows a filter selects."""
    from ftmap.profile.columns import profile_frame

    idx = frame.column(filt["column"]).index
    rows = [r for r in frame.rows
            if (r[idx] if idx < len(r) else None) == filt["value"]]
    keep = {p.id for p in live}
    narrowed = replace(frame, rows=rows,
                       columns=[c for c in frame.columns if c.id in keep])
    masked = {p.id: p for p in profile_frame(narrowed, cfg)}
    return [masked.get(p.id, p) for p in live]


def propose(frame, profiles, cat, cfg, client) -> Plan:
    if not profiles:
        return Plan(subject=None, entities=[], edges=[], bindings=[], shortlists={})

    live = [p for p in profiles if p.filled > 0]
    stray = {p.id: p for p in live
             if not p.header and not p.label
             and p.fill_rate < cfg.stray_column_fill}
    live = [p for p in live if p.id not in stray]
    if not live:
        return Plan(subject=None, entities=[], edges=[], bindings=[
            {"column": p.id, "prop": UNMAPPED, "entity": "none",
             "why": ("the column holds no value" if p.filled == 0 else
                     f"no header, and {p.filled} of {p.count} rows filled: "
                     f"stray cells, not a column the sheet declares")}
            for p in profiles
        ], shortlists={})

    live_ids = [p.id for p in live]
    live_id_set = set(live_ids)
    structure_rule = None
    if cfg.structure_mode == "heuristic":
        structure, structure_rule = heuristic_structure(live, cat, cfg)
    else:
        structure = client.complete(
            STRUCTURE_SYSTEM, structure_prompt(frame, live, cat),
            structure_schema(cat, live_ids),
        )
    subject = structure.get("subject")
    entities, key_renames = resolve_entity_keys(structure.get("entities") or [])

    entities, key_columns_resolved = resolve_key_columns(entities, live)

    _kinds_cache: list = []

    def kinds_of() -> list:
        if not _kinds_cache:
            _kinds_cache.append(kind_columns(frame, live, cfg))
        return _kinds_cache[0]

    absorbed: list[dict] = []
    if not any(e.get("filter") for e in entities):
        for kind in kinds_of():
            values = [v for v, _n in kind.counts]
            if len(values) < 2 or not all(cat.is_concrete(v) for v in values):
                continue
            parties = [e for e in entities
                       if not e.get("filter")
                       and (cat.is_party(e["schema"]) or e["schema"] in values)]
            base = parties[0] if parties else {"key": "row", "schema": values[0],
                                               "keys": []}
            keys = base.get("keys") or []
            qualifying = [p for p in live
                          if p.fill_rate >= cfg.key_fill_floor
                          and p.distinct_ratio >= cfg.bloc_key_distinct
                          and p.id != kind.column]
            if qualifying:
                best_col = max(qualifying,
                               key=lambda p: (p.distinct_ratio, -p.index))
                keys = [best_col.id]
            kept = [e for e in entities if e not in parties]
            taken = {e["key"] for e in kept}
            bloc_keys: list[str] = []
            for i, _v in enumerate(values):
                n = i + 1
                candidate = base["key"] if i == 0 else f"{base['key']}_{n}"
                while candidate in taken:
                    n += 1
                    candidate = f"{base['key']}_{n}"
                taken.add(candidate)
                bloc_keys.append(candidate)
            blocs = [{**base, "key": bloc_keys[i], "schema": v, "keys": keys,
                      "filter": {"column": kind.column, "value": v}}
                     for i, v in enumerate(values)]
            for e in parties:
                if e is base:
                    continue
                into = next((b["key"] for b in blocs if b["schema"] == e["schema"]),
                            blocs[0]["key"])
                absorbed.append({"key": e["key"], "schema": e["schema"],
                                 "into": into})
            if base in entities:
                entities = [b for e in entities
                            for b in (blocs if e is base
                                      else [] if e in parties else [e])]
            else:
                entities = [*blocs, *kept]
            break

    by_keys: dict[tuple, list[dict]] = {}
    for e in entities:
        if e.get("filter") or cat.is_descendant(e["schema"], "Interval"):
            continue
        by_keys.setdefault(tuple(e.get("keys") or ()), []).append(e)
    siblings = [group for keys, group in by_keys.items()
                if keys and len(group) > 1
                and len({g["schema"] for g in group}) > 1]
    kinds = kinds_of() if siblings else []
    if siblings and kinds and cfg.structure_mode != "heuristic":
        choices = [f"{kind.column}={value}"
                   for kind in kinds for value, _ in kind.counts]
        group = [e for sibling in siblings for e in sibling]
        answer = client.complete(
            SPLIT_SYSTEM, split_prompt(group, live, kinds),
            split_schema([e["key"] for e in group], choices),
        )
        selections = {}
        for item in answer.get("selections") or []:
            chosen = item.get("filter")
            if not chosen or chosen == NO_FILTER:
                continue
            column, _, value = str(chosen).partition("=")
            selections[item.get("entity")] = {"column": column, "value": value}
        entities = [{**e, "filter": selections[e["key"]]}
                    if e["key"] in selections else e
                    for e in entities]

    entities, subject_contradictions = resolve_subject_contradiction(
        cat, subject, entities)

    edges: list[dict] = []
    specialisations: list[dict] = []
    combinations = valid_edges(entities, cat) if len(entities) > 1 else []
    if combinations and not subject_contradictions \
            and cfg.structure_mode != "heuristic":
        answer = client.complete(
            EDGE_SYSTEM, edge_prompt(entities, live, combinations),
            edge_schema(cat, entities, combinations),
        )
        for raw in answer.get("edges") or []:
            parts = split_edge(raw.get("edge"))
            if parts is None:
                continue
            schema, src, tgt = parts
            for key, want in required_specialisations(
                    cat, entities, schema, src, tgt).items():
                for e in entities:
                    if e["key"] == key and e["schema"] != want:
                        specialisations.append(
                            {"key": key, "from": e["schema"], "to": want,
                             "edge": schema})
                        e["schema"] = want
            if any(e["schema"] == schema and e["source"] == src
                   and e["target"] == tgt for e in edges):
                continue
            edges.append({"key": raw.get("key") or schema.lower(),
                          "schema": schema, "source": src, "target": tgt})
    seen_keys = {e["key"] for e in entities}
    for edge in edges:
        if edge["key"] in seen_keys:
            base, n = f"{edge['key']}_edge", 2
            candidate = base
            while candidate in seen_keys:
                candidate, n = f"{base}{n}", n + 1
            edge["key"] = candidate
        seen_keys.add(edge["key"])

    subject_replaced = None
    if subject and entities and cat.is_concrete(subject):
        supported = any(cat.related_to(x["schema"], subject)
                        for x in (*entities, *edges))
        if not supported:
            by_id = {p.id: p for p in live}
            def _key_distinct(e):
                return max((max(by_id[k].distinct_ratio, by_id[k].distinct_ex_modal)
                            for k in e.get("keys") or [] if k in by_id),
                           default=0.0)
            used = max(entities, key=_key_distinct)["schema"]
            subject_replaced = {"declared": subject, "used": used}
            subject = used

    declared: dict[str, str] = {e["key"]: e["schema"] for e in entities}
    declared.update({e["key"]: e["schema"] for e in edges})

    declared_schemas = sorted(set(declared.values()))
    shortlists = {p.id: shortlist(p, cat, cfg, schemas=declared_schemas) for p in live}

    blocs = [e for e in entities if e.get("filter")]
    rounds: list[tuple[list[dict], list[dict], list]] = []
    if blocs:
        row_things = [e for e in entities
                      if not e.get("filter") and not cat.is_party(e["schema"])]
        for bloc in blocs:
            rounds.append(([bloc, *row_things], [],
                           _bloc_profiles(frame, cfg, bloc["filter"], live)))
        base_entities = [e for e in entities
                         if not e.get("filter") and cat.is_party(e["schema"])]
        if base_entities or edges:
            rounds.append((base_entities, edges, live))
    else:
        rounds.append((entities, edges, live))

    answered: dict[tuple[str, str], dict] = {}
    answered_by_column: dict[str, list[dict]] = {}
    unmapped_why: dict[str, str] = {}
    ever_offered: set[str] = set()
    rounds_record: list[dict] = []

    def take(col: str, entity: str, prop: str, why: str) -> None:
        """One answer for one (column, entity), whoever gave it."""
        if prop == UNMAPPED:
            unmapped_why.setdefault(col, why)
            return
        if (col, entity) not in answered:
            claim = {"column": col, "prop": prop, "entity": entity, "why": why}
            answered[(col, entity)] = claim
            answered_by_column.setdefault(col, []).append(claim)

    for round_entities, round_edges, round_profiles in rounds:
        round_declared = {e["key"]: e["schema"] for e in round_entities}
        round_declared.update({e["key"]: e["schema"] for e in round_edges})
        round_pairs = {p.id: candidate_pairs(cfg.binding_mode, shortlists[p.id],
                                             round_declared, cat)
                       for p in round_profiles}
        askable = [p for p in round_profiles if round_pairs[p.id]]
        ever_offered.update(p.id for p in askable)
        bloc = next((e.get("filter") for e in round_entities if e.get("filter")),
                    None)
        rounds_record.append({
            "declared": [[k, v] for k, v in round_declared.items()],
            "filter": bloc["column"] if bloc else None,
            "columns": [p.id for p in round_profiles],
            "askable": [p.id for p in askable],
            "pairs": {p.id: round_pairs[p.id] for p in askable},
        })
        if cfg.binding_mode in HEURISTIC_MODES:
            for p in askable:
                pair, why = heuristic_choice(
                    p, round_pairs[p.id], cat, cfg,
                    first_on_tie=cfg.binding_mode == "heuristic-first")
                entity, prop = split_binding(pair)
                take(p.id, entity, prop, why)
            continue
        queue = [askable[start: start + cfg.chunk_size]
                 for start in range(0, len(askable), cfg.chunk_size)]
        while queue:
            chunk = queue.pop(0)
            ids = [p.id for p in chunk]
            try:
                out = client.complete(
                    BINDING_SYSTEM,
                    binding_prompt(chunk, round_pairs, subject, round_declared,
                                   cat, round_edges, entities=round_entities,
                                   all_profiles=round_profiles),
                    binding_schema(ids, shortlists, round_declared, cat,
                                   round_pairs),
                    max_tokens=max(1, round(cfg.max_tokens * len(chunk) / cfg.chunk_size)),
                )
            except ModelError as err:
                if "exceed_context_size" in str(err) and len(chunk) > 1:
                    mid = len(chunk) // 2
                    queue[:0] = [chunk[:mid], chunk[mid:]]
                    continue
                raise
            for b in out.get("bindings") or []:
                col = b.get("column")
                if col not in ids:
                    continue
                raw = b.get("binding")
                if cfg.binding_mode == "property":
                    raw = assign_property(raw, round_declared, cat)
                entity, prop = split_binding(raw)
                take(col, entity, prop, b.get("why", ""))

    bindings = []
    for p in profiles:
        real = answered_by_column.get(p.id, [])
        if real:
            bindings.extend(real)
        elif p.id in unmapped_why:
            bindings.append({"column": p.id, "prop": UNMAPPED,
                             "entity": NO_ENTITY, "why": unmapped_why[p.id]})
        elif p.id in stray:
            bindings.append({"column": p.id, "prop": UNMAPPED, "entity": NO_ENTITY,
                             "why": f"no header, and {p.filled} of {p.count} rows "
                                    f"filled: stray cells, not a column the sheet "
                                    f"declares"})
        elif p.id not in ever_offered and p.id in live_id_set:
            bindings.append({"column": p.id, "prop": UNMAPPED, "entity": NO_ENTITY,
                             "why": "no declared entity's schema carries any "
                                    "candidate for this column"})
        elif p.filled == 0:
            bindings.append({"column": p.id, "prop": UNMAPPED, "entity": NO_ENTITY,
                             "why": "the column holds no value"})
        else:
            bindings.append({"column": p.id, "prop": UNMAPPED, "entity": NO_ENTITY,
                             "why": "no answer from the model for this column"})

    return Plan(subject=subject, entities=entities, edges=edges,
                bindings=bindings, shortlists=shortlists,
                key_renames=key_renames,
                key_columns_resolved=key_columns_resolved,
                specialisations=specialisations,
                absorbed=absorbed,
                subject_contradictions=subject_contradictions,
                subject_replaced=subject_replaced,
                rounds=rounds_record,
                structure_rule=structure_rule)
