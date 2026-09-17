"""The gate. The model proposes; this decides."""

from __future__ import annotations

from collections import Counter

from dataclasses import asdict, dataclass, field

from ftmap.io.frame import column_values
from ftmap.normalize.canonical import acceptance
from ftmap.plan.claims import (decline_reason, resolve_claims, share_reason,
                               shared_claims)
from ftmap.plan.keys import (key_column_reason, key_identification, propose_key, reinstate_by_dependence, reinstate_key,
                            rename_reason, resolve_entity_keys,
                            resolve_key_columns)
from ftmap.plan.candidates import HEURISTIC_MODES, assign_property, candidate_pairs
from ftmap.plan.prompt import BINDING_SYSTEM, repair_prompt
from ftmap.plan.units import contradicts as unit_contradiction
from ftmap.plan.response_schema import UNMAPPED, binding_schema, split_binding
from ftmap.plan.blocks import (ADDRESS_KEY, CONTRACT_KEY, address_block_columns,
                              attach_identification_holder, declare_address_block,
                              declare_contract, declare_identification,
                              specialise_coded_party, split_redacted_key)
from ftmap.plan.groups import bind_group_members, merge_group_twins, selection
from ftmap.plan.hierarchy import nest_organisations
from ftmap.plan.roles import (bind_by_detector, bind_edge_properties, declare_places,
                              id_from_key, machinery_header, name_from_key,
                              rehome_varying, spelled_property_beats_relation,
                              consecutive_integers, declare_parties,
                              derive_debts, derive_relations,
                              dissolve_relation_things, key_relation_on_record,
                              role_keyed, split_folded_parties)
from ftmap.plan.subject import (contradiction_reason,
                                resolve_subject_contradiction,
                                unused_reading_reason)
from ftmap.profile.shapes import shape


@dataclass(frozen=True)
class Decision:
    column: str | None
    prop: str | None
    entity: str | None
    verdict: str
    reason: str
    decided_by: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidatedPlan:
    subject: str
    entities: list[dict]
    edges: list[dict]
    bindings: list[dict] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    attachments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "entities": self.entities,
            "edges": self.edges,
            "bindings": self.bindings,
            "attachments": self.attachments,
            "decisions": [d.to_dict() for d in self.decisions],
        }


@dataclass(frozen=True)
class _Check:
    """What one binding check concluded."""

    binding: dict | None
    decision: Decision
    rejected: list[str] = field(default_factory=list)
    strength: float = 0.0
    unit_clash: bool = False
    describes_nothing: bool = False


def _row_ordinal(values: list[str]) -> bool:
    """Whether a column's filled values are consecutive integers in row order —
    the export's own row number, which describes nothing.
    """
    return consecutive_integers(values, 3)


def _twin_columns(frame, profiles, column, floor: float) -> dict[str, str]:
    """`{column id: reason}` for every column that is an earlier column written
    twice: the same header, folded, and the same values on at least `floor` of
    the rows filling both. The regiment's establishment writes «Должность по
    штату СВО» at c3 and «должность по штату сво» at c39, the second a lookup
    of the first with a zero where the lookup failed; the lease register writes
    «Назва» twice for two different things, and those differ in their values
    and both stay. The LATER column is the copy — a sheet writes its data
    before its working columns.
    """
    from ftmap.vocab.shortlist import fold
    out: dict[str, str] = {}
    by_header: dict[str, list] = {}
    for p in sorted(profiles, key=lambda p: p.index):
        if p.filled == 0 or not (p.header or "").strip():
            continue
        by_header.setdefault(fold(p.header), []).append(p)
    for folded, group in by_header.items():
        if len(group) < 2:
            continue
        first = group[0]
        a = column(first.id)
        for later in group[1:]:
            b = column(later.id)
            both = same = 0
            for x, y in zip(a, b):
                xs = str(x).strip() if x is not None else ""
                ys = str(y).strip() if y is not None else ""
                if xs and ys:
                    both += 1
                    same += xs == ys
            if both and same / both >= floor:
                out[later.id] = (f"the header «{(later.header or '').strip()}» is "
                                 f"{first.id}'s written again, and the values agree "
                                 f"on {same} of the {both} rows filling both: a "
                                 f"working copy of {first.id}, not a column of its "
                                 f"own")
    return out


def _record_fill(frame, values: list, profile) -> float:
    """The share of the frame's RECORD rows filling a column — every row that
    is not a sparse furniture row. `profile.fill_rate` counts the section
    rows a layout keeps by default (`drop_furniture = false`), and a key
    column is empty on every one of them by construction."""
    sparse = {f.index for f in frame.furniture if f.kind == "sparse"}
    if not sparse:
        return profile.fill_rate
    total = filled = 0
    for i, v in zip(frame.row_index, values):
        if i in sparse:
            continue
        total += 1
        if v not in (None, "") and str(v).strip():
            filled += 1
    return filled / total if total else profile.fill_rate


def _values_under_modal_key(frame, column: str, entity: dict) -> float | None:
    """How many distinct values `column` takes under the key of `entity` that
    carries the most rows; None when no keyed row holds a value. Not the median
    `_values_per_key` reads for an identifier: the regiment's five units hold
    one stray name each under four keys and 3 303 under the fifth, and the
    median says one. The key most rows sit under is the sheet's own reading of
    what the column varies with.
    """
    from collections import defaultdict

    from ftmap.normalize.canonical import is_sentinel
    try:
        ci = frame.column(column).index
        ks = [frame.column(k).index for k in entity["keys"]]
    except KeyError:
        return None
    f = entity.get("filter")
    fi = frame.column(f["column"]).index if f else None
    groups: dict[tuple, set] = defaultdict(set)
    rows: dict[tuple, int] = defaultdict(int)
    for r in frame.rows:
        if f and (r[fi] if fi < len(r) else None) != f["value"]:
            continue
        key = tuple((r[k] if k < len(r) else None) for k in ks)
        if any(v in (None, "") or is_sentinel(str(v)) for v in key):
            continue
        v = r[ci] if ci < len(r) else None
        if v not in (None, ""):
            groups[key].add(v)
            rows[key] += 1
    if not groups:
        return None
    modal = max(rows, key=rows.get)
    return float(len(groups[modal]))


def _values_per_key(frame, column: str, entity: dict) -> float | None:
    """The median number of distinct values `column` takes under one key of
    `entity`, over the rows its filter selects; None when no keyed row
    holds a value."""
    import statistics
    from collections import defaultdict
    try:
        ci = frame.column(column).index
        ks = [frame.column(k).index for k in entity["keys"]]
    except KeyError:
        return None
    f = entity.get("filter")
    fi = frame.column(f["column"]).index if f else None
    groups: dict[tuple, set] = defaultdict(set)
    for r in frame.rows:
        if f and (r[fi] if fi < len(r) else None) != f["value"]:
            continue
        key = tuple((r[k] if k < len(r) else None) for k in ks)
        if any(v in (None, "") for v in key):
            continue
        v = r[ci] if ci < len(r) else None
        if v not in (None, ""):
            groups[key].add(v)
    if not groups:
        return None
    return float(statistics.median(len(v) for v in groups.values()))


def effective_subject(subject: str | None, entities: list[dict],
                      bindings: list[dict], cat, profiles=None, frame=None) -> str | None:
    """The declared subject when it is a Thing some declared entity is related
    to; otherwise the schema of the bound Thing whose key identifies rows best,
    then of the one carrying the most bindings.
    """

    def related(schema: str) -> bool:
        return bool(subject) and cat.related_to(schema, subject)

    if subject and not cat.is_descendant(subject, "Interval") \
            and any(related(e["schema"]) for e in entities):
        return subject
    by_key = {e["key"]: e for e in entities}
    counts = Counter(b["entity"] for b in bindings
                     if b.get("entity") in by_key
                     and not cat.is_descendant(by_key[b["entity"]]["schema"], "Interval"))
    if not counts:
        return subject
    def identifies(key: str) -> float:
        if profiles is None and frame is None:
            return 0.0
        return key_identification(by_key[key], profiles, frame)

    best = max(counts, key=lambda k: (identifies(k), counts[k]))
    return by_key[best]["schema"]


_effective_subject = effective_subject


def _header(by_id, col: str) -> str | None:
    p = by_id.get(col)
    return getattr(p, "header", None) if p else None


def _check_binding(binding, filled, cat, cfg, header=None) -> _Check:
    """What the checks conclude about one binding. `filled` is the column's
    non-empty values: every caller reads them from `validate`'s own per-column
    memo, which filters once, so the filter is not repeated here."""
    col = binding["column"]
    prop = binding.get("prop")
    entity = binding.get("entity")
    why = binding.get("why", "")

    if prop in (None, UNMAPPED):
        return _Check(None, Decision(col, None, None, "unmapped",
                                     why or "unmapped", "model"))

    info = cat.prop(prop)
    if info is None:
        return _Check(None, Decision(col, prop, entity, "rejected",
                                     f"{prop} is not a property in the ontology",
                                     "rule"))

    clash = unit_contradiction(header, prop)
    if clash:
        return _Check(None, Decision(col, prop, entity, "rejected", clash,
                                     "rule"), unit_clash=True)

    if _row_ordinal(filled):
        return _Check(None, Decision(
            col, prop, entity, "rejected",
            "the column is the row ordinal: consecutive integers in row "
            "order, which number the rows and describe nothing", "rule"),
            describes_nothing=True)
    share, rejected = acceptance(filled, info.type_name)
    floor = cfg.accept_thresholds.get(info.type_name, 0.0)
    if share < floor:
        shapes = sorted({shape(v) for v in rejected})[:3]
        return _Check(
            None,
            Decision(col, prop, entity, "rejected",
                     f"acceptance {share:.2f} below {floor:.2f} "
                     f"for type {info.type_name}; rejected shapes "
                     f"{', '.join(shapes) or 'none'}", "rule"),
            rejected,
        )
    if info.format:
        share, rejected = acceptance(filled, info.type_name, info.format)
        if share < floor:
            shapes = sorted({shape(v) for v in rejected})[:3]
            return _Check(
                None,
                Decision(col, prop, entity, "rejected",
                         f"acceptance {share:.2f} below {floor:.2f} for the "
                         f"{info.format} format {prop} declares on top of "
                         f"type {info.type_name}; rejected shapes "
                         f"{', '.join(shapes) or 'none'}", "rule"),
                rejected,
            )
    return _Check({"column": col, "prop": prop, "entity": entity,
                   "type_name": info.type_name, "why": why},
                  Decision(col, prop, entity, "accepted", why or "accepted",
                           "model"),
                  strength=share * len(filled))


_ORG_NAME_FLOOR = 0.8
_PERSON_NAME_CEILING = 0.2


def _name_kind_check(binding: dict, profile, cat, declared: dict) -> Decision | None:
    prop = binding.get("prop")
    if prop in (None, UNMAPPED) or profile is None or not prop.endswith(":name"):
        return None
    entity = declared.get(binding.get("entity") or "")
    if entity is None or not cat.is_descendant(entity["schema"], "Person"):
        return None
    det = profile.detectors
    person = det.get("proper_name", 0.0)
    org = det.get("org_name", 0.0)
    if org >= _ORG_NAME_FLOOR and person <= _PERSON_NAME_CEILING:
        header = (profile.header or "").strip()
        return Decision(binding["column"], prop, binding.get("entity"), "rejected",
                        f"«{header}» reads as organisation names ({org:.0%}) and not as "
                        f"persons' ({person:.0%}), so it is not the name of a "
                        f"{entity['schema']}", "rule")
    return None


def _structural_check(binding: dict, cat, declared: dict) -> Decision | None:
    """The three checks that read no value: the entity is declared, its schema
    carries the property, and a cell can satisfy that property at all. Returns
    the rejection `Decision`, or `None` to proceed to the evidential check.
    """
    col = binding["column"]
    prop = binding.get("prop")
    entity = binding.get("entity")
    if prop in (None, UNMAPPED):
        return None
    info = cat.prop(prop)
    if info is None:
        return None
    ent = declared.get(entity)
    if ent is None:
        return Decision(col, prop, entity, "rejected",
                        f"entity {entity!r} was not declared", "rule")
    if not cat.carries(ent["schema"], prop):
        return Decision(col, prop, entity, "rejected",
                        f"schema {ent['schema']} does not carry {prop}", "rule")
    if not cat.bindable(ent["schema"], prop):
        return Decision(col, prop, entity, "rejected",
                        f"{prop} is satisfied by a reference to another entity, "
                        f"not by a value from a cell", "rule")
    return None


def validate(plan, profiles, cat, cfg, frame, client=None) -> ValidatedPlan:
    by_id = {p.id: p for p in profiles}
    decisions: list[Decision] = []

    _read: dict[str, list] = {}
    _read_filled: dict[str, list] = {}

    def column(col: str) -> list:
        """Every value of a column, in row order, read once."""
        got = _read.get(col)
        if got is None:
            got = column_values(frame, col)
            _read[col] = got
        return got

    def filled(col: str) -> list:
        """The non-empty values of a column, filtered once. This is what a
        value check reads; `_check_binding` no longer filters for itself."""
        got = _read_filled.get(col)
        if got is None:
            got = [v for v in column(col) if v]
            _read_filled[col] = got
        return got

    refine = cfg.refinement != "none"

    def rule(fn, empty):
        return fn if refine else (lambda *a, **k: empty)

    entity_plans, renames = resolve_entity_keys(plan.entities)
    for rename in (*plan.key_renames, *renames):
        decisions.append(Decision(
            None, rename["schema"], rename["key"], "rejected",
            rename_reason(rename), "rule"))
    entity_plans, key_columns = resolve_key_columns(entity_plans, profiles)
    for res in (*plan.key_columns_resolved, *key_columns):
        decisions.append(Decision(
            res["column"], None, res["key"], "accepted",
            key_column_reason(res), "rule"))

    for sp in getattr(plan, "specialisations", []):
        decisions.append(Decision(
            None, None, sp["key"], "accepted",
            f"specialised from {sp['from']} to {sp['to']} so {sp['edge']} could "
            f"be stated: {sp['from']} cannot satisfy that edge's range and "
            f"{sp['to']} is the only concrete subschema of it that can", "rule"))

    for ab in getattr(plan, "absorbed", []):
        decisions.append(Decision(
            None, ab["schema"], ab["key"], "rejected",
            f"an unfiltered {ab['schema']} is the {ab['schema']} bloc the "
            f"kind column derives, without its row selection; absorbed into "
            f"{ab['into']}", "rule"))

    entity_plans, contradictions = resolve_subject_contradiction(
        cat, plan.subject, entity_plans)
    readings = [*plan.subject_contradictions, *contradictions]
    for repair in readings:
        decisions.append(Decision(
            None, repair["subject"], repair["added"], "accepted",
            contradiction_reason(repair), "rule"))

    if plan.subject_replaced:
        rep = plan.subject_replaced
        decisions.append(Decision(
            None, rep["used"], None, "accepted",
            f"the declared subject {rep['declared']} names no declared "
            f"entity, so the binding prompt used {rep['used']}, the entity "
            "the plan's own key evidence supports", "rule"))

    def override_claimed(model_bindings: list[dict], claimed, reason: str) -> list[dict]:
        """The model's bindings, less the ones a rule has taken the column
        from, with a rejection line each. Two rules override a column this
        way and they differ only in the reason they give."""
        kept_model = []
        for b in model_bindings:
            if b["column"] in claimed and b.get("prop") and b["prop"] != UNMAPPED:
                decisions.append(Decision(
                    b["column"], b.get("prop"), b.get("entity"), "rejected",
                    reason, "rule"))
                continue
            kept_model.append(b)
        return kept_model

    refused_columns: dict[str, str] = {}
    for p_ in profiles:
        word = machinery_header(p_.header, p_.label)
        if word is not None:
            refused_columns[p_.id] = (f"the header carries «{word}», a word for the "
                                      f"sheet's own machinery — a lookup column or an "
                                      f"instruction to whoever edits it; the column "
                                      f"is the sheet's, not the source's")
    for b in plan.bindings:
        if b.get("prop") == UNMAPPED and "stray cells" in (b.get("why") or ""):
            refused_columns.setdefault(b["column"], b["why"])
    for col, reason in _twin_columns(frame, profiles, column, cfg.nesting_floor).items():
        refused_columns.setdefault(col, reason)

    role_parties, role_bindings, role_notes = rule(declare_parties, ([], [], []))(
        entity_plans, plan.bindings, profiles, cat, values=column,
        skip=set(refused_columns))
    entity_plans = [*entity_plans, *role_parties]
    for col, key, note in role_notes:
        decisions.append(Decision(col, None, key or None,
                                  "rejected" if note.startswith("declined") else "accepted",
                                  note, "rule"))

    from ftmap.vocab.shortlist import load_lexicon as _load_lexicon
    place_entities, place_bindings, place_notes = rule(declare_places, ([], [], []))(
        entity_plans, plan.bindings, profiles, cat,
        _load_lexicon(cfg.lexicon_path or None), column,
        declined={b["column"] for b in plan.bindings if b.get("prop") == UNMAPPED},
        skip=set(refused_columns))
    entity_plans = [*entity_plans, *place_entities]
    for col, key, note in place_notes:
        decisions.append(Decision(col, None, key, "accepted", note, "rule"))

    block_entities, block_bindings, block_notes = rule(declare_address_block, ([], [], []))(
        entity_plans, profiles)
    model_bindings = list(plan.bindings)
    if block_entities:
        entity_plans = [*entity_plans, *block_entities]
        model_bindings = override_claimed(
            model_bindings, address_block_columns(profiles),
            "overridden by the address-block rule: the column is a "
            "field of the address block and belongs to its Address, "
            "not to the entity the model bound it on")
        for col, key, note in block_notes:
            decisions.append(Decision(col, None, key,
                                      "rejected" if note.startswith("declined") else "accepted",
                                      note, "rule"))

    split_entities, rehomed, split_notes = rule(split_folded_parties, ([], [], []))(
        entity_plans, model_bindings, profiles, frame, cat)
    if split_entities:
        entity_plans = [*entity_plans, *split_entities]
        moved_cols = {b["column"] for b in rehomed}
        model_bindings = [b for b in model_bindings
                          if not (b["column"] in moved_cols
                                  and cat.prop(b.get("prop") or "") is not None
                                  and cat.prop(b["prop"]).type_name == "name")]
        model_bindings.extend(rehomed)
        for col, key, note in split_notes:
            decisions.append(Decision(col, None, key, "accepted", note, "rule"))

    contract_entities, contract_notes = rule(declare_contract, ([], []))(entity_plans, profiles)
    entity_plans = [*entity_plans, *contract_entities]
    for col, key, note in contract_notes:
        decisions.append(Decision(col, None, key, "accepted", note, "rule"))

    redacted_entities, redacted_bindings, redacted_notes = rule(split_redacted_key, ([], [], []))(
        entity_plans, model_bindings, profiles, frame, cat)
    entity_plans = [*entity_plans, *redacted_entities]
    model_bindings = [*model_bindings, *redacted_bindings]
    for col, key, note in redacted_notes:
        decisions.append(Decision(col, None, key, "accepted", note, "rule"))
    for col, key, note in rule(specialise_coded_party, [])(entity_plans, profiles):
        decisions.append(Decision(col, None, key, "accepted", note, "rule"))

    ident_entities, ident_bindings, ident_notes = rule(declare_identification, ([], [], []))(
        entity_plans, profiles, cat)
    entity_plans = [*entity_plans, *ident_entities]
    if ident_bindings:
        kept_model = override_claimed(
            model_bindings, {b["column"] for b in ident_bindings},
            "overridden by the certificate rule: the column is the "
            "Identification's number")
        model_bindings = [*kept_model, *ident_bindings]
    for col, key, note in ident_notes:
        decisions.append(Decision(col, None, key,
                                  "rejected" if note.startswith("declined") else "accepted",
                                  note, "rule"))

    model_edges = [dict(e) for e in plan.edges]
    rule_keys = {e["key"] for e in (*role_parties, *block_entities, *split_entities,
                                    *contract_entities, *redacted_entities, *ident_entities)}
    entity_plans, twin_notes = (merge_group_twins if refine
                                else (lambda ep, *a, **k: (ep, [])))(
        entity_plans, [*model_bindings, *role_bindings, *block_bindings], model_edges,
        profiles, role_keyed, lambda e: key_identification(e, profiles), prefer=rule_keys)
    for col, key, note in twin_notes:
        decisions.append(Decision(col, None, key, "accepted", note, "rule"))

    entity_key_set = {e["key"] for e in entity_plans}
    edge_plans: list[dict] = []
    for edge in model_edges:
        key = edge.get("key")
        if key in entity_key_set:
            decisions.append(Decision(
                None, edge.get("schema"), key, "rejected",
                "an edge may not reuse an entity's key; the two share one "
                "namespace and the collision would overwrite the entity", "rule"))
            continue
        if key in {e["key"] for e in edge_plans}:
            decisions.append(Decision(
                None, edge.get("schema"), key, "rejected",
                "two edges declare the same key; compiling would silently drop "
                "one of them", "rule"))
            continue
        edge_plans.append(edge)

    declared = {e["key"]: e for e in entity_plans}
    declared.update({e["key"]: e for e in edge_plans})
    bindings: list[dict] = []
    failures: list[dict] = []
    strengths: dict[str, float] = {}

    for b in [*model_bindings, *role_bindings, *block_bindings, *place_bindings]:
        col = b["column"]
        profile = by_id.get(col)
        if profile is None:
            decisions.append(Decision(col, b.get("prop"), b.get("entity"), "rejected",
                                      "no such column in this frame", "rule"))
            continue
        prop = b.get("prop")
        if col in refused_columns and prop and prop != UNMAPPED:
            decisions.append(Decision(col, prop, b.get("entity"), "rejected",
                                      refused_columns[col], "rule"))
            continue

        structural = _structural_check(b, cat, declared) \
            or _name_kind_check(b, profile, cat, declared)
        if structural is not None:
            decisions.append(structural)
            continue

        got = _check_binding(b, filled(col), cat, cfg, _header(by_id, col))
        decisions.append(got.decision)
        if got.binding is not None:
            bindings.append(got.binding)
            strengths[col] = got.strength
        elif got.describes_nothing:
            refused_columns[col] = got.decision.reason
        elif got.rejected or got.unit_clash:
            if got.unit_clash:
                refused_columns[col] = got.decision.reason
            failures.append({"column": col, "prop": prop,
                             "reason": got.decision.reason,
                             "rejected": got.rejected})

    if failures and client is not None and cfg.binding_mode not in HEURISTIC_MODES:
        cols = [f["column"] for f in failures]
        shortlists = {c: plan.shortlists.get(c, []) for c in cols}
        declared_schema = {k: v["schema"] for k, v in declared.items()}
        pairs = {c: candidate_pairs(cfg.binding_mode, shortlists[c],
                                    declared_schema, cat) for c in cols}
        out = client.complete(BINDING_SYSTEM, repair_prompt(failures),
                              binding_schema(cols, shortlists, declared_schema,
                                             cat, pairs))
        for raw in out.get("bindings") or []:
            col = raw.get("column")
            if col not in cols:
                continue
            chosen = raw.get("binding")
            if cfg.binding_mode == "property":
                chosen = assign_property(chosen, declared_schema, cat)
            entity, prop = split_binding(chosen)
            b = {"column": col, "prop": prop, "entity": entity,
                "why": raw.get("why", "")}
            structural = _structural_check(b, cat, declared) \
                or _name_kind_check(b, by_id.get(col), cat, declared)
            if structural is not None:
                decisions.append(Decision(structural.column, structural.prop,
                                          structural.entity, structural.verdict,
                                          structural.reason, "model-repair"))
                continue
            got = _check_binding(b, filled(col), cat, cfg, _header(by_id, col))
            d = got.decision
            decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                      d.reason, "model-repair"))
            if got.binding is not None:
                bindings.append(got.binding)
                strengths[col] = got.strength

    early_subject = effective_subject(plan.subject, entity_plans, bindings, cat, profiles)
    declared_now = {e["key"]: e for e in entity_plans}
    declared_now.update({e["key"]: e for e in edge_plans})
    for old, cand, note in rule(rehome_varying, [])(
            entity_plans, bindings, profiles, early_subject, cat,
            cfg.identifier_values_per_key,
            lambda col, ent: _values_under_modal_key(frame, col, ent)):
        structural = _structural_check(cand, cat, declared_now)
        if structural is not None:
            decisions.append(structural)
            continue
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            bindings = [b for b in bindings if b is not old]
            decisions.append(Decision(old["column"], old.get("prop"), old.get("entity"),
                                      "rejected", "re-homed to the row's entity: "
                                      "the column varies within this entity's key", "rule"))
            bindings.append(got.binding)
            strengths[cand["column"]] = got.strength

    from ftmap.vocab.shortlist import load_lexicon
    relation_schemata = cat.edge_schemata()
    bound_entities = {b["entity"] for b in bindings if b.get("entity")}
    late_things = [e for e in entity_plans
                   if e["key"] == CONTRACT_KEY
                   or (e["key"] not in bound_entities
                       and e["schema"] not in relation_schemata)]
    if late_things and refine:
        candidates = bind_edge_properties(late_things, entity_plans, bindings, profiles,
                                          cat, load_lexicon(cfg.lexicon_path or None))
        from ftmap.plan.roles import _BARE
        spelled = Counter(c["entity"] for c, _ in candidates
                          if c.get("leftover") not in _BARE)
        for cand, note in candidates:
            cand.pop("leftover", None)
            if cand["entity"] != CONTRACT_KEY and spelled[cand["entity"]] < 2:
                continue
            got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                                 _header(by_id, cand["column"]))
            d = got.decision
            decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                      f"{note}; {d.reason}", "rule"))
            if got.binding is not None:
                bindings.append(got.binding)
                strengths[cand["column"]] = got.strength

    declined_cols = {b["column"] for b in plan.bindings if b.get("prop") == UNMAPPED}
    rejected_cols = {f["column"] for f in failures} | set(refused_columns)
    contested = {k for r in readings for k in (r["key"], r["added"])}
    for cand, note in rule(name_from_key, [])(
            [e for e in entity_plans if e["key"] not in contested],
            bindings, profiles, cat, column,
            declined=declined_cols, rejected=rejected_cols,
            relations={e["key"] for e in edge_plans}):
        replaces = cand.pop("replaces", None)
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            if replaces:
                gone = {(b["column"], b.get("entity"), b.get("prop")) for b in replaces}
                bindings = [b for b in bindings
                            if (b["column"], b.get("entity"), b.get("prop")) not in gone]
                for b in replaces:
                    decisions.append(Decision(
                        b["column"], b.get("prop"), b.get("entity"), "rejected",
                        "overridden: the column is the party's name column, and a "
                        "party's name belongs to the party, never to the link "
                        "between parties", "rule"))
            bindings.append(got.binding)
            strengths[cand["column"]] = got.strength

    for cand, note in rule(spelled_property_beats_relation, [])(
            entity_plans, bindings, profiles, cat,
            load_lexicon(cfg.lexicon_path or None), {e["key"] for e in edge_plans},
            attempted=model_bindings):
        replaces = cand.pop("replaces", None)
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            gone = {(b["column"], b.get("entity"), b.get("prop")) for b in replaces or ()}
            bindings = [b for b in bindings
                        if (b["column"], b.get("entity"), b.get("prop")) not in gone]
            for b in replaces or ():
                decisions.append(Decision(
                    b["column"], b.get("prop"), b.get("entity"), "rejected",
                    "overridden: the header spells the party's own property, "
                    "and that is not the link's", "rule"))
            bindings.append(got.binding)
            strengths[cand["column"]] = got.strength

    for cand, note in rule(id_from_key, [])(
            [e for e in entity_plans if e["key"] not in contested],
            bindings, profiles, cat, column, declined_cols, rejected_cols,
            cfg.key_distinct_floor):
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            bindings.append(got.binding)
            strengths[cand["column"]] = got.strength

    bound_now = {b["column"] for b in bindings}
    for cand, note in rule(bind_group_members, [])(entity_plans, bindings, profiles, cat,
                                         load_lexicon(cfg.lexicon_path or None),
                                         role_keyed, UNMAPPED):
        if cand["column"] in bound_now:
            continue
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            bindings.append(got.binding)
            bound_now.add(cand["column"])
            strengths[cand["column"]] = got.strength

    bound_now = {b["column"] for b in bindings}
    for col, reason in refused_columns.items():
        if col not in bound_now:
            decisions.append(Decision(
                col, None, None, UNMAPPED,
                f"{reason}, and no other property was proposed for it", "rule"))

    floor_subject = effective_subject(plan.subject, entity_plans, bindings, cat, profiles)

    def _subject_related(schema):
        return bool(floor_subject) and (schema == floor_subject
                                        or cat.is_descendant(floor_subject, schema)
                                        or cat.is_descendant(schema, floor_subject))

    subject_answers = (floor_subject is not None
                       and any(_subject_related(o["schema"])
                               for o in entity_plans))

    entities = []
    for e in entity_plans:
        keys = []
        soft_rejected: list[tuple[str, object]] = []
        fill_rejected: list[str] = []
        row_entity = (not subject_answers
                      or (_subject_related(e["schema"])
                          and not role_keyed(e, profiles)))
        for k in e.get("keys", []):
            if k not in by_id:
                decisions.append(Decision(k, None, e["key"], "rejected",
                                          "key column does not exist", "rule"))
                continue
            p = by_id[k]
            if _record_fill(frame, column(k), p) < cfg.key_fill_floor:
                if not row_entity:
                    decisions.append(Decision(
                        k, None, e["key"], "accepted",
                        f"key fill {p.fill_rate:.2f} is below "
                        f"{cfg.key_fill_floor:.2f}, but this entity is not "
                        "the row's: rows that leave the key empty produce no "
                        "instance of it, and the rows that fill it are the "
                        "ones it names", "rule"))
                    keys.append(k)
                    continue
                decisions.append(Decision(k, None, e["key"], "rejected",
                                          f"key fill {p.fill_rate:.2f} below "
                                          f"{cfg.key_fill_floor:.2f}", "rule"))
                fill_rejected.append(k)
                continue
            effective_distinct = max(p.distinct_ratio, p.distinct_ex_modal)
            if row_entity and effective_distinct < cfg.key_distinct_floor:
                soft_rejected.append((k, p))
                decisions.append(Decision(
                    k, None, e["key"], "rejected",
                    f"key distinct {effective_distinct:.2f} below "
                    f"{cfg.key_distinct_floor:.2f}"
                    + (f" (raw {p.distinct_ratio:.2f}; its commonest value is "
                       f"{p.modal_share:.0%} of the column)"
                       if p.distinct_ex_modal > p.distinct_ratio else ""), "rule"))
                continue
            decisions.append(Decision(
                k, None, e["key"], "accepted",
                f"key fill {p.fill_rate:.2f} and distinct {p.distinct_ratio:.2f} "
                "both clear the floor", "rule"))
            keys.append(k)
        if refine and not keys and soft_rejected:
            reinstated = (reinstate_key(e, soft_rejected, bindings)
                          or reinstate_by_dependence(e, soft_rejected, bindings,
                                                     frame))
            if reinstated is not None:
                col, reason = reinstated
                decisions.append(Decision(col, None, e["key"], "accepted",
                                          reason, "rule"))
                keys.append(col)
        if refine and not keys and fill_rejected and not soft_rejected:
            proposed = propose_key(e, bindings, profiles, cfg.key_fill_floor, by_id)
            if proposed is not None:
                col, _reason = proposed
                decisions.append(Decision(
                    col, None, e["key"], "accepted",
                    f"the declared key {', '.join(fill_rejected)} is too sparse to "
                    f"identify every row, and {col}, bound to this entity, fills "
                    f"the rows it leaves; keyed on it rather than on the row "
                    f"ordinal", "rule"))
                keys.append(col)
        if refine and not keys and not e.get("keys"):
            proposed = propose_key(e, bindings, profiles, cfg.key_fill_floor, by_id)
            if proposed is not None:
                col, reason = proposed
                decisions.append(Decision(col, None, e["key"], "accepted",
                                          reason, "rule"))
                keys.append(col)
        if e.get("keys") and not keys:
            decisions.append(Decision(
                None, None, e["key"], "rejected",
                "every declared key was rejected; this entity falls back to the "
                "row ordinal, so it will produce one entity per row", "rule"))
        entities.append({**e, "keys": keys})

    checked = []
    for e in entities:
        f = e.get("filter")
        if not f:
            checked.append(e)
            continue
        col = f.get("column")
        if col not in by_id:
            decisions.append(Decision(
                col, None, e["key"], "rejected",
                "filter column does not exist, so this entity is built from "
                "every row instead of from its own", "rule"))
            checked.append({**e, "filter": None})
            continue
        values = column(col)
        hits = sum(1 for v in values if v == f.get("value"))
        if hits == 0:
            decisions.append(Decision(
                col, None, e["key"], "rejected",
                f"no row holds {f.get('value')!r} in this column, so the "
                "entity would select nothing and emit no instance", "rule"))
            continue
        if hits == len([v for v in values if v is not None]):
            decisions.append(Decision(
                col, None, e["key"], "rejected",
                "every row satisfies this filter, so it restricts nothing and "
                "the entity is built from the whole table", "rule"))
            checked.append({**e, "filter": None})
            continue
        decisions.append(Decision(
            col, None, e["key"], "accepted",
            f"built from the {hits} rows whose {col} is {f.get('value')!r}",
            "rule"))
        checked.append(e)
    entities = checked

    filter_col = {e["key"]: (e.get("filter") or {}).get("column")
                  for e in entities}
    cleared: list[dict] = []
    for b in bindings:
        if b.get("prop") not in (None, UNMAPPED) \
                and b["column"] == filter_col.get(b.get("entity")):
            decisions.append(Decision(
                b["column"], b["prop"], b["entity"], "rejected",
                "this column is the entity's own row selection; inside the "
                "bloc it holds one constant label, and what it says the "
                "filter already says", "rule"))
            continue
        cleared.append(b)
    bindings = cleared

    keyed = {e["key"]: e for e in entities}
    edge_info = cat.edge_index()
    edges = []
    for edge in edge_plans:
        src, tgt = edge.get("source"), edge.get("target")
        missing = [k for k in (src, tgt) if k not in keyed]
        if missing:
            decisions.append(Decision(edge.get("key", "?"), edge.get("schema"), None,
                                      "rejected",
                                      f"edge endpoint {missing[0]!r} was not declared",
                                      "rule"))
            continue
        if selection(keyed[src]) != selection(keyed[tgt]):
            decisions.append(Decision(
                edge.get("key", "?"), edge.get("schema"), None, "rejected",
                "its endpoints are built from different row selections, and "
                "one query can hold only one", "rule"))
            continue
        info = edge_info.get(edge.get("schema"))
        if info is None:
            decisions.append(Decision(edge.get("key", "?"), edge.get("schema"), None,
                                      "rejected", "not an edge schema", "rule"))
            continue
        if not (cat.is_descendant(keyed[src]["schema"], info.source_range)
                and cat.is_descendant(keyed[tgt]["schema"], info.target_range)):
            decisions.append(Decision(
                edge.get("key", "?"), edge.get("schema"), None, "rejected",
                f"endpoint range violated: {info.schema} wants "
                f"{info.source_range}/{info.target_range}", "rule"))
            continue
        decisions.append(Decision(
            edge.get("key", "?"), edge.get("schema"), None, "accepted",
            "endpoints declared and their schemata satisfy the endpoint range",
            "rule"))
        edges.append(edge)

    edges, tree_attachments, tree_notes, tree = (
        nest_organisations if refine
        else (lambda entities_, edges_, *a, **k: (edges_, [], [], None)))(
        entities, edges, [], column, cat, cfg.nesting_floor, profiles=profiles)
    for key, verdict, note in tree_notes:
        decisions.append(Decision(None, None, key, verdict, note, "rule"))

    fan_out = Counter((e["schema"], e["source"]) for e in edges)
    fan_in = Counter((e["schema"], e["target"]) for e in edges)
    surviving_edges = []
    for edge in edges:
        if (fan_out[(edge["schema"], edge["source"])] > 1
                or fan_in[(edge["schema"], edge["target"])] > 1):
            decisions.append(Decision(
                edge.get("key", "?"), edge.get("schema"), None, "rejected",
                f"one of its endpoints holds this relation to every "
                f"counterpart on the row; a {edge['schema']} declared toward "
                "each of them claims nothing about any one of them", "rule"))
            continue
        surviving_edges.append(edge)
    edges = surviving_edges

    if readings:
        known = {e["key"] for e in entities}
        used = {b["entity"] for b in bindings}
        endpoints = {k for e in edges for k in (e.get("source"), e.get("target"))}
        dropped: set[str] = set()
        for repair in readings:
            for key in (repair["key"], repair["added"]):
                if key not in known or key in used or key in endpoints:
                    continue
                dropped.add(key)
                decisions.append(Decision(
                    None, None, key, "rejected",
                    unused_reading_reason(repair, key), "rule"))
        entities = [e for e in entities if e["key"] not in dropped]

    keyed_now = {e["key"]: e for e in entities}
    still: list[dict] = []
    for b in bindings:
        info = cat.prop(b["prop"]) if b.get("prop") else None
        e = keyed_now.get(b.get("entity"))
        if (info is None or info.type_name != "identifier" or e is None
                or not e.get("keys") or b["column"] in e["keys"]):
            still.append(b)
            continue
        spread = _values_per_key(frame, b["column"], e)
        if spread is None or spread <= cfg.identifier_values_per_key:
            still.append(b)
            continue
        decisions.append(Decision(
            b["column"], b["prop"], e["key"], "rejected",
            f"an identifier that identifies has one value per thing; "
            f"{e['key']} is keyed on {', '.join(e['keys'])} and the typical "
            f"key holds {spread:.0f} distinct values of this column, over "
            f"{cfg.identifier_values_per_key} — it varies with something "
            f"else", "rule"))
    bindings = still

    bound_entities = {b.get("entity") for b in bindings if b.get("prop")}
    empty = [e for e in entities if e["key"] not in bound_entities]
    if empty:
        gone = {e["key"] for e in empty}
        for e in empty:
            decisions.append(Decision(
                None, None, e["key"], "rejected",
                "no column is bound to it, so it would emit one property-less "
                "entity per row and assert nothing about any of them", "rule"))
        entities = [e for e in entities if e["key"] not in gone]
        surviving_edges = []
        for edge in edges:
            hit = [k for k in (edge.get("source"), edge.get("target")) if k in gone]
            if hit:
                decisions.append(Decision(
                    edge.get("key", "?"), edge.get("schema"), None, "rejected",
                    f"its endpoint {hit[0]!r} carried no property, so this "
                    "relation would connect an anonymous node", "rule"))
                continue
            surviving_edges.append(edge)
        edges = surviving_edges

    surviving = {e["key"] for e in entities} | {e["key"] for e in edges}
    kept: list[dict] = []
    for b in bindings:
        if b["entity"] in surviving:
            kept.append(b)
        else:
            decisions.append(Decision(
                b["column"], b["prop"], b["entity"], "rejected",
                "the entity or edge this binding targets was dropped", "rule"))

    kept, displaced = resolve_claims(kept, lambda b: strengths.get(b["column"], 0.0))
    for loser, winner in displaced:
        decisions.append(Decision(
            loser["column"], loser["prop"], loser["entity"], "unmapped",
            decline_reason(loser, winner, strengths.get(loser["column"], 0.0),
                           strengths.get(winner["column"], 0.0)), "rule"))
    for b, first in shared_claims(kept):
        decisions.append(Decision(b["column"], b["prop"], b["entity"],
                                  "accepted", share_reason(b, first), "rule"))

    by_key = {e["key"]: e for e in entities}
    replicas: list[dict] = []
    claimed_columns = {(k["entity"], k["column"]) for k in kept}
    claimed_props = {(k["entity"], k["prop"].split(":", 1)[1]) for k in kept}
    for b in (kept if refine else ()):
        ent = by_key.get(b["entity"])
        fcol = (ent.get("filter") or {}).get("column") if ent else None
        if not fcol:
            continue
        name = b["prop"].split(":", 1)[1]
        for sib in entities:
            if (sib["key"] == b["entity"] or not sib.get("filter")
                    or sib["filter"].get("column") != fcol):
                continue
            info = cat.properties_of(sib["schema"]).get(name)
            if info is None or not cat.bindable(sib["schema"], info.qname):
                continue
            if ((sib["key"], b["column"]) in claimed_columns
                    or (sib["key"], name) in claimed_props):
                continue
            replicas.append({**b, "entity": sib["key"], "prop": info.qname,
                             "replica": True,
                             "why": f"replicated from {b['entity']}"})
            claimed_columns.add((sib["key"], b["column"]))
            claimed_props.add((sib["key"], info.qname.split(":", 1)[1]))
            decisions.append(Decision(
                b["column"], info.qname, sib["key"], "accepted",
                f"replicated from {b['entity']}: the column serves every "
                "bloc of the row selection that carries the property",
                "rule"))
    kept = kept + replicas

    attachments: list[dict] = list(tree_attachments)
    _substance: dict[tuple, str | None] = {}

    def _no_substance(holder: dict, g) -> str | None:
        """None when the holder has a value of its own on the group's rows;
        otherwise the reason it does not."""
        memo = (holder["key"], g)
        if memo in _substance:
            return _substance[memo]
        _substance[memo] = answer = _substance_of(holder, g)
        return answer

    def _substance_of(holder: dict, g) -> str | None:
        own = [b["column"] for b in kept
               if b.get("entity") == holder["key"] and b.get("prop")
               and (declared := cat.declared_on(b["prop"])) is not None
               and not cat.is_abstract(declared)]
        if not own:
            return (f"{holder['key']} binds no property declared on a "
                    f"concrete schema, only inherited ones")
        if g is None:
            mask = [True] * len(frame.rows)
        else:
            gi = frame.column(g[0]).index
            mask = [(r[gi] if gi < len(r) else None) == g[1] for r in frame.rows]
        for col in own:
            ci = frame.column(col).index
            if any(m and ci < len(r) and r[ci] not in (None, "")
                   for m, r in zip(mask, frame.rows)):
                return None
        where = ("on any row" if g is None
                 else f"on the rows where {g[0]} is {g[1]!r}")
        return (f"{', '.join(own)} — everything {holder['key']} binds of "
                f"its own — {'is' if len(own) == 1 else 'are'} empty {where}")

    attach_subject = (plan.subject
                      if plan.subject and cat.is_concrete(plan.subject)
                      and not cat.is_descendant(plan.subject, "Interval")
                      else effective_subject(plan.subject, entities, kept, cat, profiles))
    for holder in (entities if refine else ()):
        if cat.is_party(holder["schema"]):
            continue
        for pinfo in cat.properties_of(holder["schema"]).values():
            if (pinfo.type_name != "entity" or not pinfo.matchable
                    or pinfo.deprecated):
                continue
            rng = pinfo.range_schema or "Thing"
            def _protagonist(t, g) -> bool:
                if g is not None:
                    return True
                return (cat.is_party(t["schema"]) and bool(attach_subject)
                        and (t["schema"] == attach_subject
                             or cat.is_descendant(t["schema"], attach_subject)))

            eligible = [t for t in entities
                        if t is not holder
                        and cat.is_descendant(t["schema"], rng)
                        and _protagonist(t, selection(t))]
            if holder.get("filter"):
                target_groups = [selection(holder)]
            else:
                target_groups = sorted({selection(t) for t in eligible},
                                       key=lambda g: ("", "") if g is None else g)
            for g in target_groups:
                candidates = [t for t in eligible if selection(t) == g]
                if len(candidates) != 1:
                    continue
                target = candidates[0]
                empty = _no_substance(holder, g)
                if empty is not None:
                    decisions.append(Decision(
                        None, pinfo.qname, holder["key"], "rejected",
                        f"not attached to {target['key']}: {empty}", "rule"))
                    continue
                holder_keys = holder.get("keys") or ()
                if (len(holder_keys) == 1 and holder_keys[0] in by_id
                        and by_id[holder_keys[0]].distinct == 1
                        and (not target.get("keys")
                             or any(k in by_id and by_id[k].distinct > 1
                                    for k in target["keys"]))):
                    decisions.append(Decision(
                        None, pinfo.qname, holder["key"], "rejected",
                        f"not attached to {target['key']}: {holder['key']} is one "
                        f"thing over every row, keyed on a column of one value, "
                        f"and {target['key']} is a different one on each; one "
                        f"thing is not each row's", "rule"))
                    continue
                if any(a["entity"] == target["key"]
                       and a["target"] == holder["key"] for a in attachments):
                    continue
                attachments.append({"entity": holder["key"],
                                    "prop": pinfo.qname,
                                    "target": target["key"]})
                decisions.append(Decision(
                    None, pinfo.qname, holder["key"], "accepted",
                    f"attached to {target['key']}: the one declared entity "
                    f"{pinfo.qname} can range over in that row selection; a "
                    f"{holder['schema']} about something is about the "
                    "something", "rule"))

    subject_now = effective_subject(plan.subject, entities, kept, cat, profiles)
    role_edges, role_attachments, notes = rule(derive_relations, ([], [], []))(
        entities, edges, attachments, profiles, subject_now, cat,
        bindings=kept)
    for key, note in notes:
        decisions.append(Decision(None, None, key,
                                  "accepted" if note.startswith("derived")
                                  or " attached to " in note else "rejected",
                                  note, "rule"))
    edges = [*edges, *role_edges]
    attachments = [*attachments, *role_attachments]
    if tree:
        edges, _none, pruned, _tree = nest_organisations(
            entities, edges, attachments, column, cat, cfg.nesting_floor,
            parents=tree, profiles=profiles)
        for key, verdict, note in pruned:
            decisions.append(Decision(None, None, key, verdict, note, "rule"))

    debt_edges, debt_notes = rule(derive_debts, ([], []))(entities, edges, profiles,
                                          subject_now, cat)
    for key, note in debt_notes:
        decisions.append(Decision(None, None, key,
                                  "accepted" if note.startswith("derived") else "rejected",
                                  note, "rule"))
    edges = [*edges, *debt_edges]

    record_cands, record_notes = rule(key_relation_on_record, ([], []))(
        edges, entities, kept, profiles, frame, subject_now, cat)
    for key, note in record_notes:
        decisions.append(Decision(None, None, key,
                                  "accepted" if note.startswith("keyed on") else "rejected",
                                  note, "rule"))
    for cand in record_cands:
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"the relation's record number; {d.reason}", "rule"))
        if got.binding is not None:
            kept.append(got.binding)

    subject_now = effective_subject(plan.subject, entities, kept, cat, profiles)
    for cand, note in rule(bind_by_detector, [])(entities, kept, profiles, subject_now,
                                       cat, frame=frame,
                                       relations={e["key"] for e in edges},
                                       skip=set(refused_columns)):
        replaces = cand.pop("replaces", None)
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            if replaces:
                gone = {(b["column"], b.get("entity"), b.get("prop")) for b in replaces}
                kept = [b for b in kept
                        if (b["column"], b.get("entity"), b.get("prop")) not in gone]
                for b in replaces:
                    decisions.append(Decision(
                        b["column"], b.get("prop"), b.get("entity"), "rejected",
                        "overridden: a typed column is the party's, never the "
                        "link's nor a document's mention", "rule"))
            kept.append(got.binding)

    before_dissolve = {e["key"] for e in entities}
    entities, kept, dissolve_notes = (
        dissolve_relation_things if refine
        else (lambda entities_, edges_, kept_, cat_: (entities_, kept_, [])))(
        entities, edges, kept, cat)
    for col, key, note in dissolve_notes:
        decisions.append(Decision(col, None, key, "rejected", note, "rule"))
    dissolved = before_dissolve - {e["key"] for e in entities}
    if dissolved:
        for a in attachments:
            if a["entity"] in dissolved or a["target"] in dissolved:
                decisions.append(Decision(
                    None, a["prop"], a["entity"], "rejected",
                    f"{a['prop']} attached to {a['target']} goes with "
                    f"{a['entity'] if a['entity'] in dissolved else a['target']}, "
                    "which was dissolved into the row's relation; the relation "
                    "states its endpoints itself", "rule"))
        attachments = [a for a in attachments
                       if a["entity"] not in dissolved and a["target"] not in dissolved]

    subject_now = effective_subject(plan.subject, entities, kept, cat, profiles)
    rule_addresses = [e for e in entities
                      if e["key"] == ADDRESS_KEY or e["key"].startswith("place_")]
    for block in (rule_addresses if refine else ()):
        holders = [e for e in entities
                   if e["key"] != block["key"] and subject_now
                   and (e["schema"] == subject_now
                        or cat.is_descendant(e["schema"], subject_now))
                   and not role_keyed(e, profiles)
                   and cat.prop(f"{e['schema']}:addressEntity") is not None]
        if len(holders) == 1:
            holder = holders[0]
            qname = f"{holder['schema']}:addressEntity"
            if any(a["entity"] == holder["key"] and a["target"] == block["key"]
                   for a in attachments):
                continue
            attachments.append({"entity": holder["key"], "prop": qname,
                                "target": block["key"]})
            decisions.append(Decision(
                None, qname, holder["key"], "accepted",
                f"{qname} attached to {block['key']}: the address "
                f"{'block' if block['key'] == ADDRESS_KEY else 'the place rule declared'} "
                f"is the row's person's, and {holder['key']} ({holder['schema']}) is "
                f"the row's entity", "rule"))

    holder_attachments, holder_notes = rule(attach_identification_holder, ([], []))(
        entities, edges, attachments, profiles, subject_now, cat)
    attachments = [*attachments, *holder_attachments]
    for key, note in holder_notes:
        decisions.append(Decision(None, None, key,
                                  "accepted" if " attached to " in note else "rejected",
                                  note, "rule"))

    for cand, note in rule(bind_edge_properties, [])(edges, entities, kept, profiles, cat,
                                           load_lexicon(cfg.lexicon_path or None)):
        replaces = cand.pop("replaces", None)
        cand.pop("leftover", None)
        got = _check_binding(cand, filled(cand["column"]), cat, cfg,
                             _header(by_id, cand["column"]))
        d = got.decision
        decisions.append(Decision(d.column, d.prop, d.entity, d.verdict,
                                  f"{note}; {d.reason}", "rule"))
        if got.binding is not None:
            if replaces is not None:
                kept = [b for b in kept if b["column"] != cand["column"]]
            kept.append(got.binding)

    bound_now = {b.get("entity") for b in kept if b.get("prop")}
    stranded = {e["key"] for e in entities if e["key"] not in bound_now}
    if stranded:
        for e in entities:
            if e["key"] in stranded:
                decisions.append(Decision(
                    None, None, e["key"], "rejected",
                    "no column is bound to it, so it would emit one property-less "
                    "entity per row and assert nothing about any of them", "rule"))
        entities = [e for e in entities if e["key"] not in stranded]
        for edge in edges:
            hit = [k for k in (edge.get("source"), edge.get("target")) if k in stranded]
            if hit:
                decisions.append(Decision(
                    edge.get("key", "?"), edge.get("schema"), None, "rejected",
                    f"its endpoint {hit[0]!r} carried no property, so this "
                    "relation would connect an anonymous node", "rule"))
        gone_edges = {e["key"] for e in edges
                      if e.get("source") in stranded or e.get("target") in stranded}
        edges = [e for e in edges if e["key"] not in gone_edges]
        attachments = [a for a in attachments
                       if a["entity"] not in stranded and a["target"] not in stranded]
        kept = [b for b in kept if b.get("entity") not in gone_edges]

    return ValidatedPlan(subject=plan.subject, entities=entities, edges=edges,
                         bindings=kept, decisions=decisions,
                         attachments=attachments)
