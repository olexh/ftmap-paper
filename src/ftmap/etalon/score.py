"""Scoring one run of one source against its etalon, on all four layers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

from ftmap.etalon.document import (PROPERTY, TABLE, Etalon, _related,
                                   VERSION as ETALON_SCHEMA_VERSION)
from ftmap.vocab.catalogue import Catalogue
from ftmap.workspace import SUMMARY

EVALUATOR_VERSION = 6

AGREE = "AGREE"
DEFENSIBLE = "BOTH-DEFENSIBLE"
WRONG = "PIPELINE-WRONG"
KEY_ONLY = "KEY-ONLY"
MISSED = "MISSED"
STRETCHED = "STRETCHED"

SUBJECT_AGREE = "SUBJECT-AGREE"
SUBJECT_ACCEPTABLE = "SUBJECT-ACCEPTABLE"
SUBJECT_WRONG = "SUBJECT-WRONG"

ENTITY_AGREE = "ENTITY-AGREE"
ENTITY_GENERALISED = "ENTITY-GENERALISED"
ENTITY_MISSING = "ENTITY-MISSING"
ENTITY_EMPTY = "ENTITY-EMPTY"
ENTITY_EXTRA = "ENTITY-EXTRA"

INSTANCES_AGREE = "INSTANCES-AGREE"
INSTANCES_OVER = "INSTANCES-OVER"
INSTANCES_UNDER = "INSTANCES-UNDER"

KEYS_AGREE = "KEYS-AGREE"
KEYS_EMPTY = "KEYS-EMPTY"
KEYS_DIFFER = "KEYS-DIFFER"
KEYS_NONE_EMITTED = "KEYS-NONE-EMITTED"

EDGE_AGREE = "EDGE-AGREE"
EDGE_ACCEPTABLE = "EDGE-ACCEPTABLE"
EDGE_ENDPOINTS_WRONG = "EDGE-ENDPOINTS-WRONG"
EDGE_MISSING = "EDGE-MISSING"
EDGE_EXTRA = "EDGE-EXTRA"
EDGE_UNPRODUCIBLE = "EDGE-UNPRODUCIBLE"

ENTITY_UNPRODUCIBLE = "ENTITY-UNPRODUCIBLE"
PRODUCES_NOTHING_AGREE = "PRODUCES-NOTHING-AGREE"
PRODUCES_NOTHING_WRONG = "PRODUCES-NOTHING-WRONG"


@dataclass(frozen=True)
class ColumnRow:
    id: str
    header: str | None
    role: str
    etalon: str | None
    pipeline: str | None
    keyed: bool
    verdict: str
    caveat: str = ""
    invalid: bool = False
    bindings_produced: int = 0
    siblings_acceptable: int = 0
    extra_siblings: int = 0


@dataclass(frozen=True)
class EntityScoreRow:
    """One etalon entity, and what the run paired with it."""

    etalon_key: str | None
    etalon_schema: str | None
    pipeline_id: str | None
    pipeline_schema: str | None
    verdict: str
    etalon_keys: tuple[str, ...] = ()
    pipeline_keys: tuple[str, ...] = ()
    keys_verdict: str | None = None
    etalon_instances: int | None = None
    pipeline_instances: int | None = None
    instances_verdict: str | None = None


@dataclass(frozen=True)
class EdgeRow:
    etalon_key: str | None
    schema: str
    verdict: str
    etalon_endpoints: tuple[str, str] | None = None
    pipeline_endpoints: tuple[str | None, str | None] | None = None


@dataclass(frozen=True)
class Score:
    source: str
    subject_verdict: str
    subject_etalon: str
    subject_declared: str | None
    subject_effective: str | None
    columns: tuple[ColumnRow, ...]
    entities: tuple[EntityScoreRow, ...]
    edges: tuple[EdgeRow, ...]
    subject_effective_verdict: str = ""
    produces_nothing_verdict: str | None = None
    accounting_valid: bool | None = None

    def totals(self) -> dict:
        """The four layers as numbers, each with its own denominator."""
        prop = [r for r in self.columns if r.role in ("property", "key+property")]
        bound = [r for r in prop if r.pipeline]
        right = [r for r in bound if r.verdict in (AGREE, DEFENSIBLE)]
        declined_ok = [r for r in prop
                       if not r.pipeline and r.verdict == DEFENSIBLE]
        homeless = [r for r in self.columns
                    if r.role in ("unmappable", "not-data", "key")]
        stretched = [r for r in homeless if r.verdict == STRETCHED]
        matched = [r for r in self.entities
                   if r.keys_verdict and r.verdict != ENTITY_EMPTY]
        return {
            "versions": {"evaluator": EVALUATOR_VERSION,
                         "etalon_schema": ETALON_SCHEMA_VERSION},
            "accounting_valid": self.accounting_valid,
            "columns": {
                "total": len(self.columns),
                "with_a_property_home": len(prop),
                "bound": len(bound),
                "correct": len(right),
                "wrong": len(bound) - len(right),
                "key_only": sum(1 for r in prop if r.verdict == KEY_ONLY),
                "missed": sum(1 for r in prop if r.verdict == MISSED),
                "declined_defensibly": len(declined_ok),
                "invalid": sum(1 for r in self.columns if r.invalid),
                "acceptable_of_bound": _ratio(len(right), len(bound)),
                "acceptable_of_mappable": _ratio(len(right), len(prop)),
                "bindings_produced": sum(r.bindings_produced
                                         for r in self.columns),
                "extra_sibling_claims": sum(r.extra_siblings
                                            for r in self.columns),
            },
            "no_home": {
                "total": len(homeless),
                "declined": len(homeless) - len(stretched),
                "stretched": len(stretched),
                "specificity": _ratio(len(homeless) - len(stretched),
                                      len(homeless)),
            },
            "entities": {
                "expected": sum(1 for r in self.entities if r.etalon_key
                                and r.verdict != ENTITY_UNPRODUCIBLE),
                "matched": len(matched),
                "missing": sum(1 for r in self.entities
                               if r.verdict == ENTITY_MISSING),
                "empty": sum(1 for r in self.entities
                             if r.verdict == ENTITY_EMPTY),
                "extra": sum(1 for r in self.entities
                             if r.verdict == ENTITY_EXTRA),
                "keys_agree": sum(1 for r in matched
                                  if r.keys_verdict == KEYS_AGREE),
                "keys_empty": sum(1 for r in matched
                                  if r.keys_verdict == KEYS_EMPTY),
                "keys_differ": sum(1 for r in matched
                                   if r.keys_verdict == KEYS_DIFFER),
                "keys_none_emitted": sum(
                    1 for r in matched
                    if r.keys_verdict == KEYS_NONE_EMITTED),
                "unproducible": sum(1 for r in self.entities
                                    if r.verdict == ENTITY_UNPRODUCIBLE),
                "instances_checked": sum(1 for r in self.entities
                                         if r.instances_verdict),
                "instances_agree": sum(1 for r in self.entities
                                       if r.instances_verdict == INSTANCES_AGREE),
                "instances_over": sum(1 for r in self.entities
                                      if r.instances_verdict == INSTANCES_OVER),
                "instances_under": sum(1 for r in self.entities
                                       if r.instances_verdict == INSTANCES_UNDER),
            },
            "edges": {
                "expected": sum(1 for r in self.edges if r.etalon_key
                                and r.verdict != EDGE_UNPRODUCIBLE),
                "found": sum(1 for r in self.edges
                             if r.verdict in (EDGE_AGREE, EDGE_ACCEPTABLE)),
                "acceptable": sum(1 for r in self.edges
                                  if r.verdict == EDGE_ACCEPTABLE),
                "endpoints_wrong": sum(1 for r in self.edges
                                       if r.verdict == EDGE_ENDPOINTS_WRONG),
                "missing": sum(1 for r in self.edges
                               if r.verdict == EDGE_MISSING),
                "extra": sum(1 for r in self.edges
                             if r.verdict == EDGE_EXTRA),
                "unproducible": sum(1 for r in self.edges
                                    if r.verdict == EDGE_UNPRODUCIBLE),
            },
            "subject": self.subject_verdict,
            "subject_effective": self.subject_effective_verdict,
            "produces_nothing": self.produces_nothing_verdict,
        }


def _ratio(num: int, den: int) -> float | None:
    """`None`, not 1.0, when the denominator is empty."""
    return None if den == 0 else round(num / den, 4)


def _label(etalon: Etalon) -> str:
    """A name that identifies the SOURCE, not the file."""
    base = etalon.path or etalon.sha256[:12]
    return f"{base}#{etalon.sheet}" if etalon.sheet else base


def score(etalon: Etalon, summary: dict, cat: Catalogue | None = None,
          tolerance: float | None = None) -> Score:
    cat = cat or Catalogue.load()
    structure = summary.get("structure")
    if structure is None:
        raise ValueError(
            "this summary.json predates `emit.structure` and carries no "
            "entity keys, no edges and no per-column binding, so three of the "
            "four layers cannot be scored. Re-run the source."
        )
    bindings = _bindings_by_column(structure)
    coverage = summary.get("coverage") or {}
    keyed = {cid for e in structure["entities"] for cid in (e.get("keys") or ())}

    claim_index: dict | None = None
    if summary.get("claims") is not None:
        claim_index = {}
        for claim in summary["claims"]:
            for on in claim.get("on") or [None]:
                claim_index[(claim["column"], on, claim["prop"])] = claim

    columns = tuple(_column_row(col, bindings, coverage, keyed, cat,
                                claim_index)
                    for col in etalon.columns)
    emitted = summary.get("entities") or {}
    entities, pairing = _entity_rows(etalon, structure, cat, emitted,
                                     _tolerance(tolerance), bindings)
    edges = _edge_rows(etalon, structure, pairing)

    declared = summary.get("subject_declared")
    effective = summary.get("subject")
    nothing = None
    if etalon.produces_nothing:
        nothing = (PRODUCES_NOTHING_AGREE if not emitted
                   else PRODUCES_NOTHING_WRONG)
    return Score(
        source=_label(etalon),
        subject_verdict=_subject_verdict(etalon, declared),
        subject_effective_verdict=_subject_verdict(etalon, effective),
        produces_nothing_verdict=nothing,
        subject_etalon=etalon.subject,
        subject_declared=declared,
        subject_effective=effective,
        columns=columns, entities=entities, edges=edges,
        accounting_valid=(summary.get("coverage_totals") or {}).get(
            "accounting_valid"),
    )


def _tolerance(given: float | None) -> float:
    """The instance tolerance, resolved in ONE place."""
    if given is not None:
        return given
    from ftmap.config import packaged_config

    return packaged_config().instance_tolerance


def _accepted(col, qname: str, cat: Catalogue) -> bool:
    """Is `qname` one of the readings this column pre-registered as defensible?
    """
    return any(_same_property(qname, a, cat) for a in col.accept)


def _better_than_declining(col, qname: str, cat: Catalogue) -> bool:
    """`accept_binding`, compared like `accept`. Same argument as `_accepted`."""
    return any(_same_property(qname, a, cat) for a in col.accept_binding)


def _same_property(a: str, b: str, cat: Catalogue) -> bool:
    """One property under two spellings."""
    if a == b:
        return True
    (sa, _, na), (sb, _, nb) = a.partition(":"), b.partition(":")
    return na == nb and _related(sa, sb, cat)


def _subject_verdict(etalon: Etalon, schema: str | None) -> str:
    if schema == etalon.subject:
        return SUBJECT_AGREE
    return SUBJECT_ACCEPTABLE if etalon.allows_subject(schema) else SUBJECT_WRONG


def _bindings_by_column(structure: dict) -> dict[str, list[dict]]:
    """`structure.bindings`, each column's value as a LIST of bindings."""
    out: dict[str, list[dict]] = {}
    for col, val in (structure.get("bindings") or {}).items():
        out[col] = list(val) if isinstance(val, list) else [val]
    return out


def _column_row(col, bindings: dict, coverage: dict, keyed: set,
                cat: Catalogue, claim_index: dict | None = None) -> ColumnRow:
    cov = coverage.get(col.id) or {}
    is_key = col.id in keyed

    def _verdict(qname):
        if col.carries_property:
            if qname:
                return AGREE if _same_property(qname, col.answer, cat) else (
                    DEFENSIBLE if _accepted(col, qname, cat) else WRONG)
            return KEY_ONLY if is_key else (
                DEFENSIBLE if col.accept_decline else MISSED)
        if col.role == "key":
            if qname:
                return (DEFENSIBLE
                        if is_key or _better_than_declining(col, qname, cat)
                        else STRETCHED)
            return AGREE if is_key else MISSED
        if not qname:
            return AGREE
        return DEFENSIBLE if col.better_than_declining(qname) else STRETCHED

    rank = {AGREE: 0, DEFENSIBLE: 1, KEY_ONLY: 2, MISSED: 3, STRETCHED: 4,
            WRONG: 5}
    produced = bindings.get(col.id, [])
    candidates = produced or [None]
    judged = [(b, _verdict(b.get("prop") if b else None)) for b in candidates]
    chosen, verdict = min(judged,
                          key=lambda t: rank[t[1]])
    qname = chosen.get("prop") if chosen else None
    acceptable = sum(1 for b, v in judged
                     if b and b.get("prop") and v in (AGREE, DEFENSIBLE))
    extra = sum(1 for b, v in judged
                if b and b.get("prop") and v not in (AGREE, DEFENSIBLE)) - (
                    0 if not qname or verdict in (AGREE, DEFENSIBLE) else 1)
    if claim_index is not None:
        claim = (claim_index.get((col.id, chosen.get("on"), qname))
                 if chosen and qname else None)
        invalid = bool(claim) and claim["emitted"] == 0 and claim["rejected"] > 0
    else:
        invalid = (bool(qname) and cov.get("emitted", 0) == 0
                   and cov.get("rejected", 0) > 0)
    return ColumnRow(id=col.id, header=col.header, role=col.role,
                     etalon=col.answer, pipeline=qname, keyed=is_key,
                     verdict=verdict, caveat=col.caveat, invalid=invalid,
                     bindings_produced=len(produced),
                     siblings_acceptable=acceptable,
                     extra_siblings=max(0, extra))


def _best(pipeline: list, taken: set, want, eligible,
          columns_by_id: dict) -> dict | None:
    """The eligible pipeline entity most likely to BE this one."""
    want_columns = set(want.columns)
    want_keys = set(want.keys)
    best, best_score = None, None
    for i, p in enumerate(pipeline):
        if p["id"] in taken or not eligible(p):
            continue
        mine = columns_by_id.get(p["id"], frozenset())
        keys = set(p.get("keys") or ())
        selector = bool(want.rows and p.get("filter") == want.rows[0]
                        and p["schema"] == want.schema)
        if (not selector and not (mine & want_columns)
                and not (keys & want_keys)):
            continue
        score_ = (selector,
                  bool(keys & want_columns),
                  len(mine & want_columns),
                  p["schema"] == want.schema,
                  len(keys & want_keys),
                  -i)
        if best_score is None or score_ > best_score:
            best, best_score = p, score_
    return best


def _keys_verdict(want, got: tuple, emitted: dict, built: str) -> str:
    """Did the run give this entity the identity the etalon says it has."""
    n = emitted.get(built, 0)
    if n == 0:
        return KEYS_NONE_EMITTED
    if want.scope == TABLE:
        if n == 1:
            return KEYS_AGREE
        return KEYS_EMPTY if not got else KEYS_DIFFER
    if set(got) == set(want.keys):
        return KEYS_AGREE
    if not got and want.keys:
        return KEYS_EMPTY
    return KEYS_DIFFER


def _instances_verdict(want: int | None, got: int | None,
                       tolerance: float | None = None) -> str | None:
    """How many the run made, against how many the source holds."""
    tolerance = _tolerance(tolerance)
    if not want or got is None:
        return None
    if got > want * (1 + tolerance):
        return INSTANCES_OVER
    if got < want * (1 - tolerance):
        return INSTANCES_UNDER
    return INSTANCES_AGREE


def _entity_rows(etalon: Etalon, structure: dict, cat: Catalogue,
                 emitted: dict, tolerance: float | None = None,
                 bindings: dict[str, list[dict]] | None = None):
    """Pair each etalon entity with at most one pipeline entity."""
    pipeline = list(structure.get("entities") or [])
    edge_schemata = cat.edge_schemata()
    edge_pool = [g for g in (structure.get("edges") or []) if g.get("schema")]
    taken: set[str] = set()
    pairing: dict[str, dict] = {}
    rows: list[EntityScoreRow] = []

    def eligible(p):
        return (p["schema"] == want.schema
                or want.allows_related(p["schema"], cat)
                or cat.is_descendant(want.schema, p["schema"]))

    bindings = _bindings_by_column(structure) if bindings is None else bindings

    bound_to: dict[str, set[str]] = {}
    for column, claims in bindings.items():
        for claim in claims:
            on = claim.get("on")
            if on is not None:
                bound_to.setdefault(on, set()).add(column)
    columns_by_id = {p["id"]: bound_to.get(p["id"], set()) | set(p.get("keys") or ())
                     for p in (*pipeline, *edge_pool)}

    for want in etalon.entities:
        if not want.producible:
            rows.append(EntityScoreRow(want.key, want.schema, None, None,
                                       ENTITY_UNPRODUCIBLE))
            continue
        pool = pipeline + (edge_pool if want.schema in edge_schemata else [])
        hit = _best(pool, taken, want, eligible, columns_by_id)
        verdict = (ENTITY_AGREE if hit and hit["schema"] == want.schema
                   else ENTITY_GENERALISED)
        if hit is None:
            rows.append(EntityScoreRow(want.key, want.schema, None, None,
                                       ENTITY_MISSING, want.keys))
            continue
        taken.add(hit["id"])
        pairing[want.key] = hit
        got = tuple(hit.get("keys") or ())
        keys_verdict = _keys_verdict(want, got, emitted, hit["schema"])
        if not any(b.get("on") == hit["id"] and b.get("prop")
                   for bs in bindings.values() for b in bs):
            verdict = ENTITY_EMPTY
        rows.append(EntityScoreRow(want.key, want.schema, hit["id"],
                                   hit["schema"], verdict, want.keys, got,
                                   keys_verdict, want.instances,
                                   hit.get("instances"),
                                   _instances_verdict(want.instances,
                                                      hit.get("instances"),
                                                      tolerance)))

    for p in pipeline:
        if p["id"] not in taken:
            rows.append(EntityScoreRow(
                None, None, p["id"], p["schema"], ENTITY_EXTRA,
                pipeline_keys=tuple(p.get("keys") or ())))
    return tuple(rows), pairing


def _edge_rows(etalon: Etalon, structure: dict, pairing: dict):
    pipeline = list(structure.get("edges") or [])
    taken: set[str] = set()
    rows: list[EdgeRow] = []
    attachments = list(structure.get("attachments") or [])
    a_taken: set[int] = set()
    link_wants: list = []
    unproducible = {e.key for e in etalon.entities if not e.producible}
    for want in etalon.edges:
        if want.source in unproducible or want.target in unproducible:
            rows.append(EdgeRow(want.key, want.prop or want.schema,
                                EDGE_UNPRODUCIBLE, (want.source, want.target)))
            continue
        if want.kind == PROPERTY:
            want_src = (pairing.get(want.source) or {}).get("id")
            want_tgt = (pairing.get(want.target) or {}).get("id")
            name = (want.prop or "").split(":")[-1]
            hit = next((i for i, a in enumerate(attachments)
                        if i not in a_taken
                        and (a.get("prop") or "").split(":")[-1] == name
                        and a.get("on") == want_src
                        and a.get("target") == want_tgt), None)
            if hit is None:
                rows.append(EdgeRow(want.key, want.prop or want.schema,
                                    EDGE_MISSING, (want.source, want.target)))
                continue
            a_taken.add(hit)
            rows.append(EdgeRow(want.key, want.prop or want.schema, EDGE_AGREE,
                                (want.source, want.target),
                                (attachments[hit].get("on"),
                                 attachments[hit].get("target"))))
            continue
        want_src = (pairing.get(want.source) or {}).get("id")
        want_tgt = (pairing.get(want.target) or {}).get("id")
        readings = (want.schema, *want.accept)
        hit = next((g for g in pipeline
                    if g["id"] not in taken and g["schema"] == want.schema
                    and g.get("source") == want_src
                    and g.get("target") == want_tgt), None)
        if hit is None and want.accept:
            hit = next((g for g in pipeline
                        if g["id"] not in taken and g["schema"] in readings
                        and g.get("source") == want_src
                        and g.get("target") == want_tgt), None)
        if hit is None:
            link_wants.append(want)
            continue
        taken.add(hit["id"])
        rows.append(EdgeRow(want.key, hit["schema"],
                            EDGE_AGREE if hit["schema"] == want.schema
                            else EDGE_ACCEPTABLE,
                            (want.source, want.target),
                            (hit.get("source"), hit.get("target"))))
    for want in link_wants:
        readings = (want.schema, *want.accept)
        hit = next((g for g in pipeline if g["id"] not in taken
                    and g["schema"] in readings), None)
        if hit is None:
            rows.append(EdgeRow(want.key, want.schema, EDGE_MISSING,
                                (want.source, want.target)))
            continue
        taken.add(hit["id"])
        rows.append(EdgeRow(want.key, hit["schema"], EDGE_ENDPOINTS_WRONG,
                            (want.source, want.target),
                            (hit.get("source"), hit.get("target"))))
    for g in pipeline:
        if g["id"] not in taken:
            rows.append(EdgeRow(None, g["schema"], EDGE_EXTRA,
                                pipeline_endpoints=(g.get("source"),
                                                    g.get("target"))))
    for i, a in enumerate(attachments):
        if i not in a_taken:
            rows.append(EdgeRow(None, a.get("prop"), EDGE_EXTRA,
                                pipeline_endpoints=(a.get("on"),
                                                    a.get("target"))))
    return tuple(rows)


def find_summary(etalon: Etalon, out_dir: str) -> dict:
    """The summary for the exact bytes this etalon was written against."""
    from ftmap.workspace import open_workspace

    workspace = open_workspace(out_dir)
    index = _summary_index(workspace.root,
                           tuple((safe_id, _generation(workspace, safe_id))
                                 for safe_id in workspace.source_ids()))
    summary = index.get((etalon.sha256, etalon.sheet))
    if summary is not None:
        return summary
    raise FileNotFoundError(
        f"no source in {out_dir} has sha256 {etalon.sha256[:12]}… and sheet "
        f"{etalon.sheet!r}; the etalon is about bytes this run did not read")


def _generation(workspace, safe_id: str) -> str:
    entry = workspace.entry(safe_id)
    return entry.generation if entry else ""


@lru_cache(maxsize=4)
def _summary_index(root: str, generations: tuple[tuple[str, str], ...]
                   ) -> dict[tuple[str, str], dict]:
    """`(sha256, sheet)` to `summary.json`, for one committed generation set.
    """
    from ftmap.workspace import open_workspace

    workspace = open_workspace(root)
    out: dict[tuple[str, str], dict] = {}
    for safe_id, _generation_id in generations:
        path = os.path.join(workspace.source_dir(safe_id), SUMMARY)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        source = summary.get("source") or {}
        key = (source.get("sha256"), source.get("sheet") or "")
        out.setdefault(key, summary)
    return out


def as_markdown(score: Score) -> str:
    """The four layers as one report. Ordered subject, entities, edges,
    columns — the order the pipeline decides them in, which is also the order
    in which an error upstream forces the errors below it."""
    t = score.totals()
    out = [f"### {score.source}", "",
           f"*evaluator v{EVALUATOR_VERSION}, etalon schema "
           f"v{ETALON_SCHEMA_VERSION}*", ""]
    if score.accounting_valid is False:
        out.append("**⚠ THE RUN'S COVERAGE LEDGER DOES NOT CLOSE** — these "
                   "figures may not be published as a measurement until it "
                   "does.")
        out.append("")
    elif score.accounting_valid is None:
        out.append("*Version-1 ledger — UNVOUCHED: this summary predates the "
                   "accounting gate and cannot vouch for itself; not "
                   "publication-eligible without a schema-2 rerun.*")
        out.append("")
    out.append(f"**Subject** — etalon `{score.subject_etalon}`, pipeline "
               f"declared `{score.subject_declared}` "
               f"(**{score.subject_verdict}**), effective "
               f"`{score.subject_effective}` "
               f"(**{score.subject_effective_verdict}**)")
    if score.produces_nothing_verdict:
        out.append("")
        out.append(f"**This table should produce no entity at all** → "
                   f"**{score.produces_nothing_verdict}**")
    out += ["", "| entity | etalon | pipeline | verdict | etalon keys | "
            "pipeline keys | keys |", "|---|---|---|---|---|---|---|"]
    for r in score.entities:
        out.append(f"| {r.etalon_key or '—'} | {r.etalon_schema or '—'} | "
                   f"{r.pipeline_schema or '—'} | {r.verdict} | "
                   f"{', '.join(r.etalon_keys) or '—'} | "
                   f"{', '.join(r.pipeline_keys) or '—'} | "
                   f"{r.keys_verdict or '—'} |")
    out += ["", "| edge | schema | verdict |", "|---|---|---|"]
    if not score.edges:
        out.append("| — | — | no edge expected, none produced |")
    for r in score.edges:
        out.append(f"| {r.etalon_key or '—'} | {r.schema} | {r.verdict} |")
    out += ["", "| # | header | role | etalon | pipeline | verdict |",
            "|---|---|---|---|---|---|"]
    for r in score.columns:
        mark = " ⚠ all values refused" if r.invalid else ""
        if r.caveat:
            mark += f" · *caveat: {r.caveat}*"
        out.append(f"| {r.id} | {r.header or ''} | {r.role} | "
                   f"{r.etalon or '—'} | {r.pipeline or '—'} | "
                   f"{r.verdict}{mark} |")
    c, h = t["columns"], t["no_home"]
    out += ["", f"**Columns** {c['correct']}/{c['bound']} bound columns with "
            f"an acceptable best binding (rate {c['acceptable_of_bound']}), "
            f"{c['correct']}/{c['with_a_property_home']} of mappable "
            f"({c['acceptable_of_mappable']}), key-only {c['key_only']}, "
            f"missed {c['missed']}, declined by prior agreement "
            f"{c['declined_defensibly']}. Binding claims produced "
            f"{c['bindings_produced']}, extra sibling claims "
            f"{c['extra_sibling_claims']} — a best-of column rate, not "
            f"binding precision.",
            f"**No home** {h['declined']}/{h['total']} declined, stretched "
            f"{h['stretched']}.",
            f"**Entities** {t['entities']['matched']}/"
            f"{t['entities']['expected']} matched, unproducible by design "
            f"{t['entities']['unproducible']}, keys agree "
            f"{t['entities']['keys_agree']}, empty "
            f"{t['entities']['keys_empty']}, differ "
            f"{t['entities']['keys_differ']}, none emitted "
            f"{t['entities']['keys_none_emitted']}."
            + (f" Instances {t['entities']['instances_agree']}/"
               f"{t['entities']['instances_checked']} within tolerance, over "
               f"{t['entities']['instances_over']}, under "
               f"{t['entities']['instances_under']}."
               if t['entities']['instances_checked'] else ""),
            f"**Edges** {t['edges']['found']}/{t['edges']['expected']} "
            f"produced"
            + (f" ({t['edges']['acceptable']} under an accepted schema)"
               if t['edges']['acceptable'] else "")
            + f", missing {t['edges']['missing']}, extra "
            f"{t['edges']['extra']}, unproducible by design "
            f"{t['edges']['unproducible']}."]
    return "\n".join(out)
