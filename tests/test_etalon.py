# tests/test_etalon.py
"""The gold standard's own correctness.

An etalon is the measuring instrument, so a mistake in one is charged to the
pipeline. Most of what follows is therefore about the LOADER refusing etalons
rather than about scoring: every rejection here is a mistake phase 1 actually
made or could have made, and each one would have shifted a published number.
"""

from __future__ import annotations

import copy
import hashlib
import os

import pytest
import yaml

from ftmap.etalon import EtalonError, load
from ftmap.etalon.document import parse
from ftmap.etalon.score import (AGREE, DEFENSIBLE, EDGE_AGREE, EDGE_EXTRA,
                                EDGE_MISSING, ENTITY_EMPTY,
                                INSTANCES_OVER,
                                as_markdown,
                                EDGE_UNPRODUCIBLE, ENTITY_MISSING, KEY_ONLY,
                                KEYS_AGREE, KEYS_DIFFER, KEYS_EMPTY, KEYS_NONE_EMITTED, MISSED,
                                STRETCHED,
                                SUBJECT_ACCEPTABLE, SUBJECT_AGREE,
                                SUBJECT_WRONG, WRONG, find_summary, score)
from ftmap.vocab.catalogue import Catalogue

CAT = Catalogue.load()
ETALONS = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "docs", "measurements", "etalon")

MINIMAL = {
    "etalon": 2,
    "source": {"path": "x.csv", "sha256": "a" * 64, "sheet": ""},
    "subject": {"answer": "Airplane", "accept": ["Vehicle"]},
    "entities": [
        {"key": "plane", "schema": "Airplane", "keys": ["c0"]},
        {"key": "owner", "schema": "Company", "keys": ["c2"]},
    ],
    "edges": [{"key": "own", "kind": "link", "schema": "Ownership",
               "source": "owner", "target": "plane"}],
    "columns": [
        {"id": "c0", "header": "mark", "role": "key+property",
         "answer": "Airplane:registrationNumber"},
        {"id": "c1", "header": "serial", "role": "property",
         "answer": "Airplane:serialNumber"},
        {"id": "c2", "header": "owner", "role": "key+property",
         "answer": "Company:name"},
        {"id": "c3", "header": "mass", "role": "unmappable"},
        {"id": "c4", "header": "n/n", "role": "not-data"},
    ],
}


def doc(**changes):
    raw = copy.deepcopy(MINIMAL)
    raw.update(changes)
    return parse(raw, CAT)


def refuses(match, **changes):
    with pytest.raises(EtalonError, match=match):
        doc(**changes)


# --------------------------------------------------------------------------
# The loader
# --------------------------------------------------------------------------

def shipped(active_only: bool = False):
    """Every etalon, including the retired ones.

    `retired/` holds the answers for sources that left the measurement corpus.
    They are not scored — no run reads those bytes any more, so `find_summary`
    declines them by content — but they are still gold standards, and an
    ontology change that breaks one is a change that would have broken it while
    it counted. Keeping them loadable is what stops "retired" from quietly
    meaning "unchecked".
    """
    out = []
    for dirpath, _dirs, files in os.walk(ETALONS):
        if active_only and dirpath != ETALONS:
            continue
        out += [os.path.join(dirpath, f) for f in files if f.endswith(".etalon.yaml")]
    return sorted(out)


def test_every_shipped_etalon_loads():
    """Drift in followthemoney or in the loader breaks them here, not in a
    measurement that has already been published."""
    files = shipped()
    assert files, "no etalons found"
    for path in files:
        assert load(path, CAT).columns, path


def test_a_version_1_etalon_is_refused_rather_than_half_read():
    refuses("version 1", **{"etalon": 1})


def test_every_layer_must_be_answered_explicitly():
    """The phase-1 failure in one line: an unasked question read as agreement.
    A file with no `edges` key is not a file claiming there are no edges."""
    raw = copy.deepcopy(MINIMAL)
    del raw["edges"]
    with pytest.raises(EtalonError, match="missing 'edges'"):
        parse(raw, CAT)


def test_an_entity_must_say_what_it_is_keyed_on():
    raw = copy.deepcopy(MINIMAL)
    del raw["entities"][0]["keys"]
    with pytest.raises(EtalonError, match="`keys` is required"):
        parse(raw, CAT)


def test_a_property_the_schema_does_not_carry_is_refused():
    """THE PHASE-1 MISTAKE. `serialNumber` is declared on Airplane and is not
    inherited from Vehicle, so an etalon generalising the subject to Vehicle
    while keeping this answer would score the pipeline against an ontology that
    does not exist."""
    assert CAT.prop("Airplane:serialNumber") is not None
    assert CAT.prop("Vehicle:serialNumber") is None
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1]["answer"] = "Vehicle:serialNumber"
    ents = copy.deepcopy(MINIMAL["entities"])
    ents[0]["schema"] = "Vehicle"
    refuses("does not carry 'serialNumber'", columns=cols, entities=ents,
            subject="Vehicle", edges=[])


def test_an_answer_no_declared_entity_could_hold_is_refused():
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1]["answer"] = "Vessel:imoNumber"
    refuses("no declared entity is a Vessel", columns=cols)


def test_a_column_that_identifies_the_row_must_be_some_entity_s_key():
    ents = copy.deepcopy(MINIMAL["entities"])
    ents[0]["keys"] = []
    refuses("no declared entity keys on it", entities=ents)


def test_a_role_with_no_property_home_may_not_carry_an_answer():
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[3]["answer"] = "Airplane:model"
    refuses("carries no property", columns=cols)


def test_a_relation_followthemoney_does_not_declare_is_refused():
    """`CourtCase:court` is a string. There is no CourtCase-to-PublicBody edge
    in FtM 4.10.1, so an etalon may not expect one and score its absence."""
    refuses("is not an edge schema",
            edges=[{"key": "e", "schema": "CourtCase",
                    "source": "owner", "target": "plane"}])


def test_an_edge_endpoint_out_of_range_is_refused():
    refuses("ranges on",
            edges=[{"key": "own", "schema": "Ownership",
                    "source": "plane", "target": "owner"}])


def test_a_property_edge_must_name_an_entity_typed_property():
    refuses("is not entity-typed",
            edges=[{"key": "op", "kind": "property", "prop": "Airplane:model",
                    "source": "plane", "target": "owner"}])


def test_a_property_edge_is_accepted_and_keeps_its_qname():
    d = doc(edges=[{"key": "op", "kind": "property",
                    "prop": "Airplane:operator",
                    "source": "plane", "target": "owner"}])
    assert d.edges[0].prop == "Airplane:operator"
    assert d.edges[0].schema == "Airplane"


def test_an_etalon_without_a_sha256_is_refused():
    """A gold standard follows bytes, not filenames: a publisher reissuing a
    file under the same name must break the match, not silently re-target it."""
    refuses("sha256 is required", source={"path": "x.csv"})


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def summary(structure, coverage=None, subject="Airplane", entities=None):
    """A summary shaped like one a real run writes.

    `entities` IS THE CENSUS, and it used to be omitted here. A real
    `summary.json` always carries it — `emit.summarise` counts what came out —
    and the scorer now reads it to tell "declared and produced" from "declared
    and produced nothing", so a fixture without one describes a run that
    emitted zero of everything. Defaulted to one instance per declared schema,
    which is the ordinary case; a test about emission passes its own.
    """
    if entities is None:
        entities = {}
        for e in structure.get("entities") or ():
            entities[e["schema"]] = entities.get(e["schema"], 0) + 1
    return {"subject_declared": subject, "subject": subject,
            "structure": structure, "coverage": coverage or {},
            "entities": entities,
            "source": {"sha256": "a" * 64, "sheet": ""}}


PERFECT = {
    "entities": [{"id": "e0", "schema": "Airplane", "keys": ["c0"]},
                 {"id": "e1", "schema": "Company", "keys": ["c2"]}],
    "edges": [{"id": "g0", "schema": "Ownership", "source": "e1",
               "target": "e0"}],
    "bindings": {
        "c0": {"on": "e0", "prop": "Airplane:registrationNumber", "type": "identifier"},
        "c1": {"on": "e0", "prop": "Airplane:serialNumber", "type": "identifier"},
        "c2": {"on": "e1", "prop": "Company:name", "type": "name"},
    },
}


def test_a_list_shaped_binding_scores_like_the_dict_it_extends():
    """`structure.bindings` may value a column with a LIST of bindings —
    `column -> bindings`, the change four polymorphic-table attempts
    established as the real repair. Every archived run writes the dict shape
    and must keep scoring, so the scorer accepts both; a list of one is the
    dict."""
    st = copy.deepcopy(PERFECT)
    st["bindings"] = {c: [b] for c, b in st["bindings"].items()}
    assert score(doc(), summary(st), CAT).totals() == \
        score(doc(), summary(PERFECT), CAT).totals()


def test_a_column_bound_on_several_blocs_scores_its_best_binding():
    """One column of a polymorphic table feeds every bloc that carries the
    property — OFAC's name column belongs to the Person AND the Organization
    AND the Vessel. The verdict is the best binding's, and the entity layer
    sees the column on every entity it is bound to."""
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c2"] = [
        {"on": "e0", "prop": "Airplane:name", "type": "name"},
        {"on": "e1", "prop": "Company:name", "type": "name", "replica": True},
    ]
    s = score(doc(), summary(st), CAT)
    # The etalon binds c2 on the company; the replica is the one that agrees.
    assert s.columns[2].verdict == AGREE
    # THE WRONG SIBLING IS NOT HIDDEN BY THE RIGHT ONE. The column-level
    # verdict is an upper bound; the Airplane claim beside it surfaces as an
    # extra sibling claim, in the row and in the totals — the 2026-08-31
    # review's condition for keeping best-of at the column layer at all.
    assert s.columns[2].bindings_produced == 2
    assert s.columns[2].siblings_acceptable == 1
    assert s.columns[2].extra_siblings == 1
    assert s.totals()["columns"]["extra_sibling_claims"] == 1
    # And the company still pairs and carries its column.
    company = [r for r in s.entities if r.etalon_schema == "Company"][0]
    assert company.verdict != ENTITY_EMPTY


def test_two_acceptable_bloc_bindings_are_both_counted():
    """Two blocs answering one column correctly — the rolled-up case — leave
    no extra sibling: both are acceptable and both are counted as such."""
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c2"] = [
        {"on": "e1", "prop": "Company:name", "type": "name"},
        {"on": "e0", "prop": "LegalEntity:name", "type": "name",
         "replica": True},
    ]
    s = score(doc(), summary(st), CAT)
    assert s.columns[2].verdict == AGREE
    assert s.columns[2].siblings_acceptable == 2
    assert s.columns[2].extra_siblings == 0
    assert s.totals()["columns"]["extra_sibling_claims"] == 0


def test_the_report_identifies_its_evaluator_and_etalon_versions():
    """A score must name the ruler it was taken with: the evaluator changed
    meaning once (best-of printed as precision) without changing name."""
    from ftmap.etalon.score import as_markdown

    s = score(doc(), summary(copy.deepcopy(PERFECT)), CAT)
    t = s.totals()
    # The etalon half is the loader's own VERSION — the `etalon:` field the
    # gold files declare — never a constant that can drift from it.
    from ftmap.etalon.document import VERSION

    assert t["versions"] == {"evaluator": 6, "etalon_schema": VERSION}
    text = as_markdown(s).replace("\n", " ")
    assert "evaluator v6" in text and f"etalon schema v{VERSION}" in text
    # And the column layer says what its rate is NOT.
    assert "not binding precision" in text


def test_a_fully_engine_refused_selected_claim_is_marked_invalid():
    """Schema 2 moved engine refusals to the claim ledger; the `invalid` flag
    must read THE SELECTED binding's claim, not the disposition row (whose
    `rejected` is the canonicalizer's) and not a per-column sum."""
    st = copy.deepcopy(PERFECT)
    smry = summary(st)
    smry["claims"] = [
        {"column": "c2", "on": ["e1"], "prop": "Company:name",
         "type": "name", "canonical_type": "name", "filters": [],
         "expected": 5, "emitted": 0, "rejected": 5,
         "shortfall": 0, "over_emission": 0},
    ]
    s = score(doc(), smry, CAT)
    assert s.columns[2].verdict == AGREE
    assert s.columns[2].invalid is True


def test_an_emitting_sibling_cannot_vouch_for_a_refused_selected_claim():
    """The masking case the follow-up review named: folded per column, a
    sibling's statements made a credited-but-fully-refused binding look
    alive."""
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c2"] = [
        {"on": "e0", "prop": "Airplane:name", "type": "name"},
        {"on": "e1", "prop": "Company:name", "type": "name"},
    ]
    smry = summary(st)
    smry["claims"] = [
        {"column": "c2", "on": ["e0"], "prop": "Airplane:name",
         "type": "name", "canonical_type": "name", "filters": [],
         "expected": 5, "emitted": 5, "rejected": 0,
         "shortfall": 0, "over_emission": 0},
        {"column": "c2", "on": ["e1"], "prop": "Company:name",
         "type": "name", "canonical_type": "name", "filters": [],
         "expected": 5, "emitted": 0, "rejected": 5,
         "shortfall": 0, "over_emission": 0},
    ]
    s = score(doc(), smry, CAT)
    # The etalon's answer is the Company binding; it is credited AND flagged.
    assert s.columns[2].verdict == AGREE
    assert s.columns[2].invalid is True
    # And when the selected claim emits, the flag clears.
    smry["claims"][1]["emitted"], smry["claims"][1]["rejected"] = 5, 0
    assert score(doc(), smry, CAT).columns[2].invalid is False


def test_an_invalid_ledger_is_carried_and_shouted():
    """`accounting_valid: false` on the run's own coverage totals reaches the
    Score and the report — figures over a ledger that does not close are not
    publication figures, however clean the verdicts."""
    from ftmap.etalon.score import as_markdown

    st = copy.deepcopy(PERFECT)
    smry = summary(st)
    smry["coverage_totals"] = {"accounting_valid": False}
    s = score(doc(), smry, CAT)
    assert s.accounting_valid is False
    assert s.totals()["accounting_valid"] is False
    assert "LEDGER DOES NOT CLOSE" in as_markdown(s)
    # An archived version-1 summary predates the flag: unvouched, not broken.
    s_old = score(doc(), summary(copy.deepcopy(PERFECT)), CAT)
    assert s_old.accounting_valid is None
    assert "LEDGER DOES NOT CLOSE" not in as_markdown(s_old)


def test_a_property_edge_scores_against_the_run_s_attachments():
    """`kind: property` edges scored EDGE-UNPRODUCIBLE while nothing could
    emit an entity-typed property. `validate` now derives attachments and
    `compile` writes them as `{"entity": key}`, so the ceiling is gone: a
    property edge is found when an attachment joins the paired entities under
    the same property name, missing when none does, and an attachment no
    etalon asks for is extra — ordinary edge scoring, one row per claim."""
    d = doc(edges=[{"key": "op", "kind": "property",
                    "prop": "Airplane:operator",
                    "source": "plane", "target": "owner"}])
    st = copy.deepcopy(PERFECT)
    st["edges"] = []
    st["attachments"] = [{"prop": "Airplane:operator", "on": "e0",
                          "target": "e1"}]
    s = score(d, summary(st), CAT)
    assert [(r.etalon_key, r.verdict) for r in s.edges] == [("op", EDGE_AGREE)]

    st["attachments"] = []
    s = score(d, summary(st), CAT)
    assert [r.verdict for r in s.edges] == [EDGE_MISSING]

    st["attachments"] = [{"prop": "Airplane:operator", "on": "e0",
                          "target": "e1"},
                         {"prop": "Sanction:entity", "on": "e1",
                          "target": "e0"}]
    s = score(d, summary(st), CAT)
    assert sorted(r.verdict for r in s.edges) == sorted([EDGE_AGREE, EDGE_EXTRA])


def test_a_bloc_pairs_on_its_row_selector_when_no_column_is_its_own():
    """A polymorphic table's shared columns discriminate between its blocs
    not at all — `_owned` strips them as multi-claimant, so an etalon bloc
    often owns nothing beyond the key column the run keyed differently, and
    `work-full-bloc` showed the result: four OFAC blocs ENTITY-MISSING with
    their exact twins sitting EXTRA beside them. What both sides DO declare
    is the selector: the etalon's `rows: {column, value}` and the run's
    `filter` name the same column, and the summary publishes the column
    (never the value — a value is a cell), so selector column plus exact
    schema is the pairing evidence."""
    d = doc(entities=[
        {"key": "plane", "schema": "Airplane", "keys": ["c9"],
         "rows": {"column": "c4", "value": "Airplane"}},
        {"key": "owner", "schema": "Company", "keys": ["c9"],
         "rows": {"column": "c4", "value": "Company"}},
    ], edges=[], columns=[
        {"id": "c9", "header": "id", "role": "key"},
        {"id": "c4", "header": "kind", "role": "unmappable"},
        {"id": "c2", "header": "name", "role": "property",
         "answer": "Airplane:name"},
    ])
    st = {
        "entities": [
            {"id": "e0", "schema": "Airplane", "keys": ["c2"], "filter": "c4",
             "instances": 5},
            {"id": "e1", "schema": "Company", "keys": ["c2"], "filter": "c4",
             "instances": 7},
        ],
        "edges": [],
        "bindings": {"c2": [{"on": "e0", "prop": "Airplane:name",
                             "type": "name"},
                            {"on": "e1", "prop": "Company:name",
                             "type": "name", "replica": True}]},
    }
    s = score(d, summary(st), CAT)
    got = {r.etalon_key: (r.pipeline_id, r.verdict) for r in s.entities
           if r.etalon_key}
    assert got["plane"] == ("e0", "ENTITY-AGREE")
    assert got["owner"] == ("e1", "ENTITY-AGREE")


def test_a_run_that_matches_the_etalon_scores_clean():
    s = score(doc(), summary(PERFECT), CAT)
    assert s.subject_verdict == SUBJECT_AGREE
    assert [r.verdict for r in s.columns] == [AGREE] * 5
    assert [r.keys_verdict for r in s.entities] == [KEYS_AGREE, KEYS_AGREE]
    t = s.totals()
    assert t["columns"]["acceptable_of_bound"] == 1.0 and t["columns"]["acceptable_of_mappable"] == 1.0
    assert t["no_home"]["stretched"] == 0
    assert t["edges"] == {"expected": 1, "found": 1, "acceptable": 0,
                          "endpoints_wrong": 0, "missing": 0, "extra": 0,
                          "unproducible": 0}


def test_an_accepted_alternative_is_defensible_not_agreement():
    """A DIFFERENT property the etalon pre-registered as also right — aircraft
    J3, `Тип/модель` naming both a model and a type. Not the same property
    under another schema's name, which is agreement (see below)."""
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c1"]["prop"] = "Airplane:model"
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1]["accept"] = ["Airplane:model"]
    s = score(doc(columns=cols), summary(st), CAT)
    assert s.columns[1].verdict == DEFENSIBLE
    assert s.totals()["columns"]["acceptable_of_bound"] == 1.0


def test_a_generalised_subject_is_acceptable_and_a_foreign_one_is_not():
    assert score(doc(), summary(PERFECT, subject="Vehicle"),
                 CAT).subject_verdict == SUBJECT_ACCEPTABLE
    assert score(doc(), summary(PERFECT, subject="Audio"),
                 CAT).subject_verdict == SUBJECT_WRONG


def test_a_column_kept_as_identity_and_never_emitted_is_not_a_plain_miss():
    """Aircraft c2 and c3 in phase 2. Keying on the tail number is the right
    structural decision and losing 872 registration numbers is a real loss;
    KEY-ONLY is neither AGREE nor MISSED and is counted on its own line."""
    st = copy.deepcopy(PERFECT)
    del st["bindings"]["c0"]
    s = score(doc(), summary(st), CAT)
    assert s.columns[0].verdict == KEY_ONLY
    assert s.columns[0].keyed is True
    t = s.totals()
    assert t["columns"]["key_only"] == 1 and t["columns"]["missed"] == 0
    # It is not counted as an attempted binding, so it cannot flatter precision.
    assert t["columns"]["bound"] == 2 and t["columns"]["acceptable_of_bound"] == 1.0
    assert t["columns"]["acceptable_of_mappable"] == round(2 / 3, 4)


def test_a_mappable_column_neither_bound_nor_keyed_is_missed():
    st = copy.deepcopy(PERFECT)
    del st["bindings"]["c1"]
    assert score(doc(), summary(st), CAT).columns[1].verdict == MISSED


def test_a_pre_registered_abstention_is_defensible():
    """Courts J2 in prose, `accept_decline` in the file. A judgement made
    before the run and one made after it are not worth the same."""
    st = copy.deepcopy(PERFECT)
    del st["bindings"]["c1"]
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1]["accept_decline"] = True
    s = score(doc(columns=cols), summary(st), CAT)
    assert s.columns[1].verdict == DEFENSIBLE
    # And it does not become a binding the pipeline never made: counting it as
    # one turned the courts register's 0-of-2 into 1-of-3.
    t = s.totals()
    assert t["columns"]["bound"] == 2 and t["columns"]["correct"] == 2
    assert t["columns"]["declined_defensibly"] == 1
    assert t["columns"]["acceptable_of_mappable"] == round(2 / 3, 4)


def test_binding_a_column_with_no_home_is_stretching():
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c3"] = {"on": "e0", "prop": "Airplane:model", "type": "string"}
    s = score(doc(), summary(st), CAT)
    assert s.columns[3].verdict == STRETCHED
    assert s.totals()["no_home"]["stretched"] == 1
    assert s.totals()["no_home"]["specificity"] == 0.5


def test_a_wrong_property_is_wrong_and_costs_precision():
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c1"]["prop"] = "Airplane:model"
    s = score(doc(), summary(st), CAT)
    assert s.columns[1].verdict == WRONG
    assert s.totals()["columns"]["acceptable_of_bound"] == round(2 / 3, 4)


def test_an_entity_with_no_key_is_visible_even_when_its_columns_agree():
    """§7 gap 4. `keys: []` emits one un-deduplicated entity per row — thirty
    copies of one court — while every column bound to it scores AGREE."""
    st = copy.deepcopy(PERFECT)
    st["entities"][1]["keys"] = []
    s = score(doc(), summary(st), CAT)
    assert [r.verdict for r in s.columns] == [AGREE] * 5
    assert s.entities[1].keys_verdict == KEYS_EMPTY
    assert s.totals()["entities"]["keys_empty"] == 1


def test_a_missing_edge_is_visible_even_when_every_column_agrees():
    """§7 gap 2. The procurement source scored 12 AGREE and 59 % precision
    while producing a graph in which nothing was connected to anything."""
    st = copy.deepcopy(PERFECT)
    st["edges"] = []
    s = score(doc(), summary(st), CAT)
    assert [r.verdict for r in s.columns] == [AGREE] * 5
    assert s.edges[0].verdict == EDGE_MISSING
    assert s.totals()["edges"]["missing"] == 1


def test_a_relation_ftm_models_as_a_property_now_counts_and_can_be_missed():
    """This asserted the opposite until 2026-08-31: `bindable` excludes
    entity-typed properties from columns, so `Contract:authority` was a
    ceiling of the design and scored EDGE-UNPRODUCIBLE, out of the
    denominator. `validate` now derives attachments and `compile` emits them,
    so the ceiling is gone — a property edge no attachment answers is a MISS,
    in the denominator, like any other relation the run failed to produce."""
    s = score(doc(edges=[*MINIMAL["edges"],
                         {"key": "op", "kind": "property",
                          "prop": "Airplane:operator",
                          "source": "plane", "target": "owner"}]),
              summary(PERFECT), CAT)
    verdicts = {r.verdict for r in s.edges}
    assert EDGE_UNPRODUCIBLE not in verdicts
    t = s.totals()
    assert t["edges"]["expected"] == 2 and t["edges"]["unproducible"] == 0
    assert t["edges"]["missing"] == 1


def test_an_expected_entity_the_run_never_produced_is_missing():
    st = copy.deepcopy(PERFECT)
    st["entities"] = st["entities"][:1]
    st["edges"] = []
    s = score(doc(), summary(st), CAT)
    assert s.entities[1].verdict == ENTITY_MISSING
    assert s.totals()["entities"]["missing"] == 1


def test_an_exact_schema_match_is_never_consumed_by_a_generalisation():
    """Order matters: a pipeline Organization must not swallow the etalon's
    PublicBody while the pipeline's own PublicBody sits unmatched, which would
    score a correct run as a generalised one."""
    raw = copy.deepcopy(MINIMAL)
    raw["entities"] = [{"key": "body", "schema": "PublicBody",
                        "accept": ["Organization"], "keys": ["c2"]},
                       {"key": "plane", "schema": "Airplane", "keys": ["c0"]}]
    raw["columns"][2]["answer"] = "PublicBody:name"
    raw["edges"] = []
    st = copy.deepcopy(PERFECT)
    st["edges"] = []
    st["entities"] = [{"id": "e0", "schema": "Organization", "keys": ["c2"]},
                      {"id": "e1", "schema": "PublicBody", "keys": ["c2"]},
                      {"id": "e2", "schema": "Airplane", "keys": ["c0"]}]
    st["bindings"]["c2"]["prop"] = "PublicBody:name"
    s = score(parse(raw, CAT), summary(st), CAT)
    assert s.entities[0].pipeline_schema == "PublicBody"


def test_a_binding_whose_every_value_was_refused_is_flagged():
    s = score(doc(), summary(PERFECT, coverage={
        "c1": {"status": "bound", "values": 872, "emitted": 0, "rejected": 872,
               "declined": 0, "unaccounted": 0}}), CAT)
    assert s.columns[1].invalid is True
    assert s.totals()["columns"]["invalid"] == 1


def test_a_summary_without_the_structure_block_cannot_be_scored():
    with pytest.raises(ValueError, match="predates"):
        score(doc(), {"subject_declared": "Airplane", "coverage": {}}, CAT)


def test_an_empty_denominator_reports_nothing_rather_than_perfection():
    raw = copy.deepcopy(MINIMAL)
    raw["columns"] = [{"id": "c0", "header": "x", "role": "key"},
                      {"id": "c2", "header": "y", "role": "key"}]
    raw["edges"] = []
    t = score(parse(raw, CAT), summary(PERFECT), CAT).totals()
    assert t["columns"]["with_a_property_home"] == 0
    assert t["columns"]["acceptable_of_bound"] is None and t["columns"]["acceptable_of_mappable"] is None


def _commit(workspace, safe_id, sha256, structure=PERFECT):
    """One committed source, written the way `run_source` writes one.

    THROUGH THE WORKSPACE, NOT BY HAND. The previous version of this test made
    `<out>/a/summary.json` itself, which is the layout `find_summary` globbed
    for — so the test and the code agreed with each other and neither agreed
    with the pipeline. Generations moved artefacts to
    `sources/<safe_id>/<generation>/` and `ftmap etalon` went on reporting
    every gold standard as "no source in this run", green suite and all.
    """
    import json

    staged = workspace.staging_dir(safe_id)
    with open(os.path.join(staged, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": {"sha256": sha256, "sheet": ""},
                   "structure": structure}, fh)
    workspace.commit_source(safe_id, staged, {})


def test_the_run_is_found_by_content_not_by_path(tmp_path):
    from ftmap.workspace import create_workspace

    workspace = create_workspace(str(tmp_path / "work"))
    _commit(workspace, "a", "b" * 64)
    _commit(workspace, "b", "a" * 64)

    assert find_summary(doc(), str(tmp_path / "work"))["source"]["sha256"] == "a" * 64
    with pytest.raises(FileNotFoundError, match="did not read"):
        find_summary(doc(source={"path": "x", "sha256": "c" * 64}),
                     str(tmp_path / "work"))


def test_a_legacy_flat_run_still_scores(tmp_path):
    """The retained `work*/` evidence predates generations and must stay readable.

    `2026-08-25-etalon-zero-emission.md` §3 is nine such runs. A fix that made
    `find_summary` understand only the new layout would have made that
    comparison unrepeatable to fix a bug that postdates it.
    """
    import json

    out = tmp_path / "legacy"
    (out / "a").mkdir(parents=True)
    (out / "a" / "summary.json").write_text(json.dumps(
        {"source": {"sha256": "a" * 64, "sheet": ""}, "structure": PERFECT}))

    assert find_summary(doc(), str(out))["source"]["sha256"] == "a" * 64


def test_the_current_generation_is_scored_not_a_superseded_one(tmp_path):
    """What the glob could not have got right even at one directory deep.

    A rerun leaves the previous generation on disk until it is pruned. The
    registry names which one is current; a directory listing does not.
    """
    from ftmap.workspace import create_workspace

    workspace = create_workspace(str(tmp_path / "work"))
    _commit(workspace, "a", "a" * 64, structure=PERFECT)
    _commit(workspace, "a", "d" * 64, structure=PERFECT)   # rerun, new bytes

    assert find_summary(doc(source={"path": "x", "sha256": "d" * 64}),
                        str(tmp_path / "work"))["source"]["sha256"] == "d" * 64
    with pytest.raises(FileNotFoundError, match="did not read"):
        find_summary(doc(), str(tmp_path / "work"))


def test_every_etalon_names_bytes_that_are_actually_in_the_corpus():
    """The strongest check available without re-reading the source: the digest
    an etalon binds to is the digest of the file it names. An answer written
    against a file that was since replaced, or against the wrong sheet of a
    workbook, fails here rather than in a published measurement."""
    root = os.path.dirname(ETALONS.rstrip("/"))
    repo = os.path.dirname(os.path.dirname(root))
    for f in shipped():
        d = load(f, CAT)
        path = os.path.join(repo, d.path)
        if not d.path or not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        assert digest == d.sha256, f"{f} names {d.path} but not its bytes"


def test_two_entities_of_one_schema_are_paired_by_their_keys():
    """The aircraft register: an owner and an operator, both Company, against
    two pipeline Organizations. First-eligible-wins paired them the wrong way
    round, which reported the operator's dedup failure as the owner's and made
    the one correct Ownership edge in the run score as endpoints-wrong."""
    raw = copy.deepcopy(MINIMAL)
    raw["entities"] = [
        {"key": "plane", "schema": "Airplane", "keys": ["c0"]},
        {"key": "owner", "schema": "Company", "accept": ["Organization"],
         "keys": ["c2"]},
        {"key": "operator", "schema": "Company", "accept": ["Organization"],
         "keys": ["c1"]},
    ]
    raw["columns"][1] = {"id": "c1", "header": "operator", "role": "key+property",
                         "answer": "Company:name"}
    st = copy.deepcopy(PERFECT)
    st["entities"] = [
        {"id": "e0", "schema": "Airplane", "keys": ["c0"]},
        # The operator's entity comes FIRST and has no key at all.
        {"id": "e1", "schema": "Organization", "keys": []},
        {"id": "e2", "schema": "Organization", "keys": ["c2"]},
    ]
    st["edges"] = [{"id": "g0", "schema": "Ownership", "source": "e2",
                    "target": "e0"}]
    st["bindings"]["c1"] = {"on": "e1", "prop": "Company:name", "type": "name"}
    st["bindings"]["c2"] = {"on": "e2", "prop": "Company:name", "type": "name"}
    s = score(parse(raw, CAT), summary(st), CAT)
    by_key = {r.etalon_key: r for r in s.entities}
    assert by_key["owner"].pipeline_id == "e2"
    assert by_key["owner"].keys_verdict == KEYS_AGREE
    assert by_key["operator"].keys_verdict == KEYS_EMPTY
    assert s.edges[0].verdict == "EDGE-AGREE"


def test_one_property_under_two_spellings_is_agreement_not_disagreement():
    """`Contract:description` and `Thing:description` are the same Property
    object. Phase 2 undid this by hand three times."""
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c2"]["prop"] = "LegalEntity:name"
    s = score(doc(), summary(st), CAT)
    assert s.columns[2].verdict == AGREE


def test_a_shared_local_name_across_unrelated_schemata_is_still_wrong():
    """The guard on the rule above, and the largest finding of phase 2:
    `PublicBody:name` and `CourtCase:name` share a local name, and a court is
    not a court case."""
    raw = copy.deepcopy(MINIMAL)
    raw["entities"] = [{"key": "body", "schema": "PublicBody", "keys": ["c0"]}]
    raw["columns"] = [{"id": "c0", "header": "code", "role": "key+property",
                       "answer": "PublicBody:registrationNumber"},
                      {"id": "c1", "header": "name", "role": "property",
                       "answer": "PublicBody:name"}]
    raw["edges"] = []
    st = {"entities": [{"id": "e0", "schema": "CourtCase", "keys": ["c0"]}],
          "edges": [],
          "bindings": {"c1": {"on": "e0", "prop": "CourtCase:name",
                              "type": "name"}}}
    assert score(parse(raw, CAT), summary(st), CAT).columns[1].verdict == WRONG


def test_two_sheets_of_one_workbook_do_not_collide_in_a_report():
    """A workbook is several sources sharing one path. Keying a corpus report
    by the path alone let one sheet's score silently overwrite another's."""
    a = doc(source={"path": "wb.xlsx", "sha256": "a" * 64, "sheet": "S1"})
    b = doc(source={"path": "wb.xlsx", "sha256": "a" * 64, "sheet": "S2"})
    sa, sb = score(a, summary(PERFECT), CAT), score(b, summary(PERFECT), CAT)
    assert sa.source != sb.source
    assert len({sa.source: 1, sb.source: 1}) == 2
    assert score(doc(), summary(PERFECT), CAT).source == "x.csv"


# --------------------------------------------------------------------------
# The five things v2 could not express. Each is a real etalon's real problem.
# --------------------------------------------------------------------------

def test_a_declined_column_may_pre_register_a_binding_that_beats_declining():
    """`us_ofac_sdn`'s `identifiers` packs twelve identifier systems with the
    discriminator dropped. The etalon declines and says a pipeline switching on
    the row's schema would be BETTER — which v2 scored as STRETCHED, the
    instrument penalising the answer it exists to reward."""
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[3]["accept_binding"] = ["Vessel:imoNumber"]
    st = copy.deepcopy(PERFECT)
    st["bindings"]["c3"] = {"on": "e0", "prop": "Vessel:imoNumber",
                            "type": "identifier"}
    s = score(doc(columns=cols), summary(st), CAT)
    assert s.columns[3].verdict == DEFENSIBLE
    assert s.totals()["no_home"]["stretched"] == 0
    # A binding NOT on the list is still stretching.
    st["bindings"]["c3"]["prop"] = "Airplane:model"
    assert score(doc(columns=cols), summary(st), CAT).columns[3].verdict == STRETCHED


def test_accept_binding_is_refused_on_a_column_that_carries_a_property():
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1]["accept_binding"] = ["Airplane:model"]
    refuses("`accept_binding` is for a column the etalon DECLINES",
            columns=cols)


def test_an_entity_no_column_can_produce_is_not_counted_as_missing():
    """The procurement plan's buyer lives only in merged cell A1, above the
    header. A column-driven planner can never declare it, so scoring it
    ENTITY-MISSING loads the denominator with an answer no run can reach."""
    ents = copy.deepcopy(MINIMAL["entities"])
    ents.append({"key": "buyer", "schema": "PublicBody", "producible": False,
                 "why": "lives in a merged title cell above the header"})
    s = score(doc(entities=ents), summary(PERFECT), CAT)
    by = {r.etalon_key: r for r in s.entities}
    assert by["buyer"].verdict == "ENTITY-UNPRODUCIBLE"
    t = s.totals()["entities"]
    assert t["unproducible"] == 1 and t["missing"] == 0
    assert t["expected"] == 2, "the unproducible one is out of the denominator"


def test_an_unproducible_entity_may_not_also_be_keyed():
    ents = copy.deepcopy(MINIMAL["entities"])
    ents.append({"key": "buyer", "schema": "PublicBody", "producible": False,
                 "keys": ["c3"]})
    refuses("cannot also be keyed", entities=ents)


def test_a_table_scoped_entity_is_judged_on_how_many_came_out():
    """`keys: []` cannot distinguish "one per row, no identity available" from
    "one for the whole table, mentioned on every row". The two have opposite
    verdicts — the second IS the thirty-copies-of-one-court failure — so the
    etalon says which it means and the census settles it."""
    ents = copy.deepcopy(MINIMAL["entities"])
    ents[1] = {"key": "owner", "schema": "Company", "keys": ["c2"],
               "scope": "table"}
    one = summary(PERFECT); one["entities"] = {"Company": 1, "Airplane": 30}
    many = summary(PERFECT); many["entities"] = {"Company": 30, "Airplane": 30}
    # The census must be read under the schema the run BUILT. A run that
    # generalised Company to Organization and emitted thirty copies has zero
    # Companies, so looking up the etalon's schema scores the dedup failure as
    # agreement — the failure this scope exists to catch, passing as a pass.
    gen = copy.deepcopy(PERFECT)
    gen["entities"][1]["schema"] = "Organization"
    gen["bindings"]["c2"]["prop"] = "Organization:name"
    g = summary(gen); g["entities"] = {"Organization": 30, "Airplane": 30}
    ents_acc = copy.deepcopy(ents)
    ents_acc[1]["accept"] = ["Organization"]
    assert score(doc(entities=ents_acc), g, CAT).entities[1].keys_verdict == KEYS_DIFFER
    assert score(doc(entities=ents), one, CAT).entities[1].keys_verdict == KEYS_AGREE
    assert score(doc(entities=ents), many, CAT).entities[1].keys_verdict == KEYS_DIFFER
    # Same census, but the run declared no key at all: named, not softened.
    st = copy.deepcopy(PERFECT)
    st["entities"][1]["keys"] = []
    m2 = summary(st); m2["entities"] = {"Company": 30}
    assert score(doc(entities=ents), m2, CAT).entities[1].keys_verdict == KEYS_EMPTY


def test_a_table_whose_correct_output_is_no_entity_may_say_so():
    """`Значення` is a hidden codelist serving its sibling sheets. Its own
    etalon says a run emitting nothing from it is better than the etalon, and
    v2 forced it to declare a stand-in entity anyway."""
    raw = copy.deepcopy(MINIMAL)
    raw["entities"] = []
    raw["edges"] = []
    raw["produces_nothing"] = "a codelist consumed during the sibling sheets' ingest"
    raw["columns"] = [{"id": "c0", "header": "Символ", "role": "unmappable"},
                      {"id": "c1", "header": "Значення", "role": "unmappable"}]
    d = parse(raw, CAT)
    assert d.entities == ()
    st = {"entities": [], "edges": [], "bindings": {}}
    quiet = summary(st); quiet["entities"] = {}
    assert score(d, quiet, CAT).produces_nothing_verdict == "PRODUCES-NOTHING-AGREE"
    noisy = summary(st); noisy["entities"] = {"Thing": 209}
    assert score(d, noisy, CAT).produces_nothing_verdict == "PRODUCES-NOTHING-WRONG"


def test_an_empty_entity_list_without_a_reason_is_still_an_omission():
    raw = copy.deepcopy(MINIMAL)
    raw["entities"] = []
    raw["edges"] = []
    with pytest.raises(EtalonError, match="entities is empty"):
        parse(raw, CAT)


def test_produces_nothing_may_not_contradict_a_column_answer():
    raw = copy.deepcopy(MINIMAL)
    raw["entities"] = []
    raw["edges"] = []
    raw["produces_nothing"] = "a codelist"
    with pytest.raises(EtalonError, match="but a column is answered"):
        parse(raw, CAT)


def test_a_caveat_is_carried_and_printed_but_never_scored():
    """`reestrtz`'s D_REG is a registration date on 38 618 rows and a
    DE-registration date on 989, and the discriminator is in a column the same
    etalon declines. One column, one answer — so the format states the thing it
    cannot score rather than pretending it is not true."""
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1]["caveat"] = "inverted on the 989 de-registration rows"
    s = score(doc(columns=cols), summary(PERFECT), CAT)
    assert s.columns[1].verdict == AGREE
    assert "inverted on the 989" in as_markdown(s)


def test_the_effective_subject_is_scored_beside_the_declared_one():
    """`emit.summarise` publishes both because they disagree; scoring only the
    model's declared answer reports a phrasing-sensitive field as the finding."""
    st = summary(PERFECT, subject="Audio")
    st["subject"] = "Airplane"
    s = score(doc(), st, CAT)
    assert s.subject_verdict == SUBJECT_WRONG
    assert s.subject_effective_verdict == SUBJECT_AGREE


def test_an_accepted_reading_is_still_accepted_on_a_subschema():
    """`accept` compared like `answer`, which it was not.

    Measured on the aircraft register, whose etalon declares the subject
    `Airplane` and spells its accept lists `Vehicle:*`. Every conforming run
    answers `Airplane:registrationNumber` — and that scored PIPELINE-WRONG
    while `Vehicle:registrationNumber` scored BOTH-DEFENSIBLE. Three readings
    the annotator pre-registered as defensible were unreachable by any run
    following that same annotator's subject declaration.

    Across the twenty-source corpus this alone understated column precision by
    five points; see docs/measurements/2026-08-21-prompt-ablation.md.
    """
    d = doc(columns=[
        {"id": "c0", "header": "mark", "role": "key+property",
         "answer": "Airplane:registrationNumber"},
        {"id": "c1", "header": "type", "role": "property",
         "answer": "Airplane:model", "accept": ["Vehicle:type"]},
        {"id": "c2", "header": "owner", "role": "key+property",
         "answer": "Company:name"},
    ])
    col = d.columns[1]
    from ftmap.etalon.score import _accepted
    assert _accepted(col, "Vehicle:type", CAT), "the spelling written down"
    assert _accepted(col, "Airplane:type", CAT), "the spelling a run produces"
    # The guard `_same_property` carries stays: a shared local name on two
    # unrelated schemata is still a disagreement.
    assert not _accepted(col, "CourtCase:type", CAT)


def test_entities_sharing_a_schema_do_not_share_their_columns():
    """Column ownership derived by schema STRING put three `Person` entities
    in possession of every `Person:*` column, so none could be told from the
    others — and `score._best` ranks on exactly that overlap.

    Measured on the tax-debtor extract, which declares a debtor, a company
    director and a tax-office chief. A run that got all three right scored
    KEYS-DIFFER three times and wrong endpoints on three relations, because
    the debtor (keyed on the name column) ranked BELOW the director (keyed on
    the director column) for the debtor's own slot.

    Keys are authoritative; a column two entities could both own discriminates
    between them not at all, and is given to neither.
    """
    d = doc(
        entities=[
            {"key": "debtor", "schema": "Person", "keys": ["c0"]},
            {"key": "chief", "schema": "Person", "keys": ["c1"]},
            {"key": "office", "schema": "PublicBody", "keys": ["c2"]},
        ],
        edges=[],
        columns=[
            {"id": "c0", "header": "debtor", "role": "key+property",
             "answer": "Person:name"},
            {"id": "c1", "header": "chief", "role": "key+property",
             "answer": "Person:name"},
            {"id": "c2", "header": "office", "role": "key+property",
             "answer": "PublicBody:name"},
            {"id": "c3", "header": "office addr", "role": "property",
             "answer": "PublicBody:address"},
        ])
    owned = {e.key: set(e.columns) for e in d.entities}
    assert owned["debtor"] == {"c0"}, owned
    assert owned["chief"] == {"c1"}, owned
    # Uncontested: only one entity's schema is related to PublicBody here, so
    # the address goes to it without being named as a key.
    assert owned["office"] == {"c2", "c3"}, owned


def test_a_column_is_owned_by_an_entity_spelled_at_another_height():
    """The other half of the same defect: an entity declared `Company` whose
    columns are spelled `LegalEntity:*` owned NOTHING under string equality —
    the entity with the most evidence in the file got none of it."""
    d = doc(
        entities=[{"key": "firm", "schema": "Company", "keys": ["c0"]}],
        edges=[],
        columns=[
            {"id": "c0", "header": "name", "role": "key+property",
             "answer": "LegalEntity:name"},
            {"id": "c1", "header": "code", "role": "property",
             "answer": "LegalEntity:taxNumber"},
        ])
    assert set(d.entities[0].columns) == {"c0", "c1"}


# --------------------------------------------------------------------------
# Zero emission
# --------------------------------------------------------------------------

def test_a_declared_entity_the_run_never_emitted_is_a_key_failure():
    """Covers AE8. The instrument agreed with itself about nothing.

    `_keys_verdict` for a TABLE-scoped entity read `if n <= 1: return
    KEYS_AGREE`, on the reasoning that the etalon's claim is "the source holds
    exactly one of these" and one is what came out. Zero also satisfies
    `n <= 1`, and zero is not one: the plan DECLARED the entity, the pipeline
    emitted no instance of it, and the score reported key agreement for an
    entity that does not exist in the output.

    That is the worst shape a measurement error can take — the instrument
    reporting success for the case where the pipeline produced nothing at all.
    """
    want = copy.deepcopy(MINIMAL)
    want["entities"][1]["scope"] = "table"
    st = copy.deepcopy(PERFECT)

    # The run declares the entity and emits one. Agreement, as before.
    one = score(doc(entities=want["entities"]),
                summary(st) | {"entities": {"Airplane": 3, "Company": 1}}, CAT)
    assert one.entities[1].keys_verdict == KEYS_AGREE

    # The run declares it and emits NONE.
    none = score(doc(entities=want["entities"]),
                 summary(st) | {"entities": {"Airplane": 3}}, CAT)
    assert none.entities[1].keys_verdict == KEYS_NONE_EMITTED, (
        "a declared entity with zero emitted instances scored as agreement")

    # And the aggregate moves with it, by exactly one entity.
    assert one.totals()["entities"]["keys_agree"] - \
        none.totals()["entities"]["keys_agree"] == 1


def test_the_emission_guard_applies_to_a_row_scoped_entity_too(tmp_path):
    """The same question, on the branch that judges key COLUMNS.

    A row-scoped entity keyed on exactly the columns the etalon named, with no
    instance emitted, hit `set(got) == set(want.keys)` and returned agreement.
    The keys being right is not the claim being checked — the claim is that the
    run produced this entity, identified that way.
    """
    st = copy.deepcopy(PERFECT)
    s = score(doc(), summary(st) | {"entities": {"Company": 1}}, CAT)
    airplane = next(r for r in s.entities if r.etalon_schema == "Airplane")
    assert airplane.keys_verdict == KEYS_NONE_EMITTED
    # ...and its key COLUMNS were right all along, which is exactly why the
    # old branch agreed.
    assert set(airplane.pipeline_keys) == set(airplane.etalon_keys)
    assert s.totals()["entities"]["keys_none_emitted"] == 1

    # It still agrees when the run actually emitted some.
    s = score(doc(), summary(st) | {"entities": {"Airplane": 3, "Company": 1}},
              CAT)
    airplane = next(r for r in s.entities if r.etalon_schema == "Airplane")
    assert airplane.keys_verdict == KEYS_AGREE


def test_an_entity_the_run_declared_and_bound_nothing_to_is_not_a_match():
    """A declaration is not an entity. Nothing here asked whether the pipeline
    entity a gold-standard entity was paired with carries any property at all,
    so a plan that declares every schema it can think of scores as if it had
    found them.

    Measured on the person-graph corpus 2026-08-28: the Gemma run declared
    `Person`, `Vessel`, `Organization`, `CryptoWallet` and `Airplane` on the
    OFAC file, all keyed on the same column, and emitted 20 054 of each — the
    file holds 976 wallets and 342 airplanes. Three of the five carried no
    property whatsoever. It scored 5 of 6 entities matched.
    """
    empty = copy.deepcopy(PERFECT)
    del empty["bindings"]["c2"]          # nothing is bound to e1 any more
    s = score(doc(), summary(empty), CAT)
    owner = [r for r in s.entities if r.etalon_key == "owner"][0]
    assert owner.verdict == ENTITY_EMPTY
    t = s.totals()
    assert t["entities"]["matched"] == 1
    assert t["entities"]["empty"] == 1


def test_an_entity_may_say_how_many_instances_the_source_holds():
    doc = parse({**MINIMAL, "entities": [
        {**MINIMAL["entities"][0], "instances": 3},
        MINIMAL["entities"][1]]}, CAT)
    assert doc.entities[0].instances == 3
    assert doc.entities[1].instances is None


def test_an_instance_count_that_is_not_a_positive_number_is_refused():
    with pytest.raises(EtalonError) as e:
        parse({**MINIMAL, "entities": [
            {**MINIMAL["entities"][0], "instances": 0},
            MINIMAL["entities"][1]]}, CAT)
    assert "instances" in str(e.value)


def test_a_declaration_built_from_every_row_is_visible_against_its_count():
    """The failure the four layers could not see.

    `us_ofac_sdn` declares Person and Organization on one key column. Built
    from every row, each emits 20 054 instances from 20 079 rows; built from
    its own rows, Person emits 7 456 — the etalon's own number, stated in its
    `why` since phase 1 and until now unassertable. Both plans score the same
    on every layer, because the entity layer pairs DECLARATIONS.
    """
    doc = parse({**MINIMAL, "entities": [
        {**MINIMAL["entities"][0], "instances": 3},
        MINIMAL["entities"][1]]}, CAT)
    over = summary({**PERFECT, "entities": [
        {**PERFECT["entities"][0], "instances": 30},
        PERFECT["entities"][1]]})
    s = score(doc, over, CAT)
    plane = [r for r in s.entities if r.etalon_key == "plane"][0]
    assert plane.instances_verdict == INSTANCES_OVER
    assert s.totals()["entities"]["instances_over"] == 1

    right = summary({**PERFECT, "entities": [
        {**PERFECT["entities"][0], "instances": 3},
        PERFECT["entities"][1]]})
    ok = score(doc, right, CAT)
    assert ok.totals()["entities"]["instances_agree"] == 1


def test_an_entity_may_name_the_rows_it_is_built_from():
    """A fact about the SOURCE, not about any run: `us_ofac_sdn` says in c1
    which rows are people and which are wallets, and until the etalon could
    say so too, `instances` could not be computed for a bloc — the naive
    distinct-key count over the whole file gives 20 079 for every one of the
    five."""
    doc = parse({**MINIMAL, "entities": [
        {**MINIMAL["entities"][0], "rows": {"column": "c1", "value": "SU"}},
        MINIMAL["entities"][1]]}, CAT)
    assert doc.entities[0].rows == ("c1", "SU")
    assert doc.entities[1].rows is None


def test_a_row_selector_naming_an_undeclared_column_is_refused():
    with pytest.raises(EtalonError) as e:
        parse({**MINIMAL, "entities": [
            {**MINIMAL["entities"][0], "rows": {"column": "c9", "value": "x"}},
            MINIMAL["entities"][1]]}, CAT)
    assert "c9" in str(e.value)


def test_a_pairing_with_nothing_in_common_is_not_a_pairing():
    """`_best` returned the least-bad candidate even when nothing tied it to
    the etalon entity — no shared key, no shared column — so the FIRST etalon
    entity in the file consumed it and the one it actually matched scored
    ENTITY-MISSING.

    Measured 2026-08-29 on the tax debtors: the run built an Organization
    keyed on c6, which is exactly the etalon's `tax_office`; `subunit` (keyed
    c3, c4) took it first and the report read "749 emitted of 66" against an
    entity the run never built.
    """
    doc = parse({**MINIMAL, "edges": [], "entities": [
        {"key": "elsewhere", "schema": "Company", "keys": ["c1"]},
        {"key": "owner", "schema": "Company", "keys": ["c2"]}],
        "columns": [
            {"id": "c0", "header": "code", "role": "unmappable"},
            {"id": "c1", "header": "інше", "role": "unmappable"},
            {"id": "c2", "header": "власник", "role": "key+property",
             "answer": "Company:name"},
            {"id": "c3", "header": "mass", "role": "unmappable"},
            {"id": "c4", "header": "n/n", "role": "not-data"}]}, CAT)
    structure = {
        "entities": [{"id": "e0", "schema": "Company", "keys": ["c2"]}],
        "edges": [],
        "bindings": {"c2": {"on": "e0", "prop": "Company:name", "type": "name"}},
    }
    s = score(doc, summary(structure), CAT)
    by_key = {r.etalon_key: r for r in s.entities}
    assert by_key["elsewhere"].verdict == ENTITY_MISSING
    assert by_key["owner"].pipeline_id == "e0"


def test_instance_tolerance_is_configuration_not_a_constant():
    """The 5% band decides INSTANCES-AGREE vs OVER/UNDER in published scores
    and was sized to one measured gap — the definition of a number that
    belongs in defaults.toml and the manifest, not in a module constant."""
    from ftmap.config import Config
    from ftmap.etalon.score import (INSTANCES_AGREE, INSTANCES_OVER,
                                    _instances_verdict)

    cfg = Config.load(None)
    assert cfg.instance_tolerance == 0.05
    assert cfg.as_manifest()["etalon"]["instance_tolerance"] == 0.05
    assert _instances_verdict(100, 104, tolerance=0.05) == INSTANCES_AGREE
    assert _instances_verdict(100, 104, tolerance=0.0) == INSTANCES_OVER


def test_a_want_with_no_twin_does_not_take_the_twin_of_the_want_after_it():
    """Evaluator 3. The tax-debtor register (`work-x14`): the run produced
    the office chief's Directorship exactly and the debtor's not at all.
    Matched in one pass in etalon order, the debtor's want fell back to the
    only Directorship there was — the office chief's — and scored it
    ENDPOINTS-WRONG, leaving the office chief's want MISSING: one right
    edge reported as one wrong one and one absent one. Exact twins are
    taken first, across every want; the fallback runs after."""
    etalon = doc(
        entities=[{"key": "plane", "schema": "Airplane", "keys": ["c0"]},
                  {"key": "owner", "schema": "Company", "keys": ["c2"]},
                  {"key": "lessor", "schema": "Company", "keys": ["c1"]}],
        edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                "source": "owner", "target": "plane"},
               {"key": "lease", "kind": "link", "schema": "Ownership",
                "source": "lessor", "target": "plane"}],
        columns=[{"id": "c0", "header": "mark", "role": "key+property",
                  "answer": "Airplane:registrationNumber"},
                 {"id": "c1", "header": "lessor", "role": "key+property",
                  "answer": "Company:name"},
                 {"id": "c2", "header": "owner", "role": "key+property",
                  "answer": "Company:name"}])
    st = {
        "entities": [{"id": "e0", "schema": "Airplane", "keys": ["c0"]},
                     {"id": "e1", "schema": "Company", "keys": ["c2"]},
                     {"id": "e2", "schema": "Company", "keys": ["c1"]}],
        # Only the lessor's edge was produced.
        "edges": [{"id": "g0", "schema": "Ownership", "source": "e2", "target": "e0"}],
        "bindings": {
            "c0": {"on": "e0", "prop": "Airplane:registrationNumber", "type": "identifier"},
            "c1": {"on": "e2", "prop": "Company:name", "type": "name"},
            "c2": {"on": "e1", "prop": "Company:name", "type": "name"},
        },
    }
    s = score(etalon, summary(st), CAT)
    got = {r.etalon_key: r.verdict for r in s.edges if r.etalon_key}
    assert got == {"own": EDGE_MISSING, "lease": EDGE_AGREE}


def test_an_edge_to_an_unproducible_entity_is_a_ceiling_not_a_miss():
    """Evaluator 4. The procurement plan declares its buyer unproducible —
    it lives in a merged title cell — and its `Contract:authority` had been
    scored MISSING on every run, a figure no run could move."""
    etalon = doc(
        entities=[{"key": "plane", "schema": "Airplane", "keys": ["c0"]},
                  {"key": "owner", "schema": "Company", "keys": ["c2"]},
                  {"key": "registrar", "schema": "PublicBody", "keys": [],
                   "scope": "table", "producible": False}],
        edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                "source": "owner", "target": "plane"},
               {"key": "registered", "kind": "property", "prop": "Airplane:operator",
                "source": "plane", "target": "registrar"}])
    s = score(etalon, summary(copy.deepcopy(PERFECT)), CAT)
    got = {r.etalon_key: r.verdict for r in s.edges if r.etalon_key}
    assert got == {"own": EDGE_AGREE, "registered": EDGE_UNPRODUCIBLE}
    assert s.totals()["edges"]["unproducible"] == 1
    assert s.totals()["edges"]["missing"] == 0


def test_an_etalon_entity_of_an_edge_schema_pairs_with_the_run_s_edge():
    """Evaluator 5. The enforcement register's proceeding is a Debt keyed on
    its number: an entity in FollowTheMoney, an edge in this pipeline's own
    split. It pairs with the run's Debt edge on the edge's `keys` and the
    bindings on it, and scores its keys like any entity; the edge layer
    still scores the same Debt as a link."""
    cols = copy.deepcopy(MINIMAL["columns"])
    cols[1] = {"id": "c1", "header": "proceeding", "role": "key+property",
               "answer": "Debt:recordId"}
    d = doc(entities=MINIMAL["entities"] + [{"key": "proceeding", "schema": "Debt",
                                             "keys": ["c1"]}],
            columns=cols)
    st = copy.deepcopy(PERFECT)
    st["edges"].append({"id": "g1", "schema": "Debt", "source": "e1", "target": "e0",
                        "keys": ["c1"]})
    st["bindings"]["c1"] = {"on": "g1", "prop": "Debt:recordId", "type": "identifier"}
    got = score(d, summary(st, entities={"Airplane": 1, "Company": 1, "Debt": 1}), CAT)
    row = next(r for r in got.entities if r.etalon_key == "proceeding")
    assert row.verdict != ENTITY_MISSING and row.pipeline_id == "g1"
    assert row.keys_verdict == KEYS_AGREE
    # An edge nobody wanted as an entity is not an extra entity.
    assert not any(r.pipeline_id == "g0" for r in got.entities)


# --------------------------------------------------------------------------
# Evaluator 6: an edge may pre-register another schema as also right
# --------------------------------------------------------------------------

def test_an_edge_may_accept_another_schema_the_endpoints_can_satisfy():
    d = doc(edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                    "source": "owner", "target": "plane",
                    "accept": ["UnknownLink"]}])
    assert d.edges[0].accept == ("UnknownLink",)


def test_an_edge_s_accept_must_name_an_edge_schema_the_endpoints_fit():
    refuses("not an edge schema",
            edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                    "source": "owner", "target": "plane", "accept": ["Person"]}])
    # Directorship ranges on LegalEntity -> Organization, and the plane is
    # neither an organisation nor anything under one.
    refuses("cannot satisfy",
            edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                    "source": "owner", "target": "plane",
                    "accept": ["Directorship"]}])
    refuses("property edge is one qname",
            edges=[{"key": "op", "kind": "property", "prop": "Airplane:operator",
                    "source": "plane", "target": "owner",
                    "accept": ["Ownership"]}])


def test_a_run_s_edge_under_an_accepted_schema_is_found_not_missing_and_extra():
    """The roster (`2026-09-05-manual-corpus.md` §4): the etalon asked for a
    Membership, the run stated an Employment between the same two, and
    evaluator 5 scored one MISSING and one EXTRA on a relation the run had
    produced under the other name FollowTheMoney declares for it."""
    from ftmap.etalon.score import EDGE_ACCEPTABLE
    d = doc(edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                    "source": "owner", "target": "plane",
                    "accept": ["UnknownLink"]}])
    st = copy.deepcopy(PERFECT)
    st["edges"] = [{"id": "g0", "schema": "UnknownLink", "source": "e1",
                    "target": "e0"}]
    s = score(d, summary(st), CAT)
    assert [r.verdict for r in s.edges] == [EDGE_ACCEPTABLE]
    assert s.edges[0].schema == "UnknownLink"
    t = s.totals()["edges"]
    assert (t["found"], t["acceptable"], t["missing"], t["extra"]) == (1, 1, 0, 0)
    assert "under an accepted schema" in as_markdown(s)


def test_the_etalon_s_own_schema_takes_the_exact_twin_before_an_accepted_one():
    """A run that stated both readings: the exact twin is the agreement and
    the accepted reading is then a second relation nobody asked for, not a
    second agreement."""
    from ftmap.etalon.score import EDGE_ACCEPTABLE
    d = doc(edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                    "source": "owner", "target": "plane",
                    "accept": ["UnknownLink"]}])
    st = copy.deepcopy(PERFECT)
    st["edges"] = [{"id": "g1", "schema": "UnknownLink", "source": "e1",
                    "target": "e0"},
                   {"id": "g0", "schema": "Ownership", "source": "e1",
                    "target": "e0"}]
    s = score(d, summary(st), CAT)
    assert [r.verdict for r in s.edges] == [EDGE_AGREE, EDGE_EXTRA]
    assert EDGE_ACCEPTABLE not in {r.verdict for r in s.edges}


def test_an_accepted_schema_with_the_wrong_endpoints_is_endpoints_wrong_under_its_own_name():
    from ftmap.etalon.score import EDGE_ENDPOINTS_WRONG
    d = doc(edges=[{"key": "own", "kind": "link", "schema": "Ownership",
                    "source": "owner", "target": "plane",
                    "accept": ["UnknownLink"]}])
    st = copy.deepcopy(PERFECT)
    st["edges"] = [{"id": "g0", "schema": "UnknownLink", "source": "e0",
                    "target": "e1"}]
    s = score(d, summary(st), CAT)
    assert [r.verdict for r in s.edges] == [EDGE_ENDPOINTS_WRONG]
    assert s.edges[0].schema == "UnknownLink"
