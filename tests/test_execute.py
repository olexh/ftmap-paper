import json
import os

from ftmap.build.emit import (summarise, write_entities, write_statements)
from ftmap.build.execute import Statement, execute
from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.normalize.canonical import MULTIVALUE_JOIN
from ftmap.plan.compile import compile_mapping, normalize_frame
from ftmap.plan.validate import Decision, ValidatedPlan
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()

ROWS = [["ПІБ", "Дата народження", "Фракція"],
        ["Коваленко Іван Петрович", "17.09.1980", "Слуга народу"]]


def _frame():
    return build_frame(Grid(rows=ROWS, sheet="арк1", merges=[]),
                       "/corpus/dep.xlsx", "a" * 64, "aaaaaaaaaaaa/арк1", CFG)


def _vplan():
    return ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "party", "schema": "Organization", "keys": ["c2"]}],
        edges=[{"key": "member", "schema": "Membership",
                "source": "person", "target": "party"}],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c1", "prop": "Person:birthDate", "entity": "person",
             "type_name": "date", "why": ""},
            {"column": "c2", "prop": "Organization:name", "entity": "party",
             "type_name": "name", "why": ""},
        ],
        decisions=[Decision("c0", "Person:name", "person", "accepted", "", "model")],
    )


def _run(tmp_path):
    frame, vplan = _frame(), _vplan()
    csv_path = str(tmp_path / "n.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    entities, statements, emit_rejects = execute(
        mapping, vplan, frame, CAT, "run1", {"c0": "model"})
    return frame, vplan, entities, statements, rejects + emit_rejects


def test_entities_carry_canonical_values(tmp_path):
    _, _, entities, _, _ = _run(tmp_path)
    person = [e for e in entities if e["schema"] == "Person"][0]
    assert person["properties"]["birthDate"] == ["1980-09-17"]
    assert person["properties"]["name"] == ["Коваленко Іван Петрович"]


def test_every_statement_points_back_at_a_cell(tmp_path):
    _, _, _, statements, _ = _run(tmp_path)
    birth = [s for s in statements if s.prop == "birthDate"][0]
    assert birth.value == "1980-09-17"
    assert birth.value_raw == "17.09.1980"
    assert birth.column_id == "c1"
    assert birth.header == "Дата народження"
    assert birth.row == 1
    assert birth.sheet == "арк1"
    assert birth.sha256 == "a" * 64
    assert birth.canonicalizer == "date"
    assert birth.decided_by in {"model", "rule", "analyst"}


def test_a_multi_valued_property_is_ordered_the_same_way_every_run(tmp_path):
    """FtM's EntityProxy.get_type_values builds a bare set(), so a multi-valued
    property's order depends on the per-process string hash seed. Measured on a
    five-source run: 1 entity of 688 had its namesMentioned reordered between
    two otherwise identical runs, which broke the byte-identical claim."""
    _, _, entities, statements, _ = _run(tmp_path)
    for e in entities:
        for prop, values in e["properties"].items():
            assert values == sorted(set(values)), (e["schema"], prop)


def test_a_second_row_adds_to_an_entity_rather_than_being_lost(tmp_path):
    """Two rows mapping to one id is what declaring `keys` is FOR. An earlier
    version kept only the first row's snapshot, so a value that appeared only
    on the second row survived in statements.csv and vanished from
    entities.ftm.json — the deliverable degrading exactly when deduplication
    worked."""
    rows = [["ПІБ", "Фракція", "Сайт"],
            ["Коваленко Іван", "Слуга народу", None],
            ["Шевченко Ольга", "Слуга народу", "https://example.org"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "b" * 64, "bbbbbbbbbbbb/", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "party", "schema": "Organization", "keys": ["c1"]}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c1", "prop": "Organization:name", "entity": "party",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "Organization:website", "entity": "party",
             "type_name": "url", "why": ""},
        ],
        decisions=[],
    )
    csv_path = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    entities, statements, _ = execute(mapping, vplan, frame, CAT, "run1", {})
    party = [e for e in entities if e["schema"] == "Organization"]
    assert len(party) == 1, "both rows name one organization"
    # FtM's own `url.clean()` appends the trailing slash a bare-path URL
    # lacks — measured directly against the installed followthemoney, not an
    # artefact of this merge. `Organization:website == "https://example.org"`
    # was the wrong expectation, not the merge; the point under test is that
    # the second row's value is present at all.
    assert party[0]["properties"]["website"] == ["https://example.org/"]


def test_the_summary_s_subject_is_what_validate_read(tmp_path):
    """emit.py's derived `subject` once counted bindings per schema, edges
    included; an edge-heavy source then published the relation as its
    subject, and a source whose bindings tipped to a party published the
    party (`work-p2`). The summary now reads `validate.effective_subject`:
    the declared subject when a declared Thing is related to it — here the
    Person, whatever the edge carries — and never null while anything is
    bound."""
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    vplan.bindings = [{**b, "entity": "member"} for b in vplan.bindings]
    s = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    assert s["subject"] == "Person"
    vplan.bindings = []
    s = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    assert s["subject"] is None


def test_an_edge_reference_is_not_counted_as_unfalsifiable(tmp_path):
    """An edge reference never went through a canonicalizer and was checked
    against the edge schema's endpoint ranges, which is stronger than "the type
    accepts any string". Counting it inflated the reported share from 99/264 to
    231/396 on a real source, as a function of how many edges it declares.

    `column_id is None` is a broader set than "entity-typed": FtM's own
    `namesMentioned`, which `inline_names` injects onto the Membership
    because it links to a Person and an Organization, also has no cell
    behind it, but it is a free-text `name`-typed property, not an entity
    reference — it rightly stays IN the unfalsifiable count. Measured on
    this fixture: 4 statements have no column (`member`, `organization`, and
    two `namesMentioned`), but only the 2 entity-typed ones are excluded from
    the denominator."""
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    s = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    refs = [st for st in statements if st.column_id is None]
    assert refs, "the fixture declares an edge"
    entity_typed = [st for st in statements
                    if CAT.prop(f"{st.schema}:{st.prop}").type_name == "entity"]
    assert entity_typed, "the fixture's edge produces entity-typed statements"
    assert s["unfalsifiable"]["of_statements"] == len(statements) - len(entity_typed)


def test_statements_carry_the_ontology_s_matchable_flag(tmp_path):
    """The hook a later record-linkage stage hangs on: which asserted values
    may be used as evidence that two records describe the same thing. FtM
    already knows; a consumer should not have to reload the model to ask."""
    _, _, _, statements, _ = _run(tmp_path)
    by_prop = {s.prop: s for s in statements if s.schema == "Person"}
    assert by_prop["name"].matchable is True
    assert by_prop["birthDate"].matchable is True


def test_edge_properties_produce_no_cell_provenance(tmp_path):
    """member/organization are entity references, not cells. Do not invent one."""
    _, _, _, statements, _ = _run(tmp_path)
    refs = [s for s in statements if s.prop in ("member", "organization")]
    assert refs and all(s.column_id is None for s in refs)


def test_written_files_are_owner_only(tmp_path):
    _, _, entities, statements, _ = _run(tmp_path)
    ep = str(tmp_path / "entities.ftm.json")
    sp = str(tmp_path / "statements.csv")
    write_entities(entities, ep)
    write_statements(statements, sp)
    assert oct(os.stat(ep).st_mode)[-3:] == "600"
    assert oct(os.stat(sp).st_mode)[-3:] == "600"
    with open(ep, encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    assert set(first) == {"id", "schema", "properties"}


def test_summary_contains_no_value_from_the_source(tmp_path):
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    s = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    blob = json.dumps(s, ensure_ascii=False)
    for forbidden in ("Коваленко", "Слуга народу", "17.09.1980", "1980-09-17"):
        assert forbidden not in blob
    # name-typed bindings rest on a validator that accepts anything
    assert s["unfalsifiable"]["bindings"] == 2
    assert s["unfalsifiable"]["of_bindings"] == 3
    assert s["columns"]["total"] == 3
    assert s["columns"]["mapped"] == 3
    assert s["entities"]["Person"] == 1
    assert s["config"]["model"]["seed"] == 20260810


def test_summary_carries_the_graph_the_plan_describes(tmp_path):
    """Without this block `summary.json` says which schemata came out and what
    became of each column's values, and nothing about the shape between them —
    so an entity with no key, a missing edge layer and a column consumed as
    identity are all invisible to anything scoring the value-free artefact."""
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    st = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                   {"run_id": "run1"})["structure"]
    assert st["entities"] == [{"id": "e0", "schema": "Person", "keys": ["c0"],
                               "filter": None, "instances": 1},
                              {"id": "e1", "schema": "Organization",
                               "keys": ["c2"], "filter": None,
                               "instances": 1}]
    assert st["edges"] == [{"id": "g0", "schema": "Membership",
                            "source": "e0", "target": "e1", "keys": []}]
    # A LIST per column — `column -> bindings`; one binding is a list of one.
    assert st["bindings"]["c1"] == [{"on": "e0", "prop": "Person:birthDate",
                                     "type": "date"}]


def test_the_model_s_own_entity_keys_stay_off_the_publishable_artefact(tmp_path):
    """`key` is the one field of a plan that is free-form model text, so it is
    the one that could carry a cell value out. Positional aliases say the same
    thing — which entity, which endpoints — and say nothing else."""
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    st = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                   {"run_id": "run1"})["structure"]
    # Asserted over `structure` alone, not the whole summary: `member` is also
    # the name of Membership's own source property, so a whole-document scan
    # would fail on a real FtM property name and prove nothing.
    for declared in (e["key"] for e in (*vplan.entities, *vplan.edges)):
        assert f'"{declared}"' not in json.dumps(st, ensure_ascii=False)


# --------------------------------------------------------------------------
# What the mapping engine itself refuses.
# --------------------------------------------------------------------------

# `wikidataId` is declared with a `wikidata` format, so followthemoney accepts
# `Q6319` and refuses `NK-26WkuEefVYLmYk8aTHMC9E` — an OpenSanctions id that
# carries a digit and therefore passes the local `identifier` canonicalizer
# untouched. Measured on ua_war_sanctions.targets.simple.csv: 5042 of 5621
# values took exactly this path and left no statement, no reject line and no
# log entry, while summary.json called the binding accepted.
# The third id is the same OpenSanctions id written without its hyphen, which
# is what the LOCAL canonicalizer still refuses: since 2026-08-29 a Latin code
# carrying a separator is an identifier with or without a digit — the change
# that lets the aircraft register's `UR-ALUR` bind — so the hyphenated form now
# reaches the engine like the others. One value has to fail at each stage for
# the two stages to be told apart at all.
ENGINE_ROWS = [["id"],
               ["Q6319"],
               ["NK-26WkuEefVYLmYk8aTHMC9E"],
               ["NKAnkFpprSmpkrtMWnsUASAk"]]


def _engine_frame():
    return build_frame(Grid(rows=ENGINE_ROWS, sheet="s", merges=[]),
                       "/corpus/os.csv", "b" * 64, "bbbbbbbbbbbb/s", CFG)


def _engine_vplan():
    return ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:wikidataId",
                   "entity": "person", "type_name": "identifier", "why": ""}],
        decisions=[Decision("c0", "Person:wikidataId", "person", "accepted",
                            "", "model")],
    )


def _engine_run(tmp_path):
    frame, vplan = _engine_frame(), _engine_vplan()
    csv_path = str(tmp_path / "e.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    entities, statements, emit_rejects = execute(
        mapping, vplan, frame, CAT, "run1", {})
    return frame, vplan, entities, statements, rejects + emit_rejects


def test_a_value_the_engine_refuses_gets_a_reject_line_of_its_own(tmp_path):
    _, _, _, statements, rejects = _engine_run(tmp_path)
    assert [s.value for s in statements] == ["Q6319"]
    refused = [r for r in rejects if r["stage"] == "emit"]
    # The canonicalizer strips an identifier's internal punctuation, so the
    # value the engine was handed — and refused — is the one recorded here.
    assert [r["value"] for r in refused] == ["NK26WkuEefVYLmYk8aTHMC9E"]
    assert refused[0]["reason"] == (
        "not in the wikidata format this property is declared to carry")
    assert refused[0]["canonicalizer"] == "followthemoney"
    assert refused[0]["column"] == "c0"
    assert refused[0]["row"] == 2
    # THE PLAN'S NAME FOR THE PROPERTY, not the ontology's declaring schema.
    # FtM calls this `Thing:wikidataId`, and a reader joining the two stages'
    # reject lines on `prop` would find nothing to join.
    assert refused[0]["prop"] == "Person:wikidataId"


def test_the_two_stages_are_told_apart_on_one_column(tmp_path):
    """The third row's id carries no digit, so the LOCAL canonicalizer refuses
    it; the second row's passes that and dies at followthemoney's format gate.
    One column, two reasons, two stages, both written down."""
    _, _, _, _, rejects = _engine_run(tmp_path)
    by_stage = {r["stage"]: r for r in rejects}
    assert set(by_stage) == {"normalize", "emit"}
    assert by_stage["normalize"]["canonicalizer"] == "identifier"
    assert by_stage["normalize"]["row"] == 3
    assert by_stage["emit"]["canonicalizer"] == "followthemoney"


def test_every_bound_column_adds_up(tmp_path):
    """Both ledgers close, for every bound column.

    This is the property the whole reject path exists to make true: a value in
    a bound column either becomes a statement or is written down with a reason.
    Before the engine's own refusals were captured this held for the local
    canonicalizer alone, so a column could report 396 statements, 183 rejects
    and 5042 values that were simply gone.

    The two refusals live in two ledgers on purpose: the canonicalizer's is
    the disposition row's `rejected` (that part was never offered to the
    engine), the engine's is the claim row's `rejected` — and the claim's
    `expected == emitted + rejected` is the reconciliation that used to hide
    inside one column-level subtraction.
    """
    frame, vplan, _, statements, rejects = _engine_run(tmp_path)
    s = summarise(frame, vplan, [], statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    cov, claims = s["coverage"], s["claims"]
    assert cov["c0"] == {"status": "bound", "role": "key", "values": 3,
                         "offered": 2, "rejected": 1, "declined": 0,
                         "structural": 0, "outside_selection": 0,
                         "unaccounted": 0}
    (claim,) = claims
    assert claim["column"] == "c0" and claim["prop"] == "Person:wikidataId"
    assert claim["expected"] == 2 and claim["emitted"] == 1
    assert claim["rejected"] == 1
    assert claim["shortfall"] == 0 and claim["over_emission"] == 0

    for col, acc in cov.items():
        assert acc["status"] == "bound", col
        assert acc["offered"] + acc["rejected"] == acc["values"], col
        assert acc["unaccounted"] == 0, col
    assert s["coverage_totals"]["accounting_valid"] is True


def test_coverage_covers_a_multi_valued_cell(tmp_path):
    """One cell holding two identifiers is two values, not one. Counting cells
    instead would make the equality hold by being wrong on both sides."""
    rows = [["id"], ["12345 67890"], ["Q6319"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/m.csv", "c" * 64, "cccccccccccc/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:idNumber",
                   "entity": "person", "type_name": "identifier", "why": ""}],
        decisions=[],
    )
    csv_path = str(tmp_path / "m.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    _, statements, emit_rejects = execute(mapping, vplan, frame, CAT, "run1", {})
    s = summarise(frame, vplan, [], statements, rejects + emit_rejects, CAT,
                  CFG, {"run_id": "run1"})
    assert s["coverage"]["c0"]["values"] == 3
    assert s["coverage"]["c0"]["offered"] == 3
    assert s["coverage"]["c0"]["unaccounted"] == 0
    (claim,) = s["claims"]
    assert claim["expected"] == 3 and claim["emitted"] == 3


def _debris_run(tmp_path, rows, bindings):
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/sep.csv", "f" * 64, "ffffffffffff/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=bindings,
        decisions=[],
    )
    csv_path = str(tmp_path / "sep.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    entities, statements, emit_rejects = execute(
        mapping, vplan, frame, CAT, "run1", {})
    cov = summarise(frame, vplan, entities, statements, rejects + emit_rejects,
                    CAT, CFG, {"run_id": "run1"})["coverage"]
    return entities, statements, rejects + emit_rejects, cov


def test_a_stray_separator_is_not_an_extra_value(tmp_path):
    """One cell, one value, whatever the separator does around it.

    THE NEGATIVE `unaccounted` IS THE POINT. A canonical «Іван;» kept the
    separator, `_binding_source` told the engine to split the column on `;`,
    and the engine read two values where the source held one, while `coverage`
    counted the non-empty part only: `values: 1, emitted: 2, unaccounted: -1`.
    Not a generous pipeline — an accounting that disagrees with the
    deliverable about what a value is.

    Both policy rules are exercised on one frame: `name` is SEPARATOR_ONLY and
    `identifier` EVERY_PART, and the bug reached both.
    """
    rows = [["ПІБ", "Код"],
            ["Іван;", "1025400524313;"],
            [";Петро", ";5401103595"],
            ["Ольга; ", "76403757; "],
            ["Марія;;", "12345678;;"],
            [" ; Ганна ; ", " ; 87654321 ; "]]
    entities, statements, _, cov = _debris_run(tmp_path, rows, [
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "",
         "type_name": "name"},
        {"column": "c1", "prop": "Person:idNumber", "entity": "person",
         "type_name": "identifier", "why": ""}])

    for col in ("c0", "c1"):
        assert cov[col] == {"status": "bound",
                            "role": "key" if col == "c0" else None,
                            "values": 5, "offered": 5, "rejected": 0,
                            "declined": 0, "structural": 0,
                            "outside_selection": 0, "unaccounted": 0}, col
    # And the values themselves carry no separator and no empty string.
    names = [v for e in entities for v in e["properties"].get("name", [])]
    assert set(names) == {"Іван", "Петро", "Ольга", "Марія", "Ганна"}
    assert len(names) == 5
    assert all(s.value.strip() for s in statements)


def test_a_cell_that_is_nothing_but_separators_is_reported_not_dropped(tmp_path):
    """The decision recorded in `test_a_cell_holding_nothing_but_separators_is
    _refused_not_emptied`, seen from the accounting's end.

    Accepted as «;», the cell was one filled cell worth zero values: it
    emitted nothing, was refused by nobody, and left no trace anywhere. It is
    now one value and one reject with a reason, so the row adds up and the
    loss is visible.
    """
    rows = [["ПІБ", "Код"],
            ["Іван", "1025400524313"],
            [";", ";;"]]
    _, statements, rejects, cov = _debris_run(tmp_path, rows, [
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "",
         "type_name": "name"},
        {"column": "c1", "prop": "Person:idNumber", "entity": "person",
         "type_name": "identifier", "why": ""}])

    for col in ("c0", "c1"):
        assert cov[col] == {"status": "bound",
                            "role": "key" if col == "c0" else None,
                            "values": 2, "offered": 1, "rejected": 1,
                            "declined": 0, "structural": 0,
                            "outside_selection": 0, "unaccounted": 0}, col
    refused = {r["column"]: r for r in rejects if r["row"] == 2}
    assert set(refused) == {"c0", "c1"}
    assert refused["c0"]["stage"] == "normalize"
    assert refused["c0"]["canonicalizer"] == "namex0"
    assert refused["c1"]["canonicalizer"] == "identifierx0"
    assert all("separator" in r["reason"] for r in refused.values())
    assert len(statements) == 2


def test_a_bad_part_of_a_pack_is_rejected_without_costing_the_pack(tmp_path):
    """The engine emitted a value the guard had refused, and the arithmetic
    agreed with it, which is why this needs a test rather than a number.

    «1025400524313;ІПН» holds one identifier and one token that is not one;
    «Іван; - ; Петро» holds two aliases and a piece of debris. Both cells used
    to be accepted WHOLE — their canonicalizers had no reason to remove a `;` —
    so `_binding_source` told the engine to split the column and the engine
    emitted an idNumber reading «ІПН» and a name reading «-». `coverage` split
    the same canonical value the same way, counted the same parts, and
    reported `unaccounted: 0` over both. The loss was invisible because both
    sides made the same mistake.

    Now the bad part is refused ON ITS OWN and the values beside it survive,
    so the row closes on a bigger number rather than a smaller one:

        c0  values 4 = offered 3 (Коваленко Іван, Іван, Петро) + rejected 1
        c1  values 3 = offered 2 (the code, on two entities)   + rejected 1

    THE RECONCILIATION IS THE POINT. `values` counts every part the source
    held, which for a partly accepted cell is the surviving parts plus the
    refused ones — `build.emit._cell_canon` counts `CanonResult.refused` for
    exactly this reason, and without it the row would read `values: 3,
    offered: 3, rejected: 1, unaccounted: -1`.
    """
    rows = [["ПІБ", "Код"],
            ["Коваленко Іван", "1025400524313"],
            ["Іван; - ; Петро", "1025400524313;ІПН"]]
    entities, statements, rejects, cov = _debris_run(tmp_path, rows, [
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "",
         "type_name": "name"},
        {"column": "c1", "prop": "Person:idNumber", "entity": "person",
         "type_name": "identifier", "why": ""}])

    assert cov["c0"] == {"status": "bound", "role": "key", "values": 4,
                         "offered": 3, "rejected": 1, "declined": 0,
                         "structural": 0, "outside_selection": 0,
                         "unaccounted": 0}
    assert cov["c1"] == {"status": "bound", "role": None, "values": 3,
                         "offered": 2, "rejected": 1, "declined": 0,
                         "structural": 0, "outside_selection": 0,
                         "unaccounted": 0}
    # ONE REJECT LINE PER BAD PART, NAMING THE PART. Reporting the whole cell
    # on each would make one bad token look like the loss of every value
    # beside it, which is what the reader is here to count.
    refused = {r["column"]: r for r in rejects if r["row"] == 2}
    assert set(refused) == {"c0", "c1"}
    assert refused["c0"]["value"] == "-"
    assert refused["c1"]["value"] == "ІПН"
    assert all(r["stage"] == "normalize" for r in refused.values())
    assert refused["c0"]["canonicalizer"] == "namex2of3"
    assert refused["c1"]["canonicalizer"] == "identifierx1of2"
    # The values that were values are in the deliverable; the ones that were
    # not are nowhere, and nothing carries the join character into it.
    assert len(statements) == 5
    assert all(MULTIVALUE_JOIN not in s.value for s in statements)
    values = {v for e in entities for vs in e["properties"].values() for v in vs}
    assert {"Іван", "Петро", "1025400524313"} <= values
    assert "ІПН" not in values and "-" not in values


def test_a_pack_no_part_of_which_is_a_value_still_closes(tmp_path):
    """The other end of the same branch: nothing survives, so the cell is one
    value and one reject line about itself rather than n lines about its
    pieces, and the row still adds up."""
    rows = [["ПІБ", "Код"],
            ["Коваленко Іван", "1025400524313"],
            ["-;+", "ab;cd"]]
    _, statements, rejects, cov = _debris_run(tmp_path, rows, [
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "",
         "type_name": "name"},
        {"column": "c1", "prop": "Person:idNumber", "entity": "person",
         "type_name": "identifier", "why": ""}])

    for col in ("c0", "c1"):
        assert cov[col] == {"status": "bound",
                            "role": "key" if col == "c0" else None,
                            "values": 2, "offered": 1, "rejected": 1,
                            "declined": 0, "structural": 0,
                            "outside_selection": 0, "unaccounted": 0}, col
    refused = {r["column"]: r for r in rejects if r["row"] == 2}
    assert refused["c0"]["value"] == "-;+"
    assert refused["c1"]["value"] == "ab;cd"
    assert len(statements) == 2


def test_the_reject_summary_is_still_value_free(tmp_path):
    frame, vplan, entities, statements, rejects = _engine_run(tmp_path)
    s = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    blob = json.dumps(s, ensure_ascii=False)
    for forbidden in ("NK-26WkuEefVYLmYk8aTHMC9E", "NKAnkFpprSmpkrtMWnsUASAk",
                      "Q6319"):
        assert forbidden not in blob
    assert s["rejects"]["by_stage"] == {"normalize": 1, "emit": 1}


def test_a_row_that_produced_no_entity_still_accounts_for_its_values(tmp_path):
    """The key column is empty on row 2, so followthemoney computes no id and
    drops the whole entity — taking a perfectly good name with it. The engine
    logs that as one line about the entity and says nothing about the values,
    so the values get their own reject lines here."""
    rows = [["Код", "ПІБ"],
            ["k1", "Коваленко Іван Петрович"],
            ["", "Шевченко Ольга Миколаївна"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/k.csv", "d" * 64, "dddddddddddd/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c1", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}],
        decisions=[],
    )
    csv_path = str(tmp_path / "k.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    _, statements, emit_rejects = execute(mapping, vplan, frame, CAT, "run1", {})
    assert len(statements) == 1
    assert [r["row"] for r in emit_rejects] == [2]
    assert emit_rejects[0]["column"] == "c1"
    assert emit_rejects[0]["reason"] == (
        "no entity was produced for this row, so every value on it was dropped")
    cov = summarise(frame, vplan, [], statements, rejects + emit_rejects, CAT,
                    CFG, {"run_id": "run1"})["coverage"]
    assert cov["c1"]["unaccounted"] == 0


def test_two_columns_on_one_property_do_not_vanish_without_a_word(tmp_path):
    """`compile_mapping` keys an entity's properties by name, so two bindings
    naming one (entity, property) of a type that may NOT merge — a date is a
    fact, not a value of a set — leave the mapping with a single source and
    the losing column with values, no statements and nothing to say why."""
    rows = [["first_seen", "last_seen"],
            ["2021-03-01", "2024-01-05"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/d.csv", "e" * 64, "eeeeeeeeeeee/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:retrievedAt", "entity": "person",
                   "type_name": "date", "why": ""},
                  {"column": "c1", "prop": "Person:retrievedAt", "entity": "person",
                   "type_name": "date", "why": ""}],
        decisions=[],
    )
    csv_path = str(tmp_path / "d.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    _, statements, emit_rejects = execute(mapping, vplan, frame, CAT, "run1", {})
    lost = {b["column"] for b in vplan.bindings} - {s.column_id for s in statements}
    assert len(lost) == 1
    assert {r["column"] for r in emit_rejects} == lost
    assert emit_rejects[0]["reason"] == (
        "the compiled mapping reads no column for this property, so nothing "
        "on this column was ever offered to the engine")
    cov = summarise(frame, vplan, [], statements, rejects + emit_rejects, CAT,
                    CFG, {"run_id": "run1"})["coverage"]
    assert all(a["unaccounted"] == 0 for a in cov.values())


def test_two_columns_feeding_one_property_each_keep_their_provenance(tmp_path):
    """`countries` and `country_codes` both feed `Address:country` (the ICIJ
    addresses etalon, J3). The mapping reads both (`columns: [c1, c2]`, which
    `ftm map` reproduces standalone), and provenance is still per CELL: a
    value that came from one column is stamped with that column, a value
    both columns hold on one row is one value on the entity and TWO
    statements — one per cell that offered it — so each column's claim
    closes against its own rows and the ledger stays valid."""
    rows = [["name", "countries", "country_codes"],
            ["Kyiv, Khreshchatyk 1", "Ukraine", "UKR"],
            ["Limassol, Makariou 5", "Cyprus", None],
            ["Valletta, Republic St", None, "MLT"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/a.csv", "a" * 64, "aaaaaaaaaaaa/s", CFG)
    vplan = ValidatedPlan(
        subject="Address",
        entities=[{"key": "addr", "schema": "Address", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Address:full", "entity": "addr",
                   "type_name": "address", "why": ""},
                  {"column": "c1", "prop": "Address:country", "entity": "addr",
                   "type_name": "country", "why": ""},
                  {"column": "c2", "prop": "Address:country", "entity": "addr",
                   "type_name": "country", "why": ""}],
        decisions=[],
    )
    csv_path = str(tmp_path / "a.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    assert mapping["entities"]["addr"]["properties"]["country"] == {
        "columns": ["c1", "c2"], "split": MULTIVALUE_JOIN}
    entities, statements, emit_rejects = execute(mapping, vplan, frame, CAT,
                                                 "run1", {})
    assert emit_rejects == []
    by_row = {}
    for s in statements:
        if s.prop == "country":
            by_row.setdefault(s.row, []).append((s.column_id, s.value))
    assert sorted(by_row[1]) == [("c1", "ua"), ("c2", "ua")]
    assert by_row[2] == [("c1", "cy")]
    assert by_row[3] == [("c2", "mt")]
    assert {e["properties"]["country"][0] for e in entities} == {"ua", "cy", "mt"}
    s = summarise(frame, vplan, entities, statements, rejects + emit_rejects,
                  CAT, CFG, {"run_id": "run1"})
    claims = {c["column"]: c for c in s["claims"] if c["prop"] == "Address:country"}
    assert claims["c1"]["expected"] == 2 and claims["c1"]["emitted"] == 2
    assert claims["c2"]["expected"] == 2 and claims["c2"]["emitted"] == 2
    assert all(c["shortfall"] == 0 and c["over_emission"] == 0
               for c in claims.values())
    assert s["coverage"]["c1"]["status"] == "bound"
    assert s["coverage"]["c2"]["status"] == "bound"
    assert s["coverage_totals"]["accounting_valid"] is True


def test_a_displaced_duplicate_binding_is_voiced_even_when_the_column_is_tapped(tmp_path):
    """The mask behind OFAC's silent 7 456. `compile_mapping` keys an
    entity's properties by local name, so of two surviving bindings for one
    (entity, property) only the last column is read. The `_NO_SOURCE` guard
    asked "is this COLUMN tapped by anyone in the query" — and another
    entity's tap on the displaced column answered yes, so not one reject was
    written. The gate now forbids the duplicate upstream; this pins the
    guard's granularity — (entity, property, column), not column — so any
    future displacement is voiced instead of silent."""
    rows = [["Прізвище", "Код", "Опис"],
            ["Коваленко", "1025400524313", "справа"],
            ["Шевченко", "5401103595", "провадження"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/dup.csv", "9" * 64, "999999999999/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "note", "schema": "Note", "keys": ["c0"]}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            # Two claims on (person, idNumber); compile keeps the LAST source.
            {"column": "c1", "prop": "Person:idNumber", "entity": "person",
             "type_name": "identifier", "why": ""},
            {"column": "c2", "prop": "Person:idNumber", "entity": "person",
             "type_name": "string", "why": ""},
            # Another entity taps the displaced column — the old mask.
            {"column": "c1", "prop": "Note:description", "entity": "note",
             "type_name": "string", "why": ""},
        ],
        decisions=[],
    )
    csv_path = str(tmp_path / "dup.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    _, statements, emit_rejects = execute(mapping, vplan, frame, CAT, "run1", {})
    displaced = [r for r in emit_rejects
                 if r["column"] == "c1" and r["prop"] == "Person:idNumber"]
    assert len(displaced) == 2, emit_rejects
    assert all("reads no column" in r["reason"] for r in displaced)
    # And the claim ledger closes on it: the displaced claim's rows are
    # rejected, not silently gone.
    s = summarise(frame, vplan, [], statements, rejects + emit_rejects, CAT,
                  CFG, {"run_id": "r"})
    claim = next(c for c in s["claims"]
                 if c["column"] == "c1" and c["prop"] == "Person:idNumber")
    assert claim["emitted"] == 0 and claim["rejected"] == 2
    assert claim["shortfall"] == 0
    assert s["coverage_totals"]["accounting_valid"] is True


def test_a_binding_dropped_before_emission_is_still_accounted_for(tmp_path):
    """A run that loses all of its output must not read as fully reconciled.

    `coverage` iterated the SURVIVING bindings, so a source whose every
    binding was rejected reported `coverage: {}` — no column, and therefore
    `unaccounted == 0` everywhere. Observed on
    `ua_war_sanctions.targets.simple.csv`: 5 621 rows in, 0 statements out,
    and an accounting that said everything added up because nothing was left
    to add.

    Schema 2 refines the destination without reopening the hole: `c0` here is
    the surviving entity's KEY, so its cells are consumed even though the
    property binding on the column was rejected — `structural`, with the
    rejection still in `decisions`. The reviewer's live case was the ICIJ
    intermediaries id column: 32 values reported as `dropped` while the
    surviving LegalEntity was being keyed on every one of them.
    """
    frame = _frame()
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[],
        decisions=[
            Decision("c0", "Organization:name", "person", "rejected",
                     "schema Person does not carry Organization:name", "rule"),
            Decision("c1", "unmapped", "none", "unmapped", "a row ordinal",
                     "model"),
            # A key-column verdict: same `column`, no property. Not a binding,
            # so it must not be mistaken for one.
            Decision("c0", None, "person", "accepted", "key fill 1.00", "rule"),
        ],
    )
    cov = summarise(frame, vplan, [], [], [], CAT, CFG, {"run_id": "r"})["coverage"]
    assert cov["c0"] == {"status": "structural", "role": "key", "values": 1,
                         "offered": 0, "rejected": 0, "declined": 0,
                         "structural": 1, "outside_selection": 0,
                         "unaccounted": 0}
    # The column the model declined never had output to lose, so its values are
    # `declined` and none of them is unaccounted — but it is still reported,
    # because a coverage report over the columns that survived is a coverage
    # report with the losses left out of the denominator.
    assert cov["c1"] == {"status": "unmapped", "role": None, "values": 1,
                         "offered": 0, "rejected": 0, "declined": 1,
                         "structural": 0, "outside_selection": 0,
                         "unaccounted": 0}


def test_a_dropped_column_no_survivor_consumes_is_still_a_loss(tmp_path):
    """The other half of the structural refinement: a key or binding of a
    declaration that did NOT survive protects nothing. The reviewer's live
    case was the procurement sheet's `процедура` column, which keyed a
    declaration `validate` later dropped — its values are output that was
    lost, and `structural` must not absorb them."""
    frame = _frame()
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[],
        decisions=[
            Decision("c1", "Organization:name", "org", "rejected",
                     "the declaration this bound to was dropped", "rule"),
        ],
    )
    cov = summarise(frame, vplan, [], [], [], CAT, CFG, {"run_id": "r"})["coverage"]
    assert cov["c1"] == {"status": "dropped", "role": None, "values": 1,
                         "offered": 0, "rejected": 0, "declined": 0,
                         "structural": 0, "outside_selection": 0,
                         "unaccounted": 1}


def test_a_repaired_column_is_not_reported_as_lost(tmp_path):
    """The last word on a column wins. A first-pass rejection the repair round
    settled — into an accepted binding, or into an honest `unmapped` — is not
    a binding lost before emission, and reporting it as one would put an
    `unaccounted` figure on a column nothing went wrong with."""
    frame = _frame()
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}],
        decisions=[
            Decision("c0", "Person:birthDate", "person", "rejected",
                     "acceptance 0.00 below 0.90 for type date", "rule"),
            Decision("c0", "Person:name", "person", "accepted", "names",
                     "model-repair"),
            Decision("c1", "Person:name", "person", "rejected",
                     "acceptance 0.00 below 0.90 for type name", "rule"),
            Decision("c1", None, None, "unmapped", "nothing fits",
                     "model-repair"),
        ],
    )
    cov = summarise(frame, vplan, [], [], [], CAT, CFG, {"run_id": "r"})["coverage"]
    assert cov["c0"]["status"] == "bound"
    assert cov["c1"]["status"] == "unmapped"
    assert cov["c1"]["unaccounted"] == 0


def test_a_run_that_binds_nothing_does_not_report_a_reconciled_run(tmp_path):
    """Zero surviving bindings over a non-empty frame.

    Measured on `reestrtz_2026_sample.csv`: 39 607 rows and 17 columns in, 0
    statements out, one propertyless entity per row written — and
    `coverage: {}`, so no column failed to add up and `summary.json` read as a
    fully reconciled run. `columns.mapped: 0` was on record but is not framed
    as an outcome, and `coverage`, the field that exists to make loss visible,
    was the silent one. It now reports all 17 columns and 652 220 values
    against 0 emitted.

    Declining is not a defect, so none of this is `unaccounted`: the values
    are `declined`, which is a destination and not a residue. What makes the
    run legible is that every column is REPORTED, and that `coverage_totals`
    puts what the source held next to what came out of it.
    """
    frame = _frame()
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[],
        decisions=[Decision(c, "unmapped", "none", "unmapped",
                            "no answer from the model for this column", "model")
                   for c in ("c0", "c1", "c2")],
    )
    s = summarise(frame, vplan, [], [], [], CAT, CFG, {"run_id": "r"})
    assert s["statements"] == 0
    assert set(s["coverage"]) == {"c0", "c1", "c2"}
    # `c0` is the surviving declaration's key, so it is consumed rather than
    # declined; the other two are honest abstentions.
    assert s["coverage"]["c0"]["status"] == "structural"
    assert all(a == {"status": "unmapped", "role": None, "values": 1,
                     "offered": 0, "rejected": 0, "declined": 1,
                     "structural": 0, "outside_selection": 0, "unaccounted": 0}
               for c, a in s["coverage"].items() if c != "c0")
    # The one place the whole file's arithmetic is stated: three values in,
    # none out. The ONE boolean is about the ledger, not the pipeline.
    assert s["coverage_totals"] == {
        "columns": 3,
        "statuses": {"bound": 0, "structural": 1, "dropped": 0,
                     "unmapped": 2, "undecided": 0},
        "values": 3, "offered": 0, "rejected": 0, "declined": 2,
        "structural": 1, "outside_selection": 0,
        "unaccounted": 0,
        "claims": {"claims": 0, "expected": 0, "emitted": 0, "rejected": 0,
                   "shortfall": 0, "over_emission": 0},
        "accounting_valid": True,
    }


def test_a_column_no_decision_mentions_is_unaccounted_not_declined(tmp_path):
    """`undecided` is the silent omission itself, and must not read as an
    abstention. A column with no verdict on record has no reason attached to
    it, so its values are unaccounted for, exactly as a dropped binding's are.
    It cannot happen in this pipeline — `propose` answers every column and
    `validate` records every answer — which is why the arithmetic has to keep
    saying so if one ever stops."""
    frame = _frame()
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[],
        decisions=[],
    )
    s = summarise(frame, vplan, [], [], [], CAT, CFG, {"run_id": "r"})
    assert all(a["status"] == "undecided" and a["declined"] == 0
               and a["unaccounted"] == a["values"] == 1
               for c, a in s["coverage"].items() if c != "c0")
    assert s["coverage"]["c0"]["status"] == "structural"
    assert s["coverage_totals"]["unaccounted"] == 2


def test_every_coverage_row_states_where_each_value_went(tmp_path):
    """One invariant over all four statuses: nothing is counted twice, and
    nothing the source held is left out of the row that reports its column."""
    frame, vplan, _, statements, rejects = _engine_run(tmp_path)
    s = summarise(frame, vplan, [], statements, rejects, CAT, CFG,
                  {"run_id": "run1"})
    for col, a in s["coverage"].items():
        assert a["status"] in ("bound", "structural", "dropped", "unmapped",
                               "undecided"), col
        assert a["values"] == (a["offered"] + a["rejected"] + a["declined"]
                               + a["structural"] + a["outside_selection"]
                               + a["unaccounted"]), col
        assert all(a[f] >= 0 for f in ("values", "offered", "rejected",
                                       "declined", "structural",
                                       "outside_selection", "unaccounted")), col
    totals = s["coverage_totals"]
    assert totals["columns"] == len(s["coverage"])
    assert sum(totals["statuses"].values()) == len(s["coverage"])
    for field in ("values", "offered", "rejected", "declined", "structural",
                  "unaccounted"):
        assert totals[field] == sum(a[field] for a in s["coverage"].values())
    assert totals["accounting_valid"] is True


def test_a_value_the_cell_did_not_produce_carries_no_cell_address(tmp_path):
    """FollowTheMoney fills some properties itself — an edge's `namesMentioned`
    is derived from its endpoints' captions — and every value of a property used
    to inherit the column bound to that property. Measured on the aircraft
    register: one Ownership edge carried two `namesMentioned` statements, both
    claiming column c8 row 6, one of them holding the operator's value from c10.
    The column also reported `emitted 1744` against `values 872` and drove the
    run's `unaccounted` negative — the arithmetic was the symptom, the false
    provenance was the fault."""
    rows = [["ПІБ", "Організація", "Підписант"],
            ["Коваленко Іван", "ТОВ Ромашка", "Шевченко Ольга"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv",
                        "a" * 64, "aaaaaaaaaaaa/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "org", "schema": "Organization", "keys": ["c1"]}],
        edges=[{"key": "member", "schema": "Membership",
                "source": "person", "target": "org"}],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c1", "prop": "Organization:name", "entity": "org",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "Membership:namesMentioned",
             "entity": "member", "type_name": "name", "why": ""},
        ],
        decisions=[])
    csv_path = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    _entities, statements, _rej = execute(mapping, vplan, frame, CAT, "run1", {})

    named = [s for s in statements if s.prop == "namesMentioned"]
    assert len(named) > 1, "the engine derives captions beside the bound value"
    from_cell = [s for s in named if s.value == "Шевченко Ольга"]
    derived = [s for s in named if s.value != "Шевченко Ольга"]
    assert from_cell and derived, "one bound value and at least one derived"
    # The bound one keeps its address.
    assert from_cell[0].column_id == "c2" and from_cell[0].row is not None
    assert from_cell[0].value_raw == "Шевченко Ольга"
    # The derived ones must not claim a cell they never came from.
    for s in derived:
        assert s.column_id is None, f"{s.value!r} claims column {s.column_id}"
        assert s.row is None and s.header is None and s.value_raw is None
        # NOR A CANONICALIZER. The address was stripped and this was not, so a
        # derived value still named the function that had processed a
        # DIFFERENT column's cell. `statements.csv` publishes the field, and a
        # reader auditing which canonicalizer produced which value would be
        # reading an answer about a cell that never fed this statement — the
        # same false provenance the address fix was for, one column over.
        assert s.canonicalizer is None, (
            f"{s.value!r} credits the canonicalizer {s.canonicalizer!r} of a "
            "column that did not produce it")
    # The bound one still names its own, or this test would pass by making the
    # field always empty.
    assert from_cell[0].canonicalizer

    # And the arithmetic that was the symptom: a derived value credited to a
    # column made the column's claim emit more than its rows offered, which
    # schema 2 reports as `over_emission` — never as a negative that a loss
    # elsewhere could cancel.
    from ftmap.build.emit import ledgers, _fold_output, coverage_totals
    cov, claims = ledgers(frame, vplan, _fold_output(statements, []))
    c2 = next(c for c in claims if c["column"] == "c2")
    assert c2["over_emission"] == 0
    assert coverage_totals(cov, claims)["accounting_valid"] is True


def test_the_streaming_core_and_the_collecting_wrapper_agree_exactly(tmp_path):
    """The failure this shape invites: the fast path writes something else.

    `execute()` is now a wrapper that collects what `execute_into` hands out,
    so the two cannot disagree by construction — which is exactly why it is
    worth asserting, because "collect it differently for speed" is the change
    someone will make later.
    """
    from ftmap.build.execute import Statement, execute_into

    rows = [["ПІБ", "Дата народження", "Фракція"],
            ["Коваленко Іван Петрович", "17.09.1980", "Слуга народу"],
            ["Шевченко Ольга Іванівна", "не вказано", "Слуга народу"],
            ["Мельник Петро", "01.02.1990", "Голос"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv",
                        "a" * 64, "aaaaaaaaaaaa/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "party", "schema": "Organization", "keys": ["c2"]}],
        edges=[{"key": "member", "schema": "Membership",
                "source": "person", "target": "party"}],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c1", "prop": "Person:birthDate", "entity": "person",
             "type_name": "date", "why": ""},
            {"column": "c2", "prop": "Organization:name", "entity": "party",
             "type_name": "name", "why": ""},
        ], decisions=[])
    csv_path = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)

    collected = execute(mapping, vplan, frame, CAT, "run1", {})

    streamed_statements, streamed_rejects, streamed_entities = [], [], []
    entities = execute_into(mapping, vplan, frame, CAT, "run1", {},
                            on_statement=streamed_statements.append,
                            on_reject=streamed_rejects.append,
                            on_entity=streamed_entities.append)

    assert entities == collected[0]
    assert streamed_statements == collected[1]
    assert streamed_rejects == collected[2]
    # `on_entity` fires once per DISTINCT id, at first sight — before any later
    # row merges into it — so the identities match while the property sets may
    # not, and the returned list is the merged form.
    assert [e["id"] for e in streamed_entities] == [e["id"] for e in entities]
    assert streamed_statements, "the fixture produced no statements"


def test_the_streamed_files_are_byte_identical_to_the_collected_ones(tmp_path):
    """The sinks are the same writers, opened once — asserted on the bytes.

    A run that streams and a run that collects have to produce the same
    `statements.csv` and the same `rejects.jsonl`, or the golden corpus is
    measuring one path while production takes the other.
    """
    from ftmap.build.emit import statement_writer, write_statements
    from ftmap.normalize.report import reject_writer, write_rejects

    statements = [Statement(entity_id="e1", schema="Person", prop="name",
                            value="Коваленко Іван", value_raw="Коваленко Іван",
                            file="/x.csv", sheet="", column_id="c0",
                            header="ПІБ", row=1, sha256="a" * 64,
                            run_id="run1", decided_by="model",
                            canonicalizer="whitespace", matchable=True)]
    rejects = [{"row": 2, "column": "c1", "prop": "Person:birthDate",
                "value": "не вказано", "reason": "unparseable",
                "canonicalizer": "date", "stage": "normalize"}]

    bulk_s, bulk_r = str(tmp_path / "b.csv"), str(tmp_path / "b.jsonl")
    write_statements(statements, bulk_s)
    write_rejects(rejects, bulk_r)

    stream_s, stream_r = str(tmp_path / "s.csv"), str(tmp_path / "s.jsonl")
    with statement_writer(stream_s) as write:
        for s in statements:
            write(s)
    with reject_writer(stream_r) as write:
        for r in rejects:
            write(r)

    import pathlib as _pathlib

    assert _pathlib.Path(stream_s).read_bytes() == _pathlib.Path(bulk_s).read_bytes()
    assert _pathlib.Path(stream_r).read_bytes() == _pathlib.Path(bulk_r).read_bytes()
    assert oct(os.stat(stream_s).st_mode)[-3:] == "600"
    assert oct(os.stat(stream_r).st_mode)[-3:] == "600"


def test_a_filtered_entity_says_which_column_selects_it_and_not_which_value(tmp_path):
    """The value-free artefact stays value-free.

    Which rows an entity is built from is structure, and `summary.json` is
    what the etalon and the review app read — so the COLUMN belongs in it. The
    value does not: it is a cell, `us_ofac_sdn` happens to spell its kinds in
    FtM's own vocabulary but a Ukrainian register spells them `ФОП` and
    `Товариство`, and `tests/test_boundary.py` exists because that distinction
    was got wrong twice.
    """
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    filtered = ValidatedPlan(
        subject=vplan.subject,
        entities=[{**vplan.entities[0],
                   "filter": {"column": "c2", "value": "Слуга народу"}}]
        + list(vplan.entities[1:]),
        edges=vplan.edges, bindings=vplan.bindings)
    st = summarise(frame, filtered, entities, statements, rejects, CAT, CFG,
                   {"run_id": "run1"})["structure"]
    assert st["entities"][0]["filter"] == "c2"
    assert st["entities"][1].get("filter") is None
    assert "Слуга народу" not in json.dumps(st, ensure_ascii=False)


def test_the_summary_says_how_many_instances_each_declared_entity_produced(tmp_path):
    """A declaration and what it produced are different facts, and only the
    first was published.

    `summary.json` counted entities BY SCHEMA, so a plan declaring two
    entities of one schema, or one entity built from every row of a table that
    holds several kinds, read the same as a plan that got it right. Measured
    2026-08-28: `us_ofac_sdn` emitted 20 054 Person from a file whose Person
    rows number 7 456, and nothing in the value-free artefact could say so.
    A count is not a value, so this belongs here.
    """
    frame, vplan, entities, statements, rejects = _run(tmp_path)
    st = summarise(frame, vplan, entities, statements, rejects, CAT, CFG,
                   {"run_id": "run1"})["structure"]
    assert [e["instances"] for e in st["entities"]] == [1, 1]


def test_a_hand_deleted_entity_block_still_writes_no_source_rejects(tmp_path):
    """The `_NO_SOURCE` backstop exists for a mapping this pipeline did not
    compile — a hand-edited document, a replayed plan. Scoping it to each
    query's own entities silently skipped a binding whose entity appears in
    NO query: its values fell into `unaccounted` with no reject line saying
    why."""
    frame, vplan = _frame(), _vplan()
    csv_path = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    del mapping["entities"]["party"]
    del mapping["entities"]["member"]  # the edge that referenced it
    _, _, rejects = execute(mapping, vplan, frame, CAT, "run1", {})
    orphaned = [r for r in rejects if r["column"] == "c2"]
    assert orphaned, "the deleted entity's column wrote no reject at all"
    assert all("reads no column" in r["reason"] for r in orphaned)


def test_a_quoted_cell_the_engine_unquotes_is_still_its_own_column_s(tmp_path):
    """The MPs table (`work-c3b`): four columns fed `Person:name`, the engine
    stored «"Фітнес-центр"» from `old_post` without its quotes, no column's
    canonical reading matched, and the statement was handed to the first
    column that held anything — the full name — which over-emitted by one
    while `old_post` fell short by one. The ledger read as invalid on a run
    that had lost nothing. Compared without quotes, case and spacing, the
    cell that produced the value says so."""
    # Cyrillic headers: a Latin header over a Cyrillic first row reads as a
    # label row to the layout, and this test is about the tap, not that.
    rows = [["ПІБ", "Попередня посада"],
            ["Дмитрук Артем Геннадійович", '"Фітнес-центр"'],
            ["Ісаєнко Дмитро Валерійович", "доцент кафедри"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/corpus/m.csv", "b" * 64, "bbbbbbbbbbbb/s", CFG)
    assert len(frame.rows) == 2
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "mp", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "mp",
                   "type_name": "name", "why": ""},
                  {"column": "c1", "prop": "Person:name", "entity": "mp",
                   "type_name": "name", "why": ""}],
        decisions=[],
    )
    csv_path = str(tmp_path / "m.csv")
    rejects = normalize_frame(frame, vplan, csv_path)
    mapping = compile_mapping(vplan, frame, csv_path)
    entities, statements, emit_rejects = execute(mapping, vplan, frame, CAT,
                                                 "run1", {})
    by_row = {}
    for st in statements:
        if st.prop == "name":
            by_row.setdefault(st.row, []).append(st.column_id)
    assert sorted(by_row[1]) == ["c0", "c1"], by_row
    s = summarise(frame, vplan, entities, statements, rejects + emit_rejects,
                  CAT, CFG, {"run_id": "run1"})
    claims = {c["column"]: c for c in s["claims"] if c["prop"] == "Person:name"}
    assert claims["c0"]["emitted"] == 2 and claims["c0"]["over_emission"] == 0
    assert claims["c1"]["emitted"] == 2 and claims["c1"]["shortfall"] == 0
    assert s["coverage_totals"]["accounting_valid"] is True
