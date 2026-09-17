import csv as csvmod
import os

import pytest
import yaml
from followthemoney import model as ftm_model
from followthemoney.exc import InvalidMapping
from followthemoney.mapping import QueryMapping

from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.build.execute import execute
from ftmap.plan.compile import (NO_ENTITY_REASON, UnsafeWorkspacePath,
                                compile_mapping, compile_queries,
                                ensure_mappable_root,
                                normalize_frame, unmappable_reason,
                                write_mapping)
from ftmap.plan.validate import ValidatedPlan
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()

ROWS = [["ПІБ", "Дата народження", "Фракція", "Посада"],
        ["Коваленко Іван Петрович", "17.09.1980", "Слуга народу", "депутат"],
        ["Шевченко Ольга Іванівна", "не вказано", "Слуга народу", "секретар"]]


def _vplan():
    return ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0", "c1"]},
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
            {"column": "c3", "prop": "Membership:role", "entity": "member",
             "type_name": "string", "why": ""},
        ],
    )


def _frame():
    return build_frame(Grid(rows=ROWS, sheet="s", merges=[]),
                       "/x.csv", "0" * 64, "sid/s", CFG)


def test_a_plan_with_no_entity_is_refused_by_followthemoney_and_caught_first(tmp_path):
    """The condition `unmappable_reason` exists to catch, asserted against the
    engine rather than against a belief about it: `QueryMapping` raises
    `InvalidMapping` on the document an entity-less plan compiles to. 20 of 244
    private-corpus sources reached that raise, and a source that raises writes
    no summary — so its columns left the corpus coverage totals with it."""
    empty = ValidatedPlan(subject="Vehicle", entities=[], edges=[], bindings=[])
    mapping = compile_mapping(empty, _frame(), str(tmp_path / "n.csv"))
    assert mapping["entities"] == {}
    with pytest.raises(InvalidMapping):
        QueryMapping(ftm_model, mapping, key_prefix="sid/s")

    assert unmappable_reason(empty) == NO_ENTITY_REASON
    assert unmappable_reason(_vplan()) is None


def test_an_attachment_compiles_to_the_same_entity_reference_an_edge_gets(tmp_path):
    """`Sanction:entity` and its kin are emitted exactly the way an edge's
    endpoints always were — `{"entity": <plan key>}` in the query — so `ftm
    map` runs the document unchanged and the engine, not ftmap, wires the
    reference. The attachment is validate's, derived; compile only writes it."""
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "sanction", "schema": "Sanction", "keys": []}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "Sanction:program", "entity": "sanction",
             "type_name": "string", "why": ""},
        ],
        attachments=[{"entity": "sanction", "prop": "Sanction:entity",
                      "target": "person"}],
    )
    mapping = compile_mapping(vplan, _frame(), str(tmp_path / "n.csv"))
    assert mapping["entities"]["sanction"]["properties"]["entity"] == \
        {"entity": "person"}
    # And the engine accepts the document as written.
    QueryMapping(ftm_model, mapping, key_prefix="sid/s")


def test_normalized_csv_holds_canonical_values_and_reports_rejects(tmp_path):
    out = str(tmp_path / "n.csv")
    rejects = normalize_frame(_frame(), _vplan(), out)
    with open(out, encoding="utf-8") as fh:
        rows = list(csvmod.DictReader(fh))
    assert rows[0]["c1"] == "1980-09-17"
    assert rows[1]["c1"] == ""
    assert rows[0]["_row"] == "1"
    assert [r["value"] for r in rejects] == ["не вказано"]
    assert oct(os.stat(out).st_mode)[-3:] == "600"


def test_mapping_is_valid_ftm_and_produces_the_three_entities(tmp_path):
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), _vplan(), out)
    mapping = compile_mapping(_vplan(), _frame(), out)
    qm = QueryMapping(ftm_model, mapping, key_prefix="sid/s")
    schemata = []
    for record in qm.source.records:
        schemata.append(sorted(e.schema.name for e in qm.map(record).values()))
    assert schemata[0] == ["Membership", "Organization", "Person"]


def test_edge_endpoints_are_wired_to_the_entity_ids(tmp_path):
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), _vplan(), out)
    qm = QueryMapping(ftm_model, compile_mapping(_vplan(), _frame(), out),
                      key_prefix="sid/s")
    record = next(iter(qm.source.records))
    ents = qm.map(record)
    assert ents["member"].get("member") == [ents["person"].id]
    assert ents["member"].get("organization") == [ents["party"].id]


def test_entity_ids_are_stable_across_runs(tmp_path):
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), _vplan(), out)
    mapping = compile_mapping(_vplan(), _frame(), out)
    ids = []
    for _ in range(2):
        qm = QueryMapping(ftm_model, mapping, key_prefix="sid/s")
        ids.append([e.id for r in qm.source.records for e in qm.map(r).values()])
    assert ids[0] == ids[1]


def test_written_yaml_is_the_document_the_ftm_cli_expects(tmp_path):
    """Measured: the bare query makes `ftm map` emit nothing at all, which
    reads like an empty source rather than a malformed file."""
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), _vplan(), out)
    path = str(tmp_path / "m.yml")
    write_mapping(compile_mapping(_vplan(), _frame(), out), path,
                  _frame().source_id)
    with open(path, encoding="utf-8") as fh:
        got = yaml.safe_load(fh)
    query = got[_frame().source_id]["queries"][0]
    assert "entities" in query and "csv_url" in query


def test_an_edge_carries_its_own_properties(tmp_path):
    """A Membership has a `role` that belongs to neither endpoint. Without edge
    keys as legal binding targets, no relationship property could be populated
    at all."""
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), _vplan(), out)
    qm = QueryMapping(ftm_model, compile_mapping(_vplan(), _frame(), out),
                      key_prefix="sid/s")
    record = next(iter(qm.source.records))
    ents = qm.map(record)
    assert ents["member"].get("role") == ["депутат"]


def test_a_keyless_entity_falls_back_to_the_row_number(tmp_path):
    vplan = _vplan()
    vplan.entities[0]["keys"] = []
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), vplan, out)
    mapping = compile_mapping(vplan, _frame(), out)
    assert mapping["entities"]["person"]["keys"] == ["_row"]


def test_two_keyless_entities_do_not_share_an_id(tmp_path):
    """Measured against the engine: without `key_literal`, a Person and an
    Organization that both fall back to `_row` on the same row hash to one
    identical id, and the id-based dedup downstream then drops one of them
    while its statements keep pointing at the survivor."""
    vplan = _vplan()
    for e in vplan.entities:
        e["keys"] = []
    out = str(tmp_path / "n.csv")
    normalize_frame(_frame(), vplan, out)
    qm = QueryMapping(ftm_model, compile_mapping(vplan, _frame(), out),
                      key_prefix="sid/s")
    record = next(iter(qm.source.records))
    ids = {k: v.id for k, v in qm.map(record).items()}
    assert len(set(ids.values())) == len(ids), ids


# The first id carries no digit AND no separator, which is what this fixture
# needs: as of 2026-08-29 a Latin code WITH a separator is an identifier
# without one, which is what lets the aircraft register's `UR-ALUR` bind. The
# same OpenSanctions id written without its hyphen is still refused, and still
# for the reason this test is about.
OSINT_ROWS = [["id", "name"],
              ["NKAnkFpprSmpkrtMWnsUASAk", "Товариство Схід"],
              ["NK-228jBYSTdUSvbZvsKsiHh6", "Товариство Захід"]]


def _osint_frame(rows=None):
    return build_frame(Grid(rows=rows or OSINT_ROWS, sheet="s", merges=[]),
                       "/x.csv", "0" * 64, "sid/s", CFG)


def _osint_plan(bind_the_key: bool):
    """`c0` keys the Organization. Whether it is ALSO bound to a property is
    the variable."""
    bindings = [{"column": "c1", "prop": "Organization:name", "entity": "org",
                 "type_name": "name", "why": ""}]
    if bind_the_key:
        bindings.insert(0, {"column": "c0", "prop": "Organization:registrationNumber",
                            "entity": "org", "type_name": "identifier", "why": ""})
    return ValidatedPlan(
        subject="Organization",
        entities=[{"key": "org", "schema": "Organization", "keys": ["c0"]}],
        edges=[], bindings=bindings)


def test_a_key_refused_as_a_property_still_identifies_its_row(tmp_path):
    """Measured on `ua_war_sanctions.targets.simple.csv`: `c0` is both
    `keys: ["c0"]` and bound to an `identifier` property. 183 of its 5621
    OpenSanctions ids carry no digit, so `_canon_identifier` refused them, the
    cell was written empty, `EntityMapping.compute_key` found no key material
    and `QueryMapping.map` dropped the WHOLE ROW — 1645 good values on the
    other nine columns went with it. "Is this a usable identifier?" and "which
    entity is this row about?" are different questions about one cell, and the
    first one's answer must not decide the second."""
    frame = _osint_frame()
    vplan = _osint_plan(bind_the_key=True)
    out = str(tmp_path / "n.csv")
    rejects = normalize_frame(frame, vplan, out)
    # The property verdict is unchanged: the first id really is not one.
    assert [(r["column"], r["row"]) for r in rejects] == [("c0", 1)]

    qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                      key_prefix="sid/s")
    mapped = [qm.map(record) for record in qm.source.records]
    assert [sorted(m) for m in mapped] == [["org"], ["org"]], mapped
    # The row survives WITH the rest of its content, which is what was lost.
    assert [m["org"].get("name") for m in mapped] == [["Товариство Схід"],
                                                      ["Товариство Захід"]]
    # And it survives without inventing the property that was refused.
    assert mapped[0]["org"].get("registrationNumber") == []
    assert mapped[1]["org"].get("registrationNumber") == ["NK228jBYSTdUSvbZvsKsiHh6"]
    assert len({m["org"].id for m in mapped}) == 2


def test_binding_a_key_column_does_not_move_the_entity_id(tmp_path):
    """The invariant behind the fix, stated as the thing a reader can check:
    an entity's id is a function of its key columns' cells and nothing else.
    Before, `NK-228jBYSTdUSvbZvsKsiHh6` seeded the hash as
    `NK228jBYSTdUSvbZvsKsiHh6` when the column happened to carry an
    `identifier` binding — so an id depended on a decision about a PROPERTY,
    and adding or removing a binding silently renamed every entity in the
    source."""
    frame = _osint_frame()
    ids = []
    for bound, name in ((True, "bound.csv"), (False, "unbound.csv")):
        vplan = _osint_plan(bind_the_key=bound)
        out = str(tmp_path / name)
        normalize_frame(frame, vplan, out)
        qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                          key_prefix="sid/s")
        ids.append([qm.map(r)["org"].id for r in qm.source.records])
    assert ids[0] == ids[1]


def test_the_key_collapses_whitespace_and_nothing_else(tmp_path):
    """What "light normalization" buys and what it refuses to buy. Two
    spellings of one id that differ only in invisible whitespace are one
    entity; two that differ in a character the source actually wrote — the
    hyphen `_canon_identifier` strips — are two."""
    rows = [["id", "name"],
            ["NK-228 jBYST", "a"], ["  NK-228  jBYST  ", "b"],
            ["NK228jBYST", "c"]]
    frame = _osint_frame(rows)
    vplan = _osint_plan(bind_the_key=True)
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)
    qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                      key_prefix="sid/s")
    ids = [qm.map(r)["org"].id for r in qm.source.records]
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_an_empty_key_cell_still_produces_no_entity(tmp_path):
    """The row-drop this fix narrows, not one it removes. A row with no key
    material genuinely cannot be identified, and `build.execute` reports every
    value on it as lost — that reporting must keep its job."""
    rows = [["id", "name"], ["", "Товариство Схід"]]
    frame = _osint_frame(rows)
    vplan = _osint_plan(bind_the_key=True)
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)
    qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                      key_prefix="sid/s")
    assert [qm.map(r) for r in qm.source.records] == [{}]


def test_key_literal_boundary_is_unambiguous_when_one_key_prefixes_another(tmp_path):
    """`key_literal` alone is not enough while FtM concatenates it with the key
    column's value with no separator (`EntityMapping.compute_key`): an entity
    keyed "x" with key value "21" would hash identically to one keyed "x2"
    with key value "1", since "x" + "21" == "x2" + "1" as raw bytes. Measured
    on a real public deputies file: `voting_identifier_edge` (Membership, row
    21) collided with `voting_identifier_edge2` (Employment, row 1) for
    exactly this reason — `propose.py`'s own collision-rename scheme names a
    second edge by appending a digit to the first, which is exactly the shape
    that later collides with a plain digit row index."""
    rows = [["A", "B"], ["21", "1"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "x", "schema": "Person", "keys": ["c0"]},
                  {"key": "x2", "schema": "Person", "keys": ["c1"]}],
        edges=[],
    )
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)
    qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                      key_prefix="sid/s")
    record = next(iter(qm.source.records))
    ids = {k: v.id for k, v in qm.map(record).items()}
    assert len(set(ids.values())) == len(ids), ids


def test_a_year_column_survives_normalization_and_a_measurement_column_does_not(tmp_path):
    """The end of the road for the МВС vehicle registry's numeric columns.
    `normalize_frame` writes what the engine reads, so the column decision has
    to reach here too or a year column that passed the gate arrives empty.
    MAKE_YEAR keeps every value and reaches `Vehicle:buildDate`; CAPACITY,
    bound to the same property and type, keeps none — and every cell it loses
    has a reject line naming the column, not a silent blank.

    Values are verbatim from `reestrtz_2026_sample.csv`.
    """
    rows = [["MAKE_YEAR", "CAPACITY"],
            ["2015", "3471"],
            ["2013", "1956"],
            ["1949", "2993"],
            ["2026", "998"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    vplan = ValidatedPlan(
        subject="Vehicle",
        entities=[{"key": "car", "schema": "Vehicle", "keys": ["c0"]}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Vehicle:buildDate", "entity": "car",
             "type_name": "date", "why": ""},
            {"column": "c1", "prop": "Vehicle:registrationDate",
             "entity": "car", "type_name": "date", "why": ""},
        ],
    )
    out = str(tmp_path / "n.csv")
    rejects = normalize_frame(frame, vplan, out)
    with open(out, encoding="utf-8") as fh:
        got = list(csvmod.DictReader(fh))
    assert [r["c0"] for r in got] == ["2015", "2013", "1949", "2026"]
    assert [r["c1"] for r in got] == ["", "", "", ""]
    assert {r["column"] for r in rejects} == {"c1"}
    assert len(rejects) == 4
    assert all("column" in r["reason"] for r in rejects)

    qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                      key_prefix="sid/s")
    record = next(iter(qm.source.records))
    assert qm.map(record)["car"].get("buildDate") == ["2015"]


def _two_column_key_plan():
    """One entity keyed on two columns, and nothing else to interfere."""
    return ValidatedPlan(
        subject="Person", entities=[
            {"key": "person", "schema": "Person", "keys": ["c0", "c1"]}],
        edges=[], bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""}])


def _ids_for(tmp_path, rows, vplan, name="n.csv"):
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    out = str(tmp_path / name)
    normalize_frame(frame, vplan, out)
    qm = QueryMapping(ftm_model, compile_mapping(vplan, frame, out),
                      key_prefix="sid/s")
    return [{k: e.id for k, e in qm.map(record).items()}
            for record in qm.source.records]


def test_distinct_composite_key_tuples_do_not_share_an_id(tmp_path):
    """`("a", "bc")` and `("ab", "c")` are two entities, and were one.

    FollowTheMoney's `EntityMapping.compute_key` sorts the key values and
    concatenates them with no separator — `digest.update(value)` per value, in
    `followthemoney/mapping/entity.py`. Sorted, both tuples above are
    `[b"a", b"bc"]` and `[b"ab", b"c"]`, and both concatenate to `abc`. Two
    different people, one id, and the id-based dedup in `build.execute` then
    drops one of them while its statements keep pointing at the survivor.

    The sort is worth noting on its own: the ORDER of `keys:` in the mapping
    document cannot fix this, because the engine does not preserve it.
    """
    rows = [["Перше", "Друге"], ["a", "b"], ["a", "bc"], ["ab", "c"]]
    ids = _ids_for(tmp_path, rows, _two_column_key_plan())
    assert ids[1]["person"] != ids[2]["person"], "distinct tuples merged"
    assert len({row["person"] for row in ids}) == 3


def test_a_single_column_key_keeps_the_id_it_already_had(tmp_path):
    """The composite encoding is for multi-column keys ONLY.

    A single-column key is unambiguous already — there is nothing to
    concatenate — so touching it would move every id in every published run
    for no defect. This asserts the id is exactly what the engine produces
    from the plain key field, computed here rather than pasted, so it stays
    true if the key form legitimately changes for some other reason.
    """
    rows = [["ПІБ", "Друге"],
            ["Коваленко Іван Петрович", "x"], ["Шевченко Ольга", "y"]]
    plan = ValidatedPlan(
        subject="Person", entities=[
            {"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[], bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""}])
    ids = _ids_for(tmp_path, rows, plan)

    mapping = compile_mapping(plan, build_frame(
        Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG),
        str(tmp_path / "n.csv"))
    assert mapping["entities"]["person"]["keys"] == ["c0__key"]
    assert len({row["person"] for row in ids}) == 2


def test_a_composite_key_of_entirely_empty_cells_still_produces_no_entity(tmp_path):
    """The one way a length-prefixed encoding could invent entities.

    `compute_key` returns None when every key value is empty, and that is what
    keeps a blank row from becoming an entity. An encoding that wrote
    `0:|0:|` for a row of empty key cells would be non-empty, and every blank
    row in the corpus would start emitting a property-less entity.
    """
    # A third, filled column, or `build_frame` classifies the blank row as
    # furniture and drops it before the engine ever sees it — which would make
    # this test pass without testing anything.
    plan = ValidatedPlan(
        subject="Person", entities=[
            {"key": "person", "schema": "Person", "keys": ["c0", "c1"]}],
        edges=[], bindings=[
            {"column": "c2", "prop": "Person:position", "entity": "person",
             "type_name": "string", "why": ""}])
    rows = [["ПІБ", "Друге", "Посада"],
            ["Коваленко Іван", "х", "депутат"],
            ["", "", "секретар"]]
    ids = _ids_for(tmp_path, rows, plan)
    assert len(ids) == 2
    assert "person" in ids[0]
    assert "person" not in ids[1], "an all-empty composite key made an entity"


def test_an_edge_over_two_multi_column_endpoints_is_unambiguous(tmp_path):
    """The edge inherits its endpoints' key fields, so it inherits the defect.

    An edge keys on the union of both endpoints' key fields — two or more
    fields whenever either endpoint has more than one — so a fix applied only
    to entities would leave every Membership, Ownership and Directorship in
    the corpus still concatenating without a boundary.
    """
    plan = ValidatedPlan(
        subject="Person", entities=[
            {"key": "person", "schema": "Person", "keys": ["c0", "c1"]},
            {"key": "party", "schema": "Organization", "keys": ["c2"]}],
        edges=[{"key": "member", "schema": "Membership",
                "source": "person", "target": "party"}],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}])
    rows = [["Перше", "Друге", "Третє"],
            ["a", "bc", "org"], ["ab", "c", "org"]]
    ids = _ids_for(tmp_path, rows, plan)
    assert ids[0]["member"] != ids[1]["member"], "distinct edges merged"
    assert ids[0]["person"] != ids[1]["person"]


# Measured against FollowTheMoney 4.10.2's own loader, not assumed. A directory
# named for each of these was created, a mapping written pointing into it, and
# `QueryMapping(...).source.records` asked for its rows:
#
#   'with space'   OK        'hash#one'      FileNotFoundError .../hash
#   'Кирилиця'     OK        'quest?ion'     FileNotFoundError .../quest
#   'pct%20here'   OK        'dollar$HOME'   FileNotFoundError .../dollar/Users/…
#   'brack[et]'    OK        'tab\there'     FileNotFoundError .../tabhere/…
#   'star*x'       OK        'cr\rhere'      FileNotFoundError .../crhere/…
#   "quote'q"      OK        'lf\nhere'      FileNotFoundError .../lfhere/…
#   'amp&and'      OK
#   'semi;colon'   OK
#
# `#` and `?` truncate at the fragment and the query. `$HOME` is EXPANDED —
# the loader ran a shell-style substitution on a path we handed it. Tab, CR
# and LF are stripped. `%` is not decoded, which is why it stays supported.
UNSAFE_ROOTS = ["a#b", "a?b", "a$HOME", "a\tb", "a\rb", "a\nb"]
SAFE_ROOTS = ["plain", "with space", "Кирилиця", "pct%20here", "star*x"]


@pytest.mark.parametrize("name", SAFE_ROOTS)
def test_a_supported_workspace_path_round_trips_through_the_ftm_loader(
        tmp_path, name):
    """The claim `write_mapping` exists to support, asserted against the engine.

    `ftm map <file>.yml` reproducing a run without this pipeline is the whole
    argument that the model is a front end and not the runtime. It is only
    true if the loader can open the CSV the mapping names, and the mapping
    names it as an unencoded `file://` URL.
    """
    root = tmp_path / name
    root.mkdir()
    out = str(root / "n.csv")
    normalize_frame(_frame(), _vplan(), out)
    mapping = compile_mapping(_vplan(), _frame(), out)
    qm = QueryMapping(ftm_model, mapping, key_prefix="sid/s")
    assert sum(1 for _ in qm.source.records) == 2


@pytest.mark.parametrize("name", UNSAFE_ROOTS)
def test_an_unsafe_workspace_root_is_refused_before_anything_runs(tmp_path, name):
    """Fail closed, and fail EARLY.

    Each of these makes the loader read a different file than the one the
    mapping names — and `$` makes it read a path the run never chose at all.
    Refusing at the door costs a message; discovering it after inventory and a
    corpus of model calls costs the run.
    """
    with pytest.raises(UnsafeWorkspacePath) as excinfo:
        ensure_mappable_root(str(tmp_path / name))
    # The message has to name the character, or the operator is told their
    # perfectly ordinary-looking directory is bad and not which byte did it.
    assert repr(name[1]) in str(excinfo.value)
    # And the check must not be refusing everything.
    ensure_mappable_root(str(tmp_path / "fine"))


def test_the_synthetic_key_field_is_invisible_to_profiling_and_accounting(
        tmp_path):
    """A composite key field is identity material, not source data.

    It is written into the normalized CSV and read by the engine, and it must
    not appear anywhere a COLUMN is counted: `columns.total`, `values`,
    `emitted`, `unaccounted`, or any per-column coverage row. Measured across
    the golden corpus, every one of those figures is byte-identical before and
    after this encoding — 69 columns, 42 mapped, 2 624 values, 1 539 emitted,
    `unaccounted` 0 — and this is the unit-level guard on the same claim.
    """
    from ftmap.build.emit import coverage, coverage_totals
    from ftmap.build.execute import execute
    from ftmap.plan.compile import composite_field
    from ftmap.profile.columns import profile_frame

    frame, vplan = _frame(), _vplan()
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)

    # It really is in the file, or the rest of this asserts nothing.
    with open(out, encoding="utf-8") as fh:
        fields = next(csvmod.reader(fh))
    assert composite_field("person") in fields

    profiles = profile_frame(frame, CFG)
    assert all(not p.id.startswith("__ck_") for p in profiles)

    mapping = compile_mapping(vplan, frame, out)
    _ents, statements, _rej = execute(mapping, vplan, frame, CAT, "run1", {})
    cov = coverage(frame, vplan, statements, [])
    assert all(not column.startswith("__ck_") for column in cov)
    totals = coverage_totals(cov)
    # Four source columns over two data rows. A synthetic field counted as a
    # column would make this 5 and 10, and every coverage rate in every
    # published run would be computed over a denominator the source does not
    # have.
    assert totals["columns"] == len(frame.columns) == 4
    assert totals["values"] == 8


def _split_plan():
    """One table, two kinds of row, declared by a column.

    The OpenSanctions shape: `c1` says what each row IS, and a single
    non-conditional plan can only ever be right about one of its values.
    """
    return ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Person"}},
                  {"key": "org", "schema": "Organization", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Organization"}}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c0", "prop": "Organization:name", "entity": "org",
             "type_name": "name", "why": ""},
        ],
    )


SPLIT_ROWS = [["name", "schema"],
              ["Коваленко Іван Петрович", "Person"],
              ["ТОВ Ромашка", "Organization"],
              ["Шевченко Ольга Іванівна", "Person"]]


def _split_frame():
    return build_frame(Grid(rows=SPLIT_ROWS, sheet="s", merges=[]),
                       "/x.csv", "0" * 64, "sid/s", CFG)


def test_entities_that_filter_on_different_values_compile_to_separate_queries(tmp_path):
    """FollowTheMoney puts `filters` on the SOURCE, so one query cannot hold
    two row selections. A mapping document may hold several queries, which is
    the form this compiles to."""
    out = str(tmp_path / "n.csv")
    normalize_frame(_split_frame(), _split_plan(), out)
    queries = compile_queries(_split_plan(), _split_frame(), out)
    assert len(queries) == 2
    assert [sorted(q["entities"]) for q in queries] == [["org"], ["person"]]
    assert [q["filters"] for q in queries] == [{"c1__is": "Organization"},
                                               {"c1__is": "Person"}]


def test_the_filter_column_is_written_verbatim_for_the_engine_to_match_on(tmp_path):
    """The filter compares strings against the record the engine reads, so the
    column has to be IN that record and has to hold the source's own spelling —
    not the canonical form, which is what every other field holds."""
    out = str(tmp_path / "n.csv")
    normalize_frame(_split_frame(), _split_plan(), out)
    with open(out, encoding="utf-8") as fh:
        rows = list(csvmod.DictReader(fh))
    assert [r["c1__is"] for r in rows] == ["Person", "Organization", "Person"]


def test_a_filtered_entity_is_built_from_its_own_rows_only(tmp_path):
    """The measurement this exists for: on `us_ofac_sdn` one plan declared
    Person, Organization, Vessel, CryptoWallet and Airplane and emitted 20 054
    of EACH from 20 079 rows — 976 of which are wallets. An entity must see
    the rows its filter selects and no others."""
    out = str(tmp_path / "n.csv")
    frame, vplan = _split_frame(), _split_plan()
    normalize_frame(frame, vplan, out)
    entities, _statements, _rejects = execute(
        compile_queries(vplan, frame, out), vplan, frame, CAT, "run1", {})
    by_schema = {}
    for e in entities:
        by_schema.setdefault(e["schema"], []).append(e)
    assert len(by_schema["Person"]) == 2
    assert len(by_schema["Organization"]) == 1


def test_a_row_thing_follows_every_bloc_and_its_attachment_stays_home(tmp_path):
    """An unfiltered non-party thing — the Sanction that describes every row
    — must live in EACH bloc's query: FollowTheMoney resolves an entity
    reference inside one query, so a sanction left in a base query could
    attach to nothing, and a base query holding only the sanction would emit
    one designation per row about nobody. Each copy keeps only the
    attachment whose target shares its query; a dangling `{"entity": ...}`
    reference is pruned rather than shipped for the engine to refuse."""
    plan = _split_plan()
    plan.entities.append({"key": "sanction", "schema": "Sanction", "keys": []})
    plan.bindings.append({"column": "c1", "prop": "Sanction:program",
                          "entity": "sanction", "type_name": "string",
                          "why": ""})
    plan.attachments.extend([
        {"entity": "sanction", "prop": "Sanction:entity", "target": "person"},
        {"entity": "sanction", "prop": "Sanction:entity", "target": "org"},
    ])
    out = str(tmp_path / "n.csv")
    normalize_frame(_split_frame(), plan, out)
    queries = compile_queries(plan, _split_frame(), out)
    assert [sorted(q["entities"]) for q in queries] == \
        [["org", "sanction"], ["person", "sanction"]]
    for q in queries:
        target = "org" if "org" in q["entities"] else "person"
        assert q["entities"]["sanction"]["properties"]["entity"] == \
            {"entity": target}
    # And the whole document still satisfies the engine.
    for q in queries:
        QueryMapping(ftm_model, q, key_prefix="sid/s")


def test_a_binding_in_another_query_is_not_reported_as_read_by_nothing(tmp_path):
    """`_NO_SOURCE` means "the compiled mapping reads no column for this
    property". With one query per kind of row that has to be asked of the
    QUERY, not of the plan: a column bound on an entity in another query is
    read, just not here.

    Asked of the plan, every row of every query emitted a false reject for
    every other query's columns — and `emit.coverage` counts emitted and
    rejected independently, so a value counted in both drives `unaccounted`
    negative, which the accounting exists to make impossible.
    """
    rows = [["name", "schema", "case"],
            ["Коваленко Іван Петрович", "Person", "справа 1"],
            ["Шевченко Ольга Іванівна", "Person", "справа 2"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Person"}},
                  {"key": "case", "schema": "CourtCase", "keys": ["c2"]}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "CourtCase:name", "entity": "case",
             "type_name": "name", "why": ""},
        ],
    )
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)
    _entities, statements, rejects = execute(
        compile_queries(vplan, frame, out), vplan, frame, CAT, "run1", {})
    emitted = {(s.column_id, s.value) for s in statements if s.column_id}
    for reject in rejects:
        assert (reject["column"], reject["value"]) not in emitted, (
            f"{reject['column']} was both emitted and rejected: {reject['reason']}")
    assert not [r for r in rejects if r["column"] == "c0"]


def test_values_on_rows_outside_every_selection_are_counted_as_such(tmp_path):
    """Row selections mean a bound column has values on rows its own entity is
    not built from.

    `us_ofac_sdn`'s name column holds a name on all 20 079 rows while the
    Person entity is built from 7 456 of them, and the plan answers two of the
    file's eight kinds. Those values are not emitted and the engine never
    refused them — it was never offered them — so before this they fell into
    `unaccounted`, the bucket that means the count and the deliverable
    disagree about what a value is. The loss is real, deliberate, and now has
    its own name.
    """
    rows = [["name", "schema"],
            ["Коваленко Іван Петрович", "Person"],
            ["ТОВ Ромашка", "Organization"],
            ["Судно Дніпро", "Vessel"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Person"}}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}],
    )
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)
    _entities, statements, rejects = execute(
        compile_queries(vplan, frame, out), vplan, frame, CAT, "run1", {})
    assert [s.value for s in statements] == ["Коваленко Іван Петрович"]

    from ftmap.build.emit import coverage
    row = coverage(frame, vplan, statements, rejects)["c0"]
    assert row["offered"] == 1
    assert row["outside_selection"] == 2
    assert row["unaccounted"] == 0


def test_a_column_bound_on_two_kinds_is_outside_neither(tmp_path):
    """One column may feed several entities — the OFAC name column feeds the
    Person and the Organization — and then a row belonging to either is
    covered. Counting per binding instead of per column reported the same
    value as emitted AND outside, and the corpus went to `unaccounted: -616`.
    """
    rows = [["name", "schema"],
            ["Коваленко Іван Петрович", "Person"],
            ["ТОВ Ромашка", "Organization"],
            ["Судно Дніпро", "Vessel"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Person"}},
                  {"key": "org", "schema": "Organization", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Organization"}}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""},
                  {"column": "c0", "prop": "Organization:name", "entity": "org",
                   "type_name": "name", "why": ""}],
    )
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, vplan, out)
    _entities, statements, rejects = execute(
        compile_queries(vplan, frame, out), vplan, frame, CAT, "run1", {})
    from ftmap.build.emit import ledgers, _fold_output
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    row = cov["c0"]
    assert row["offered"] == 2 and row["outside_selection"] == 1
    assert row["unaccounted"] == 0
    # And each bloc's claim reconciles against its OWN rows — the fan-out
    # that version 1 subtracted from one column count.
    by_prop = {c["prop"]: c for c in claims}
    assert by_prop["Person:name"]["expected"] == 1
    assert by_prop["Person:name"]["emitted"] == 1
    assert by_prop["Organization:name"]["expected"] == 1
    assert by_prop["Organization:name"]["emitted"] == 1


def test_an_edge_bound_column_is_counted_outside_its_endpoints_selection():
    """`_selection_of` was built from `vplan.entities` alone, so a column bound
    to an EDGE looked unfiltered — covered on every row — while the engine
    only ever read the filtered rows. Every role value on a non-selected row
    landed in `unaccounted`, reading as broken accounting."""
    from ftmap.build.emit import ledgers, _fold_output

    rows = [["Тип", "Назва", "Роль"],
            ["член", "Коваленко Іван", "голова"],
            ["інше", "Шевченко Ольга", "секретар"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    sel = {"column": "c0", "value": "член"}
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"],
                   "filter": sel},
                  {"key": "party", "schema": "Organization", "keys": ["c1"],
                   "filter": sel}],
        edges=[{"key": "member", "schema": "Membership",
                "source": "person", "target": "party"}],
        bindings=[
            {"column": "c1", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "Membership:role", "entity": "member",
             "type_name": "string", "why": ""},
        ],
    )
    cov, _claims = ledgers(frame, vplan, _fold_output([], []))
    # The `інше` row is outside the shared selection for BOTH columns: the
    # entity-bound name and the edge-bound role alike.
    assert cov["c1"]["outside_selection"] == 1
    assert cov["c2"]["outside_selection"] == 1


def test_the_written_mapping_reproduces_the_ids_not_only_the_entities(tmp_path):
    """THE ENVELOPE KEY IS NOT A LABEL, IT IS THE KEY PREFIX. FollowTheMoney
    derives an entity id from the mapping's keys prefixed by the dataset name.
    `execute` runs the query with `key_prefix=frame.source_id`; `write_mapping`
    wrapped the same query under the literal `ftmap`. So the document on disk
    reproduced the entities and not their identities.

    Measured 2026-08-30 on the ICIJ entities file: `ftm map --no-sign` produced
    2 916 `Organization` whose schemata and properties matched the run's own
    value for value, and **not one of the 2 916 ids matched**. Rewriting the
    envelope key to the source id matched all 2 916.

    `README.md` claims every run reproduces standalone with `ftm map`. That was
    true of the entities and false of their ids, which is the half a second
    source joins on.
    """
    frame = _frame()
    out = str(tmp_path / "n.csv")
    normalize_frame(frame, _vplan(), out)
    path = str(tmp_path / "m.yml")
    write_mapping(compile_mapping(_vplan(), frame, out), path,
                  dataset=frame.source_id)
    with open(path, encoding="utf-8") as fh:
        document = yaml.safe_load(fh)

    [dataset] = list(document)
    standalone = set()
    for query in document[dataset]["queries"]:
        qm = QueryMapping(ftm_model, query, key_prefix=dataset)
        with open(out, encoding="utf-8") as fh:
            for record in csvmod.DictReader(fh):
                for proxy in qm.map(record).values():
                    standalone.add(proxy.id)

    vplan = _vplan()
    entities, _statements, _rejects = execute(
        compile_mapping(vplan, frame, out), vplan, frame, CAT, "run1", {})
    ours = {e["id"] for e in entities}
    assert ours and standalone == ours


def test_a_thing_stays_in_the_base_query_when_a_party_is_still_there():
    """The tax debtors (`work-c11`): an unfiltered debtor beside one filtered
    sibling, and an unfiltered Asset. Copied into the sibling's query the
    Asset emitted twice on every redacted row."""
    from ftmap.plan.compile import compile_queries
    frame = build_frame(Grid(rows=[["tin_s", "name", "shot_name"],
                                   ["14360570", "ТОВ «Ромашка»", "податок на прибуток"],
                                   ["**********", "Коваленко Іван", "земельний податок"]],
                             sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    vplan = ValidatedPlan(
        subject="LegalEntity",
        entities=[{"key": "debtor", "schema": "Organization", "keys": ["c0"]},
                  {"key": "debtor_redacted", "schema": "Person", "keys": ["c1"],
                   "filter": {"column": "c0", "value": "**********"}},
                  {"key": "asset", "schema": "Asset", "keys": ["c2"]}],
        edges=[],
        bindings=[{"column": "c1", "prop": "Organization:name", "entity": "debtor", "type_name": "name", "why": ""},
                  {"column": "c1", "prop": "Person:name", "entity": "debtor_redacted", "type_name": "name", "why": ""},
                  {"column": "c2", "prop": "Asset:name", "entity": "asset", "type_name": "name", "why": ""}],
        decisions=[])
    queries = compile_queries(vplan, frame, "/x.csv")
    holders = [q for q in queries if "asset" in q["entities"]]
    assert len(holders) == 1 and "filters" not in holders[0]


def test_an_edge_s_own_key_joins_its_endpoints_fields():
    """A Debt keyed on the proceeding number is one Debt per proceeding; the
    endpoints alone would fold a debtor's proceedings under one creditor
    into one. The edge's own key column joins the layout, and the record
    carries its key field like an entity's."""
    from ftmap.plan.compile import key_layout
    v = _vplan()
    v.edges.append({"key": "debt", "schema": "Debt", "source": "person",
                    "target": "party", "keys": ["c3"]})
    layout = key_layout(v)
    assert set(layout["member"]) == set(layout["person"]) | set(layout["party"])
    assert set(layout["debt"]) == set(layout["member"]) | {"c3__key"}
