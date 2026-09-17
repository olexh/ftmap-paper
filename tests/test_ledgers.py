"""Coverage schema 2: the cases the version-1 ledger could not express.

The 2026-08-31 review found the corpus aggregate publishing `unaccounted:
-63 019` — a signed total in which +7 737 of real loss and -70 756 of
legitimate multi-binding fan-out cancelled each other — because one ledger
was asked to be both a source-value disposition and a statement
reconciliation. These are the regression tests for the split; the ordinary
single-binding arithmetic keeps its tests in `test_execute.py` and
`test_compile.py`.
"""

from ftmap.build.emit import (_fold_output, coverage_totals, ledgers,
                              merge_totals, summarise)
from ftmap.build.execute import execute
from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.plan.compile import compile_queries, normalize_frame
from ftmap.plan.validate import Decision, ValidatedPlan
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()


def _frame(rows):
    return build_frame(Grid(rows=rows, sheet="s", merges=[]),
                       "/x.csv", "0" * 64, "sid/s", CFG)


def _run(tmp_path, frame, vplan):
    out = str(tmp_path / "n.csv")
    rejects = normalize_frame(frame, vplan, out)
    _entities, statements, emit_rejects = execute(
        compile_queries(vplan, frame, out), vplan, frame, CAT, "run1", {})
    return statements, rejects + emit_rejects


POLY_ROWS = [["name", "code", "schema"],
             ["Коваленко Іван Петрович", "1025400524313", "Person"],
             ["ТОВ Ромашка", "32855961", "Organization"],
             ["Судно Дніпро", "12345678", "Vessel"]]


def test_one_column_bound_under_two_types_reconciles_per_claim(tmp_path):
    """The war-sanctions `identifiers` case: one column, one bloc reading it
    as an identifier and another as text. The deliverable is normalized under
    ONE of those — `normalize_frame`'s last-binding dict — so the claim rows
    carry both the declared and the canonical type, and each reconciles
    against its own rows instead of fighting over one column count."""
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"],
                   "filter": {"column": "c2", "value": "Person"}},
                  {"key": "org", "schema": "Organization", "keys": ["c1"],
                   "filter": {"column": "c2", "value": "Organization"}}],
        edges=[],
        bindings=[
            {"column": "c1", "prop": "Person:idNumber", "entity": "person",
             "type_name": "identifier", "why": ""},
            {"column": "c1", "prop": "Organization:description",
             "entity": "org", "type_name": "text", "why": ""},
        ],
    )
    frame = _frame(POLY_ROWS)
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))

    by_prop = {c["prop"]: c for c in claims}
    id_claim = by_prop["Person:idNumber"]
    txt_claim = by_prop["Organization:description"]
    # The canonical type is one and the same for both — the column was
    # normalized once — and the declared types differ, which is the fact the
    # reviewer's case needed on record.
    assert id_claim["canonical_type"] == txt_claim["canonical_type"] == "text"
    assert id_claim["type"] == "identifier" and txt_claim["type"] == "text"
    # Each claim closes on its own bloc's row.
    for claim in (id_claim, txt_claim):
        assert claim["expected"] == 1
        assert claim["expected"] == claim["emitted"] + claim["rejected"]
        assert claim["shortfall"] == 0 and claim["over_emission"] == 0
    # And the column's own disposition covers both blocs' rows once each.
    assert cov["c1"]["offered"] == 2
    assert cov["c1"]["outside_selection"] == 1  # the Vessel row
    assert coverage_totals(cov, claims)["accounting_valid"] is True


def test_a_filtered_bloc_and_an_unfiltered_thing_share_a_column(tmp_path):
    """A bloc's binding and a row-thing's binding on one column.

    The claim ledger surfaces a fact version 1 could not express:
    `compile_queries` copies an unfiltered row-thing into each FILTERED
    query, so the engine only ever reads it on the blocs' rows — here the
    Person row alone, while the thing's own declaration selects all three.
    That narrowing is a real, deliberate loss and it lands as `shortfall` on
    the thing's claim, with the ledger still valid: shortfall is loss
    reported, not accounting broken. Version 1 folded the same fact into the
    signed column residual where fan-out could cancel it.
    """
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c2", "value": "Person"}},
                  {"key": "note", "schema": "Note", "keys": ["c1"]}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c0", "prop": "Note:description", "entity": "note",
             "type_name": "text", "why": ""},
        ],
    )
    frame = _frame(POLY_ROWS)
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    by_prop = {c["prop"]: c for c in claims}
    # The unfiltered thing declares every row; the bloc declares its own.
    assert by_prop["Note:description"]["expected"] == 3
    assert by_prop["Note:description"]["filters"] == []
    assert by_prop["Person:name"]["expected"] == 1
    assert by_prop["Person:name"]["filters"] == ["c2"]
    # The engine reads the copied thing on the bloc's rows only, and the two
    # uncovered rows are a claim-level shortfall, listed, not cancelled.
    assert by_prop["Note:description"]["emitted"] == 1
    assert by_prop["Note:description"]["shortfall"] == 2
    assert by_prop["Person:name"]["emitted"] == 1
    # Every row is covered by the unfiltered claim, so nothing is outside.
    assert cov["c0"]["offered"] == 3
    assert cov["c0"]["outside_selection"] == 0
    total = coverage_totals(cov, claims)
    assert total["claims"]["shortfall"] == 2
    assert total["accounting_valid"] is True


def test_a_packed_cell_is_expected_under_each_claim_s_own_split_policy(tmp_path):
    """The OFAC over-emission artefact, distilled. The engine splits a source
    per BINDING type (`compile._binding_source`), while the column is
    normalized once under the canonical (last) type — so an identifier claim
    over a text-normalized column reads `id1;id2` as two values where the
    canonical count sees one cell. Counting every claim under the canonical
    type published an impossible `over_emission` on a correct run."""
    rows = [["code", "schema"],
            ["12345678;87654321", "Person"],
            ["11223344;44332211", "Organization"]]
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Person"}},
                  {"key": "org", "schema": "Organization", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "Organization"}}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:idNumber", "entity": "person",
             "type_name": "identifier", "why": ""},
            # Last binding: the canonical type of the column is `text`.
            {"column": "c0", "prop": "Organization:description",
             "entity": "org", "type_name": "text", "why": ""},
        ],
    )
    frame = _frame(rows)
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    by_prop = {c["prop"]: c for c in claims}
    # The identifier claim expects the SPLIT parts of its bloc's cell; the
    # text claim expects the whole cell; both close against the engine.
    assert by_prop["Person:idNumber"]["expected"] == 2
    assert by_prop["Organization:description"]["expected"] == 1
    for claim in claims:
        assert claim["over_emission"] == 0, claim
        assert claim["shortfall"] == 0, claim
    # And the disposition still counts each cell once, under the canonical
    # type: two covered cells, whole.
    assert cov["c0"]["offered"] == 2
    assert coverage_totals(cov, claims)["accounting_valid"] is True


def test_a_rejected_sibling_leaves_the_survivor_s_claim_clean(tmp_path):
    """One surviving binding beside one the gate rejected: the column stays
    `bound`, the survivor's claim closes, and the rejection stays in
    `decisions` rather than becoming a phantom claim row."""
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}],
        decisions=[
            Decision("c0", "Person:alias", "person", "rejected",
                     "two bindings named one (entity, property)", "rule"),
        ],
    )
    frame = _frame(POLY_ROWS)
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    assert cov["c0"]["status"] == "bound"
    assert [c["prop"] for c in claims if c["column"] == "c0"] == ["Person:name"]
    (claim,) = (c for c in claims if c["column"] == "c0")
    assert claim["expected"] == 3 == claim["emitted"]
    assert coverage_totals(cov, claims)["accounting_valid"] is True


def test_a_column_used_only_as_a_row_selection_filter_is_structural(tmp_path):
    """The war-sanctions `schema` column: no binding may survive on it —
    `validate` forbids an entity binding its own filter column — and every
    derived bloc selects on it. 5 621 values were reported as `dropped`."""
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c2", "value": "Person"}}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}],
        decisions=[
            Decision("c2", "Person:description", "person", "rejected",
                     "an entity may not bind its own selection column", "rule"),
        ],
    )
    frame = _frame(POLY_ROWS)
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, _claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    assert cov["c2"] == {"status": "structural", "role": "filter",
                         "values": 3, "offered": 0, "rejected": 0,
                         "declined": 0, "structural": 3,
                         "outside_selection": 0, "unaccounted": 0}


def test_two_same_schema_entities_on_one_column_keep_separate_claims():
    """Follow-up review P1: keyed by (column, schema, property), two distinct
    plan entities of one schema shared a claim row, and one's shortfall could
    cancel the other's over-emission before the residual split. The identity
    is (column, plan entity, property) — statements and engine rejects carry
    the plan key — so opposing residuals surface separately and the invalid
    half poisons the flag."""
    from ftmap.build.emit import Tally

    # Ukrainian headers, or the layout stage reads the first Cyrillic data
    # row as a label row under Latin headers and eats the chair's only row.
    rows = [["ПІБ", "Роль"],
            ["Коваленко Іван", "голова"],
            ["Шевченко Ольга", "член"],
            ["Петренко Марія", "член"]]
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "chair", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "голова"}},
                  {"key": "member", "schema": "Person", "keys": ["c0"],
                   "filter": {"column": "c1", "value": "член"}}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:name", "entity": "chair",
             "type_name": "name", "why": ""},
            {"column": "c0", "prop": "Person:name", "entity": "member",
             "type_name": "name", "why": ""},
        ],
    )
    tally = Tally()
    # The chair's claim emitted nothing (shortfall 1); the member's emitted
    # four from two expected values (over-emission 2). Merged, the residuals
    # would have netted to a clean-looking +1; separated, both surface.
    tally.emitted_claims[("c0", "member", "name")] = 4
    cov, claims = ledgers(_frame(rows), vplan, tally)
    assert len(claims) == 2
    by_on = {c["on"][0]: c for c in claims}
    assert by_on["e0"]["expected"] == 1 and by_on["e0"]["shortfall"] == 1
    assert by_on["e0"]["over_emission"] == 0
    assert by_on["e1"]["expected"] == 2 and by_on["e1"]["over_emission"] == 2
    assert by_on["e1"]["shortfall"] == 0
    assert coverage_totals(cov, claims)["accounting_valid"] is False


def test_claim_rows_are_checked_not_trusted():
    """Follow-up review P2: `coverage_totals` validated disposition rows and
    the over-emission sum, but a hand-fed claim row with a negative count or
    an inconsistent residual passed. The gate now requires every claim count
    non-negative and `expected + over_emission == emitted + rejected +
    shortfall` per row."""
    ok = {"column": "c0", "on": ["e0"], "prop": "Person:name",
          "type": "name", "canonical_type": "name", "filters": [],
          "expected": 2, "emitted": 2, "rejected": 0,
          "shortfall": 0, "over_emission": 0}
    assert coverage_totals({}, [ok])["accounting_valid"] is True
    negative = {**ok, "emitted": -2}
    assert coverage_totals({}, [negative])["accounting_valid"] is False
    inconsistent = {**ok, "emitted": 1}  # residual hidden: no shortfall
    assert coverage_totals({}, [inconsistent])["accounting_valid"] is False


def test_corpus_totals_never_cancel_a_loss_against_an_over_emission():
    """The defect itself, at the fold: one source with a shortfall and one
    with an over-emission must publish BOTH, and the invalid ledger must
    poison the conjunction — their signed sum is exactly the -63 019 this
    schema exists to make impossible."""
    losing = {"columns": 1,
              "statuses": {"bound": 1, "structural": 0, "dropped": 0,
                           "unmapped": 0, "undecided": 0},
              "values": 10, "offered": 10, "rejected": 0, "declined": 0,
              "structural": 0, "outside_selection": 0, "unaccounted": 0,
              "claims": {"claims": 1, "expected": 10, "emitted": 3,
                         "rejected": 0, "shortfall": 7, "over_emission": 0},
              "accounting_valid": True}
    bleeding = {"columns": 1,
                "statuses": {"bound": 1, "structural": 0, "dropped": 0,
                             "unmapped": 0, "undecided": 0},
                "values": 10, "offered": 10, "rejected": 0, "declined": 0,
                "structural": 0, "outside_selection": 0, "unaccounted": 0,
                "claims": {"claims": 1, "expected": 10, "emitted": 17,
                           "rejected": 0, "shortfall": 0, "over_emission": 7},
                "accounting_valid": False}
    total = merge_totals([losing, bleeding])
    assert total["claims"]["shortfall"] == 7
    assert total["claims"]["over_emission"] == 7
    assert total["accounting_valid"] is False
    # A clean corpus of the clean source alone stays valid.
    assert merge_totals([losing])["accounting_valid"] is True


def test_a_version_1_total_cannot_vouch_for_a_corpus():
    """Archived runs carry no flag and no claim block; folding one in leaves
    the corpus flag False rather than silently vouching for a ledger that
    cannot say it is valid."""
    v1 = {"columns": 1,
          "statuses": {"bound": 1, "dropped": 0, "unmapped": 0,
                       "undecided": 0},
          "values": 5, "emitted": 5, "rejected": 0, "declined": 0,
          "outside_selection": 0, "unaccounted": 0}
    total = merge_totals([v1])
    assert total["accounting_valid"] is False
    assert total["values"] == 5


def test_summary_publishes_the_schema_version_and_the_claims(tmp_path):
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person",
                   "type_name": "name", "why": ""}],
    )
    frame = _frame(POLY_ROWS)
    statements, rejects = _run(tmp_path, frame, vplan)
    s = summarise(frame, vplan, [], statements, rejects, CAT, CFG,
                  {"run_id": "r"})
    assert s["coverage_schema"] == 2
    assert s["coverage_totals"]["accounting_valid"] is True
    (claim,) = s["claims"]
    assert claim["on"] == ["e0"]
    # Value-free: column ids, aliases, qnames, types, counts — and the filter
    # is a COLUMN, never the value it selects on.
    import json
    blob = json.dumps(s, ensure_ascii=False)
    for forbidden in ("Коваленко", "Ромашка", "Дніпро", "1025400524313"):
        assert forbidden not in blob


def test_a_value_the_engine_rewrote_is_credited_to_the_cell_with_an_unclaimed_part(tmp_path):
    """Two columns feeding one property on one entity, and the ledger
    reconciling each against its own cells. Four shapes over four rows: the
    same value in both cells (one stored value, one statement per cell),
    different values, a value followthemoney rewrites past the loose match
    — an empty address component, «, ,» collapsed to «,» — and one cell of
    the pair holding nothing (whitespace, booked as empty and offered to
    no claim). The rejected value is on the same entity's IMO column, a
    single-column property: a mergeable type's canonicalizer is free text
    and refuses only an empty cell, and for the typed mergeable types it
    runs the engine's own validator first, so neither a canonicalizer nor
    an engine reject can reach a two-column claim; the closure with a
    reject in it is covered on the entity, not on the pair.

    The rewritten value used to go to the FIRST column holding anything:
    over-emission 1 on it, shortfall 1 on the cell that produced it, the
    source total closing and `accounting_valid` False — the lease register
    under B′, H and H′ (2026-09-10/11). It now goes to the cell whose
    offered part the matched values have not used up. Per claim and per
    source the ledger closes, and the address statements are the same
    seven, one of them on the other column.
    """
    vplan = ValidatedPlan(
        subject="Vessel",
        entities=[{"key": "vessel", "schema": "Vessel", "keys": ["c0"]}],
        edges=[],
        bindings=[
            {"column": "c1", "prop": "Vessel:address", "entity": "vessel",
             "type_name": "address", "why": ""},
            {"column": "c2", "prop": "Vessel:address", "entity": "vessel",
             "type_name": "address", "why": ""},
            {"column": "c3", "prop": "Vessel:imoNumber", "entity": "vessel",
             "type_name": "identifier", "why": ""},
        ],
    )
    frame = _frame([
        ["id", "адреса 1", "адреса 2", "IMO"],
        ["1", "вул. Шевченка, 10", "вул. Шевченка, 10", "9074729"],
        ["2", "вул. Лесі Українки 5", "просп. Перемоги, 1", "9176187"],
        ["3", "вул. Гоголя, 7", "вул.  Пушкіна, , 3", "1234568"],
        ["4", "вул. Франка, 2", "   ", "9074729"],
    ])
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    by_col = {c["column"]: c for c in claims}
    # Addresses: every accepted cell's one part is a statement on that cell.
    assert by_col["c1"]["expected"] == 4 == by_col["c1"]["emitted"], by_col["c1"]
    assert by_col["c2"]["expected"] == 3 == by_col["c2"]["emitted"], by_col["c2"]
    for col in ("c1", "c2"):
        assert by_col[col]["shortfall"] == 0 == by_col[col]["over_emission"]
    by_cell = {(st.row, st.column_id) for st in statements if st.prop == "address"}
    assert len(by_cell) == 7
    assert cov["c2"]["offered"] == 3 and cov["c2"]["rejected"] == 0
    # The bad check digit is the engine's reject, on its own claim.
    assert by_col["c3"]["expected"] == 4
    assert by_col["c3"]["emitted"] == 3 and by_col["c3"]["rejected"] == 1
    assert by_col["c3"]["shortfall"] == 0 == by_col["c3"]["over_emission"]
    totals = coverage_totals(cov, claims)
    assert totals["accounting_valid"] is True
    assert totals["claims"]["over_emission"] == 0 == totals["claims"]["shortfall"]


def test_a_loosely_matching_value_is_credited_only_to_a_cell_with_an_unclaimed_part(tmp_path):
    """The ship register's `ДРСУ 2020` under the no-model arms: two name
    cells on one row, one quoted and one upper-cased. The engine strips
    the quotes and keeps the case, so it stores two values: one is the
    second cell's canonical exactly, the other matches no cell exactly and
    both loosely. The loose pass used to credit both cells with it — the
    second cell twice in all, over-emission 1. It now credits the cell
    whose part is still unclaimed. And a value two cells offer, one
    exactly and one loosely, is credited to both, as each offered it.
    """
    vplan = ValidatedPlan(
        subject="Vessel",
        entities=[{"key": "vessel", "schema": "Vessel", "keys": ["c0"]}],
        edges=[],
        bindings=[
            {"column": "c1", "prop": "Vessel:name", "entity": "vessel",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "Vessel:name", "entity": "vessel",
             "type_name": "name", "why": ""},
        ],
    )
    frame = _frame([
        ["id", "назва", "назва латиною"],
        ["1", '"Дніпро"', "ДНІПРО"],
        ["2", 'ТОВ "Ромашка"', "ТОВ Ромашка"],
        ["3", "Славутич", "SLAVUTYCH"],
    ])
    statements, rejects = _run(tmp_path, frame, vplan)
    cov, claims = ledgers(frame, vplan, _fold_output(statements, rejects))
    by_col = {c["column"]: c for c in claims}
    for col in ("c1", "c2"):
        assert by_col[col]["expected"] == 3 == by_col[col]["emitted"], by_col[col]
        assert by_col[col]["shortfall"] == 0 == by_col[col]["over_emission"]
    assert coverage_totals(cov, claims)["accounting_valid"] is True
