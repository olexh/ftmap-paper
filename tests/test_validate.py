from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.normalize.canonical import acceptance
from ftmap.plan.propose import Plan
from ftmap.plan.validate import validate
from ftmap.profile.columns import profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()


def _setup(rows):
    f = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    return f, profile_frame(f, CFG)


def _plan(bindings, entities=None, edges=None, subject="Person"):
    return Plan(subject=subject,
                entities=entities or [{"key": "person", "schema": "Person", "keys": ["c0"]}],
                edges=edges or [], bindings=bindings, shortlists={})


def test_a_good_binding_survives_and_carries_its_type():
    frame, profiles = _setup([["ПІБ", "Дата"], ["Коваленко Іван", "17.09.1980"],
                              ["Шевченко Ольга", "01.02.1990"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"},
        {"column": "c1", "prop": "Person:birthDate", "entity": "person", "why": "dates"},
    ]), profiles, CAT, CFG, frame)
    by_col = {b["column"]: b for b in v.bindings}
    assert by_col["c1"]["type_name"] == "date"
    assert len(v.bindings) == 2


def test_a_property_the_schema_does_not_carry_is_dropped():
    frame, profiles = _setup([["ПІБ"], ["Коваленко Іван"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Vessel:imoNumber", "entity": "person", "why": "x"},
    ]), profiles, CAT, CFG, frame)
    assert v.bindings == []
    d = [d for d in v.decisions if d.column == "c0"][0]
    assert d.verdict == "rejected" and "does not carry" in d.reason


def test_a_property_only_a_reference_can_satisfy_is_dropped():
    """The grammar cannot express this binding any more, so what reaches here
    is a hand-edited or replayed plan — and it must not pass. Nothing else can
    stop it: `entity` is a FREE_TYPE, so the evidential check scores the column
    1.0 and agrees with the binding the engine will refuse value by value.

    Measured on the МВС vehicle registry: `Vehicle:addressEntity` on a column
    of service-centre labels, 39 607 values, 0 emitted.

    The reason must say WHICH test failed. `Address:full` on an entity
    declared `Person` is a different fault with a different repair, and both
    reading "does not carry" would send an analyst looking in the wrong
    place — here the schema does carry the property."""
    frame, profiles = _setup([["Підрозділ"], ["ТСЦ 8045"], ["ТСЦ 6141"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:addressEntity", "entity": "person",
          "why": "the service centre"}],
    ), profiles, CAT, CFG, frame)
    assert v.bindings == []
    d = [d for d in v.decisions if d.column == "c0"][0]
    assert d.verdict == "rejected"
    assert "reference to another entity" in d.reason
    assert "does not carry" not in d.reason
    assert CAT.carries("Person", "Person:addressEntity")


# A COLUMN OF OPENSANCTIONS IDS. Each is one token carrying a digit, so the
# local `identifier` canonicalizer takes it; none is a Q-number, so
# followthemoney's `wikidata` format check takes none of them. Measured on
# ua_war_sanctions.targets.simple.csv: 5042 of 5621 values on the `id` column
# died exactly there, one at a time, after the gate had called the binding
# accepted. On the private corpus "not in the wikidata format this property is
# declared to carry" was the largest single reject class of the run, 1469
# values.
OS_IDS = [["id"], ["NK-228jBYSTdUSvbZvsKsiHh6"], ["NK-23p2d4vMT5sJtQ845GyzJt"],
          ["NK-24ZomtS59XsEEB94Qcqy7S"], ["NK-26WkuEefVYLmYk8aTHMC9E"]]
# The same property, on a column that really does hold Q-numbers.
QIDS = [["wikidata"], ["Q6319"], ["Q42"], ["Q7259"], ["Q123456789"]]


def test_a_column_that_cannot_meet_a_declared_format_is_rejected():
    """`wikidataId` is `identifier`-typed AND declares a `wikidata` format,
    and until this check the second half was nobody's question. `bindable`
    lets the pair through — an identifier is satisfiable by a cell, unlike an
    entity reference — the shortlist offers it, the grammar admits it, and
    acceptance scored it with OUR `identifier` canonicalizer, which knows
    nothing about Q-numbers. The refusal happened in the engine instead, value
    by value, after summary.json had called the binding accepted.

    The reason must name the FORMAT, not the type: these values are perfectly
    good identifiers and the repair is a different property, not a different
    column."""
    frame, profiles = _setup(OS_IDS)
    v = validate(_plan([
        {"column": "c0", "prop": "Person:wikidataId", "entity": "person",
         "why": "the record id"},
    ]), profiles, CAT, CFG, frame)
    assert v.bindings == []
    d = [d for d in v.decisions if d.column == "c0"][0]
    assert d.verdict == "rejected"
    assert "wikidata format" in d.reason
    # NOT VACUOUS, and this is the whole point of the change: every other gate
    # passes this binding. Person carries the property, a cell can satisfy it,
    # and the column clears the `identifier` floor outright.
    assert CAT.bindable("Person", "Person:wikidataId")
    assert acceptance([r[0] for r in OS_IDS[1:]], "identifier")[0] >= \
        CFG.accept_thresholds["identifier"]


def test_a_column_that_meets_the_declared_format_still_binds():
    """The reason Route 1 is the check and not an exclusion. A
    format-carrying property stays reachable for a column that can satisfy it
    — the same sanctions file emitted 396 real Q-numbers — so excluding these
    16 properties from the candidates the way entity-typed ones are excluded
    would lose real data."""
    frame, profiles = _setup(QIDS)
    v = validate(_plan([
        {"column": "c0", "prop": "Person:wikidataId", "entity": "person",
         "why": "Q-numbers"},
    ]), profiles, CAT, CFG, frame)
    assert [b["prop"] for b in v.bindings] == ["Person:wikidataId"]
    d = [d for d in v.decisions if d.column == "c0"][0]
    assert d.verdict == "accepted"


def test_a_property_with_no_format_is_judged_on_its_type_alone():
    """The narrower question is only asked where the ontology asks it. Just 16
    of followthemoney's 518 non-stub properties declare a format, and the
    other 502 must not start being measured against one that does not exist —
    `idNumber` is the neighbouring property these very values belong to."""
    frame, profiles = _setup(OS_IDS)
    v = validate(_plan([
        {"column": "c0", "prop": "Person:idNumber", "entity": "person",
         "why": "the record id"},
    ]), profiles, CAT, CFG, frame)
    assert [b["prop"] for b in v.bindings] == ["Person:idNumber"]
    assert CAT.prop("Person:idNumber").format is None


def test_values_that_fail_the_type_validator_reject_the_binding():
    frame, profiles = _setup([["Дата"], ["79253902086"], ["79114585963"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:birthDate", "entity": "person", "why": "header says date"},
    ]), profiles, CAT, CFG, frame)
    assert v.bindings == []
    d = [d for d in v.decisions if d.column == "c0"][0]
    assert d.verdict == "rejected" and "acceptance" in d.reason


def test_unmapped_is_recorded_not_omitted():
    frame, profiles = _setup([["№ з/п"], ["1"], ["2"]])
    v = validate(_plan([
        {"column": "c0", "prop": "unmapped", "entity": "none", "why": "row ordinal"},
    ]), profiles, CAT, CFG, frame)
    d = [d for d in v.decisions if d.column == "c0"][0]
    assert d.verdict == "unmapped" and d.reason == "row ordinal"


def test_a_key_column_that_is_mostly_empty_is_rejected():
    """And the name bound to the entity, which fills the rows the code
    leaves, keys it instead of the row ordinal (2026-09-05, the regiment's
    person on a service number absent on its open billets)."""
    frame, profiles = _setup([["ПІБ", "Код"], ["Коваленко Іван", "1"],
                              ["Шевченко Ольга", None], ["Мельник Петро", None]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == ["c0"]
    assert any(d.column == "c1" and d.verdict == "rejected" and "key fill" in d.reason
               for d in v.decisions)
    assert any(d.column == "c0" and "too sparse" in d.reason for d in v.decisions)


def test_a_key_that_clears_the_floor_is_recorded_as_accepted():
    """decisions.jsonl was 'one line per column decision across its whole
    life' (§8.3) in name only for keys and edges: only their rejections were
    recorded, never a surviving key or edge, so a reader could not tell "no
    key was ever checked" from "the key was checked and it passed"."""
    frame, profiles = _setup([["ПІБ", "Код"], ["Коваленко Іван", "1"],
                              ["Шевченко Ольга", "2"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == ["c1"]
    accepted = [d for d in v.decisions if d.column == "c1" and d.verdict == "accepted"]
    assert accepted, v.decisions


def test_an_edge_that_validates_is_recorded_as_accepted():
    frame, profiles = _setup([["ПІБ", "Партія"], ["Коваленко Іван", "Слуга народу"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"},
         {"column": "c1", "prop": "Organization:name", "entity": "party", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "party", "schema": "Organization", "keys": ["c1"]}],
        edges=[{"key": "member", "schema": "Membership",
                "source": "person", "target": "party"}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.edges] == ["member"]
    accepted = [d for d in v.decisions if d.column == "member" and d.verdict == "accepted"]
    assert accepted, v.decisions


def test_an_edge_may_not_reuse_an_entity_key():
    """Observed twice on live model output. Entities and edges share one
    namespace in `declared` and one dict in `compile_mapping`, so a collision
    silently overwrites the entity and the engine then fails on a mapping that
    reads as well-formed."""
    frame, profiles = _setup([["ПІБ"], ["Коваленко Іван"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        edges=[{"key": "person", "schema": "Membership",
                "source": "person", "target": "person"}],
    ), profiles, CAT, CFG, frame)
    assert v.edges == []
    assert any("namespace" in d.reason for d in v.decisions)
    # The entity survives, and its own binding must not be judged against the
    # edge's schema. An earlier version rejected it with "schema Membership
    # does not carry Person:name", which is both false and misleading.
    assert [b["prop"] for b in v.bindings] == ["Person:name"]


def test_key_collisions_are_renamed_not_dropped_for_entities_and_for_edges():
    """C3: the collision guard covered edge-vs-entity but not entity-vs-entity
    or edge-vs-edge — the latter had no dedicated test either — and
    `declared = {e["key"]: e for e in plan.entities}` is a dict comprehension,
    so the second of any pair silently overwrote the first. Demonstrated on
    real output: a plan declaring `who -> Person` and `who -> Organization`
    rejected an innocent `Person:name` binding with the false reason "schema
    Organization does not carry Person:name" and only the Organization entity
    survived — a whole declared entity gone with nothing in decisions.jsonl
    saying so. Renamed rather than dropped, the same shape `propose.py`
    already uses for a colliding edge key (propose.py:87-95)."""
    # Entity vs entity. Both entities carry a binding, because an entity no
    # column is bound to is dropped for being an empty shell — a separate rule,
    # tested separately, and not what this test is about.
    frame, profiles = _setup([["ПІБ", "Компанія"], ["Коваленко Іван", "ТОВ Альфа"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "who", "why": "names"},
         {"column": "c1", "prop": "Organization:name", "entity": "who_2",
          "why": "companies"}],
        entities=[{"key": "who", "schema": "Person", "keys": ["c0"]},
                  {"key": "who", "schema": "Organization", "keys": []}],
    ), profiles, CAT, CFG, frame)

    assert len(v.entities) == 2
    assert {e["schema"] for e in v.entities} == {"Person", "Organization"}
    assert any("two entities declared the key" in d.reason for d in v.decisions)
    # The Person binding survives, judged against its OWN schema — not against
    # whichever entity happened to end up under "who" in an overwritten dict.
    assert [b["prop"] for b in v.bindings] == ["Person:name", "Organization:name"]

    # Edge vs edge: `compile_mapping` builds one dict keyed on edge key, so
    # two edges declaring the same key collide silently there even though
    # each one individually validates.
    frame2, profiles2 = _setup([["ПІБ", "Партія"], ["Коваленко Іван", "Слуга народу"]])
    v2 = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"},
         {"column": "c1", "prop": "Organization:name", "entity": "party", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "party", "schema": "Organization", "keys": ["c1"]}],
        edges=[{"key": "m", "schema": "Membership", "source": "person", "target": "party"},
               {"key": "m", "schema": "Membership", "source": "person", "target": "party"}],
    ), profiles2, CAT, CFG, frame2)

    assert len(v2.edges) == 1
    assert any("two edges declare the same key" in d.reason for d in v2.decisions)


def test_an_edge_with_an_undeclared_endpoint_is_dropped():
    frame, profiles = _setup([["ПІБ"], ["Коваленко Іван"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        edges=[{"key": "m", "schema": "Membership", "source": "person", "target": "party"}],
    ), profiles, CAT, CFG, frame)
    assert v.edges == []
    assert any("party" in d.reason for d in v.decisions)


def test_an_edge_whose_endpoint_range_is_wrong_is_dropped():
    frame, profiles = _setup([["ПІБ", "Фракція"], ["Коваленко Іван", "Слуга народу"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"},
         {"column": "c1", "prop": "Person:position", "entity": "other", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "other", "schema": "Person", "keys": ["c1"]}],
        edges=[{"key": "m", "schema": "Membership", "source": "person", "target": "other"}],
    ), profiles, CAT, CFG, frame)
    assert v.edges == []
    assert any("range" in d.reason for d in v.decisions)


def test_a_rejection_names_the_shapes_that_failed_not_the_values():
    """A bare acceptance figure cannot be told apart from a genuinely bad
    column, and that ambiguity hid a canonicalizer gap on real data. Shapes are
    value-free, so decisions.jsonl stays publishable."""
    frame, profiles = _setup([["Дата"], ["79253902086"], ["79114585963"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:birthDate", "entity": "person", "why": "x"},
    ]), profiles, CAT, CFG, frame)
    reason = [d for d in v.decisions if d.column == "c0"][0].reason
    assert "rejected shapes" in reason
    assert "d+" in reason
    assert "79253902086" not in reason


def test_an_entity_that_loses_every_key_says_so():
    """The aggregate fact: this entity now produces one entity per row, which
    changes what deduplication means downstream."""
    frame, profiles = _setup([["Примітка", "Код"], ["має вищу освіту", "1"],
                              ["працює у раді", None], ["ветеран", None]])
    # Nothing keyable is bound — a note is not a name — so the fallback is
    # the row ordinal, said out loud.
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:notes", "entity": "person", "why": "notes"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == []
    fallback = [d for d in v.decisions if d.entity == "person" and d.column is None]
    assert fallback and "row ordinal" in fallback[0].reason


def test_one_repair_round_at_most():
    frame, profiles = _setup([["Дата"], ["79253902086"], ["79114585963"]])

    class Repairer:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, schema, max_tokens=None):
            self.calls += 1
            return {"bindings": [{"column": "c0", "binding": "unmapped",
                                  "why": "phones, not dates"}]}

    client = Repairer()
    v = validate(_plan([
        {"column": "c0", "prop": "Person:birthDate", "entity": "person", "why": "x"},
    ]), profiles, CAT, CFG, frame, client=client)
    assert client.calls == 1
    assert v.bindings == []
    assert [d.verdict for d in v.decisions if d.decided_by == "model-repair"] == ["unmapped"]


def test_repair_round_also_enforces_structural_checks():
    """The repair path re-ran only `_check_binding` (catalogue membership and
    evidential acceptance), not the schema-carries-property or
    entity-declared checks the primary path applies. `Address:full` is a real
    property of type `address`, a FREE_TYPE that accepts anything, so a
    repaired answer binding it to the `person` entity (schema Person, which
    does not carry it) must still be rejected structurally rather than
    silently accepted."""
    frame, profiles = _setup([["Дата"], ["79253902086"], ["79114585963"]])

    class WrongEntityRepairer:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, schema, max_tokens=None):
            self.calls += 1
            return {"bindings": [{"column": "c0", "binding": "person|Address:full",
                                  "why": "actually an address"}]}

    client = WrongEntityRepairer()
    v = validate(_plan([
        {"column": "c0", "prop": "Person:birthDate", "entity": "person", "why": "x"},
    ]), profiles, CAT, CFG, frame, client=client)
    assert v.bindings == []
    repaired = [d for d in v.decisions if d.decided_by == "model-repair"][0]
    assert repaired.verdict == "rejected" and "does not carry" in repaired.reason


def test_a_plan_that_contradicts_its_own_subject_is_resolved_here_too():
    """The backstop for a plan this stage did not build — hand-edited, or
    replayed from a cache written before `propose` settled this. A single
    entity declared as a schema the plan's own subject rules out is the МВС
    vehicle registry's failure: subject `Vehicle`, one `Organization`, every
    candidate for all 17 columns `Organization:*`, 0 statements from 39 607
    rows. The subject is declared as an entity here too, so a binding this
    stage never offered a grammar for can still be checked against it."""
    frame, profiles = _setup([["MODEL"], ["SENS"], ["OCTAVIA"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Vehicle:model", "entity": "vehicle",
          "why": "models"}],
        entities=[{"key": "veh", "schema": "Organization", "keys": ["c0"]}],
        subject="Vehicle",
    ), profiles, CAT, CFG, frame)
    assert [(e["key"], e["schema"]) for e in v.entities] == [("vehicle", "Vehicle")]
    # The declared subject is not rewritten: `summary.json` publishes it as
    # `subject_declared`, and it is the answer that exposed the contradiction.
    assert v.subject == "Vehicle"
    assert [(b["entity"], b["prop"]) for b in v.bindings] == [
        ("vehicle", "Vehicle:model")]
    said = [d for d in v.decisions if "its only declared entity was" in d.reason]
    assert len(said) == 1 and said[0].verdict == "accepted"
    dropped = [d for d in v.decisions if "no accepted binding" in d.reason]
    assert len(dropped) == 1 and dropped[0].entity == "veh"


def test_a_promoted_reading_an_edge_needs_falls_with_its_edge():
    """The unused half of a contradicted pair survives the reading rule when an
    edge references it — and is then dropped anyway, taking the edge, because
    it carries no property.

    REVERSED DELIBERATELY on 2026-08-28. The rule this test used to assert kept
    the entity so the edge would not point at something missing. Measured on
    the person-graph corpus, that is the wrong way round: an edge whose
    endpoint has no property connects an anonymous node and states nothing
    checkable — 23 807 `CourtCaseParty` and 1 744 `Ownership` of exactly that
    shape in one run. The edge is what goes."""
    # «Назва», not «Власник»: a role word over a column of firm names now
    # declares the party the header names (`plan.roles.declare_parties`),
    # and this test is about the contradicted pair, not about that.
    frame, profiles = _setup([["MODEL", "Назва"],
                              ["SENS", "ТОВ Ромашка"],
                              ["OCTAVIA", "ПАТ Мрія"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Vehicle:model", "entity": "vehicle",
          "why": "models"}],
        entities=[{"key": "veh", "schema": "Organization", "keys": ["c0"]}],
        edges=[{"key": "own", "schema": "Ownership",
                "source": "veh", "target": "vehicle"}],
        subject="Vehicle",
    ), profiles, CAT, CFG, frame)
    assert {e["key"] for e in v.entities} == {"vehicle"}
    assert v.edges == []
    # Not the reading rule: that one spared it because the edge referenced it.
    assert [d for d in v.decisions if "no accepted binding" in d.reason] == []
    assert any("carried no property" in d.reason for d in v.decisions)


def test_a_subject_the_ontology_does_not_have_promotes_nothing():
    """This resolves a disagreement between two answers, it does not invent
    one. A subject that names no schema at all — only reachable from a
    hand-edited plan, since the grammar closes that enum — must leave the plan
    alone rather than declare an entity as a name FtM cannot build."""
    frame, profiles = _setup([["ПІБ"], ["Коваленко Іван"], ["Шевченко Ольга"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        subject="Автомобіль",
    ), profiles, CAT, CFG, frame)
    assert [(e["key"], e["schema"]) for e in v.entities] == [("person", "Person")]
    assert [d for d in v.decisions if "its only declared entity was" in d.reason] == []
    assert len(v.bindings) == 1




def test_two_columns_may_not_both_claim_one_date_or_number_property():
    """`compile_mapping` keys an entity's properties by name, so a second
    binding on one property overwrote the first and the losing column reached
    emission as an accepted binding the mapping never read — 144 333 values on
    corpus-external. For a `number` or a `date` the columns are two FACTS
    wearing one slot (`first_seen`/`last_seen` -> retrievedAt; four Vehicle
    figures -> amount), so the column with more usable values keeps the
    property and the other is declined."""
    frame, profiles = _setup([["first_seen", "last_seen"],
                              ["2021-03-01", "2024-01-05"],
                              ["2022-07-09", None],
                              ["2023-01-01", None]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:retrievedAt", "entity": "person", "why": "a"},
        {"column": "c1", "prop": "Person:retrievedAt", "entity": "person", "why": "b"},
    ]), profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c0"]
    declined = [d for d in v.decisions if d.column == "c1"][-1]
    assert declined.verdict == "unmapped" and declined.decided_by == "rule"
    assert "c0 already binds Person:retrievedAt" in declined.reason


def test_two_columns_of_a_multi_valued_type_both_feed_the_property():
    """THE CONTEST IS LEGITIMATE FOR A DATE AND WRONG FOR A COUNTRY. The ICIJ
    addresses etalon (J3) binds `countries` AND `country_codes` to
    `Address:country` — "the type canonicalizes name and code to the same
    value, and this is not a duplicate answer to be avoided" — and the ship
    register binds «Тип судна» and «Призначення» both to `Vessel:type`,
    "type is multi-valued, so both columns feed it". FollowTheMoney's own
    data model agrees: a country, a name, a type, a phone is a SET of values.
    Three prompt-side experiments established that the model will not name
    a second-choice property when told the first is taken; the deterministic
    layer is where the etalon says the answer lives."""
    frame, profiles = _setup([["countries", "country_codes"],
                              ["Ukraine", "UKR"],
                              ["Cyprus", "CYP"],
                              ["Malta", "MLT"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Address:country", "entity": "addr", "why": "a"},
        {"column": "c1", "prop": "Address:country", "entity": "addr", "why": "b"},
    ], entities=[{"key": "addr", "schema": "Address", "keys": []}],
        subject="Address"), profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c0", "c1"]
    shared = [d for d in v.decisions if d.column == "c1" and d.decided_by == "rule"
              and d.verdict == "accepted"]
    assert shared and "c0 also binds Address:country" in shared[-1].reason
    assert not any(d.verdict == "unmapped" for d in v.decisions if d.column == "c1")


def test_two_string_columns_still_contest_one_property():
    """A string names a fact and says nothing about which. Measured on the
    ship register: engine type, hold type and boiler type all bound to
    `Vessel:type` beside «Призначення», hull material to `description` —
    seven columns the gold standard calls unmappable, and letting them
    share would publish `Vessel:type: [прогулянкове, Rotax, 0]`. The one
    pair the etalon does bind to one string property, «Тип судна» and
    «Призначення», is the stated cost of this rule."""
    frame, profiles = _setup([["Тип судна", "Тип головних механізмів"],
                              ["самохідне моторне", "Rotax"],
                              ["несамохідне", "Mercruiser"],
                              ["самохідне моторне", "Yamaha"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Vessel:type", "entity": "v", "why": "a"},
        {"column": "c1", "prop": "Vessel:type", "entity": "v", "why": "b"},
    ], entities=[{"key": "v", "schema": "Vessel", "keys": []}],
        subject="Vessel"), profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c0"]
    assert "c0 already binds Vessel:type" in \
        [d for d in v.decisions if d.column == "c1"][-1].reason


def test_a_declined_column_is_declined_after_it_was_accepted():
    """The order is what `emit._column_status` reads: last word wins, so the
    decline has to come after the acceptance it supersedes or the column is
    still reported as bound and its values as lost."""
    frame, profiles = _setup([["first_seen", "last_seen"],
                              ["2021-03-01", "2024-01-05"],
                              ["2022-07-09", None]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:retrievedAt", "entity": "person", "why": "a"},
        {"column": "c1", "prop": "Person:retrievedAt", "entity": "person", "why": "b"},
    ]), profiles, CAT, CFG, frame)
    verdicts = [d.verdict for d in v.decisions if d.column == "c1"]
    assert verdicts == ["accepted", "unmapped"]


def test_a_contested_property_goes_to_the_column_with_more_values():
    """Not to whichever binding the loop reached last, which is what the dict
    in `compile_mapping` used to decide."""
    frame, profiles = _setup([["Мало", "Багато"],
                              ["2021-03-01", "2024-01-05"],
                              [None, "2022-07-09"],
                              [None, "2023-01-01"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:retrievedAt", "entity": "person", "why": "a"},
        {"column": "c1", "prop": "Person:retrievedAt", "entity": "person", "why": "b"},
    ], entities=[{"key": "person", "schema": "Person", "keys": []}]),
        profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c1"]
    reason = [d for d in v.decisions if d.column == "c0"][-1].reason
    assert "3 values against this column's 1" in reason


def test_a_tied_property_goes_to_the_earlier_column():
    """Deterministic, and decided by the source's own order rather than by the
    order this list happened to be built in."""
    frame, profiles = _setup([["Перше", "Друге"],
                              ["2021-03-01", "2024-01-05"],
                              ["2022-07-09", "2023-01-01"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:retrievedAt", "entity": "person", "why": "a"},
        {"column": "c1", "prop": "Person:retrievedAt", "entity": "person", "why": "b"},
    ], entities=[{"key": "person", "schema": "Person", "keys": []}]),
        profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c0"]
    assert "equally many" in [d for d in v.decisions if d.column == "c1"][-1].reason


def test_one_property_on_two_different_entities_is_not_a_collision():
    """The claim is an ENTITY's property. Two entities each carrying their own
    `name` is the ordinary shape of a two-entity plan, not a contest."""
    frame, profiles = _setup([["ПІБ", "Установа"],
                              ["Коваленко Іван", "ТОВ Ромашка"],
                              ["Шевченко Ольга", "ТОВ Волошка"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "a"},
        {"column": "c1", "prop": "Organization:name", "entity": "org", "why": "b"},
    ], entities=[{"key": "person", "schema": "Person", "keys": []},
                 {"key": "org", "schema": "Organization", "keys": []}]),
        profiles, CAT, CFG, frame)
    assert sorted(b["column"] for b in v.bindings) == ["c0", "c1"]


def test_two_qnames_for_one_property_are_one_claim():
    """`Thing:name` and `Person:name` are the same property, and
    `compile_mapping` keys on the name after the colon — so two bindings under
    two qnames would collide there while looking distinct anywhere else."""
    frame, profiles = _setup([["ПІБ", "Друге ім'я"],
                              ["Коваленко Іван", "Шевченко Ольга"],
                              ["Мельник Петро", "Бондаренко Анна"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Thing:name", "entity": "person", "why": "a"},
        {"column": "c1", "prop": "Person:name", "entity": "person", "why": "b"},
    ], entities=[{"key": "person", "schema": "Person", "keys": []}]),
        profiles, CAT, CFG, frame)
    # One claim: a name is multi-valued, so both columns feed it, and the
    # second is recorded as SHARING the property the first already holds —
    # under the first's spelling, because the claim is one property.
    assert [b["column"] for b in v.bindings] == ["c0", "c1"]
    shared = [d for d in v.decisions if d.column == "c1" and d.decided_by == "rule"]
    assert shared and "c0 also binds" in shared[-1].reason


# --------------------------------------------------------------------------
# The header's own unit, and the entity that declared no identity at all.
# Both rules exist because the etalon corpus measured what happens without
# them: 19 of 121 homeless columns bound anyway, 19 of 56 entities keyed on
# nothing.
# --------------------------------------------------------------------------

def test_a_row_ordinal_is_refused_every_property_and_asked_about_no_more():
    """Measured on the lease register: `№ з/п`, bound to `RealEstate:area`,
    then WON the contest for `area` against the real area column on value
    count. The row number describes nothing; it is refused before any
    evidential check, never re-asked, and booked as declined."""
    rows = [["№ з/п", "Площа, кв.м"]] + [[str(i + 1), "12.5" if i % 3 else None]
                                         for i in range(9)]
    frame, profiles = _setup(rows)
    v = validate(_plan([
        {"column": "c0", "prop": "RealEstate:area", "entity": "o", "why": "n"},
        {"column": "c1", "prop": "RealEstate:area", "entity": "o", "why": "a"},
    ], entities=[{"key": "o", "schema": "RealEstate", "keys": []}],
        subject="RealEstate"), profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c1"]
    c0 = [d for d in v.decisions if d.column == "c0"]
    assert c0[0].verdict == "rejected" and "row ordinal" in c0[0].reason
    assert not any(d.decided_by == "model-repair" for d in c0)
    # Closed as declined so the accounting reads the column as passed over.
    assert c0[-1].verdict == "unmapped"


def test_a_register_s_own_numbers_are_not_mistaken_for_an_ordinal():
    """Distinct and numeric is not consecutive: a register's ids are the
    register's ids, and gaps are the tell."""
    rows = [["Код", "Назва"]] + [[str(n), f"ТОВ {n}"]
                                 for n in (101, 102, 104, 105, 109)]
    frame, profiles = _setup(rows)
    v = validate(_plan([
        {"column": "c0", "prop": "Company:registrationNumber", "entity": "c", "why": ""},
        {"column": "c1", "prop": "Company:name", "entity": "c", "why": ""},
    ], entities=[{"key": "c", "schema": "Company", "keys": ["c0"]}],
        subject="Company"), profiles, CAT, CFG, frame)
    assert sorted(b["column"] for b in v.bindings) == ["c0", "c1"]


def test_a_column_whose_header_states_its_unit_is_refused_a_money_property():
    """`Ширина, м` is a perfectly good `number`, so every evidential check
    passes it and `Vessel:amount` takes it. The contradiction is between the
    header and the ontology, and nothing below this rule can see it."""
    frame, profiles = _setup([["Назва", "Ширина, м"], ["Нептун", "4.2"],
                              ["Аврора", "5.1"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Vessel:name", "entity": "v", "why": "names"},
        {"column": "c1", "prop": "Vessel:amount", "entity": "v", "why": "number"},
    ], entities=[{"key": "v", "schema": "Vessel", "keys": ["c0"]}],
        subject="Vessel"), profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c0"]
    d = [d for d in v.decisions if d.column == "c1"][0]
    assert d.verdict == "rejected"
    assert "'м'" in d.reason and "monetary" in d.reason
    # Value-free: the reason quotes the header's unit, never a cell.
    for value in ("4.2", "5.1", "Нептун"):
        assert value not in d.reason


def test_a_money_column_that_declares_no_unit_is_untouched():
    """The rule refuses a contradiction; it does not decide what money looks
    like. `zn_all` on the tax-debtor register is money and says so nowhere."""
    frame, profiles = _setup([["name", "zn_all"], ["ТОВ А", "1200,00"],
                              ["ТОВ Б", "980,50"]])
    v = validate(_plan([
        {"column": "c1", "prop": "Asset:amount", "entity": "a", "why": "debt"},
    ], entities=[{"key": "a", "schema": "Asset", "keys": ["c0"]}],
        subject="Asset"), profiles, CAT, CFG, frame)
    assert [b["column"] for b in v.bindings] == ["c1"]


def test_an_entity_that_declared_no_key_is_given_one_from_its_own_bindings():
    """`keys: []` is one un-deduplicated entity per row — thirty copies of one
    court, 872 operators for 65. The alternative to a proposed key is not a
    better key, it is the row ordinal."""
    frame, profiles = _setup([["ЄДРПОУ", "Назва"], ["42257456", "Суд"],
                              ["42257456", "Суд"], ["42257456", "Суд"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Organization:registrationNumber",
         "entity": "o", "why": "code"},
        {"column": "c1", "prop": "Organization:name", "entity": "o",
         "why": "name"},
    ], entities=[{"key": "o", "schema": "Organization", "keys": []}],
        subject="Organization"), profiles, CAT, CFG, frame)
    # The identifier wins over the name: an ЄДРПОУ identifies, a name describes.
    assert v.entities[0]["keys"] == ["c0"]
    d = [d for d in v.decisions
         if d.entity == "o" and d.column == "c0" and d.prop is None][0]
    assert d.verdict == "accepted" and "no key was declared" in d.reason


def test_a_proposed_key_is_never_taken_from_a_number_or_a_date():
    """The tempting rule — key on the most distinct bound column — is how a
    register of 39 607 vehicles gets keyed on a two-valued flag. A number, a
    date and a string describe an entity; they do not identify it."""
    frame, profiles = _setup([["Рік", "Дата"], ["2006", "17.09.1980"],
                              ["2007", "01.02.1990"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Vehicle:buildDate", "entity": "v", "why": "y"},
        {"column": "c1", "prop": "Vehicle:registrationDate", "entity": "v",
         "why": "d"},
    ], entities=[{"key": "v", "schema": "Vehicle", "keys": []}],
        subject="Vehicle"), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == []


def test_a_key_the_evidence_rejected_is_not_quietly_replaced():
    """An entity whose declared keys were all REJECTED is a different case from
    one that declared none: the plan had an opinion and the evidence refused
    it. Substituting a column the plan never nominated would silence the
    falls-back-to-the-row-ordinal report, which is what actually happened."""
    frame, profiles = _setup([["id", "Опис"], ["", "суд першої інстанції"],
                              ["", "апеляційний суд"], ["7", "суд"]])
    v = validate(_plan([
        {"column": "c1", "prop": "Organization:description", "entity": "o", "why": "n"},
    ], entities=[{"key": "o", "schema": "Organization", "keys": ["c0"]}],
        subject="Organization"), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == []
    assert any("falls back to the row ordinal" in d.reason
               for d in v.decisions if d.entity == "o")
    # With a NAME bound that fills the rows, the replacement is made and
    # said — never quietly: the sparse key's rejection stays on the record
    # beside the line that names what keys the entity instead.
    frame, profiles = _setup([["id", "Назва"], ["", "Суд"], ["", "Суд"],
                              ["7", "Суд"]])
    v = validate(_plan([
        {"column": "c1", "prop": "Organization:name", "entity": "o", "why": "n"},
    ], entities=[{"key": "o", "schema": "Organization", "keys": ["c0"]}],
        subject="Organization"), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == ["c1"]
    assert any(d.column == "c0" and d.verdict == "rejected" and "key fill" in d.reason
               for d in v.decisions)
    assert any(d.column == "c1" and "too sparse" in d.reason and "rather than on the row ordinal" in d.reason
               for d in v.decisions)


def test_a_column_refused_for_what_it_is_is_declined_not_counted_as_lost():
    """`emit.coverage` books a column whose binding did not survive as
    `dropped` and every value it held as `unaccounted`, because the ordinary
    case is real output going missing. Nothing goes missing here: the header
    says the column is a width in metres and no monetary property was ever
    going to emit from it. Measured on the first run with the unit rule in
    place, the other reading sent 208 083 values of two ship sheets into
    `unaccounted` — a rule doing its job, reported as the largest accounting
    failure in the run."""
    from ftmap.build.emit import coverage, coverage_totals

    frame, profiles = _setup([["Назва", "Ширина, м"], ["Нептун", "4.2"],
                              ["Аврора", "5.1"]])
    v = validate(_plan([
        {"column": "c0", "prop": "Vessel:name", "entity": "v", "why": "n"},
        {"column": "c1", "prop": "Vessel:amount", "entity": "v", "why": "n"},
    ], entities=[{"key": "v", "schema": "Vessel", "keys": ["c0"]}],
        subject="Vessel"), profiles, CAT, CFG, frame)

    cov = coverage(frame, v, [], [])
    assert cov["c1"]["status"] == "unmapped"
    assert cov["c1"]["declined"] == cov["c1"]["values"] == 2
    assert cov["c1"]["unaccounted"] == 0
    assert coverage_totals(cov)["statuses"]["dropped"] == 0
    # The refusal is still on record beside the decline, so a reader learns
    # WHICH property was refused and why, not merely that nothing was bound.
    reasons = [d.reason for d in v.decisions if d.column == "c1"]
    assert any("monetary value" in r for r in reasons)
    assert any("no other property was proposed" in r for r in reasons)


def test_a_key_that_repeats_is_reinstated_rather_than_left_with_none():
    """`key_distinct_floor` exists to prefer a sharper key over a blunter one.
    It was also, silently, deciding that an entity with no sharp key should
    have NO key — which is not a refusal to identify, it is one copy per row.
    The state-enterprise register nominated the managing body's name, 79
    distinct bodies over 3 009 rows, and emitted 3 009 organisations."""
    rows = [["Підприємство", "Орган управління"]]
    rows += [[f"ДП {i}", "ФДМУ" if i % 2 else "Мінагро"] for i in range(20)]
    frame, profiles = _setup(rows)
    v = validate(_plan([
        {"column": "c0", "prop": "Company:name", "entity": "co", "why": "n"},
        {"column": "c1", "prop": "Organization:name", "entity": "mgr", "why": "n"},
    ], entities=[{"key": "co", "schema": "Company", "keys": ["c0"]},
                 {"key": "mgr", "schema": "Organization", "keys": ["c1"]}],
        subject="Company"), profiles, CAT, CFG, frame)
    mgr = [e for e in v.entities if e["key"] == "mgr"][0]
    assert mgr["keys"] == ["c1"], "two managers, not twenty"
    d = [d for d in v.decisions if d.entity == "mgr" and d.verdict == "accepted"
         and d.prop is None][0]
    assert "reinstated" in d.reason
    # The floor still did its job on record: the rejection is not erased.
    assert any("below" in d.reason for d in v.decisions if d.entity == "mgr")


def test_a_repeating_key_is_not_reinstated_from_a_column_that_describes():
    """Only a name or an identifier names something that recurs. A number, a
    date or a string is a description that happens to repeat, and keying on one
    is how 39 607 vehicles become two entities under a natural/juridical flag."""
    rows = [["Модель", "PERSON"]]
    rows += [[f"Model {i}", "P" if i % 2 else "J"] for i in range(20)]
    frame, profiles = _setup(rows)
    v = validate(_plan([
        {"column": "c0", "prop": "Vehicle:model", "entity": "v", "why": "m"},
        {"column": "c1", "prop": "Vehicle:type", "entity": "v", "why": "flag"},
    ], entities=[{"key": "v", "schema": "Vehicle", "keys": ["c1"]}],
        subject="Vehicle"), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == []


def test_an_entity_no_column_is_bound_to_is_not_emitted():
    """A declared entity with no property binding is an empty shell.

    Measured on the person-graph corpus, 2026-08-28: 10 of the Gemma run's 36
    declared entities and 7 of the Qwen run's 25 had no column bound to them
    at all, and they were emitted anyway — 23 807 property-less `Person` for
    the court decisions' judge, 20 054 each of `Vessel`, `CryptoWallet` and
    `Airplane` on the OFAC file. An entity with no property asserts nothing,
    carries no evidence, and inflates every entity count that includes it.
    """
    frame, profiles = _setup([["ПІБ", "Суд"], ["Коваленко Іван", "Конотопський"],
                              ["Шевченко Ольга", "Сумський"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "judge", "schema": "Person", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.entities] == ["person"]
    d = [d for d in v.decisions if d.entity == "judge" and d.verdict == "rejected"][0]
    assert "no column is bound to it" in d.reason


def test_an_edge_falls_with_the_empty_entity_it_pointed_at():
    """The edge is not evidence either once its endpoint is gone.

    The same runs emitted 23 807 `CourtCaseParty` and 1 744 `Ownership` whose
    one endpoint was a property-less entity. A relation to an anonymous node
    states nothing that can be checked.
    """
    frame, profiles = _setup([["ПІБ", "Компанія"], ["Коваленко Іван", "ТОВ Альфа"],
                              ["Шевченко Ольга", "ПрАТ Бета"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "company", "schema": "Company", "keys": ["c1"]}],
        edges=[{"key": "job", "schema": "Directorship", "source": "person",
                "target": "company"}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.entities] == ["person"]
    assert v.edges == []


def test_an_edge_fanned_to_every_counterparty_is_dropped_whole():
    """The loose-roster arm (`work-full-thing3`, 2026-08-31): 17 of its 18
    spurious edges were one entity wired to two or more counterparts under
    one schema — a Documentation from the certificate to EVERY party, a
    ContractAward from the charter to all four. A relation answered
    identically for every counterparty says "connected to something", which
    is the same always-available, information-free answer that kept
    `UnknownLink` out of the menu. The single-counterpart edge is the claim
    worth keeping; a fan is dropped whole, because nothing in it says which
    strand was meant."""
    frame, profiles = _setup([["ПІБ", "Керівник", "Засновник"],
                              ["Коваленко Іван", "Шевченко Ольга", "Бондаренко Ігор"],
                              ["Мельник Петро", "Ткаченко Ніна", "Гончар Олесь"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Person:name", "entity": "chief", "why": "x"},
         {"column": "c2", "prop": "Person:name", "entity": "founder", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "chief", "schema": "Person", "keys": ["c1"]},
                  {"key": "founder", "schema": "Person", "keys": ["c2"]}],
        edges=[{"key": "a1", "schema": "Associate", "source": "person",
                "target": "chief"},
               {"key": "a2", "schema": "Associate", "source": "person",
                "target": "founder"}],
    ), profiles, CAT, CFG, frame)
    assert v.edges == []
    reasons = [d.reason for d in v.decisions if "every counterpart" in d.reason]
    assert len(reasons) == 2


def test_a_single_counterpart_edge_survives_the_fan_gate():
    frame, profiles = _setup([["ПІБ", "Компанія"],
                              ["Коваленко Іван", "ТОВ Альфа"],
                              ["Шевченко Ольга", "ПрАТ Бета"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Company:name", "entity": "company", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "company", "schema": "Company", "keys": ["c1"]}],
        edges=[{"key": "job", "schema": "Directorship", "source": "person",
                "target": "company"}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.edges] == ["job"]


def test_a_thing_attaches_its_entity_typed_property_to_the_one_party_it_can_mean():
    """Sanction:entity, Identification:holder, Contract:authority — 18 of the
    corpus's expected relations are entity-typed properties, scored
    EDGE-UNPRODUCIBLE while nothing could emit one. The compiler has always
    written `{"entity": key}` for edge endpoints; the same mechanism serves
    these, and when a declared non-party thing has exactly ONE co-resident
    entity its property can range over, no model call is needed to say which:
    the war-sanctions row holds one Person and one Sanction, and a Sanction
    about something is about the something. Deterministic, and never shown to
    the model — every prompt-side attempt at the relation layer is in
    `2026-08-20-negative-results.md`."""
    frame, profiles = _setup([["ПІБ", "Програма"],
                              ["Коваленко Іван", "UA-WS-MILIND"],
                              ["Шевченко Ольга", "UA-WS-KAB"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Sanction:programId", "entity": "sanction", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "sanction", "schema": "Sanction", "keys": []}],
    ), profiles, CAT, CFG, frame)
    assert v.attachments == [{"entity": "sanction", "prop": "Sanction:entity",
                              "target": "person"}]
    d = [d for d in v.decisions if "attached" in d.reason][0]
    assert d.decided_by == "rule"
    # riskSource is entity-typed too and FollowTheMoney marks it unmatchable —
    # an attachment that cannot help match anything is metadata, not a claim.
    assert not any(a["prop"].endswith("riskSource") for a in v.attachments)


def test_two_readings_of_the_protagonist_attach_nothing():
    """Two declared entities of the subject's own schema — an owner and a
    charterer, both Person on a Person-subject row — is a question, and the
    edge call already declined it. A non-subject party is no candidate at
    all: the procurement plan's Contract attached its authority to the
    SUPPLIER when the buyer was unproducible, so a lone bystander is not a
    fallback protagonist (measured, `work-full-att` v2)."""
    frame, profiles = _setup([["ПІБ", "Власник", "Програма"],
                              ["Коваленко Іван", "Шевченко Ольга", "UA-WS-1"],
                              ["Мельник Петро", "Ткаченко Ніна", "UA-WS-2"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Person:name", "entity": "owner", "why": "x"},
         {"column": "c2", "prop": "Sanction:programId", "entity": "sanction", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "owner", "schema": "Person", "keys": ["c1"]},
                  {"key": "sanction", "schema": "Sanction", "keys": []}],
    ), profiles, CAT, CFG, frame)
    assert v.attachments == []


def test_a_bystander_party_is_no_attachment_target():
    """Subject Person; the Company on the row is not the protagonist, so the
    Sanction attaches to the Person and never to the Company — even were the
    Person absent."""
    frame, profiles = _setup([["Компанія", "Програма"],
                              ["ТОВ Альфа", "UA-WS-1"],
                              ["ПрАТ Бета", "UA-WS-2"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Company:name", "entity": "company", "why": "x"},
         {"column": "c1", "prop": "Sanction:programId", "entity": "sanction", "why": "x"}],
        entities=[{"key": "company", "schema": "Company", "keys": ["c0"]},
                  {"key": "sanction", "schema": "Sanction", "keys": []}],
        subject="Person",
    ), profiles, CAT, CFG, frame)
    assert v.attachments == []


def test_attachment_runs_thing_to_party_and_never_thing_to_thing():
    """`Person:addressEntity` ranges on Address and a Person beside one
    Address would fire — but the wanted cases all run thing-to-party, and
    both other directions are where the junk lives: the dry run found
    `Address:proof -> Document` beside `Document:addressEntity -> Address`
    on one source, a circular entity dependency followthemoney's compiler
    refuses outright (measured: the intermediaries file FAILED on it). A
    party attaches nothing, and nothing attaches to a thing."""
    frame, profiles = _setup([["ПІБ", "Адреса", "Джерело"],
                              ["Коваленко Іван", "м. Київ, вул. Хрещатик 1", "A"],
                              ["Шевченко Ольга", "м. Львів, пл. Ринок 2", "B"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Address:full", "entity": "addr", "why": "x"},
         {"column": "c2", "prop": "Document:title", "entity": "doc", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "addr", "schema": "Address", "keys": ["c1"]},
                  {"key": "doc", "schema": "Document", "keys": ["c2"]}],
    ), profiles, CAT, CFG, frame)
    # addr and doc could each attach to the other (proof / addressEntity);
    # person is the only legal target, and each thing may claim it only if it
    # is the single candidate — here it is, so only party-targeted
    # attachments exist and no thing points at a thing.
    assert all(a["target"] == "person" for a in v.attachments)


def test_an_unfiltered_thing_attaches_once_per_bloc_of_the_protagonist():
    """The polymorphic case per-bloc binding exists for: each row is one
    kind plus the sanction that designates it. The sanction is unfiltered —
    it describes every row — and each bloc's copy must point at that bloc's
    protagonist, one attachment per bloc, resolved inside that bloc's query
    (`compile_queries` distributes the copies). The bloc must still be the
    subject's own reading, and still the only candidate in its group."""
    frame, profiles = _setup([["назва", "тип", "програма"],
                              ["Коваленко Іван", "особа", "UA-1"],
                              ["Кравець Олег", "фізособа", "UA-2"]])
    v = validate(_filtered(
        [{"key": "p1", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}},
         {"key": "p2", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "фізособа"}},
         {"key": "sanction", "schema": "Sanction", "keys": []}],
        [{"column": "c0", "prop": "Person:name", "entity": "p1", "why": "x"},
         {"column": "c0", "prop": "Person:name", "entity": "p2", "why": "x"},
         {"column": "c2", "prop": "Sanction:programId", "entity": "sanction",
          "why": "x"}],
    ), profiles, CAT, CFG, frame)
    got = sorted((a["prop"], a["target"]) for a in v.attachments
                 if a["entity"] == "sanction")
    assert got == [("Sanction:entity", "p1"), ("Sanction:entity", "p2")]


def test_an_entity_s_own_filter_column_may_not_bind_on_it():
    """Per-bloc binding showed the model its kind column with a constant
    value — `Company` on every row of the Company bloc — and it bound it as
    the bloc's `name`, which then CONTESTED the real name column and won
    (war sanctions, `work-full-bloc`: c2, the name, emitted nothing). The
    selection column's value IS the selection; on its own bloc it is a
    constant label, not a property."""
    frame, profiles = _setup([["назва", "тип"],
                              ["ТОВ Ромашка", "Company"],
                              ["Коваленко Іван", "Person"],
                              ["ПрАТ Мрія", "Company"]])
    v = validate(_filtered(
        [{"key": "org", "schema": "Company", "keys": ["c0"],
          "filter": {"column": "c1", "value": "Company"}}],
        [{"column": "c0", "prop": "Company:name", "entity": "org", "why": "x"},
         {"column": "c1", "prop": "Company:name", "entity": "org", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    got = {(b["column"], b["prop"]) for b in v.bindings}
    assert ("c0", "Company:name") in got
    assert not any(b["column"] == "c1" for b in v.bindings)
    d = [d for d in v.decisions if d.column == "c1"
         and "selection" in d.reason][0]
    assert d.verdict == "rejected"


def test_a_bloc_is_its_own_group_s_protagonist_whatever_its_schema():
    """OFAC's designation targets a Vessel bloc as readily as a Person bloc,
    and inside a bloc's query the bloc IS the row's protagonist — the
    subject-reading and party guards exist for the UNFILTERED case, where a
    lone bystander was measured taking an attachment meant for an absent
    buyer. In a filtered group nothing is absent: the group holds exactly
    its bloc."""
    frame, profiles = _setup([["назва", "тип", "програма"],
                              ["Корабель Мрія", "Vessel", "UA-1"],
                              ["ТОВ Ромашка", "Organization", "UA-2"]])
    v = validate(_filtered(
        [{"key": "vessel", "schema": "Vessel", "keys": ["c0"],
          "filter": {"column": "c1", "value": "Vessel"}},
         {"key": "org", "schema": "Organization", "keys": ["c0"],
          "filter": {"column": "c1", "value": "Organization"}},
         {"key": "sanction", "schema": "Sanction", "keys": []}],
        [{"column": "c0", "prop": "Vessel:name", "entity": "vessel", "why": "x"},
         {"column": "c0", "prop": "Organization:name", "entity": "org", "why": "x"},
         {"column": "c2", "prop": "Sanction:programId", "entity": "sanction",
          "why": "x"}],
    ), profiles, CAT, CFG, frame)
    got = sorted(a["target"] for a in v.attachments
                 if a["entity"] == "sanction")
    assert got == ["org", "vessel"]


def test_a_filtered_thing_attaches_only_inside_its_own_selection():
    """A FILTERED thing belongs to its bloc's rows and may not reference an
    entity built from different ones — FollowTheMoney resolves the reference
    inside one query. (An UNFILTERED thing is the opposite case: it follows
    every bloc — see the per-bloc test above.)"""
    frame, profiles = _setup([["назва", "тип", "програма"],
                              ["Коваленко Іван", "особа", "UA-1"],
                              ["ТОВ Ромашка", "організація", "UA-2"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}},
         {"key": "sanction", "schema": "Sanction", "keys": [],
          "filter": {"column": "c1", "value": "організація"}}],
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c2", "prop": "Sanction:programId", "entity": "sanction", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    assert v.attachments == []


def _filtered(entities, bindings, edges=None):
    return _plan(bindings, entities=entities, edges=edges or [])


def test_a_binding_on_one_bloc_replicates_to_the_siblings_that_carry_it():
    """`column -> bindings`. On a polymorphic table one column serves every
    bloc: OFAC writes the party's name in c2 whether the row is a Person or
    an Organization, and a binding that names ONE entity starves the others —
    the 2026-08-29 run left the Person bloc with a birth date, a country and
    no name. A binding on a filtered entity is therefore replicated onto its
    filter-siblings that carry the property; the model's own answers are
    never overwritten, and a property the sibling's schema cannot hold does
    not travel (`birthDate` stays the Person's)."""
    frame, profiles = _setup([["назва", "тип", "дата"],
                              ["Коваленко Іван", "особа", "17.09.1980"],
                              ["ТОВ Ромашка", "організація", "01.02.1990"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}},
         {"key": "org", "schema": "Organization", "keys": ["c0"],
          "filter": {"column": "c1", "value": "організація"}}],
        [{"column": "c0", "prop": "Organization:name", "entity": "org", "why": "x"},
         {"column": "c2", "prop": "Person:birthDate", "entity": "person", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    got = {(b["column"], b["entity"]): b for b in v.bindings}
    assert ("c0", "person") in got, v.bindings
    assert got[("c0", "person")]["prop"] == "Person:name"
    assert got[("c0", "person")].get("replica") is True
    assert got[("c0", "org")].get("replica") is None
    # birthDate does not land on the Organization: no LegalEntity property
    # spells it.
    assert ("c2", "org") not in got


def test_a_replica_does_not_overwrite_the_model_s_own_binding():
    frame, profiles = _setup([["назва", "тип"],
                              ["Коваленко Іван", "особа"],
                              ["ТОВ Ромашка", "організація"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}},
         {"key": "org", "schema": "Organization", "keys": ["c0"],
          "filter": {"column": "c1", "value": "організація"}}],
        [{"column": "c0", "prop": "Organization:name", "entity": "org", "why": "x"},
         {"column": "c0", "prop": "Person:alias", "entity": "person", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    got = {(b["column"], b["entity"]): b for b in v.bindings}
    assert got[("c0", "person")]["prop"] == "Person:alias"
    assert got[("c0", "person")].get("replica") is None


def test_a_replica_never_makes_a_second_claim_on_one_entity_property():
    """The OFAC idNumber displacement, 2026-08-31. The Person answered
    `c0 -> Person:idNumber` itself; the Organization answered
    `c7 -> Organization:idNumber`; replication then offered the Person a
    SECOND idNumber from c7 — and `compile_mapping`, which keys an entity's
    properties by local name, silently kept whichever came last. 7 456 of the
    model's own values vanished with no reject; the claim ledger's shortfall
    was what surfaced it. A replica fills a gap, never contests a slot —
    on any column."""
    frame, profiles = _setup([["ід", "тип", "код"],
                              ["1234567890", "особа", "12345678"],
                              ["9876543210", "організація", "87654321"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}},
         {"key": "org", "schema": "Organization", "keys": ["c0"],
          "filter": {"column": "c1", "value": "організація"}}],
        [{"column": "c0", "prop": "Person:idNumber", "entity": "person",
          "why": "x"},
         {"column": "c2", "prop": "Organization:idNumber", "entity": "org",
          "why": "x"}],
    ), profiles, CAT, CFG, frame)
    by_entity_prop = {}
    for b in v.bindings:
        key = (b["entity"], b["prop"].split(":", 1)[1])
        by_entity_prop.setdefault(key, []).append(b["column"])
    # One claim per (entity, property), everywhere — the model's own answer
    # holds its slot and no replica lands beside it.
    assert all(len(cols) == 1 for cols in by_entity_prop.values()), \
        by_entity_prop
    assert by_entity_prop.get(("person", "idNumber")) == ["c0"]


def test_an_unfiltered_entity_gets_no_replicas():
    """Replication is the polymorphic-table rule and nothing wider: without a
    row selection two entities genuinely share every row, and spraying one
    column's property onto both would re-create the very ambiguity the
    filters resolve."""
    frame, profiles = _setup([["назва", "інша"],
                              ["Коваленко Іван", "щось"],
                              ["ТОВ Ромашка", "інше"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"]},
         {"key": "org", "schema": "Organization", "keys": ["c1"]}],
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Organization:name", "entity": "org", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    assert not any(b.get("replica") for b in v.bindings)


def test_a_filter_naming_a_column_that_is_not_there_is_dropped_not_obeyed():
    """The entity is still real; only its row selection was written badly. A
    plan that named a column the table does not have would otherwise select no
    rows and take the entity with it."""
    frame, profiles = _setup([["ПІБ", "тип"], ["Коваленко Іван", "особа"],
                              ["ТОВ Ромашка", "організація"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c9", "value": "особа"}}],
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    assert v.entities[0].get("filter") is None
    assert any("filter column does not exist" in d.reason for d in v.decisions)


def test_a_filter_on_a_value_no_row_holds_drops_the_entity():
    """It selects nothing, so the entity is a declaration that can emit no
    instance. Dropping it says so once, instead of leaving an entity in the
    plan and an empty query in the mapping."""
    frame, profiles = _setup([["ПІБ", "тип"], ["Коваленко Іван", "особа"],
                              ["ТОВ Ромашка", "організація"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "судно"}},
         {"key": "org", "schema": "Organization", "keys": ["c0"],
          "filter": {"column": "c1", "value": "організація"}}],
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c0", "prop": "Organization:name", "entity": "org", "why": "y"}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.entities] == ["org"]
    assert any("no row holds" in d.reason for d in v.decisions)


def test_a_filter_every_row_satisfies_is_dropped_as_a_no_op():
    """A column with one value selects the whole table, which is what an
    unfiltered entity already does. Keeping it would put a `filters` block in
    the mapping that reads as a restriction and restricts nothing."""
    frame, profiles = _setup([["ПІБ", "тип"], ["Коваленко Іван", "особа"],
                              ["Шевченко Ольга", "особа"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}}],
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    assert v.entities[0].get("filter") is None
    assert any("every row" in d.reason for d in v.decisions)


def test_an_edge_across_two_row_selections_is_refused():
    """FollowTheMoney resolves an entity reference inside ONE query, and a
    query holds one selection. An edge between two differently-filtered
    entities cannot be compiled, so it is refused here with a reason rather
    than dropped silently in `compile_queries`."""
    frame, profiles = _setup([["ПІБ", "тип"], ["Коваленко Іван", "особа"],
                              ["ТОВ Ромашка", "організація"]])
    v = validate(_filtered(
        [{"key": "person", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "особа"}},
         {"key": "org", "schema": "Organization", "keys": ["c0"],
          "filter": {"column": "c1", "value": "організація"}}],
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c0", "prop": "Organization:name", "entity": "org", "why": "y"}],
        edges=[{"key": "job", "schema": "Directorship", "source": "person",
                "target": "org"}],
    ), profiles, CAT, CFG, frame)
    assert v.edges == []
    assert any("different row selections" in d.reason for d in v.decisions)


def test_the_distinct_floor_is_the_row_entity_s_question_not_every_entity_s():
    """`key_distinct_floor` asks "is this distinct enough to identify a ROW".
    That is the right question for the entity a row IS, and the wrong one for
    every other entity in a table, which recurs by construction.

    Measured over the 19 etalons on 2026-08-29: 25 of the key columns a
    careful annotator declared are below the 0.5 floor — the tax office named
    on 749 of 21 529 rows, the managing body on 79 of 3 009, the judge, the
    charterer, the workplace. The floor rejects them all, and what the entity
    falls back to is the row ordinal: the tax debtors emitted 21 529 entities
    where the source holds 66.
    """
    rows = [["ПІБ", "Орган"]] + [[f"Особа {i}", "ДПІ Сумська" if i % 2 else "ДПІ Львівська"]
                                 for i in range(10)]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "PublicBody:name", "entity": "office", "why": "y"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "office", "schema": "PublicBody", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert [e["keys"] for e in v.entities] == [["c0"], ["c1"]]
    assert not any("distinct" in d.reason and d.entity == "office"
                   for d in v.decisions if d.verdict == "rejected")


def test_a_party_named_on_some_rows_keeps_the_key_that_names_it():
    """THE FILL FLOOR IS THE ROW ENTITY'S QUESTION TOO. A key column empty on
    a third of rows cannot identify the rows it is empty on — true, and the
    right objection for the entity a row IS. For every other entity it is
    backwards: an owner named on 63 % of the ship register's rows is an
    entity on those rows and nothing on the rest, which is exactly what
    followthemoney does with an empty key (`EntityMapping.compute_key`
    returns None and the row emits no such entity). Refusing the key sent the
    entity to the row ordinal instead: 20 251 owners emitted where the sheet
    names 10 026, one per row, most of them nameless. The same gate the
    distinct floor already uses."""
    rows = [["Судно", "Власник"]] + [
        [f"Судно {i}", f"Власник {i % 3}" if i % 3 else None] for i in range(12)]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Vessel:name", "entity": "vessel", "why": "x"},
         {"column": "c1", "prop": "Person:name", "entity": "owner", "why": "y"}],
        entities=[{"key": "vessel", "schema": "Vessel", "keys": ["c0"]},
                  {"key": "owner", "schema": "Person", "keys": ["c1"]}],
        subject="Vessel",
    ), profiles, CAT, CFG, frame)
    assert [e["keys"] for e in v.entities] == [["c0"], ["c1"]]
    kept = [d for d in v.decisions if d.column == "c1" and d.entity == "owner"
            and d.prop is None and d.verdict == "accepted"]
    assert kept and "not the row" in kept[0].reason, v.decisions


def test_the_row_entity_still_loses_a_key_too_sparse_to_identify_a_row():
    rows = [["Судно", "Код"]] + [
        [f"Судно {i}", str(i) if i % 3 else None] for i in range(12)]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Vessel:name", "entity": "vessel", "why": "x"}],
        entities=[{"key": "vessel", "schema": "Vessel", "keys": ["c1"]}],
        subject="Vessel",
    ), profiles, CAT, CFG, frame)
    # The sparse code is still lost; the name that fills every row keys the
    # vessel in its place (2026-09-05).
    assert v.entities[0]["keys"] == ["c0"]
    assert any(d.verdict == "rejected" and "key fill" in d.reason
               for d in v.decisions), v.decisions


def test_the_row_entity_still_loses_a_key_too_coarse_to_identify_a_row():
    """The floor keeps its job where the question is its own: an entity the
    table is ABOUT, keyed on a column that repeats, is one entity standing for
    many rows."""
    rows = [["Тип", "Модель"]] + [["Легковий" if i % 2 else "Вантажний", f"Модель {i}"]
                                  for i in range(10)]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Vehicle:type", "entity": "car", "why": "x"},
         {"column": "c1", "prop": "Vehicle:model", "entity": "car", "why": "y"}],
        entities=[{"key": "car", "schema": "Vehicle", "keys": ["c0"]}],
        subject="Vehicle",
    ), profiles, CAT, CFG, frame)
    assert any("distinct" in d.reason for d in v.decisions
               if d.verdict == "rejected")


def test_the_distinct_floor_holds_when_the_subject_is_absent():
    """The floor is the row entity's question — but a plan whose subject is
    None cannot answer which entity that is, and 'unknown' must fall back to
    asking everyone, not no one. Gated purely on the subject, a 2-valued
    status column keyed the whole table into two merged entities."""
    rows = [["ПІБ", "Статус"]] + [
        [f"Особа Номер {'Перша Друга Третя Четверта Пята Шоста Сьома Восьма Девята Десята'.split()[i]}",
         "Чинна" if i % 2 else "Недіюча"] for i in range(10)]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": ""}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
        subject=None,
    ), profiles, CAT, CFG, frame)
    assert any(d.verdict == "rejected" and "key distinct" in d.reason
               for d in v.decisions), v.decisions


def test_the_distinct_floor_reaches_a_row_entity_specialised_past_the_subject():
    """Subject `Organization`, entity `Company`: the entity IS the row, one
    rung more specific than the subject says. Checking only schema == subject
    or subject-descends-from-schema skipped exactly this case."""
    rows = [["Назва", "Статус"]] + [
        [f"ТОВ Ромашка {i}", "Чинна" if i % 2 else "Недіюча"] for i in range(10)]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Company:name", "entity": "company", "why": ""}],
        entities=[{"key": "company", "schema": "Company", "keys": ["c1"]}],
        subject="Organization",
    ), profiles, CAT, CFG, frame)
    assert any(d.verdict == "rejected" and "key distinct" in d.reason
               for d in v.decisions), v.decisions


def test_a_replaced_subject_leaves_a_line_in_decisions():
    """`propose` records the repair in `plan.subject_replaced` and the field's
    own comment says `validate` turns it into a Decision — like `key_renames`
    and `subject_contradictions` before it. It did not: the replacement
    reached summary.json as the model's own answer with no line anywhere."""
    import dataclasses
    frame, profiles = _setup([["ПІБ"], ["Коваленко Іван"]])
    p = dataclasses.replace(
        _plan([{"column": "c0", "prop": "Person:name", "entity": "person",
                "why": ""}]),
        subject_replaced={"declared": "Audio", "used": "Person"})
    v = validate(p, profiles, CAT, CFG, frame)
    assert any("Audio" in (d.reason or "") and "Person" in (d.reason or "")
               for d in v.decisions), v.decisions


def test_a_thing_with_nothing_of_its_own_on_a_bloc_does_not_attach_there():
    """OFAC (`work-x13`): the Sanction, once over every row, attached to the
    CryptoWallet bloc too, and every one of those sanctions was empty —
    `program_ids` is blank on all 976 wallet rows. `publisher` is filled on
    every row, but it is `Interval`'s, not the Sanction's own; substance is
    a property declared on the holder's concrete schema with a value on the
    bloc's rows."""
    frame, profiles = _setup([["назва", "тип", "програма", "джерело"],
                              ["Коваленко Іван", "Person", "UA-1", "ofac"],
                              ["Корабель Мрія", "Vessel", None, "ofac"],
                              ["Кравець Олег", "Person", "UA-2", "ofac"]])
    v = validate(_filtered(
        [{"key": "p", "schema": "Person", "keys": ["c0"],
          "filter": {"column": "c1", "value": "Person"}},
         {"key": "v", "schema": "Vessel", "keys": ["c0"],
          "filter": {"column": "c1", "value": "Vessel"}},
         {"key": "sanction", "schema": "Sanction", "keys": []}],
        [{"column": "c0", "prop": "Person:name", "entity": "p", "why": "x"},
         {"column": "c0", "prop": "Vessel:name", "entity": "v", "why": "x"},
         {"column": "c2", "prop": "Sanction:programId", "entity": "sanction", "why": "x"},
         {"column": "c3", "prop": "Sanction:publisher", "entity": "sanction", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    got = sorted(a["target"] for a in v.attachments if a["entity"] == "sanction")
    assert got == ["p"], v.attachments
    d = [d for d in v.decisions if d.entity == "sanction"
         and d.reason.startswith("not attached to v")]
    assert d, [d.reason for d in v.decisions if d.entity == "sanction"]
    assert ("c2 — everything sanction binds of its own — is empty on the rows "
            "where c1 is 'Vessel'") in d[0].reason


def test_an_identifier_that_varies_within_its_key_is_not_that_entity_s():
    """ICIJ entities (`work-x16-cold`): `node_id`, one per row, bound as the
    idNumber of the provider keyed on `service_provider`; Mossack Fonseca
    came out with 584 identifiers. The same column on the company keyed on
    its name is one per key and stays."""
    frame, profiles = _setup([["node_id", "name", "service_provider"],
                              ["NK-1041", "ТОВ Ромашка", "Mossack Fonseca"],
                              ["NK-2077", "ПрАТ Мрія", "Mossack Fonseca"],
                              ["NK-3309", "КП Дніпро", "Mossack Fonseca"],
                              ["NK-4812", "ТОВ Зоря", "Appleby"],
                              ["NK-5150", "ТОВ Світанок", "Appleby"],
                              ["NK-6666", "ПП Крок", "Appleby"]])
    v = validate(_plan(
        [{"column": "c1", "prop": "Company:name", "entity": "company", "why": "x"},
         {"column": "c2", "prop": "LegalEntity:name", "entity": "provider", "why": "x"},
         {"column": "c0", "prop": "LegalEntity:idNumber", "entity": "provider", "why": "x"}],
        entities=[{"key": "company", "schema": "Company", "keys": ["c1"]},
                  {"key": "provider", "schema": "LegalEntity", "keys": ["c2"]}],
        subject="Company",
    ), profiles, CAT, CFG, frame)
    # Re-homed, not dropped: the column varies within the provider's key and
    # the company — the row's entity — carries `idNumber`, which is exactly
    # where the second half of this test says the column belongs. Until
    # 2026-09-05 the identifier rule dropped it and the company went without.
    assert [b["entity"] for b in v.bindings if b["column"] == "c0"] == ["company"]
    d = [d for d in v.decisions if d.column == "c0" and d.verdict == "rejected"
         and d.entity == "provider" and "re-homed" in d.reason]
    assert d

    v = validate(_plan(
        [{"column": "c1", "prop": "Company:name", "entity": "company", "why": "x"},
         {"column": "c0", "prop": "Company:idNumber", "entity": "company", "why": "x"}],
        entities=[{"key": "company", "schema": "Company", "keys": ["c1"]}],
        subject="Company",
    ), profiles, CAT, CFG, frame)
    assert any(b["column"] == "c0" and b["entity"] == "company" for b in v.bindings)


def test_a_declared_key_one_to_one_with_the_bound_name_is_reinstated():
    """The Rada's MPs: parties keyed on `party_id`, 15 values over 469 rows,
    refused by the distinct floor and keyed on the row instead — 469 parties.
    The id is one-to-one with `party_name`, bound on the same entity."""
    # `org_code`, not `party_id`: a role word in the key header makes the
    # entity a named party, never the row's, and the floor would not apply.
    rows = [["org_code", "org_name", "full_name"]]
    parties = [("226", "Слуга Народу"), ("227", "ОПЗЖ"),
               ("228", "Європейська Солідарність"), ("229", "Голос")]
    names = ["Коваленко Іван", "Шевченко Ольга", "Бондаренко Петро",
             "Мельник Андрій", "Ткаченко Марія", "Кравець Олег",
             "Бойко Сергій", "Лисенко Юрій", "Іваненко Дарія",
             "Петренко Олена", "Гнатюк Василь", "Романюк Ігор"]
    for i, name in enumerate(names):
        pid, pname = parties[i % 4]
        rows.append([pid, pname, name])
    # Four ids over twelve rows, evenly: 0.33 distinct, and 0.33 past the
    # commonest value too, so the distinct floor refuses the key.
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c1", "prop": "Organization:name", "entity": "party", "why": "x"},
         {"column": "c2", "prop": "Person:name", "entity": "mp", "why": "x"}],
        entities=[{"key": "mp", "schema": "Person", "keys": ["c2"]},
                  {"key": "party", "schema": "Organization", "keys": ["c0"]}],
        # The floor is the row entity's question; make the party the row's
        # so the floor refuses its 3-of-8 key and the rule has work to do.
        subject="Organization",
    ), profiles, CAT, CFG, frame)
    party = [e for e in v.entities if e["key"] == "party"][0]
    assert party["keys"] == ["c0"], party
    d = [d for d in v.decisions if d.entity == "party" and d.column == "c0"
         and "one-to-one with c1" in d.reason]
    assert d and d[0].verdict == "accepted"


def test_under_an_interval_subject_the_row_s_entity_is_the_one_most_bound():
    """Боярка (`work-c1`): the model declared `Occupancy` as the subject, so
    no entity was the row's by schema and the three Memberships the role
    lexicon could read stayed ambiguous. The effective subject — the Thing
    with the most bindings, the Person — is what the derivation reads."""
    frame, profiles = _setup([["familyName", "name", "partyName", "factionName"],
                              ["Коваленко", "Іван", "Слуга Народу", "Фракція Слуга Народу"],
                              ["Шевченко", "Ольга", "За майбутнє", "Фракція За майбутнє"],
                              ["Бондаренко", "Петро", "Слуга Народу", "Фракція Слуга Народу"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:lastName", "entity": "deputy", "why": "x"},
         {"column": "c1", "prop": "Person:firstName", "entity": "deputy", "why": "x"},
         {"column": "c2", "prop": "Organization:name", "entity": "party", "why": "x"},
         {"column": "c3", "prop": "Organization:name", "entity": "faction", "why": "x"}],
        entities=[{"key": "deputy", "schema": "Person", "keys": ["c0", "c1"]},
                  {"key": "party", "schema": "Organization", "keys": ["c2"]},
                  {"key": "faction", "schema": "Organization", "keys": ["c3"]}],
        subject="Occupancy",
    ), profiles, CAT, CFG, frame)
    assert sorted((e["schema"], e["source"], e["target"]) for e in v.edges) == [
        ("Membership", "deputy", "faction"), ("Membership", "deputy", "party")]


def test_the_post_beside_a_derived_membership_is_the_membership_s_role():
    frame, profiles = _setup([["familyName", "factionName", "factionPost"],
                              ["Коваленко", "Фракція Слуга Народу", "член фракції"],
                              ["Шевченко", "Фракція За майбутнє", "голова"],
                              ["Бондаренко", "Фракція Слуга Народу", "член фракції"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:lastName", "entity": "deputy", "why": "x"},
         {"column": "c1", "prop": "Organization:name", "entity": "faction", "why": "x"}],
        entities=[{"key": "deputy", "schema": "Person", "keys": ["c0"]},
                  {"key": "faction", "schema": "Organization", "keys": ["c1"]}],
        subject="Person",
    ), profiles, CAT, CFG, frame)
    assert [(e["schema"], e["source"], e["target"]) for e in v.edges] == [
        ("Membership", "deputy", "faction")]
    role = [b for b in v.bindings if b["column"] == "c2"]
    # The derived edge is keyed by the party column's entity: the faction.
    assert role and role[0]["prop"] == "Membership:role" \
        and role[0]["entity"] == "faction_membership"


def test_two_role_worded_name_columns_folded_into_one_party_are_split():
    """The MPs: `party_text` (nominator) and `party_name` (membership) both
    bound as one Organization's name, keyed on `party_id`. They disagree on
    most rows that fill both, so the nominator becomes its own party and
    the role lexicon relates both."""
    # Rada-shaped ids, not 1..8: a run of consecutive integers is a row
    # ordinal to the gate and would empty the MP of its one binding.
    rows = [["id", "full_name", "party_text", "party_id", "party_name"]]
    data = [("21253", "Коваленко Іван", "Слуга Народу", "50", "Безпартійний"),
            ("21039", "Шевченко Ольга", "Слуга Народу", "226", "Слуга Народу"),
            ("20877", "Бондаренко Петро", "Слуга Народу", "50", "Безпартійний"),
            ("21402", "Мельник Андрій", "Голос", "229", "Голос"),
            ("20051", "Ткаченко Марія", "Слуга Народу", "50", "Безпартійний"),
            ("21188", "Кравець Олег", "Слуга Народу", "50", "Безпартійний"),
            ("20690", "Бойко Сергій", "Батьківщина", "17", "Батьківщина"),
            ("21331", "Лисенко Юрій", "Слуга Народу", "50", "Безпартійний")]
    rows += [list(r) for r in data]
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c1", "prop": "Person:name", "entity": "mp", "why": "x"},
         {"column": "c2", "prop": "Organization:name", "entity": "party", "why": "x"},
         {"column": "c4", "prop": "Organization:name", "entity": "party", "why": "x"}],
        entities=[{"key": "mp", "schema": "Person", "keys": ["c0"]},
                  {"key": "party", "schema": "Organization", "keys": ["c3"]}],
        subject="Person",
    ), profiles, CAT, CFG, frame)
    keys = {e["key"]: e for e in v.entities}
    assert "c2" in keys and keys["c2"]["schema"] == "Organization" and keys["c2"]["keys"] == ["c2"]
    got = {(b["column"], b["entity"]) for b in v.bindings if b["prop"] == "Organization:name"}
    assert got == {("c2", "c2"), ("c4", "party")}
    assert sorted((e["schema"], e["source"], e["target"]) for e in v.edges) == [
        ("Membership", "mp", "c2"), ("Membership", "mp", "party")]


def test_an_unbound_phone_column_binds_to_the_party_whose_header_it_shares():
    """The debtors register: `ORG_PHONE_NUM` beside `ORG_NAME`, declined by
    the model on format, 95 % phones to the detector."""
    # Whole numbers only: a pack with a local extension («…, 62-08-31»)
    # fails the value check's acceptance floor part by part, which is the
    # canonicalizer's own policy for phones and not this rule's subject.
    # No spaces inside a number: the phone canonicalizer splits a cell on
    # whitespace as well as on separators, so «(0362) 62-08-30» is two parts
    # to the value check and one of them is not a number — the
    # canonicalizer's own policy, not this rule's subject.
    frame, profiles = _setup([["DEBTOR_NAME", "ORG_NAME", "ORG_PHONE_NUM"],
                              ["Коваленко Іван", "Відділ ДВС у м. Рівному", "+380362620830"],
                              ["ТОВ «Денкар»", "Відділ ДВС у м. Луцьку", "+380332721515"],
                              ["Шевченко Ольга", "Відділ ДВС у м. Рівному", "0362620830"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "debtor", "why": "x"},
         {"column": "c1", "prop": "PublicBody:name", "entity": "office", "why": "x"}],
        entities=[{"key": "debtor", "schema": "Person", "keys": ["c0"]},
                  {"key": "office", "schema": "PublicBody", "keys": ["c1"]}],
        subject="Person",
    ), profiles, CAT, CFG, frame)
    phone = [b for b in v.bindings if b["column"] == "c2"]
    assert phone and phone[0]["prop"] == "PublicBody:phone" and phone[0]["entity"] == "office"


def test_an_institution_column_is_not_a_person_s_name():
    """The MPs' `college` column — 5 % filled, every value an institution —
    came back as a second `Person:name` on a person already keyed on
    `full_name` (`work-c15-cold`). The `org_name` detector reads the words
    only an organisation carries, so the binding is refused with a line.
    ONE DIRECTION ONLY: `proper_name` fires on «ДПІ Сумська» as readily as
    on a person, so a body's `name` is never refused on it, and a column
    the detectors cannot read is left to the model."""
    frame, profiles = _setup([
        ["ПІБ", "college", "Орган"],
        ["Коваленко Іван Петрович", "Київський національний університет", "ДПІ Сумська"],
        ["Шевченко Ольга Іванівна", "Львівський політехнічний інститут", "ДПІ Львівська"],
        ["Бондаренко Петро Сидорович", "Харківська державна академія", "ДПІ Сумська"],
        ["Мельник Андрій Васильович", "Одеський медичний університет", "ДПІ Львівська"],
    ])
    assert profiles[2].detectors.get("proper_name", 0) >= 0.8   # the trap
    v = validate(_plan([
        {"column": "c0", "prop": "Person:name", "entity": "person", "why": "names"},
        {"column": "c1", "prop": "Person:name", "entity": "person", "why": "college"},
        {"column": "c2", "prop": "PublicBody:name", "entity": "body", "why": "body"},
    ], entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                 {"key": "body", "schema": "PublicBody", "keys": ["c2"]}]),
        profiles, CAT, CFG, frame)
    assert {b["column"] for b in v.bindings} == {"c0", "c2"}
    reasons = {d.column: d.reason for d in v.decisions if d.verdict == "rejected"}
    assert "organisation" in reasons["c1"] and "Person" in reasons["c1"]


def test_the_effective_subject_is_the_bound_thing_whose_key_identifies_rows():
    """The enforcement register's plan (`work-p2`): subject `Debt`, no Debt
    entity; the office carried three bindings and 1 584 distinct names
    over 52 737 rows, the debtor two bindings and a name distinct on 95 %
    of them. The count alone chose the office; the key chooses the
    debtor, which is what the row is about."""
    from ftmap.plan.validate import _effective_subject
    rows = [["DEBTOR_NAME", "DEBTOR_BIRTHDATE", "ORG_NAME", "ORG_PHONE_NUM", "EMAIL_ADDR"]]
    for i in range(20):
        rows.append([f"Особа {i}", "01.01.1980", f"Відділ {i % 2}", f"(04142) 3-08-3{i % 2}",
                     f"vdvs{i % 2}@just.gov.ua"])
    frame, profiles = _setup(rows)
    entities = [{"key": "debtor", "schema": "Person", "keys": ["c0"]},
                {"key": "office", "schema": "PublicBody", "keys": ["c2"]}]
    bindings = [{"column": "c0", "prop": "Person:name", "entity": "debtor"},
                {"column": "c1", "prop": "Person:birthDate", "entity": "debtor"},
                {"column": "c2", "prop": "PublicBody:name", "entity": "office"},
                {"column": "c3", "prop": "PublicBody:phone", "entity": "office"},
                {"column": "c4", "prop": "PublicBody:email", "entity": "office"}]
    assert _effective_subject("Debt", entities, bindings, CAT, profiles) == "Person"
    # Without profiles the count decides, as before.
    assert _effective_subject("Debt", entities, bindings, CAT) == "PublicBody"


def test_key_identification_reads_the_same_from_profiles_and_from_the_frame():
    """`emit` has the frame and no profiles; `validate` has both. The two
    readings of one entity must agree, or the summary's subject and the
    validator's would differ on the same run."""
    from ftmap.plan.keys import key_identification
    rows = [["name", "code"]] + [[f"Особа {i}", "12345678" if i % 10 == 0 else ""]
                                 for i in range(40)]
    frame, profiles = _setup(rows)
    for keys in (["c0"], ["c1"], ["c0", "c1"], []):
        e = {"key": "x", "schema": "Person", "keys": keys}
        assert abs(key_identification(e, profiles) - key_identification(e, frame=frame)) < 1e-9
    assert key_identification({"keys": ["c0"]}, profiles) > key_identification({"keys": ["c1"]}, profiles)


def test_an_empty_declared_thing_is_revived_only_by_two_spelled_fields():
    """The court's purchases (`work-p6-cold`): the model declared a Contract
    keyed on the lot's description and bound nothing to it, and it was
    dropped as property-less while «Процедура» and «Назва закупівлі» spelled
    its fields exactly. Two such headers revive it; one generic word does
    not — an Organization keyed on VINs with «Назва» beside it is the empty
    entity the drop exists for."""
    frame, profiles = _setup([
        ["Опис", "Процедура", "Назва закупівлі", "Учасник"],
        ["Лот 1", "відкриті торги", "Папір офісний", "ТОВ «Ромашка»"],
        ["Лот 2", "переговорна", "Тонер", "ТОВ «Бузок»"],
        ["Лот 3", "відкриті торги", "Стільці", "ПП «Дуб»"],
    ])
    v = validate(_plan([
        {"column": "c3", "prop": "LegalEntity:name", "entity": "supplier", "why": "x"},
    ], entities=[{"key": "contract", "schema": "Contract", "keys": ["c0"]},
                 {"key": "supplier", "schema": "LegalEntity", "keys": ["c3"]}],
        subject="Contract"), profiles, CAT, CFG, frame)
    assert {e["key"] for e in v.entities} == {"contract", "supplier"}
    assert {(b["column"], b["prop"]) for b in v.bindings if b["entity"] == "contract"} == {
        ("c1", "Contract:procedure"), ("c2", "Contract:name")}
    frame, profiles = _setup([
        ["VIN", "Назва", "MODEL"],
        ["Y6DA69700A0000199", "ЗАЗ", "SENS"],
        ["XW8ZZZ61ZDG000123", "SKODA", "OCTAVIA"],
    ])
    v = validate(_plan([
        {"column": "c2", "prop": "Vehicle:model", "entity": "vehicle", "why": "x"},
    ], entities=[{"key": "vehicle", "schema": "Vehicle", "keys": ["c0"]},
                 {"key": "org", "schema": "Organization", "keys": ["c0"]}],
        subject="Vehicle"), profiles, CAT, CFG, frame)
    assert {e["key"] for e in v.entities} == {"vehicle"}


def test_an_organisation_keyed_on_a_column_of_words_survives_with_that_column_as_its_name():
    """The establishment table (`2026-09-05-manual-corpus.md`): the structure
    call declared an Organization keyed on «Рота», the binding call declined
    the column, and the organisation was dropped as property-less with the
    person's relation to it. See `plan.roles.name_from_key`."""
    frame, profiles = _setup([["ФИО", "Рота"],
                              ["Коваленко Іван", "1 мсв"],
                              ["Шевченко Ольга", "2 мсв"],
                              ["Бондаренко Ігор", "1 мсв"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "unmapped", "entity": "none", "why": "no candidate"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "platoon", "schema": "Organization", "keys": ["c1"]}],
        edges=[{"key": "e", "schema": "Employment", "source": "person",
                "target": "platoon"}],
    ), profiles, CAT, CFG, frame)
    assert {e["key"] for e in v.entities} == {"person", "platoon"}
    named = [b for b in v.bindings if b["column"] == "c1"]
    assert named and named[0]["prop"] == "Organization:name" \
        and named[0]["entity"] == "platoon"
    assert [e["key"] for e in v.edges] == ["e"]
    d = [d for d in v.decisions if d.column == "c1" and d.verdict == "accepted"][0]
    assert d.decided_by == "rule" and "named from its key" in d.reason


def test_a_person_s_name_bound_to_the_relation_s_names_mentioned_is_the_person_s_name():
    frame, profiles = _setup([["ПІБ", "Посада"],
                              ["Коваленко Іван", "стрілець"],
                              ["Шевченко Ольга", "командир"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Occupancy:namesMentioned", "entity": "held", "why": "x"},
         {"column": "c1", "prop": "Position:name", "entity": "post", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "post", "schema": "Position", "keys": ["c1"]}],
        edges=[{"key": "held", "schema": "Occupancy", "source": "person", "target": "post"}],
    ), profiles, CAT, CFG, frame)
    by = {(b["column"], b["entity"]): b["prop"] for b in v.bindings}
    assert by[("c0", "person")] == "Person:name"
    assert ("c0", "held") not in by
    assert [e["key"] for e in v.edges] == ["held"]
    assert any(d.verdict == "rejected" and "never to the link" in d.reason
               and d.prop == "Occupancy:namesMentioned" for d in v.decisions)


def test_a_thing_a_late_override_empties_is_dropped_with_what_pointed_at_it():
    """The forensic contacts sheet: a Note keyed on the name and the number
    block, its one binding the block as `phoneMentioned`. The detector rule
    gives the numbers to the contact, and the Note — property-less now —
    must not be emitted because the empty-entity drop ran before that."""
    rows = [["Имя", "Абоненты"]]
    for i in range(10):
        rows.append([f"Особа {i}", f"Mobile-: +38050123456{i}"])
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Note:phoneMentioned", "entity": "note", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "note", "schema": "Note", "keys": ["c0", "c1"]}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.entities] == ["person"]
    assert {(b["column"], b["entity"], b["prop"]) for b in v.bindings} == {
        ("c0", "person", "Person:name"), ("c1", "person", "Person:phone")}
    assert sum(1 for d in v.decisions if d.entity == "note"
               and "no column is bound" in d.reason) == 1


def test_a_birth_date_bound_to_the_occupancy_is_the_person_s():
    frame, profiles = _setup([["ПІБ", "Дата народження", "Посада"],
                              ["Коваленко Іван", "17.09.1980", "стрілець"],
                              ["Шевченко Ольга", "01.02.1990", "командир"]])
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Occupancy:date", "entity": "held", "why": "x"},
         {"column": "c2", "prop": "Position:name", "entity": "post", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "post", "schema": "Position", "keys": ["c2"]}],
        edges=[{"key": "held", "schema": "Occupancy", "source": "person", "target": "post"}],
    ), profiles, CAT, CFG, frame)
    by = {(b["column"], b["entity"]): b["prop"] for b in v.bindings}
    assert by[("c1", "person")] == "Person:birthDate" and ("c1", "held") not in by
    assert [e["key"] for e in v.edges] == ["held"]


def test_a_row_entity_whose_key_is_too_sparse_is_keyed_on_the_name_that_fills_the_rows():
    """The regiment's establishment: the person keyed on a service number
    absent on the open billets, the name on every row. The fill floor is
    right about the number; the row ordinal is not the alternative."""
    rows = [["ФИО", "Личный номер"]]
    for i in range(10):
        rows.append([f"Особа {i}", f"АБ-12345{i}" if i < 7 else None])
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Person:idNumber", "entity": "person", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert [e["keys"] for e in v.entities] == [["c0"]]
    d = [d for d in v.decisions if d.column == "c0" and d.verdict == "accepted"
         and "too sparse" in d.reason]
    assert d


def test_a_redaction_constant_in_the_key_column_is_no_key():
    from ftmap.plan.validate import _values_under_modal_key
    rows = [["Код", "Назва"]] + [["**********", f"ТОВ Фірма {i}"] for i in range(8)] \
        + [["12345678", "ТОВ Альфа"], ["12345678", "ТОВ Альфа"]]
    frame, _profiles = _setup(rows)
    entity = {"key": "org", "schema": "LegalEntity", "keys": ["c0"]}
    assert _values_under_modal_key(frame, "c1", entity) == 1.0


def test_a_thing_the_sheet_holds_one_of_is_not_each_row_s():
    """The forensic contacts sheet: a UserAccount keyed on «Account», the
    device owner's id on every row, attached as owned by every contact."""
    rows = [["Name", "Account", "Service"]]
    for i in range(6):
        rows.append([f"Особа {i}", "1234567890", "Telegram"])
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c2", "prop": "UserAccount:service", "entity": "acct", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]},
                  {"key": "acct", "schema": "UserAccount", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert not any(a["prop"] == "UserAccount:owner" for a in v.attachments)
    assert any("one thing is not each row's" in d.reason for d in v.decisions)


def test_the_key_floor_counts_records_not_the_section_rows_a_layout_keeps():
    """The brigade roster: eleven section rows kept as sparse furniture,
    the name column 83 % filled over the rows and 91 % over the records."""
    from ftmap.plan.validate import _record_fill
    rows = [["№", "Ф.И.О.", "Звание", "Телефон"]]
    for section in ("1 рота", "2 рота"):
        rows.append([section, None, None, None])           # a section row: sparse
        for i in range(5):
            rows.append([str(i + 1), f"Особа {section[0]}{i}", "рядовой", f"+38050123456{i}"])
    frame, profiles = _setup(rows)
    name = [p for p in profiles if p.id == "c1"][0]
    from ftmap.io.frame import column_values
    assert name.fill_rate < 0.9
    assert _record_fill(frame, column_values(frame, "c1"), name) == 1.0
    v = validate(_plan(
        [{"column": "c1", "prop": "Person:name", "entity": "person", "why": "x"}],
        entities=[{"key": "person", "schema": "Person", "keys": ["c1"]}],
    ), profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == ["c1"]


def test_the_sheet_s_machinery_and_a_header_written_twice_are_refused():
    """«впр л/н» is a VLOOKUP copy of the name column; «должность по штату
    сво» is «Должность по штату СВО» written again with the same values."""
    rows = [["Ф.И.О.", "Должность по штату СВО", "впр л/н", "должность по штату сво"]]
    for i in range(6):
        rows.append([f"Особа {i}", "стрелок" if i % 2 else "командир", f"Особа {i}",
                     "стрелок" if i % 2 else "командир"])
    frame, profiles = _setup(rows)
    v = validate(_plan(
        [{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
         {"column": "c1", "prop": "Person:position", "entity": "person", "why": "x"},
         {"column": "c2", "prop": "Person:alias", "entity": "person", "why": "x"},
         {"column": "c3", "prop": "Person:summary", "entity": "person", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    assert {b["column"] for b in v.bindings} == {"c0", "c1"}
    assert any(d.column == "c2" and d.verdict == "rejected" and "machinery" in d.reason
               for d in v.decisions)
    assert any(d.column == "c3" and d.verdict == "rejected" and "working copy" in d.reason
               for d in v.decisions)
