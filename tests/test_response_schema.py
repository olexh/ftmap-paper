import json

import jsonschema
import pytest

from ftmap.plan.response_schema import (NO_ENTITY, UNMAPPED, binding_schema,
                                        edge_schema, split_binding, split_edge,
                                        split_schema, valid_edges,
                                        structure_schema, valid_pairs)
from ftmap.vocab.catalogue import Catalogue

CAT = Catalogue.load()
ENTS = [{"key": "person", "schema": "Person"},
        {"key": "party", "schema": "Organization"}]


def test_structure_schema_only_allows_concrete_schemata():
    s = structure_schema(CAT, ["c0", "c1"])
    subject_enum = s["properties"]["subject"]["enum"]
    assert "Person" in subject_enum
    assert "Thing" not in subject_enum


def test_structure_schema_accepts_a_well_formed_answer():
    s = structure_schema(CAT, ["c0", "c1"])
    jsonschema.validate(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}]},
        s,
    )


def test_structure_keys_are_confined_to_the_columns_that_exist():
    """The measured failure: asked for an entity's key columns, the model
    answered with the source HEADER (`id`) rather than the column id (`c0`) —
    the header being the only part of the prompt's `c0: id` line that carries
    meaning. Every key was rejected as non-existent, both entities fell back to
    the row ordinal, and the run emitted one Person and one Organization per
    row of `ua_war_sanctions.targets.simple.csv`, 11 242 entities from 5 621
    rows, deduplicating nothing."""
    s = structure_schema(CAT, ["c0", "c1"])
    jsonschema.validate(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0", "c1"]}]}, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"subject": "Person",
             "entities": [{"key": "person", "schema": "Person", "keys": ["id"]}]}, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"subject": "Person",
             "entities": [{"key": "person", "schema": "Person", "keys": ["c2"]}]}, s)


def test_structure_schema_refuses_an_empty_column_list():
    with pytest.raises(ValueError, match="at least one column"):
        structure_schema(CAT, [])


def test_edge_endpoints_are_confined_to_keys_that_exist():
    """In one call the keys do not exist yet, so the endpoints cannot be an
    enum. Measured on three real files, the model then answered with a CSV
    header, with a schema name, and once by accident."""
    s = edge_schema(CAT, ENTS)
    jsonschema.validate({"edges": [{"key": "member",
                                    "edge": "Membership|person|party"}]}, s)
    for bad in ("Membership|Person|partyId",   # the CSV header / schema name
                "Membership|party|person",     # endpoints the wrong way round
                "Family|person|party"):        # a schema these two cannot satisfy
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"edges": [{"key": "member", "edge": bad}]}, s)


def test_an_edge_schema_its_endpoints_cannot_satisfy_is_unsayable():
    """The measured defect: `schema` was open across all sixteen edge schemata
    while only the endpoints were closed, so `Family` between an Organization
    and a Contract was well-formed and refused a moment later. Over the
    19-source etalon corpus, 14 of 19 declared edges were rejected on endpoint
    ranges and 9 of the 14 were exactly this — `Family` between things that are
    not two people."""
    combos = valid_edges(ENTS, CAT)
    assert "Membership|person|party" in combos
    assert not any(c.startswith("Family|") for c in combos), \
        "Family ranges Person to Person; one Person and one Organization is not that"
    # An entity is never related to itself.
    assert not any(c.endswith("|person|person") or c.endswith("|party|party")
                   for c in combos)


def test_two_people_can_be_family_and_one_cannot():
    two = [{"key": "a", "schema": "Person"}, {"key": "b", "schema": "Person"}]
    assert "Family|a|b" in valid_edges(two, CAT)
    assert valid_edges([{"key": "a", "schema": "Person"}], CAT) == []


def test_a_malformed_edge_answer_reads_as_no_edge_rather_than_crashing():
    assert split_edge("Membership|person|party") == ("Membership", "person", "party")
    for bad in (None, "", "Membership", "Membership|person", "a|b|c|d", "||"):
        assert split_edge(bad) is None


# --- Change 2: one closed enum of (entity, property) PAIRS, not two ---
#
# The measured defect: `prop` and `entity` were two independent enums, so
# nothing stopped the model from choosing a real candidate property AND a
# real declared entity that cannot carry it. 45% of all validation
# rejections were exactly that: "schema lacks the property". Encoding one
# string per valid pair — `"person|Person:name"` — makes the wrong
# combination something the grammar cannot even produce, rather than
# something `validate.py` catches after the fact.


def test_valid_pairs_only_includes_pairs_the_schema_can_carry():
    declared = {"person": "Person", "party": "Organization"}
    pairs = valid_pairs(["Person:name", "Organization:name"], declared, CAT)
    assert "person|Person:name" in pairs
    assert "party|Organization:name" in pairs
    # The cross pairings are real candidate, real entity, wrong together.
    assert "person|Organization:name" not in pairs
    assert "party|Person:name" not in pairs


def test_valid_pairs_drops_a_qname_no_declared_entity_can_carry():
    # `Organization:cageCode`, not `Organization:name`: since candidates are
    # re-spelled onto the entity's own schema, a Person offered
    # `Organization:name` is offered `Person:name` — the same property, which
    # a Person really does carry. The property here is one a Person does not.
    declared = {"person": "Person"}
    assert valid_pairs(["Organization:cageCode"], declared, CAT) == []


def test_valid_pairs_ignores_a_qname_outside_the_ontology():
    declared = {"person": "Person"}
    assert valid_pairs(["Person:notAProperty"], declared, CAT) == []


def test_valid_pairs_drops_a_property_only_a_reference_can_satisfy():
    """Carried by the schema, and unsatisfiable by any column: an entity-typed
    property takes the id of another entity, which comes from an edge endpoint
    and never from a cell. Measured on the МВС vehicle registry: a column of
    service-centre labels bound to `Vehicle:addressEntity`, 39 607 values
    offered, 0 emitted, 39 607 refused by followthemoney's entity validator."""
    declared = {"vehicle": "Vehicle", "link": "Membership"}
    pairs = valid_pairs(
        ["Vehicle:addressEntity", "Vehicle:model", "Membership:organization",
         "Membership:role"], declared, CAT)
    assert "vehicle|Vehicle:model" in pairs
    assert "link|Membership:role" in pairs
    assert "vehicle|Vehicle:addressEntity" not in pairs
    # An edge's own endpoint is the sharpest case: `compile_mapping` writes it
    # as `{"entity": <key>}` first, and a binding on the same property name
    # would overwrite the reference that makes the edge an edge.
    assert "link|Membership:organization" not in pairs


def test_valid_pairs_mirrors_the_structural_check_exactly():
    """`valid_pairs` and `validate.py`'s `_structural_check` must agree on
    every (entity, property) pair, because they share the same underlying
    test — `Catalogue.bindable`. This walks the two independently rather than
    importing one from the other, so a future edit to either side that
    breaks the agreement fails here."""
    declared = {"person": "Person", "party": "Organization"}
    candidates = ["Person:name", "Person:birthDate", "Person:addressEntity",
                  "Organization:name", "Organization:jurisdiction",
                  "Organization:parent", "Membership:role"]
    pairs = set(valid_pairs(candidates, declared, CAT))
    for entity, schema in declared.items():
        for qname in candidates:
            expected = CAT.bindable(schema, qname)
            assert (f"{entity}|{qname}" in pairs) == expected, (entity, qname)
    # Not vacuous: the two entity-typed candidates above are carried by their
    # schemata and still absent, so the walk is testing more than `carries`.
    assert CAT.carries("Person", "Person:addressEntity")
    assert "person|Person:addressEntity" not in pairs


def test_split_binding_round_trips_a_pair():
    assert split_binding("person|Person:name") == ("person", "Person:name")


def test_split_binding_treats_unmapped_and_malformed_input_the_same():
    assert split_binding(UNMAPPED) == (NO_ENTITY, UNMAPPED)
    assert split_binding(None) == (NO_ENTITY, UNMAPPED)
    assert split_binding("garbage-with-no-pipe") == (NO_ENTITY, UNMAPPED)
    assert split_binding("") == (NO_ENTITY, UNMAPPED)


def test_binding_schema_offers_pairs_not_independent_fields():
    s = binding_schema(["c0"], {"c0": ["Person:name"]}, {"person": "Person"}, CAT)
    branch = s["properties"]["bindings"]["prefixItems"][0]
    assert "binding" in branch["properties"]
    assert "prop" not in branch["properties"]
    assert "entity" not in branch["properties"]
    assert branch["properties"]["binding"]["enum"] == ["person|Person:name", UNMAPPED]


def test_binding_schema_makes_a_wrong_pairing_structurally_impossible():
    """The whole point of Change 2. Both `Person:name` and `Organization:name`
    are real candidates and both `person` and `party` are real declared
    entities, but pairing the property with the entity that cannot carry it
    is not a document the grammar can express at all — `jsonschema.validate`
    rejects it exactly like an out-of-shortlist property always did."""
    s = binding_schema(["c0"], {"c0": ["Person:name", "Organization:name"]},
                       {"person": "Person", "party": "Organization"}, CAT)
    jsonschema.validate(
        {"bindings": [{"column": "c0", "binding": "person|Person:name", "why": "x"}]}, s)
    jsonschema.validate(
        {"bindings": [{"column": "c0", "binding": "party|Organization:name", "why": "x"}]}, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"bindings": [{"column": "c0", "binding": "person|Organization:name", "why": "x"}]}, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"bindings": [{"column": "c0", "binding": "party|Person:name", "why": "x"}]}, s)


def test_binding_schema_still_confines_each_column_to_its_own_candidates():
    s = binding_schema(["c0", "c1"],
                       {"c0": ["Person:name"], "c1": ["Person:birthDate"]},
                       {"person": "Person"}, CAT)
    ok = {"bindings": [
        {"column": "c0", "binding": "person|Person:name", "why": "names"},
        {"column": "c1", "binding": "unmapped", "why": "unclear"},
    ]}
    jsonschema.validate(ok, s)
    # Full length, so this fails on the cross-column candidate and not merely
    # on being short — see `test_binding_schema_requires_an_answer_per_column`.
    bad = {"bindings": [
        {"column": "c0", "binding": "person|Person:birthDate", "why": "x"},
        {"column": "c1", "binding": "unmapped", "why": "x"},
    ]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, s)


def test_binding_schema_requires_an_answer_per_column():
    """Stopping early has to be ungrammatical, not merely disappointing.

    Under `minItems: 1` a short array was legal, `propose` filled the silence
    with `unmapped`, and a column the model never considered was indistinguish-
    able from one it declined. Declining is still available — `unmapped` is in
    every column's enum — but it now has to be said out loud.
    """
    s = binding_schema(["c0", "c1", "c2"],
                       {"c0": ["Person:name"], "c1": ["Person:name"],
                        "c2": ["Person:name"]},
                       {"person": "Person"}, CAT)
    short = {"bindings": [{"column": "c0", "binding": "person|Person:name", "why": "x"}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(short, s)

    full = {"bindings": [
        {"column": "c0", "binding": "person|Person:name", "why": "names"},
        {"column": "c1", "binding": "unmapped", "why": "declined, out loud"},
        {"column": "c2", "binding": "unmapped", "why": "declined, out loud"},
    ]}
    jsonschema.validate(full, s)

    over = {"bindings": full["bindings"] + [
        {"column": "c0", "binding": "unmapped", "why": "x"}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(over, s)

    # Right length, wrong content: c0 answered twice and c2 never. Length
    # bounds alone allowed this, and the model produced it — on the last
    # column of a chunk, which is where the loss concentrated.
    duplicated = {"bindings": [
        {"column": "c0", "binding": "person|Person:name", "why": "names"},
        {"column": "c1", "binding": "unmapped", "why": "x"},
        {"column": "c0", "binding": "unmapped", "why": "x"},
    ]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(duplicated, s)

    # Right columns, wrong order — also ungrammatical, because position i
    # carries column i's `const`.
    reordered = {"bindings": [full["bindings"][1], full["bindings"][0],
                              full["bindings"][2]]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(reordered, s)


def test_binding_schema_rejects_an_undeclared_entity_key():
    s = binding_schema(["c0"], {"c0": ["Person:name"]}, {"person": "Person"}, CAT)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"bindings": [{"column": "c0", "binding": "party|Person:name", "why": "x"}]}, s)


def test_every_object_is_strict_mode_safe():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(structure_schema(CAT, ["c0", "c1"]))
    walk(binding_schema(["c0"], {"c0": ["Person:name"]}, {"person": "Person"}, CAT))
    walk(edge_schema(CAT, ENTS))
    json.dumps(structure_schema(CAT, ["c0", "c1"]))


def test_an_empty_column_list_names_its_own_failure():
    """Otherwise the server answers HTTP 400 'failed to parse grammar', which
    says nothing about the empty sheet that caused it."""
    with pytest.raises(ValueError, match="at least one column"):
        binding_schema([], {}, {"person": "Person"}, CAT)


def test_unknownlink_is_not_offered_though_the_ontology_declares_it():
    """`UnknownLink` ranges Thing to Thing, so it is offerable between every
    pair of entities in every plan and competes with the specific answer
    everywhere at once. Measured: 5 of 20 declared edges were UnknownLink and
    none matched an etalon. A record table never states an untyped link."""
    assert any(e.schema == "UnknownLink" for e in CAT.edges())
    combos = valid_edges(ENTS, CAT)
    assert combos, "the plan still has real options"
    assert not any(c.startswith("UnknownLink|") for c in combos)


def test_a_candidate_spelled_on_a_sibling_schema_is_offered_on_the_entity_s_own():
    """Retrieval returns ONE spelling of a property; the entity may need
    another.

    `name` is declared on `Thing`, so the catalogue indexes it under every
    schema that inherits it, and `carries("LegalEntity", "Organization:name")`
    is False — `LegalEntity` does not descend from `Organization`. Retrieval
    returning `Organization:name` for a column of supplier names therefore
    offered that column NOTHING for a declared `LegalEntity` supplier, while
    the same property under its own spelling was available all along.

    Measured on `ua_annual_procurement_plan_2023q3` c5 (`з ким укладено
    договір`) on 2026-08-29: the only offered pairs put the supplier's name on
    the BUYER, the model bound `Thing:sourceUrl` instead, the engine refused
    it, and the supplier entity was dropped for carrying no property.
    """
    pairs = valid_pairs(["Organization:name"], {"supplier": "LegalEntity"}, CAT)
    assert pairs == ["supplier|LegalEntity:name"]


def test_a_property_the_entity_does_not_have_is_still_not_offered():
    """Re-spelling is not a licence to invent. `serialNumber` is declared on
    `Airplane` and a `Vehicle` does not carry it — the etalon comparison notes
    this exact pair as an ontology fact worth getting right."""
    assert valid_pairs(["Airplane:serialNumber"], {"car": "Vehicle"}, CAT) == []


def test_a_candidate_the_entity_carries_under_its_own_name_is_offered_once():
    pairs = valid_pairs(["LegalEntity:name", "Organization:name"],
                        {"party": "LegalEntity"}, CAT)
    assert pairs == ["party|LegalEntity:name"]


def test_split_schema_pins_each_entity_to_its_own_position():
    """`items` with an entity enum let the model answer one sibling twice and
    never the other — the same grammar failure `binding_schema` documents and
    fixes with `prefixItems` + `const`. The omitted sibling then kept no
    filter and emitted from every row."""
    s = split_schema(["person", "wallet"], ["c1=Person", "c1=CryptoWallet"])
    sel = s["properties"]["selections"]
    assert "items" not in sel
    assert sel["minItems"] == sel["maxItems"] == 2
    consts = [p["properties"]["entity"]["const"] for p in sel["prefixItems"]]
    assert consts == ["person", "wallet"]
