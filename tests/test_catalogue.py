from ftmap.vocab.catalogue import Catalogue


def test_person_properties_are_present_and_typed():
    cat = Catalogue.load()
    props = cat.properties_of("Person")
    assert "birthDate" in props
    assert props["birthDate"].type_name == "date"
    assert props["birthDate"].qname == "Person:birthDate"
    assert props["name"].matchable is True


def test_qname_lookup_resolves_inherited_properties():
    cat = Catalogue.load()
    # `name` is declared on Thing, so Person carries it under its own qname.
    assert cat.prop("Thing:name") is not None
    assert cat.prop("Person:name") is not None
    assert cat.prop("Person:notAProperty") is None


def test_abstract_schemata_are_not_offered_as_subjects():
    # FollowTheMoney 4.10.1 ships 69 schemata, 5 of them abstract.
    cat = Catalogue.load()
    names = cat.concrete_schemata()
    assert "Person" in names
    assert "Thing" not in names  # abstract
    assert len(names) == 64


def test_edges_carry_their_endpoint_ranges():
    cat = Catalogue.load()
    edges = {e.schema: e for e in cat.edges()}
    m = edges["Membership"]
    assert m.source_prop == "member"
    assert m.target_prop == "organization"
    assert m.target_range == "Organization"


def test_descendant_relation():
    cat = Catalogue.load()
    assert cat.is_descendant("Person", "LegalEntity") is True
    assert cat.is_descendant("Person", "Organization") is False


def test_carries_is_true_only_when_the_schema_can_hold_the_property():
    """`carries` is the ontology half of the test `_structural_check`
    (validate.py) and the grammar's pair-builder (response_schema.py) both
    apply, so the two cannot disagree about what a wrong pairing is."""
    cat = Catalogue.load()
    assert cat.carries("Person", "Person:name") is True
    assert cat.carries("Person", "Organization:name") is False
    assert cat.carries("Organization", "Person:name") is False
    # Not a property in the ontology at all.
    assert cat.carries("Person", "Person:notAProperty") is False


def test_bindable_refuses_a_property_no_cell_could_ever_satisfy():
    """`carries` is necessary and not sufficient, and the gap was measured:
    Vehicle really does carry `addressEntity`, so a column of service-centre
    labels on the МВС vehicle registry was offered it, took it, and had all
    39 607 of its values refused by followthemoney's entity validator.

    An entity-typed property takes the id of another entity, which this
    pipeline computes as a hash — no cell can hold one."""
    cat = Catalogue.load()
    assert cat.carries("Vehicle", "Vehicle:addressEntity") is True
    assert cat.is_entity_reference("Vehicle:addressEntity") is True
    assert cat.bindable("Vehicle", "Vehicle:addressEntity") is False
    # An edge's own endpoints are the same class, and worse: a binding on one
    # overwrites the reference `compile_mapping` wrote there.
    assert cat.bindable("Membership", "Membership:organization") is False
    # What a column CAN satisfy is untouched, including on the edge itself.
    assert cat.bindable("Vehicle", "Vehicle:model") is True
    assert cat.bindable("Membership", "Membership:role") is True
    # Still false for the reasons `carries` was already false for.
    assert cat.bindable("Person", "Organization:name") is False
    assert cat.bindable("Person", "Person:notAProperty") is False
    assert cat.is_entity_reference("Person:notAProperty") is False


def test_a_declared_format_travels_with_the_property():
    """The ontology's extra condition on a value, carried so the gate can ask
    it. Only 16 of followthemoney 4.10.1's 518 non-stub properties declare a
    format — every one of them `identifier`-typed — but inheritance spreads
    those 16 over 101 of the catalogue's 2641 qnames, `wikidataId` alone
    reaching 39 schemata from `Thing`. A gate that read the format off the
    DECLARING schema only would miss all but the 16.
    """
    cat = Catalogue.load()
    assert cat.prop("Thing:wikidataId").format == "wikidata"
    # Inherited, and the inherited form is what every other artefact names.
    assert cat.prop("Person:wikidataId").format == "wikidata"
    assert cat.prop("Organization:wikidataId").format == "wikidata"
    # The same type, no format: `idNumber` takes any identifier.
    assert cat.prop("Person:idNumber").type_name == "identifier"
    assert cat.prop("Person:idNumber").format is None
    assert cat.prop("Person:name").format is None

    with_format = [p for p in cat.all_props() if p.format]
    assert len(with_format) == 101
    assert {p.type_name for p in with_format} == {"identifier"}
    # A FORMAT IS NOT A REASON TO WITHHOLD THE PROPERTY. It narrows what a
    # column may hold; it does not put the property beyond every cell the way
    # an entity reference does, and the sanctions file emitted 396 real
    # Q-numbers through this very property. The check is evidential, and it
    # lives where the values are (`plan/validate._check_binding`).
    assert cat.bindable("Person", "Person:wikidataId") is True


def test_the_declaring_schema_of_a_property_is_known():
    cat = Catalogue.load()
    assert cat.declared_on("Sanction:program") == "Sanction"
    assert cat.declared_on("Sanction:publisher") == "Interval"
    assert cat.declared_on("License:contractDate") == "Contract"
    assert cat.declared_on("Person:nothingOfTheSort") is None
    assert cat.is_abstract("Interval") and cat.is_abstract("Thing")
    assert not cat.is_abstract("Sanction")
