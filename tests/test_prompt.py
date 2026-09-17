from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.plan.prompt import (BINDING_MARKER, BINDING_SYSTEM, EDGE_MARKER,
                               kind_columns, split_prompt, SPLIT_MARKER,
                               REPAIR_MARKER, STRUCTURE_MARKER, STRUCTURE_SYSTEM,
                               binding_prompt, edge_prompt, repair_prompt,
                               structure_prompt)
from ftmap.profile.columns import profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()

ROWS = [["ПІБ", "Дата народження", "Фракція"],
        ["Коваленко Іван Петрович", "17.09.1980", "Слуга народу"]]


def _fp():
    f = build_frame(Grid(rows=ROWS, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    return f, profile_frame(f, CFG)


def test_instructions_are_english_and_forbid_translation():
    assert "translate" in STRUCTURE_SYSTEM.lower()
    assert "translate" in BINDING_SYSTEM.lower()
    for word in ("Ukrainian", "Russian"):
        assert word in BINDING_SYSTEM


def test_structure_prompt_carries_headers_verbatim():
    frame, profiles = _fp()
    text = structure_prompt(frame, profiles)
    assert "Дата народження" in text
    assert "c1" in text
    assert "rows: 1" in text


def test_binding_prompt_shows_samples_and_candidates():
    """The candidate list is now pairs, not bare properties, and the entity
    each one would attach to must be explicit — the project's own measured
    lesson is that prompt text is not cosmetic, it changes what the model
    declares, so hiding which entity a property would bind to would leave
    the prompt no longer describing what the grammar actually allows."""
    frame, profiles = _fp()
    declared = {"person": "Person", "party": "Organization"}
    pairs = {"c0": ["person|Person:name"], "c1": ["person|Person:birthDate"],
            "c2": ["party|Organization:name"]}
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]},
                {"key": "party", "schema": "Organization", "keys": []}]
    text = binding_prompt(profiles, pairs, "Person", declared, CAT,
                          entities=entities)
    assert "Коваленко Іван Петрович" in text
    assert "person (Person), identified by c0 (ПІБ)" in text
    # The candidate LINE itself makes the entity explicit, e.g.
    # "person (Person) -> Person:birthDate (Birth date)" — not just the
    # qname appearing somewhere and not just "unmapped", which is boilerplate
    # present regardless of what candidates were built.
    assert "person (Person)" in text
    assert "Person:birthDate (Birth date)" in text
    assert "party (Organization)" in text
    assert "Organization:name" in text


def test_repair_prompt_names_the_rejected_values():
    text = repair_prompt([
        {"column": "c1", "prop": "Person:birthDate", "reason": "date validator",
         "rejected": ["не вказано", "-"]},
    ])
    assert "c1" in text and "не вказано" in text and "Person:birthDate" in text


def test_each_prompt_carries_its_own_marker_and_no_others():
    """I5: `_CachedPlanClient` (and every test double) dispatches a replayed
    prompt on these markers instead of re-typing prompt prose independently.
    Each builder already asserts its own marker on its own output; this test
    is the external check that the five are mutually exclusive too, which is
    the property the dispatch logic actually depends on.

    `split_prompt` was left out of this table while `SPLIT_MARKER` was left out
    of the dispatch, so the one invariant test that would have named the gap
    had the same gap. A prompt builder absent here is a prompt the replay
    client is not shown to handle."""
    frame, profiles = _fp()
    structure = {"subject": "Person",
                 "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
                 "edges": []}
    declared = {"person": "Person"}
    pairs = {"c0": ["person|Person:name"], "c1": ["person|Person:birthDate"]}
    kind_frame = _kind_frame()
    kind_profiles = [p for p in profile_frame(kind_frame, CFG) if p.filled > 0]

    by_marker = {
        STRUCTURE_MARKER: structure_prompt(frame, profiles),
        SPLIT_MARKER: split_prompt(
            [{"key": "person", "schema": "Person", "keys": ["c0"]},
             {"key": "org", "schema": "Organization", "keys": ["c0"]}],
            kind_profiles, kind_columns(kind_frame, kind_profiles, CFG)),
        EDGE_MARKER: edge_prompt(structure["entities"], profiles),
        BINDING_MARKER: binding_prompt(profiles, pairs, "Person", declared, CAT,
                                       entities=structure["entities"]),
        REPAIR_MARKER: repair_prompt([
            {"column": "c0", "prop": "Person:name", "reason": "x", "rejected": ["a"]},
        ]),
    }
    for marker, text in by_marker.items():
        assert marker in text
        for other in by_marker:
            if other != marker:
                assert other not in text, (marker, other)


def test_the_structure_call_is_shown_the_values_it_decides_from():
    """The structure call answers what one row IS — the subject, every entity,
    the columns identifying each — and it constrains everything after it. It
    used to decide from headers, fill rates and shape strings alone, while the
    binding call was shown samples. Measured: the subject was wrong on 15 of 20
    sources, and the ICIJ officers table — headers `node_id, name, countries,
    country_codes, sourceID, valid_until, note` — was declared an `Asset`. Its
    values are people's names."""
    from ftmap.io.layout import build_frame
    from ftmap.io.tabular import Grid
    from ftmap.plan.prompt import structure_prompt
    from ftmap.profile.columns import profile_frame

    frame = build_frame(Grid(rows=[["name", "countries"],
                                   ["Коваленко Іван Петрович", "Ukraine"],
                                   ["Шевченко Ольга Миколаївна", "Cyprus"]],
                             sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    text = structure_prompt(frame, profile_frame(frame, CFG))
    assert "Коваленко Іван Петрович" in text, "a name is what makes this a Person"
    assert "values:" in text
    # The header line is still there; the values are added, not substituted.
    assert "c0: name" in text and "filled" in text


def test_the_binding_prompt_says_what_the_properties_mean():
    """1 258 of 2 641 qnames carry a description differing from the label, and
    it is where the distinguishing information lives — `idNumber` is "used
    mainly for people and their national ID cards". Every etalon that got a
    hard column right cited one of these; the model was shown none of them."""
    from ftmap.plan.prompt import binding_prompt
    from ftmap.plan.response_schema import valid_pairs
    from ftmap.vocab.catalogue import Catalogue
    from ftmap.io.layout import build_frame
    from ftmap.io.tabular import Grid
    from ftmap.profile.columns import profile_frame

    cat = Catalogue.load()
    frame = build_frame(Grid(rows=[["id"], ["7"]], sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    profs = profile_frame(frame, CFG)
    declared = {"p": "Person", "o": "Organization"}
    pairs = {profs[0].id: valid_pairs(["Person:idNumber", "Organization:name"],
                                      declared, cat)}
    text = binding_prompt(profs, pairs, "Person", declared, cat,
                          entities=[{"key": "p", "schema": "Person",
                                     "keys": [profs[0].id]},
                                    {"key": "o", "schema": "Organization",
                                     "keys": []}])
    assert "what these properties mean" in text
    assert "national ID cards" in text
    # Deduplicated: one line per qname however many candidates offer it.
    assert text.count("Person:idNumber:") == 1


def test_a_property_whose_description_repeats_its_label_is_not_listed():
    """The glossary exists to add information. A description identical to the
    label adds none and costs context the wide sources do not have."""
    from ftmap.plan.prompt import binding_prompt
    from ftmap.vocab.catalogue import Catalogue
    from ftmap.io.layout import build_frame
    from ftmap.io.tabular import Grid
    from ftmap.profile.columns import profile_frame

    cat = Catalogue.load()
    frame = build_frame(Grid(rows=[["x"], ["1"]], sheet="s", merges=[]),
                        "/x.csv", "0" * 64, "sid/s", CFG)
    profs = profile_frame(frame, CFG)
    # Vehicle:registrationNumber carries no description at all.
    assert not (cat.prop("Vehicle:registrationNumber").description or "").strip()
    pairs = {profs[0].id: ["v|Vehicle:registrationNumber"]}
    text = binding_prompt(profs, pairs, "Vehicle", {"v": "Vehicle"}, cat,
                          entities=[{"key": "v", "schema": "Vehicle",
                                     "keys": [profs[0].id]}])
    assert "Vehicle:registrationNumber:" not in text


def test_the_structure_system_prompt_says_what_generalising_costs():
    """Measured on the state-enterprise register: both models declared two
    `Organization` entities, which is linguistically correct and deletes the
    file's only relation. `Ownership` ranges its target on `Asset`; `Company`
    descends from `Asset` and `Organization` does not, so `Ownership` was not
    in the menu the edge call was offered. The model never declined the
    relationship — it was never shown it."""
    from ftmap.plan.prompt import STRUCTURE_SYSTEM
    from ftmap.vocab.catalogue import Catalogue

    cat = Catalogue.load()
    assert cat.is_descendant("Company", "Asset")
    assert not cat.is_descendant("Organization", "Asset")
    assert "most specific schema" in STRUCTURE_SYSTEM
    # It must also say when the general schema is right, or it trades one
    # systematic error for the opposite one.
    assert "LegalEntity" in STRUCTURE_SYSTEM


def test_structure_system_asks_for_things_as_well_as_parties():
    """Fifteen of the entities the gold standards name were never declared
    and almost none is a party — `ContractAward`, `Contract`,
    `Identification`, `Address`, `Document`, `Debt`. This clause is the only
    change that has ever moved the entity layer (+7 matched, +4 keys
    agreeing at `work-full-thing4`); its fan-out into spurious edges is
    handled downstream by `validate`'s fan gate, not by hedging the prompt —
    the tight form that put restraint here gave the gain back, twice
    (`2026-08-20-negative-results.md`)."""
    from ftmap.plan.prompt import STRUCTURE_SYSTEM

    assert "NOT EVERYTHING A ROW DESCRIBES IS A PARTY" in STRUCTURE_SYSTEM
    for word in ("contract", "licence", "certificate", "court", "debt"):
        assert word in STRUCTURE_SYSTEM.lower(), word


def test_edge_system_does_not_discourage_answering():
    """The same restraint has now cost measurable edges twice.

    First as "No relationship is a valid answer" in `edge_prompt`, where it took
    declared edges from 19 to 5 across the corpus. Then, three lines above the
    comment recording that, as "and only where the table states the
    relationship" in EDGE_SYSTEM — worth 3 points of column precision and every
    spurious edge on the five-source ablation set.

    The grammar already makes a wrong answer unsayable and an empty array is
    still expressible, so neither prompt needs to argue for silence. This test
    exists because prose that sounds careful is easy to reintroduce.
    """
    from ftmap.plan.prompt import EDGE_SYSTEM

    for phrase in ("only where", "valid answer", "rather than leave it out"):
        assert phrase not in EDGE_SYSTEM, phrase


def test_the_edge_menu_does_not_annotate_the_roles():
    """Naming the two positions — "(c3 is the owner, c1 is the asset)" — was
    written to stop the direction being a coin flip. Ablated over five sources,
    two replicates: removing it raised found edges from 2 to 3 and took
    wrong-direction edges from 1 to 0. It caused the error it was written to
    prevent, so the menu stays bare."""
    frame, profiles = _fp()
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]},
                {"key": "party", "schema": "Organization", "keys": ["c2"]}]
    text = edge_prompt(entities, profiles, ["Membership|person|party"])
    assert "Membership|person|party" in text
    assert " is the " not in text


ROW_KIND = [["name", "schema", "note"],
            ["Коваленко Іван Петрович", "Person", "a"],
            ["ТОВ Ромашка", "Organization", "b"],
            ["Шевченко Ольга Іванівна", "Person", "c"],
            ["1CF46Rfbp97absrs7zb7dFfZS6qBXUm9EP", "CryptoWallet", "d"]]


def _kind_frame():
    return build_frame(Grid(rows=ROW_KIND, sheet="s", merges=[]),
                       "/x.csv", "0" * 64, "sid/s", CFG)


def test_the_split_prompt_names_the_columns_that_say_what_a_row_is():
    """The one question a single non-conditional plan cannot answer, asked.

    `us_ofac_sdn` declares its rows' schema in a column taking eight values,
    and a plan that ignores that column claims every row is every kind: the
    2026-08-28 run emitted 20 054 CryptoWallet from a file holding 976. The
    column is right there and the prompt never mentioned it.
    """
    frame = _kind_frame()
    profiles = [p for p in profile_frame(frame, CFG) if p.filled > 0]
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]},
                {"key": "org", "schema": "Organization", "keys": ["c0"]}]
    text = split_prompt(entities, profiles, kind_columns(frame, profiles, CFG))
    assert "c1 (schema)" in text
    assert "Person (2)" in text and "CryptoWallet (1)" in text
    # The unique column is not a kind column: it selects one row per value.
    assert "c0 (name)" not in text.split("what KIND")[1]
    # And the structure call is not the place this is asked.
    assert "what KIND" not in structure_prompt(frame, profiles)


def test_a_column_with_a_value_per_row_is_not_offered_as_a_kind():
    frame = _kind_frame()
    profiles = [p for p in profile_frame(frame, CFG) if p.filled > 0]
    assert [c.column for c in kind_columns(frame, profiles, CFG)] == ["c1"]


def test_the_binding_roster_says_which_column_identifies_each_entity():
    """THE ROSTER IS THE ONLY PLACE THIS CALL LEARNS WHAT AN ENTITY IS, and
    until now it read `person (Person)` and nothing else — the alias and the
    schema. On a table declaring two entities of one schema, or five blocs of
    a polymorphic one, that is not enough to tell a candidate's entity apart
    from its neighbour, and the model has been leaning on the ALIAS to carry
    the difference because the structure call happens to name each entity
    after the column it keyed it on.

    Measured 2026-08-30: replacing that alias with the entity's schema —
    `person` for `c2` — cost 13 correct bindings across the corpus, because
    the pointer to the key column went with it.

    `edge_prompt` had the same problem and was repaired the same way: bare
    names produced no edges at all on three files, "because the keys say
    nothing", so it shows each entity's key columns and their headers. This
    is that repair, one call over.
    """
    frame, profiles = _fp()
    declared = {"person": "Person", "party": "Organization"}
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]},
                {"key": "party", "schema": "Organization", "keys": ["c2"]}]
    pairs = {"c0": ["person|Person:name"], "c1": ["person|Person:birthDate"],
             "c2": ["party|Organization:name"]}
    text = binding_prompt(profiles, pairs, "Person", declared, CAT,
                          entities=entities)
    assert "person (Person), identified by c0 (ПІБ)" in text
    assert "party (Organization), identified by c2" in text


def test_a_keyless_entity_is_named_by_the_column_it_came_from():
    """An entity with no key is not an entity with no description. The edge
    prompt's own note records what happens without this: four entities of the
    ship register all read `no key column`, indistinguishable, though their
    headers name an owner and a charterer written twice."""
    frame, profiles = _fp()
    declared = {"c1": "Person"}
    entities = [{"key": "c1", "schema": "Person", "keys": []}]
    text = binding_prompt(profiles, {"c0": ["c1|Person:name"]}, "Person",
                          declared, CAT, entities=entities)
    assert "no key column; named after c1 (Дата народження)" in text


def test_the_roster_sees_key_columns_outside_the_chunk():
    """`binding_prompt` receives one chunk of the askable columns, but an
    entity's key column can sit in another chunk — or be unanswerable and in
    no chunk at all. Built from the chunk alone, the roster told the model
    `no key column` about an entity that has one, on every chunk past the
    one holding it. `all_profiles` is the whole live table."""
    frame, profiles = _fp()
    chunk = [p for p in profiles if p.id != "c0"]
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]}]
    text = binding_prompt(chunk, {"c1": ["person|Person:birthDate"]}, "Person",
                          {"person": "Person"}, CAT, entities=entities,
                          all_profiles=profiles)
    assert "person (Person), identified by c0 (ПІБ)" in text
    assert "no key column" not in text


def test_the_structure_call_is_told_what_the_schemata_mean():
    """The grammar offers 64 concrete schemata by name and nothing else, and
    a name is where the subject layer has failed on every measurement since
    2026-08-20 — `Audio` for a ship register, `TaxRoll` for a procurement
    table. `2026-08-20-negative-results.md` §1: closing off a wrong option
    only moves the mass to the next one; "the wrong subject is an absence of
    evidence". This is the evidence, from the ontology's own words, the way
    the binding call has carried its property glossary since the ablation
    that measured its removal harmful at scale."""
    from ftmap.vocab.catalogue import Catalogue

    cat = Catalogue.load()
    frame = build_frame(Grid(rows=[["ПІБ"], ["Коваленко Іван"]], sheet="s",
                             merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    text = structure_prompt(frame, profiles, cat)
    assert "what these schemata mean, in the ontology's own words:" in text
    assert "  Company: Company — A corporation, usually for profit." in text
    assert "  TaxRoll: Tax roll" in text
    # Every schema the grammar can answer with is on the menu, described or
    # not, and the question still comes last.
    for name in cat.concrete_schemata():
        assert f"  {name}: " in text, name
    assert text.rstrip().endswith("asked for separately.")
    # Without the catalogue the prompt is what it was, so the replay client's
    # markers and every older test see the same text.
    assert "what these schemata mean" not in structure_prompt(frame, profiles)


def test_the_organisation_name_detector_is_not_shown_to_the_model():
    """Its work is retrieval and the rules; shown, it re-rolled the corpus
    (`work-c3`)."""
    from dataclasses import replace
    frame, profiles = _fp()
    p = replace(profiles[0], detectors={"org_name": 0.9, "legal_name": 0.4})
    text = binding_prompt([p, *profiles[1:]], {"c0": ["person|Person:name"]}, "Person",
                          {"person": "Person"}, CAT,
                          entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}])
    assert "org_name" not in text and "legal_name 40%" in text
