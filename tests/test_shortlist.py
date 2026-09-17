from ftmap.config import Config
from ftmap.profile.columns import ColumnProfile
from ftmap.vocab.catalogue import Catalogue
from ftmap.vocab.shortlist import load_lexicon, shortlist

CFG = Config.load(None)
CAT = Catalogue.load()


def _p(header, detectors=None, shapes=None, samples=None, label=None):
    return ColumnProfile(
        id="c0", index=0, header=header, label=label, count=10, filled=10,
        fill_rate=1.0, distinct=10, distinct_ratio=1.0, min_len=3, max_len=30,
        shapes=shapes or [], detectors=detectors or {}, samples=samples or [],
    )


def test_lexicon_entries_all_resolve_in_the_ontology():
    """A spelling pointing at a qname that does not exist is a silent no-op."""
    for qname in load_lexicon():
        if qname.startswith("_"):
            continue  # metadata, e.g. "_provenance" — not a qname
        assert CAT.prop(qname) is not None, qname


def test_lexicon_keys_are_all_properties_a_column_could_satisfy():
    """A key that no column can be bound to is a spelling that can never
    match: retrieval does not offer an entity-typed property and the grammar
    cannot express a binding to one, so such an entry would be unmatchable and
    the file would overstate what it covers. Two were —
    `Membership:organization` and `Occupancy:post` — and their observed
    spellings moved to the properties that hold those values."""
    for qname in load_lexicon():
        if qname.startswith("_"):
            continue
        assert not CAT.is_entity_reference(qname), qname
    lex = load_lexicon()
    assert "фракція" in lex["Organization:name"]
    assert "займана посада" in lex["Position:name"]


def test_retrieval_never_offers_a_property_only_a_reference_can_satisfy():
    """An address column is where this bites: `Person:addressEntity` scores
    well on the header and can never hold a value. Measured on the sanctions
    file, 6 of the 12 pairs offered for its `addresses` column were
    entity-typed — half the candidate list spent on bindings followthemoney
    would refuse value by value."""
    got = shortlist(_p("Адреса"), CAT, CFG, schemas=["Person", "Organization"])
    assert got, "the column retrieved nothing at all"
    assert [q for q in got if CAT.prop(q).type_name == "entity"] == []
    # THE PROPERTY, NOT THE SPELLING. One slot per property NAME, and which
    # schema's spelling of `address` takes that slot is not this test's
    # question — `valid_pairs` re-spells whichever survives onto the entity
    # that will carry it, which is the reason the dedupe is safe at all.
    assert [q for q in got if q.split(":", 1)[1] == "address"]


def test_ukrainian_header_retrieves_the_right_property():
    got = shortlist(_p("Дата народження", {"date": 1.0}), CAT, CFG, subject="Person")
    assert "Person:birthDate" in got


def test_detector_evidence_pulls_in_type_compatible_candidates():
    got = shortlist(_p("Дані", {"phone_ua": 0.95}), CAT, CFG, subject="Person")
    assert any(CAT.prop(q).type_name == "phone" for q in got)


def test_neighbours_are_derived_from_the_ontology_not_hand_listed():
    from ftmap.vocab.shortlist import neighbours

    person = neighbours(CAT, "Person")
    # FtM itself says a Person stands at one end of a Membership and carries an
    # addressEntity, so both must be reachable without anyone writing them down.
    for expected in ("Person", "Membership", "Organization", "Address", "Occupancy"):
        assert expected in person, expected
    # It narrows: a Person row is not about a vessel.
    assert "Vessel" not in person
    assert len(person) < len(CAT.concrete_schemata())
    # And it generalizes to a subject nobody tuned it for.
    assert "Vessel" in neighbours(CAT, "Vessel")
    assert len(neighbours(CAT, "Vessel")) < len(person)


def test_a_person_subject_can_still_reach_another_entity_s_properties():
    """A deputies row carries a person AND the party they sit for. If the
    subject narrowed the scope to Person alone, the party column could never
    be bound and the multi-entity plan would be unreachable."""
    from ftmap.vocab.shortlist import _scope

    # The structural claim, independent of any similarity score: Organization
    # properties are IN SCOPE for a Person subject at all.
    assert "Organization:name" in set(_scope(CAT, "Person"))

    got = shortlist(_p("фракція", samples=["Слуга народу", "Голос"]),
                    CAT, CFG, subject="Person")
    # THE PROPERTY, ON A PARTY. «фракція» is written under `Organization:name`
    # and `Company` inherits it, so one slot per name may hold either
    # spelling; `valid_pairs` re-spells whichever survives onto the entity
    # that carries it. What must hold is that a party's `name` is on the
    # list at all for a Person subject.
    assert any(q.split(":", 1)[1] == "name"
               and CAT.is_descendant(q.split(":", 1)[0], "Organization")
               for q in got), got


def test_shortlist_is_bounded_and_deterministic():
    a = shortlist(_p("ПІБ"), CAT, CFG, subject="Person")
    b = shortlist(_p("ПІБ"), CAT, CFG, subject="Person")
    assert a == b
    assert len(a) <= CFG.shortlist_size


# --- Change 1: retrieval scoped to declared entity schemas, not neighbours(subject) ---
#
# The measured defect: `neighbours(subject)` widens the scoped ranking to
# every schema a Person could conceivably connect to — Organization,
# Membership, Address, Occupancy, 27 schemata for Person alone — but a
# binding can only ever attach to an entity the model actually DECLARED for
# this source. Most of that neighbourhood is not usable, and 78% of offered
# slots turned out to name a property no declared entity could hold.


def test_schemas_param_scopes_to_exactly_those_schemas_not_the_neighbourhood():
    """Organization is IN neighbours(Person) — a Person row can connect to an
    org — but a source that declared no Organization entity has nothing that
    could carry Organization:name. `schemas=` must not widen past what it is
    given, the way `subject=` deliberately does."""
    from ftmap.vocab.shortlist import _scope

    assert "Organization:name" in set(_scope(CAT, "Person"))  # old behaviour, unchanged
    assert "Organization:name" not in set(_scope(CAT, None, schemas=["Person"]))
    assert "Person:name" in set(_scope(CAT, None, schemas=["Person"]))


def test_schemas_param_retrieves_candidates_for_a_declared_schema():
    got = shortlist(_p("Дата народження", {"date": 1.0}), CAT, CFG, schemas=["Person"])
    assert "Person:birthDate" in got


def test_schemas_param_covers_every_schema_it_is_given():
    """Two entities declared (Person and Organization) means both schemata's
    properties are retrievable, exactly like the old subject-based
    neighbourhood did for this same case — but now because both were
    actually declared, not because Organization happens to sit in Person's
    ontology neighbourhood."""
    got = shortlist(_p("фракція", samples=["Слуга народу", "Голос"]),
                    CAT, CFG, schemas=["Person", "Organization"])
    assert any(q.split(":", 1)[1] == "name"
               and CAT.is_descendant(q.split(":", 1)[0], "Organization")
               for q in got), got


def test_schemas_param_still_unions_with_the_unrestricted_ranking():
    """A wrong or too-narrow declared-schema set must not cap the source
    either — the same reasoning `shortlist`'s docstring gives for `subject`
    applies unchanged to `schemas`: retrieval makes no decision, so a wrong
    one costs nothing."""
    got = shortlist(_p("Телефон", {"phone_ua": 0.95}), CAT, CFG, schemas=["Vessel"])
    assert any(CAT.prop(q).type_name == "phone" for q in got)


def test_schemas_param_is_bounded_and_deterministic():
    a = shortlist(_p("ПІБ"), CAT, CFG, schemas=["Person"])
    b = shortlist(_p("ПІБ"), CAT, CFG, schemas=["Person"])
    assert a == b
    assert len(a) <= CFG.shortlist_size


def test_one_property_does_not_take_twelve_of_the_twenty_four_slots():
    """`emailMentioned` is declared on `Analyzable` and inherited by twelve
    schemata, so retrieval returns twelve spellings of ONE property and the
    list has room for half as many real candidates as it looks like it has.

    Measured over the person-graph corpus on 2026-08-29: 88 of 169 shortlists
    spent 8 or more of their 24 slots on a single property name, 27 of them
    exactly 12. The land valuers' `employmentCity` column was offered twelve
    `*:emailMentioned` and neither `Address:city` nor `Address:region`; the
    court decisions' judge column was offered no name property at all.

    Deduplicating by NAME is safe because `valid_pairs` re-spells whatever
    survives onto the entity that will carry it.
    """
    profile = _p(header="email", detectors={"email": 1.0},
                 samples=["ivan@example.com", "olha@example.com"])
    out = shortlist(profile, CAT, CFG, subject="Person")
    names = [q.split(":", 1)[1] for q in out]
    assert len(names) == len(set(names)), f"a name is repeated: {out}"


def test_a_schema_declared_twice_does_not_halve_the_candidate_list():
    """A SOURCE MAY DECLARE TWO ENTITIES OF ONE SCHEMA, and the ship register
    declares three `Person` — an owner, a charterer, and the charterer's
    representative. `_scope` enumerated each declared schema's properties
    again, so `Person:position` and `Person:birthPlace` occupied three slots
    apiece of the scoped ranking, which is truncated to `shortlist_size`
    BEFORE the one-slot-per-name dedupe in `shortlist` removes the copies.
    The scoped side then contributed about eight real candidates where it
    should contribute twenty-four.

    Measured 2026-08-30 on `ua_ship_register_2026-07-01.xlsx#ДРСУ`, whose
    declared schemata are `Vessel, Person, LegalEntity, Person, Person`: the
    top of the ranking for `Модель судна` was `address, position, position,
    position, birthPlace, birthPlace, birthPlace`, and `Vessel:model`,
    `Vessel:type` and `Vessel:registrationPort` were nowhere in the list. Eight
    of the ten properties the gold standards name that retrieval never offered
    are `Vessel:*` on this one file.

    Declaring a schema twice says nothing about the properties it carries, so
    the two lists must be identical.
    """
    p = _p("Модель судна")
    once = shortlist(p, CAT, CFG, schemas=["Vessel", "Person", "LegalEntity"])
    thrice = shortlist(p, CAT, CFG,
                       schemas=["Vessel", "Person", "LegalEntity",
                                "Person", "Person"])
    assert once == thrice


def test_retrieval_reads_the_spellings_the_label_catalogue_already_carries():
    """TWO FILES HELD THE SAME KIND OF DATA AND ONLY THE SMALLER ONE WAS READ.
    `lexicon_uk.json` carries 124 observed header spellings over 35 properties
    and is what retrieval matched against; `labels_uk.json` carries 370 over
    144, is the hand-maintained catalogue the project notes call the missing `uk`
    locale, and was read only by the review app for its analyst-facing labels.
    123 qnames had spellings the matcher never saw — `вулиця` for
    `Address:street`, `тривалість дзвінка` for `Call:duration`, `широта` and
    `довгота` for the coordinates.

    Nothing is authored here. Every spelling was already in the repository,
    already hand-maintained, and already inside the policy `labels_uk.json`
    states for itself: header forms observed in the corpora, not synonyms
    invented to help the matcher score better.

    This is the principled form of the retrieval gap that FtM's Russian locale
    exposed and could not fix — see `2026-08-20-negative-results.md`, where the
    proxy locale bound `КодЄДРПОУ` to `Organization:okpoCode`, a property
    FollowTheMoney describes as a Russian industry classifier.
    """
    lex = load_lexicon()
    assert "вулиця" in lex.get("Address:street", [])
    assert "тривалість дзвінка" in lex.get("Call:duration", [])
    # ...and the spellings that were only in the smaller file are still there.
    assert lex.get("LegalEntity:country")

    got = shortlist(_p("вулиця"), CAT, CFG, schemas=["Address"])
    assert "Address:street" in got


# --- 2026-09-03: spellings resolve by inheritance, and the fuzz tail is cut ---


def test_an_inherited_spelling_reaches_the_schema_that_carries_the_property():
    """«назва» is written under `Thing:name`; a Contract carries `name` one
    rung down and used to score it 0.0 on the scoped side, so the answer
    reached the model only through the unrestricted ranking's slots."""
    from ftmap.vocab.shortlist import shortlist_scored

    scoped = shortlist_scored(_p("назва закупівлі"), CAT, CFG, schemas=["Contract"])
    assert scoped and scoped[0][0] == "Contract:name", scoped[:3]
    assert scoped[0][1] >= 1.0


def test_a_spelling_under_one_root_does_not_reach_an_unrelated_root():
    """FollowTheMoney declares `title` on Person (an honorific) and on
    Document (a heading). A spelling written for one is not evidence about
    the other; the locale carries both roots explicitly where they do mean
    the same thing (`Interval:publisher` beside `Thing:publisher`)."""
    from ftmap.vocab.shortlist import resolved_spellings

    lex = {"Person:title": ["звання"], "Thing:publisher": ["джерело"],
           "Interval:publisher": ["джерело"]}
    assert resolved_spellings(lex, CAT, "Document:title") == []
    assert resolved_spellings(lex, CAT, "Person:title") == ["звання"]
    assert resolved_spellings(lex, CAT, "Company:publisher") == ["джерело"]
    assert resolved_spellings(lex, CAT, "Sanction:publisher") == ["джерело"]


def test_a_character_level_near_miss_is_not_a_candidate():
    """«Опис» scored 0.40 against «рнокпп» (a taxNumber spelling) on
    character overlap alone, and under the old ranking that put a tax number
    on a description column's list. Below the floor the header says nothing,
    and with no detector on the values there is nothing to offer."""
    got = shortlist(_p("Опис"), CAT, CFG, schemas=["LegalEntity"])
    assert "LegalEntity:taxNumber" not in got
    # The real match is untouched: the header IS a spelling of description.
    assert any(q.endswith(":description") for q in got), got


def test_the_prime_and_the_modifier_letter_fold_to_an_apostrophe():
    from ftmap.vocab.shortlist import _fold

    assert _fold("Прізвище Ім′я По-батькові") == _fold("прізвище ім'я по-батькові")
    assert _fold("Імʼя") == _fold("Ім'я")


def test_snake_case_is_a_word_separator():
    from ftmap.vocab.shortlist import _fold

    assert _fold("country_codes") == "country codes"
    assert _fold("valid_until") == _fold("Valid Until")
    got = shortlist(_p("country_codes", {"country": 1.0}), CAT, CFG,
                    schemas=["Address"])
    assert got and got[0] == "Address:country", got


def test_an_organisation_column_with_a_mute_header_still_reaches_name():
    """«З ким укладено договір» carries no name spelling, and once the fuzz
    tail was cut its `name` candidate went with it (`work-x6`: the
    procurement plan's supplier entity lost its only column). The values
    say what the header does not: they are organisations, written as a
    register writes them, and the detector puts `name` back on the list."""
    from ftmap.profile.columns import profile_frame
    from ftmap.io.layout import build_frame
    from ftmap.io.tabular import Grid

    rows = [["З ким укладено договір"], ['ТОВ "РОМАШКА"'], ["ПрАТ «Київстар»"],
            ['ДП "УКРВОДШЛЯХ"'], ["Wind Rose LLC"]]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv",
                        "0" * 64, "sid/s", CFG)
    profile = profile_frame(frame, CFG)[0]
    assert profile.detectors.get("legal_name", 0) >= 0.5
    got = shortlist(profile, CAT, CFG, schemas=["LegalEntity"])
    assert any(q.split(":", 1)[1] == "name" for q in got), got


def test_camel_case_is_a_word_boundary_on_the_header_side_only():
    from ftmap.vocab.shortlist import _fold, _fold_header
    assert _fold_header("employmentCity") == "employment city"
    assert _fold_header("qualCertIssueDate") == "qual cert issue date"
    assert _fold_header("КодЄДРПОУ") == "кодєдрпоу"
    assert _fold_header("sourceID") == "sourceid" and _fold_header("ibcRUC") == "ibcruc"
    assert _fold_header("IMO") == "imo" and _fold_header("ПІБ") == "піб"
    assert _fold_header("node_id") == "node id"
    # A property name keeps its shape: the words are in its label already,
    # and folding the name re-ranked every shortlist in the corpus.
    assert _fold("retrievedAt") == "retrievedat"
