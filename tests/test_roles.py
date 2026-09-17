# tests/test_roles.py
"""Relations read off the headers of party columns.

Every case is a header from `corpus-external/`, and the expected relation
is what that source's etalon asks for.
"""

from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.plan.propose import Plan
from ftmap.plan.roles import (declare_parties, derive_relations, group_roles,
                              load_roles, role_in, roles_in)
from ftmap.plan.validate import validate
from ftmap.profile.columns import ColumnProfile, profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()
ROLES = load_roles()


def _p(cid, header, label=None, group=None, detectors=None, distinct=1.0, fill=1.0):
    return ColumnProfile(
        id=cid, index=int(cid[1:]), header=header, label=label, count=10,
        filled=int(10 * fill), fill_rate=fill, distinct=10, distinct_ratio=distinct,
        min_len=3, max_len=30, shapes=[], detectors=detectors or {}, samples=[],
        group=group)


def test_the_lexicon_names_only_relations_the_ontology_declares():
    edges = {e.schema for e in CAT.edges()}
    for relation, sides in ROLES.items():
        if relation in edges:
            assert set(sides) <= {"source", "target"}, relation
        else:
            info = CAT.prop(relation)
            assert info is not None and info.type_name == "entity", relation
            assert set(sides) <= {"source", "target"}, relation


def test_a_role_word_is_read_from_the_header_or_its_label_row():
    assert role_in("Власники", "Owners", ROLES) == ("Ownership", "source", "власники")
    assert role_in("Назва юр. (фрахтувальник)", None, ROLES) == (
        "Vessel:operator", "target", "фрахтувальник")
    assert role_in("Експлуатант/ Орендар", "Operator/  Lessee", ROLES) == (
        "Vehicle:operator", "target", "експлуатант")
    assert role_in("sti_chief_name", None, ROLES) == ("Directorship", "source", "chief")
    assert role_in("Код суб'єкта управління", None, ROLES) == (
        "Ownership", "source", "суб'єкта управління")
    assert role_in("ПІБ", None, ROLES) is None
    assert role_in(None, None, ROLES) is None


def test_an_owner_column_owns_the_row_s_vessel():
    """The ship register: «ПІБ» and «Назва (юр)» are the owner columns and
    the sheet's Vessel is the row. Two owner columns, two Ownerships, each
    from one column to the one vessel — not a fan."""
    profiles = [_p("c0", "Реєстраційний №"), _p("c14", "ПІБ", "Судновласник"),
                _p("c15", "Назва (юр)", "Судновласник"),
                _p("c16", "Назва юр. (фрахтувальник)")]
    entities = [{"key": "vessel", "schema": "Vessel", "keys": ["c0"]},
                {"key": "c14", "schema": "Person", "keys": ["c14"]},
                {"key": "c15", "schema": "Company", "keys": ["c15"]},
                {"key": "c16", "schema": "Company", "keys": ["c16"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles,
                                                 "Vessel", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Ownership", "c14", "vessel"), ("Ownership", "c15", "vessel")]
    assert attachments == [{"entity": "vessel", "prop": "Vessel:operator",
                            "target": "c16"}]
    assert all("derived from" in n or "attached to" in n for _, n in notes)


def test_a_role_column_follows_the_entity_whose_header_it_shares_a_token_with():
    """`sti_chief_name` is the chief of the `sti_name` office, not of the
    debtor beside it; `chief_name` shares a token with nothing and is the
    row's own director."""
    profiles = [_p("c1", "tin_s"), _p("c2", "name"), _p("c5", "chief_name"),
                _p("c6", "c_sti"), _p("c7", "sti_name"), _p("c8", "sti_chief_name")]
    entities = [{"key": "debtor", "schema": "Company", "keys": ["c1", "c2"]},
                {"key": "chief", "schema": "Person", "keys": ["c5"]},
                {"key": "office", "schema": "PublicBody", "keys": ["c6", "c7"]},
                {"key": "office_chief", "schema": "Person", "keys": ["c8"]}]
    edges, _, notes = derive_relations(entities, [], [], profiles, "Company", CAT)
    got = {(e["schema"], e["source"], e["target"]) for e in edges}
    assert ("Directorship", "office_chief", "office") in got
    assert ("Directorship", "chief", "debtor") in got
    assert len(got) == 2


def test_a_counterpart_out_of_range_is_named_and_nothing_is_derived():
    """`work-x13`, the tax-debtor register: `chief_name` is the debtor's
    chief and the debtor is a LegalEntity, which cannot stand at the
    organisation end of a Directorship. Among the in-range candidates alone
    the tax office was lone and took the chief. The choice is made over
    every entity on the row and only then checked against the range. The
    tax office is a LegalEntity too, under a LegalEntity subject; what makes
    the debtor the row's entity is its key — 99 % distinct against the
    office's 749 codes over 21 529 rows."""
    profiles = [_p("c1", "tin_s", distinct=0.99), _p("c2", "name"),
                _p("c5", "chief_name"), _p("c6", "c_sti", distinct=0.03),
                _p("c7", "sti_name"), _p("c8", "sti_chief_name")]
    entities = [{"key": "c1", "schema": "LegalEntity", "keys": ["c1"]},
                {"key": "c7", "schema": "PublicBody", "keys": ["c6"]},
                {"key": "c5", "schema": "Person", "keys": ["c5"]},
                {"key": "c8", "schema": "Person", "keys": ["c8"]}]
    edges, _, notes = derive_relations(entities, [], [], profiles, "LegalEntity", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Directorship", "c8", "c7")]
    note = dict(notes)["c5"]
    assert "c1 (LegalEntity), cannot stand at the other end" in note


def test_an_ambiguous_counterpart_derives_nothing_and_says_so():
    profiles = [_p("c0", "Власник"), _p("c1", "Судно"), _p("c2", "Літак")]
    entities = [{"key": "owner", "schema": "Person", "keys": ["c0"]},
                {"key": "ship", "schema": "Vessel", "keys": ["c1"]},
                {"key": "plane", "schema": "Airplane", "keys": ["c2"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles,
                                                 None, CAT)
    assert edges == [] and attachments == []
    assert notes and "ambiguous" in notes[0][1]


def test_a_relation_the_model_already_declared_is_not_derived_twice():
    profiles = [_p("c0", "Власники"), _p("c2", "Реєстраційний знак")]
    entities = [{"key": "owner", "schema": "LegalEntity", "keys": ["c0"]},
                {"key": "plane", "schema": "Airplane", "keys": ["c2"]}]
    declared = [{"key": "own", "schema": "Ownership", "source": "owner",
                 "target": "plane"}]
    edges, _, _ = derive_relations(entities, declared, [], profiles, "Airplane", CAT)
    assert edges == []


def test_a_party_that_cannot_stand_at_the_role_s_end_is_refused_with_a_reason():
    """`Vessel:operator` ranges on LegalEntity; a Vessel named «Фрахтувальник»
    would be nonsense, and the note says why nothing was derived."""
    profiles = [_p("c0", "Фрахтувальник"), _p("c1", "Судно")]
    entities = [{"key": "v2", "schema": "Vessel", "keys": ["c0"]},
                {"key": "v", "schema": "Vessel", "keys": ["c1"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles,
                                                 "Vessel", CAT)
    assert edges == [] and attachments == []
    assert notes == []  # a Vessel is not a party; nothing was even tried


def test_the_derived_relation_reaches_the_validated_plan_with_a_decision():
    frame = build_frame(Grid(rows=[["Судно", "Власник"],
                                   ["Нептун", "Коваленко Іван"],
                                   ["Аврора", "Шевченко Ольга"]],
                             sheet="s", merges=[]), "/x.csv", "0" * 64,
                        "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    plan = Plan(subject="Vessel",
                entities=[{"key": "vessel", "schema": "Vessel", "keys": ["c0"]},
                          {"key": "owner", "schema": "Person", "keys": ["c1"]}],
                edges=[], shortlists={},
                bindings=[{"column": "c0", "prop": "Vessel:name",
                           "entity": "vessel", "why": "x"},
                          {"column": "c1", "prop": "Person:name",
                           "entity": "owner", "why": "y"}])
    v = validate(plan, profiles, CAT, CFG, frame)
    assert [(e["schema"], e["source"], e["target"]) for e in v.edges] == [
        ("Ownership", "owner", "vessel")]
    d = [d for d in v.decisions if d.entity == "owner_ownership"]
    assert d and d[0].verdict == "accepted" and "«Власник»" in d[0].reason


def test_a_role_word_on_a_bound_name_column_counts_when_the_key_is_a_code():
    """The lease register keys its holder on «Код за ЄДРПОУ» and names it in
    «Балансоутримувач»; ICIJ's provider is keyed on nothing and named in
    `service_provider`. The role word is on the column bound to the party,
    and that is where it is read."""
    profiles = [_p("c0", "№ з/п"), _p("c3", "Код за ЄДРПОУ"),
                _p("c4", "Балансоутримувач"), _p("c8", "Назва об'єкта")]
    entities = [{"key": "holder", "schema": "Organization", "keys": ["c3"]},
                {"key": "object", "schema": "RealEstate", "keys": ["c8"]}]
    bindings = [{"column": "c3", "prop": "Organization:registrationNumber",
                 "entity": "holder"},
                {"column": "c4", "prop": "Organization:name", "entity": "holder"},
                {"column": "c8", "prop": "RealEstate:name", "entity": "object"}]
    edges, _, _ = derive_relations(entities, [], [], profiles, "RealEstate",
                                   CAT, bindings=bindings)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Ownership", "holder", "object")]

    profiles = [_p("c0", "node_id"), _p("c1", "name"), _p("c14", "service_provider")]
    entities = [{"key": "company", "schema": "LegalEntity", "keys": ["c0"]},
                {"key": "provider", "schema": "Organization", "keys": []}]
    bindings = [{"column": "c1", "prop": "LegalEntity:name", "entity": "company"},
                {"column": "c14", "prop": "Organization:name", "entity": "provider"}]
    edges, _, _ = derive_relations(entities, [], [], profiles, "Company", CAT,
                                   bindings=bindings)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Representation", "provider", "company")]


def test_every_role_word_in_a_text_is_read_in_text_order():
    assert roles_in("Власник, фрактувальник", ROLES) == [
        ("Ownership", "source", "власник"),
        ("Vessel:operator", "target", "фрактувальник")]
    assert roles_in("Код суб'єкта управління", ROLES) == [
        ("Ownership", "source", "суб'єкта управління")]
    assert roles_in("Довжина, м", ROLES) == []


def test_a_group_naming_one_role_names_it_for_the_whole_band():
    profiles = [_p("c0", "№"), _p("c1", "ПІБ", group="Власник"),
                _p("c2", "Код ЄДРПОУ", group="Власник"), _p("c3", "Судно")]
    got = group_roles(profiles, ROLES)
    assert got == {"c1": (("Ownership", "source", "власник"), "«Власник» over c1–c2"),
                   "c2": (("Ownership", "source", "власник"), "«Власник» over c1–c2")}


def test_a_group_listing_two_roles_lists_them_in_column_order():
    """The ship register: the band is owner, owner, charterer, charterer,
    and only the third column says so itself. The first role holds until
    a column's own header names the second; the fourth inherits it."""
    g = "Власник, фрактувальник"
    profiles = [_p("c13", "IMO"), _p("c14", "ПІБ", group=g),
                _p("c15", "Назва (юр)", group=g),
                _p("c16", "Назва юр. (фрахтувальник)", group=g),
                _p("c17", "Прізвище Ім′я По-батькові", group=g),
                _p("c18", "Довжина, м", group="Характеристики")]
    got = {k: v[0] for k, v in group_roles(profiles, ROLES).items()}
    assert got == {"c14": ("Ownership", "source", "власник"),
                   "c15": ("Ownership", "source", "власник"),
                   "c17": ("Vessel:operator", "target", "фрактувальник")}


def test_the_column_s_own_header_wins_over_its_group():
    profiles = [_p("c0", "Директор", group="Власник"),
                _p("c1", "ПІБ", group="Власник")]
    got = {k: v[0] for k, v in group_roles(profiles, ROLES).items()}
    assert got == {"c1": ("Ownership", "source", "власник")}
    assert role_in("Директор", None, ROLES) == ("Directorship", "source", "директор")


def test_the_ship_register_s_owners_own_the_vessel_through_the_group_header():
    g = "Власник, фрактувальник"
    profiles = [_p("c0", "Реєстраційний №"), _p("c14", "ПІБ", group=g),
                _p("c15", "Назва (юр)", group=g),
                _p("c16", "Назва юр. (фрахтувальник)", group=g),
                _p("c17", "Прізвище Ім′я По-батькові (фрахтувальник)", group=g)]
    entities = [{"key": "c3", "schema": "Vessel", "keys": ["c0"]},
                {"key": "c14", "schema": "Person", "keys": ["c14"]},
                {"key": "c15", "schema": "LegalEntity", "keys": ["c15"]},
                {"key": "c16", "schema": "LegalEntity", "keys": ["c16"]},
                {"key": "c17", "schema": "Person", "keys": ["c17"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles,
                                                 "Vessel", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Ownership", "c14", "c3"), ("Ownership", "c15", "c3")]
    assert [a["target"] for a in attachments] == ["c16", "c17"]
    note = dict(notes)["c14_ownership"]
    assert "«Власник, фрактувальник» over c14–c17, the group header of «ПІБ» (c14)" in note


def test_a_free_column_headed_by_a_role_over_names_is_declared_a_party():
    """The tax-debtor register: `chief_name` and `sti_chief_name` hold
    personal names under a director's role word and the roster had no
    entity for them; ICIJ's `service_provider` holds firm names under a
    provider's. A flag column under a role word («Фрахтувальник судна
    юридична особа?») holds no name and declares nothing; a column the
    plan already binds is left to the plan."""
    profiles = [_p("c1", "tin_s"), _p("c2", "name", detectors={"legal_name": 0.6}),
                _p("c5", "chief_name", detectors={"proper_name": 1.0}),
                _p("c8", "sti_chief_name", detectors={"proper_name": 1.0}),
                _p("c14", "service_provider",
                   detectors={"proper_name": 0.91, "legal_name": 0.32}),
                _p("c15", "Фрахтувальник судна юридична особа?"),
                _p("c16", "Власники", detectors={"legal_name": 0.9})]
    entities = [{"key": "debtor", "schema": "LegalEntity", "keys": ["c1"]},
                {"key": "c5", "schema": "Note", "keys": []}]
    bindings = [{"column": "c2", "prop": "LegalEntity:name", "entity": "debtor"},
                {"column": "c16", "prop": "LegalEntity:name", "entity": "debtor"},
                # The model's own answer for c5 — the one this rule revisits.
                {"column": "c5", "prop": "unmapped", "entity": "none",
                 "why": "chief_name contains person names, but no candidate "
                        "links a person to these entities."}]
    new, binds, notes = declare_parties(entities, bindings, profiles, CAT)
    assert new == [{"key": "c5_2", "schema": "Person", "keys": ["c5"]},
                   {"key": "c8", "schema": "Person", "keys": ["c8"]},
                   {"key": "c14", "schema": "LegalEntity", "keys": ["c14"]}]
    assert [(b["column"], b["prop"], b["entity"]) for b in binds] == [
        ("c5", "Person:name", "c5_2"), ("c8", "Person:name", "c8"),
        ("c14", "LegalEntity:name", "c14")]
    assert [n[0] for n in notes] == ["c5", "c8", "c14"]
    assert "«chief» names the party of a Directorship" in notes[0][2]
    assert "100% of its values read as names" in notes[0][2]


def test_a_declared_party_reaches_the_plan_and_its_relation_is_derived():
    frame = build_frame(Grid(rows=[["Назва", "Директор", "Код ЄДРПОУ"],
                                   ["ТОВ Ромашка", "Коваленко Іван Петрович", "12345678"],
                                   ["ПрАТ Мрія", "Шевченко Ольга Іванівна", "87654321"],
                                   ["КП Дніпро", "Бондаренко Петро Ілліч", "11223344"]],
                             sheet="s", merges=[]), "/x.csv", "0" * 64,
                        "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    plan = Plan(subject="Organization",
                entities=[{"key": "org", "schema": "Organization", "keys": ["c2"]}],
                edges=[], shortlists={},
                bindings=[{"column": "c0", "prop": "Organization:name",
                           "entity": "org", "why": "x"},
                          {"column": "c2", "prop": "Organization:registrationNumber",
                           "entity": "org", "why": "x"}])
    v = validate(plan, profiles, CAT, CFG, frame)
    assert [(e["key"], e["schema"], e["keys"]) for e in v.entities] == [
        ("org", "Organization", ["c2"]), ("c1", "Person", ["c1"])]
    assert any(b["column"] == "c1" and b["prop"] == "Person:name"
               and b["entity"] == "c1" for b in v.bindings)
    assert [(e["schema"], e["source"], e["target"]) for e in v.edges] == [
        ("Directorship", "c1", "org")]
    d = [d for d in v.decisions if d.entity == "c1" and d.column == "c1"
         and "declared from" in d.reason]
    assert d and d[0].decided_by == "rule"


def test_a_person_s_party_faction_and_commission_are_three_memberships():
    """The ПКМУ-835 deputies layout (`work-c1`): the model declared all three
    Memberships from one person and the fan gate dropped them as a fan. Each
    is one column's claim — «партія», «фракція», «комісія» in its header —
    about one organisation, and is derived here after the gate."""
    profiles = [_p("c2", "familyName"), _p("c10", "partyName"),
                _p("c16", "factionName"), _p("c18", "commissionName"),
                _p("c28", "addressThoroughfare")]
    entities = [{"key": "deputy", "schema": "Person", "keys": ["c2"]},
                {"key": "party", "schema": "Organization", "keys": ["c10"]},
                {"key": "faction", "schema": "Organization", "keys": ["c16"]},
                {"key": "commission", "schema": "PublicBody", "keys": ["c18"]},
                {"key": "reception", "schema": "Address", "keys": ["c28"]}]
    edges, _, notes = derive_relations(entities, [], [], profiles, "Person", CAT)
    assert sorted((e["schema"], e["source"], e["target"]) for e in edges) == [
        ("Membership", "deputy", "commission"), ("Membership", "deputy", "faction"),
        ("Membership", "deputy", "party")]


def test_a_workplace_column_names_the_employer_of_the_row_s_person():
    """The NAZK register: `CORR_WORK_PLACE` beside three name columns."""
    profiles = [_p("c0", "CORRUPTIONER_LAST_NAME"), _p("c3", "CORR_WORK_PLACE"),
                _p("c10", "COURT_CASE_NUM")]
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]},
                {"key": "employer", "schema": "Organization", "keys": ["c3"]},
                {"key": "case", "schema": "CourtCase", "keys": ["c10"]}]
    edges, _, _ = derive_relations(entities, [], [], profiles, "Person", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Employment", "person", "employer")]


def test_a_post_column_is_never_declared_a_party():
    """`factionPost` holds «Член фракції ПП ВО «Свобода»»: the faction is
    named in it and the organisation detector fires, but the header says it
    is the member's post — a Membership's role, not a second organisation."""
    profiles = [_p("c2", "familyName"),
                _p("c16", "factionName", detectors={"org_name": 1.0}),
                _p("c17", "factionPost", detectors={"org_name": 1.0})]
    entities = [{"key": "deputy", "schema": "Person", "keys": ["c2"]}]
    new, binds, notes = declare_parties(entities, [], profiles, CAT)
    assert [e["keys"] for e in new] == [["c16"]]


def test_a_leftover_column_that_spells_a_relation_s_property_binds_to_it():
    """`factionName | factionPost`: once the relation's own word (faction)
    is taken out of the header, `post` spells Membership:role exactly."""
    from ftmap.plan.roles import bind_edge_properties
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c2", "familyName"), _p("c16", "factionName"),
                _p("c17", "factionPost"), _p("c18", "commissionName"),
                _p("c19", "commissionPost"), _p("c20", "areaId")]
    entities = [{"key": "deputy", "schema": "Person", "keys": ["c2"]},
                {"key": "faction", "schema": "Organization", "keys": ["c16"]},
                {"key": "commission", "schema": "Organization", "keys": ["c18"]}]
    edges = [{"key": "deputy_membership", "schema": "Membership",
              "source": "deputy", "target": "faction"},
             {"key": "deputy_membership_2", "schema": "Membership",
              "source": "deputy", "target": "commission"}]
    bindings = [{"column": "c2", "prop": "Person:lastName", "entity": "deputy"},
                {"column": "c16", "prop": "Organization:name", "entity": "faction"},
                {"column": "c18", "prop": "Organization:name", "entity": "commission"}]
    got = bind_edge_properties(edges, entities, bindings, profiles, CAT, load_lexicon(None))
    assert [(c["column"], c["prop"], c["entity"]) for c, _ in got] == [
        ("c17", "Membership:role", "deputy_membership"),
        ("c19", "Membership:role", "deputy_membership_2")]
    assert "_endpoint_tokens" not in edges[0]


def test_a_header_that_fits_two_relations_alike_binds_to_neither():
    from ftmap.plan.roles import bind_edge_properties
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c0", "ПІБ"), _p("c1", "Партія"), _p("c2", "Фракція"), _p("c3", "Посада")]
    entities = [{"key": "p", "schema": "Person", "keys": ["c0"]},
                {"key": "party", "schema": "Organization", "keys": ["c1"]},
                {"key": "faction", "schema": "Organization", "keys": ["c2"]}]
    edges = [{"key": "m1", "schema": "Membership", "source": "p", "target": "party"},
             {"key": "m2", "schema": "Membership", "source": "p", "target": "faction"}]
    got = bind_edge_properties(edges, entities, [], profiles, CAT, load_lexicon(None))
    assert got == []


def test_a_post_column_is_never_split_off_as_a_party():
    """Боярка (`work-c9`): `commissionName` and `commissionPost` both bound
    as the commission's name disagree on every row — the post is not the
    name — and the split made a PublicBody out of «член комісії»."""
    from ftmap.io.layout import build_frame
    from ftmap.io.tabular import Grid
    from ftmap.plan.roles import split_folded_parties
    from ftmap.profile.columns import profile_frame
    frame = build_frame(Grid(rows=[["commissionName", "commissionPost"],
                                   ["Комісія з питань бюджету", "член комісії"],
                                   ["Комісія з питань освіти", "голова"],
                                   ["Комісія з питань бюджету", "заступник голови"]],
                             sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    entities = [{"key": "commission", "schema": "PublicBody", "keys": ["c0"]}]
    bindings = [{"column": "c0", "prop": "PublicBody:name", "entity": "commission"},
                {"column": "c1", "prop": "PublicBody:name", "entity": "commission"}]
    assert split_folded_parties(entities, bindings, profiles, frame, CAT) == ([], [], [])


def test_a_supplier_s_award_is_of_the_one_contract_on_the_row():
    """The court's purchases: the buyer is the row's entity by bindings and
    cannot be awarded; the one Contract on the row is what the award is of."""
    profiles = [_p("c0", "Найменування"), _p("c8", "Опис"), _p("c15", "Учасник"), _p("c19", "Посилання")]
    entities = [{"key": "buyer", "schema": "PublicBody", "keys": ["c0"]},
                {"key": "contract", "schema": "Contract", "keys": ["c19"]},
                {"key": "supplier", "schema": "LegalEntity", "keys": ["c15"]}]
    edges, _, notes = derive_relations(entities, [], [], profiles, "PublicBody", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("ContractAward", "contract", "supplier")]
    assert "only Contract on the row" in dict(notes)["supplier_contractaward"]


def test_a_relation_thing_s_key_column_is_free_for_the_contract_s_own_facts():
    from ftmap.plan.roles import bind_edge_properties
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c1", "процедура"), _p("c5", "з ким укладено договір")]
    entities = [{"key": "c1", "schema": "ContractAward", "keys": ["c1"]},
                {"key": "contract_row", "schema": "Contract", "keys": []},
                {"key": "c5", "schema": "LegalEntity", "keys": ["c5"]}]
    got = bind_edge_properties([entities[1]], entities, [], profiles, CAT, load_lexicon(None))
    assert [(c["column"], c["prop"], c["entity"]) for c, _ in got] == [
        ("c1", "Contract:procedure", "contract_row")]


def test_a_counterpart_no_stronger_evidence_settles_is_the_nearest_by_column():
    """The enforcement register (`work-c15-cold`): `EMP_FULL_FIO` names an
    employee, and both `PUBLISHER` and `ORG_NAME` are bodies in range. No
    key header shares a token with the officer's, the row's entity is the
    debtor and cannot employ anyone, so the relation was not derived. An
    export lays a record out in reading order — a thing, then its fields,
    then the next thing — so among the candidates the range admits, the
    one nearest the role column is the counterpart, when one is strictly
    nearest. The debtor's code column (`DEBTOR_CODE`, keyed on by a second
    entity) is in the debtor's own prefix group and is never a counterpart
    of the debtor."""
    profiles = [_p("c0", "DEBTOR_NAME", distinct=0.95), _p("c1", "DEBTOR_BIRTHDATE"),
                _p("c2", "DEBTOR_CODE", fill=0.08), _p("c3", "PUBLISHER"), _p("c4", "ORG_NAME"),
                _p("c5", "ORG_PHONE_NUM"), _p("c6", "EMP_FULL_FIO"), _p("c7", "EMP_PHONE_NUM")]
    entities = [{"key": "debtor", "schema": "Person", "keys": ["c0"]},
                {"key": "code", "schema": "LegalEntity", "keys": ["c2"]},
                {"key": "issuer", "schema": "PublicBody", "keys": ["c3"]},
                {"key": "office", "schema": "PublicBody", "keys": ["c4"]},
                {"key": "officer", "schema": "Person", "keys": ["c6"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles, "Person", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Employment", "officer", "office")]
    assert any("nearest" in n for _, n in notes)


def test_a_debtor_column_states_a_debt_to_the_nearest_body_in_range():
    """«Боржник»/`DEBTOR_NAME` says the row's party owes; FtM's Debt is the
    relation, and its creditor is the party the file names for it — by a
    creditor word («стягувач», «кредитор») when the header carries one,
    else the nearest body in range that is not in a stated relation of its
    own. Two bodies at the same distance, or none, derive nothing: a Debt
    to nobody is not an edge this pipeline can write."""
    from ftmap.plan.roles import derive_debts
    profiles = [_p("c0", "DEBTOR_NAME", distinct=0.95), _p("c1", "DEBTOR_BIRTHDATE"),
                _p("c2", "DEBTOR_CODE", fill=0.08), _p("c3", "PUBLISHER"), _p("c4", "ORG_NAME"),
                _p("c5", "ORG_PHONE_NUM"), _p("c6", "EMP_FULL_FIO"), _p("c7", "EMP_PHONE_NUM")]
    # The code-keyed entity first, as the model listed it: the name-keyed
    # debtor still owes, because its key identifies rows.
    entities = [{"key": "code", "schema": "LegalEntity", "keys": ["c2"]},
                {"key": "debtor", "schema": "Person", "keys": ["c0"]},
                {"key": "issuer", "schema": "PublicBody", "keys": ["c3"]},
                {"key": "office", "schema": "PublicBody", "keys": ["c4"]},
                {"key": "officer", "schema": "Person", "keys": ["c6"]}]
    edges, notes = derive_debts(entities, [], profiles, "Person", CAT)
    assert [(e["schema"], e["source"], e["target"]) for e in edges] == [
        ("Debt", "debtor", "issuer")]
    # A creditor word wins over distance.
    profiles2 = [_p("c0", "Боржник"), _p("c1", "Орган"), _p("c2", "Стягувач")]
    entities2 = [{"key": "debtor", "schema": "LegalEntity", "keys": ["c0"]},
                 {"key": "body", "schema": "PublicBody", "keys": ["c1"]},
                 {"key": "creditor", "schema": "LegalEntity", "keys": ["c2"]}]
    edges, _ = derive_debts(entities2, [], profiles2, "LegalEntity", CAT)
    assert [(e["source"], e["target"]) for e in edges] == [("debtor", "creditor")]
    # No party in range: no edge, and a note that says why.
    edges, notes = derive_debts(entities2[:1], [], profiles2[:1], "LegalEntity", CAT)
    assert edges == [] and "no Debt derived" in notes[0][1]


def test_a_branch_column_names_the_parent_of_the_row_s_organisation():
    """«Назва філії» beside the enterprise's name: the branch is the
    holder of `Organization:parent`, and the row's organisation is the
    parent. The first role word read on the SOURCE side of an entity-typed
    property — the column names the holder, not the target."""
    profiles = [_p("c0", "Назва підприємства"), _p("c1", "Код ЄДРПОУ"),
                _p("c2", "Назва філії"), _p("c3", "Код філії")]
    entities = [{"key": "enterprise", "schema": "Company", "keys": ["c1"]},
                {"key": "branch", "schema": "Organization", "keys": ["c3"]}]
    bindings = [{"column": "c2", "prop": "Organization:name", "entity": "branch"}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles, "Company", CAT,
                                                 bindings=bindings)
    assert edges == []
    assert attachments == [{"entity": "branch", "prop": "Organization:parent",
                            "target": "enterprise"}]


def test_the_row_s_relation_is_keyed_on_the_record_number_the_row_carries():
    """The enforcement register: `VP_ORDERNUM` is distinct on every row,
    numeric, keys nothing and is bound to nothing, the debtor repeats, and
    the Debt is the row's one relation. The edge is keyed on it and the
    number is its recordId. A row ordinal is not a record number; a second
    such column derives nothing and says so."""
    from ftmap.plan.roles import key_relation_on_record
    rows = [["DEBTOR_NAME", "PUBLISHER", "VP_ORDERNUM", "N"]]
    for i in range(12):
        rows.append([f"Особа {i // 2}", f"Суд {i % 3}", str(70000000 + i * 7), str(i + 1)])
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    entities = [{"key": "debtor", "schema": "Person", "keys": ["c0"]},
                {"key": "issuer", "schema": "PublicBody", "keys": ["c1"]}]
    edges = [{"key": "debtor_debt", "schema": "Debt", "source": "debtor", "target": "issuer"}]
    cands, notes = key_relation_on_record(edges, entities, [], profiles, frame, "Person", CAT)
    assert edges[0]["keys"] == ["c2"]
    assert [(c["column"], c["prop"], c["entity"]) for c in cands] == [
        ("c2", "Debt:recordId", "debtor_debt")]
    assert "keyed on «VP_ORDERNUM»" in notes[0][1]
    # The debtor keyed per row: the number could be its own, nothing moves.
    edges2 = [{"key": "d", "schema": "Debt", "source": "debtor", "target": "issuer"}]
    entities2 = [{"key": "debtor", "schema": "Person", "keys": ["c2"]},
                 {"key": "issuer", "schema": "PublicBody", "keys": ["c1"]}]
    assert key_relation_on_record(edges2, entities2, [], profiles, frame, "Person", CAT) == ([], [])


def test_an_exact_relation_spelling_beats_the_model_s_binding_on_an_endpoint():
    """Боярка: `commissionPost` bound as `PublicBody:name` on the commission
    (`work-c15-cold`). Once «commission» is taken out the header spells
    Membership's `role`; the column is not a key and was bound on an
    endpoint of that Membership, so the rule offers the relation's binding
    in place of the model's. A bare leftover («status», «date») moves
    nothing."""
    from ftmap.plan.roles import bind_edge_properties
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c0", "votingIdentifier"), _p("c1", "commissionName"),
                _p("c2", "commissionPost"), _p("c3", "commissionStatus")]
    entities = [{"key": "deputy", "schema": "Person", "keys": ["c0"]},
                {"key": "commission", "schema": "PublicBody", "keys": ["c1"]}]
    edges = [{"key": "seat", "schema": "Membership", "source": "deputy",
              "target": "commission"}]
    bindings = [{"column": "c1", "prop": "PublicBody:name", "entity": "commission"},
                {"column": "c2", "prop": "PublicBody:name", "entity": "commission"},
                {"column": "c3", "prop": "PublicBody:status", "entity": "commission"}]
    out = bind_edge_properties(edges, entities, bindings, profiles, CAT, load_lexicon(None))
    got = {(c["column"], c["prop"], c.get("replaces")) for c, _ in out}
    assert got == {("c2", "Membership:role", "PublicBody:name")}


def test_a_phone_column_of_packs_binds_on_the_canonicalizer_s_rate_not_the_prompt_s():
    """`ORG_PHONE_NUM`: 0.44 to the single-number detector the prompt shows,
    0.95 to the canonicalizer with the office's second line taken. The
    binding reads the second, off the prompt; see `_pack_rates`."""
    from ftmap.plan.roles import bind_by_detector
    rows = [["ORG_NAME", "ORG_PHONE_NUM"]]
    for i in range(10):
        rows.append([f"Відділ {i}", f"(04142) 3-08-3{i}, 3-00-0{i}" if i < 9 else "0442345678"])
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    assert profiles[1].detectors.get("phone_ua", 0) < 0.8
    entities = [{"key": "office", "schema": "PublicBody", "keys": ["c0"]}]
    out = bind_by_detector(entities, [], profiles, "PublicBody", CAT, frame=frame)
    assert [(c["column"], c["prop"], c["entity"]) for c, _ in out] == [
        ("c1", "PublicBody:phone", "office")]
    assert bind_by_detector(entities, [], profiles, "PublicBody", CAT) == []


def test_a_binding_on_an_endpoint_s_prefix_twin_yields_to_the_exact_spelling_too():
    """The enforcement register: the Debt runs from the name-keyed debtor,
    `VD_CAT` is bound on the code-keyed LegalEntity the model declared
    beside it (`work-p3`). By the file's own naming the two are one
    debtor, so the binding sits on the endpoint and the category moves to
    the Debt's description."""
    from ftmap.plan.roles import bind_edge_properties
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c0", "DEBTOR_NAME"), _p("c2", "DEBTOR_CODE", fill=0.08),
                _p("c3", "PUBLISHER"), _p("c10", "VD_CAT", distinct=0.01)]
    entities = [{"key": "code", "schema": "LegalEntity", "keys": ["c2"]},
                {"key": "debtor", "schema": "Person", "keys": ["c0"]},
                {"key": "issuer", "schema": "PublicBody", "keys": ["c3"]}]
    edges = [{"key": "debtor_debt", "schema": "Debt", "source": "debtor", "target": "issuer"}]
    bindings = [{"column": "c10", "prop": "LegalEntity:description", "entity": "code"}]
    out = bind_edge_properties(edges, entities, bindings, profiles, CAT, load_lexicon(None))
    assert [(c["column"], c["prop"], c.get("replaces")) for c, _ in out] == [
        ("c10", "Debt:description", "LegalEntity:description")]


# --------------------------------------------------------------------------
# 2026-09-05: a word two relations name, a parent of the holder's kind, and
# a thing named by the one column that identifies it
# --------------------------------------------------------------------------

def test_a_word_two_relations_name_is_read_in_the_lexicon_s_order():
    """«підрозділ» is the branch whose parent is the row's organisation AND
    the unit the row's person belongs to. `roles_in` keeps both, the file's
    order first; `role_in` still answers one, the first."""
    both = roles_in("Підрозділ", ROLES)
    assert [r[0] for r in both] == ["Organization:parent", "Membership"]
    assert role_in("Підрозділ", None, ROLES) == ("Organization:parent", "source", "підрозділ")
    from ftmap.plan.roles import roles_for
    assert [r[0] for r in roles_for("Підрозділ", None, ROLES)] == [
        "Organization:parent", "Membership"]
    # A longer spelling still wins over a shorter one it contains.
    assert role_in("Код суб'єкта управління", None, ROLES)[0] == "Ownership"


def test_a_unit_column_beside_a_person_is_the_person_s_membership_not_the_person_as_parent():
    """The roster (`79 МСП.csv`): a Person keyed on the name and an
    Organization keyed on «Підрозділ». The parent reading fails — an
    organisation's parent is an organisation, whatever range the ontology
    declares — and the membership reading is the one the row can say."""
    profiles = [_p("c1", "ПІБ (російською)", detectors={"proper_name": 0.98}),
                _p("c5", "Підрозділ", distinct=0.15)]
    entities = [{"key": "c1", "schema": "Person", "keys": ["c1"]},
                {"key": "c5", "schema": "Organization", "keys": ["c5"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles, "Person", CAT)
    assert attachments == []
    assert edges == [{"key": "c5_membership", "schema": "Membership",
                      "source": "c1", "target": "c5"}]
    assert any("derived from «Підрозділ»" in n for _k, n in notes)


def test_a_subunit_column_beside_an_organisation_is_still_its_branch():
    """The tax-debtor register's shape: the parent reading comes first in the
    lexicon and the row's organisation can be a parent, so nothing changes."""
    profiles = [_p("c0", "Назва підприємства"), _p("c1", "Код ЄДРПОУ"),
                _p("c2", "Відокремлений підрозділ")]
    entities = [{"key": "enterprise", "schema": "Company", "keys": ["c1"]},
                {"key": "sub", "schema": "Organization", "keys": ["c2"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles, "Company", CAT)
    assert edges == []
    assert attachments == [{"entity": "sub", "prop": "Organization:parent",
                            "target": "enterprise"}]


def test_a_thing_keyed_on_a_column_of_words_and_bound_to_nothing_is_named_by_it():
    from ftmap.plan.roles import name_from_key
    profiles = [_p("c5", "ФИО"), _p("c9", "Рота"), _p("c15", "Личный номер"),
                _p("c1", "Должность")]
    entities = [{"key": "c5", "schema": "Person", "keys": ["c5"]},
                {"key": "c9", "schema": "Organization", "keys": ["c9"]},
                {"key": "c15", "schema": "Identification", "keys": ["c15"]},
                {"key": "c1", "schema": "Position", "keys": ["c1"]}]
    bindings = [{"column": "c5", "prop": "Person:name", "entity": "c5"},
                {"column": "c1", "prop": "Person:position", "entity": "c5"}]
    text = {"c9": ["1 мсв", "2 мсв", "3 мсв"], "c15": ["АБ-123456", "ВГ-654321"],
            "c1": ["стрелок", "командир"], "c5": ["Коваленко Іван"]}
    got = name_from_key(entities, bindings, profiles, CAT, lambda c: text[c],
                        declined={"c9", "c15"})
    by_col = {c["column"]: (c["prop"], c["entity"]) for c, _n in got}
    # The organisation keyed on the unit column: named by it.
    assert by_col["c9"] == ("Organization:name", "c9")
    # The post: named by the column the person's `position` string is on.
    assert by_col["c1"] == ("Position:name", "c1")
    assert any("post's name said of the post" in n for _c, n in got)
    # The Identification is neither a party nor a thing named by its key, and
    # the Person is already bound; neither is touched.
    assert "c15" not in by_col and "c5" not in by_col


def test_a_thing_keyed_on_a_column_of_codes_is_not_named_by_it():
    from ftmap.plan.roles import name_from_key, text_share
    profiles = [_p("c0", "ПІБ"), _p("c1", "Код ЄДРПОУ")]
    entities = [{"key": "p", "schema": "Person", "keys": ["c0"]},
                {"key": "org", "schema": "Organization", "keys": ["c1"]}]
    bindings = [{"column": "c0", "prop": "Person:name", "entity": "p"}]
    for codes in (["12345678", "87654321"], ["Y6DA69700A0000199", "XW8ZZZ61ZDG000123"],
                  ["АБ-123456", "ВГ-654321"], ["АА1234ВВ", "КА5678ІВ"]):
        text = {"c1": codes, "c0": ["Коваленко Іван"]}
        assert text_share(codes) < 0.9
        assert name_from_key(entities, bindings, profiles, CAT, lambda c: text[c],
                             declined={"c1"}) == []
    assert text_share(["1 мсв", "2 мср", "ТОВ Альфа", "в/ч 11111"]) == 0.75


def test_a_party_keyed_on_a_post_column_is_not_named_by_it():
    """`factionPost` holds «Член фракції …»: words, and a role word in the
    header — and a party keyed on it by mistake is not named by a post."""
    from ftmap.plan.roles import name_from_key
    profiles = [_p("c0", "ПІБ"), _p("c1", "factionPost")]
    entities = [{"key": "p", "schema": "Person", "keys": ["c0"]},
                {"key": "org", "schema": "Organization", "keys": ["c1"]}]
    bindings = [{"column": "c0", "prop": "Person:name", "entity": "p"}]
    text = {"c1": ["член фракції", "голова"], "c0": ["Коваленко Іван"]}
    assert name_from_key(entities, bindings, profiles, CAT, lambda c: text[c],
                         declined={"c1"}) == []


def test_a_column_the_model_never_declined_or_whose_binding_was_refused_names_nothing():
    """The rule revisits the binding call's `unmapped` and nothing else: a
    binding the value check refused is a wrong reading, not an absent one."""
    from ftmap.plan.roles import name_from_key
    profiles = [_p("c0", "ПІБ", detectors={"proper_name": 0.9}), _p("c1", "Рота")]
    entities = [{"key": "p", "schema": "Person", "keys": ["c0"]},
                {"key": "org", "schema": "Organization", "keys": ["c1"]}]
    text = {"c1": ["1 мсв"], "c0": ["Коваленко Іван"]}
    assert name_from_key(entities, [], profiles, CAT, lambda c: text[c]) == []
    assert name_from_key(entities, [], profiles, CAT, lambda c: text[c],
                         declined={"c1"}, rejected={"c1"}) == []
    assert name_from_key(entities, [], profiles, CAT, lambda c: text[c],
                         declined={"c0"}, rejected={"c1"}) != []


def test_a_parent_is_never_chosen_by_layout():
    """The establishment table: the unit, the company and the section each
    under a level word, beside a person who cannot be anyone's parent.
    `nearest_in_range` would make the company the unit's parent and the
    unit the company's — a cycle. A parent is the row's organisation or
    nothing; the tree reads the rest from the values."""
    profiles = [_p("c5", "ФИО", detectors={"proper_name": 0.9}),
                _p("c11", "Подразделение", distinct=0.0), _p("c12", "войсковая часть", distinct=0.0),
                _p("c31", "Отделение", distinct=0.17)]
    entities = [{"key": "c5", "schema": "Person", "keys": ["c5"]},
                {"key": "c11", "schema": "Organization", "keys": ["c11"]},
                {"key": "c12", "schema": "Organization", "keys": ["c12"]},
                {"key": "c31", "schema": "Organization", "keys": ["c31"]}]
    edges, attachments, notes = derive_relations(entities, [], [], profiles, "Person", CAT)
    assert not any(a["prop"].endswith(":parent") for a in attachments)
    # The membership reading stands in for each: the person is in the unit.
    assert {(e["schema"], e["source"], e["target"]) for e in edges} == {
        ("Membership", "c5", "c11"), ("Membership", "c5", "c12"), ("Membership", "c5", "c31")}


def test_a_party_s_name_column_taken_by_a_relation_is_given_back_to_the_party():
    """The roster, second run: the person keyed on «ПІБ» and «ПІБ» bound to
    the Occupancy's `namesMentioned` — the person nameless, the link
    carrying the name. A party's name belongs to the party."""
    from ftmap.plan.roles import name_from_key
    profiles = [_p("c1", "ПІБ (російською)", detectors={"proper_name": 0.98}),
                _p("c6", "Посада")]
    entities = [{"key": "c1", "schema": "Person", "keys": ["c1"]},
                {"key": "c6", "schema": "Position", "keys": ["c6"]}]
    bindings = [{"column": "c1", "prop": "Occupancy:namesMentioned", "entity": "c1_edge"},
                {"column": "c6", "prop": "Position:name", "entity": "c6"}]
    text = {"c1": ["Коваленко Іван", "Шевченко Ольга"], "c6": ["стрелок"]}
    got = name_from_key(entities, bindings, profiles, CAT, lambda c: text[c],
                        relations={"c1_edge"})
    assert [(c["column"], c["prop"], c["entity"]) for c, _n in got] == [
        ("c1", "Person:name", "c1")]
    assert got[0][0]["replaces"] == [bindings[0]]
    assert "never to the link" in got[0][1]
    # Bound to a party — not a relation — the column is settled and left alone.
    other = [{"column": "c1", "prop": "Person:alias", "entity": "c1"}]
    assert name_from_key(entities, other, profiles, CAT, lambda c: text[c],
                         relations={"c1_edge"}) == []


def test_a_typed_column_taken_by_a_relation_s_string_is_the_party_s():
    """The roster: «Номер телефону», eight of ten values phones, bound to the
    Occupancy's `constituency`. A party's phone belongs to the party."""
    from ftmap.plan.roles import bind_by_detector
    rows = [["ПІБ", "Номер телефону", "Посада"]]
    for i in range(10):
        rows.append([f"Особа {i}", f"+38050123456{i}", "стрілець"])
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    entities = [{"key": "c0", "schema": "Person", "keys": ["c0"]},
                {"key": "c2", "schema": "Position", "keys": ["c2"]}]
    taken = [{"column": "c1", "prop": "Occupancy:constituency", "entity": "c0_edge"}]
    out = bind_by_detector(entities, taken, profiles, "Person", CAT, frame=frame,
                           relations={"c0_edge"})
    assert [(c["column"], c["prop"], c["entity"]) for c, _ in out] == [
        ("c1", "Person:phone", "c0")]
    assert out[0][0]["replaces"] == taken
    # Bound to a party — or to a relation's own phone — it is settled.
    assert bind_by_detector(entities, [{"column": "c1", "prop": "Person:phone",
                                        "entity": "c0"}], profiles, "Person", CAT,
                            frame=frame, relations={"c0_edge"}) == []


def test_a_person_keyed_on_a_column_of_codes_the_model_declined_is_identified_by_it():
    from ftmap.plan.roles import id_from_key
    profiles = [_p("c0", "telegramId", distinct=1.0), _p("c1", "firstName")]
    entities = [{"key": "c0", "schema": "Person", "keys": ["c0"]}]
    bindings = [{"column": "c1", "prop": "Person:firstName", "entity": "c0"}]
    text = {"c0": ["1234567", "7654321", "1122334"], "c1": ["Іван"]}
    got = id_from_key(entities, bindings, profiles, CAT, lambda c: text[c],
                      {"c0"}, set(), 0.5)
    assert [(c["column"], c["prop"], c["entity"]) for c, _n in got] == [
        ("c0", "Person:idNumber", "c0")]
    # Not declined, refused, words, or an organisation: nothing.
    assert id_from_key(entities, bindings, profiles, CAT, lambda c: text[c], set(), set(), 0.5) == []
    assert id_from_key(entities, bindings, profiles, CAT, lambda c: text[c], {"c0"}, {"c0"}, 0.5) == []
    words = {"c0": ["Коваленко Іван"], "c1": ["Іван"]}
    assert id_from_key(entities, bindings, profiles, CAT, lambda c: words[c], {"c0"}, set(), 0.5) == []
    org = [{"key": "c0", "schema": "Organization", "keys": ["c0"]}]
    assert id_from_key(org, [], profiles, CAT, lambda c: text[c], {"c0"}, set(), 0.5) == []


def test_a_column_that_varies_within_the_key_is_the_row_s():
    """«Местонахождение», 44 settlements, bound as the one unit's address;
    the regiment's name column, 3 584 names, bound as the name of one of
    five units — under the unit most rows sit in it takes thousands."""
    from ftmap.plan.roles import rehome_varying
    from ftmap.plan.validate import _values_under_modal_key
    rows = [["ФИО", "войсковая часть", "Местонахождение", "Телефон части"]]
    for i in range(12):
        rows.append([f"Особа {i}", "в/ч 11111", f"Село {i % 6}", "0441234567" if i % 2 else "0447654321"])
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    entities = [{"key": "person", "schema": "Person", "keys": ["c0"]},
                {"key": "unit", "schema": "Organization", "keys": ["c1"]}]
    bindings = [{"column": "c0", "prop": "Person:name", "entity": "person"},
                {"column": "c1", "prop": "Organization:name", "entity": "unit"},
                {"column": "c2", "prop": "Organization:address", "entity": "unit"},
                {"column": "c3", "prop": "Organization:phone", "entity": "unit"}]
    got = rehome_varying(entities, bindings, profiles, "Person", CAT, 2,
                         lambda c, e: _values_under_modal_key(frame, c, e))
    assert [(o["column"], n["prop"], n["entity"]) for o, n, _r in got] == [
        ("c2", "Person:address", "person")]
    # Two phone numbers on one unit are the unit's; the name is its key.
    assert not any(o["column"] in ("c1", "c3") for o, _n, _r in got)


def test_a_phone_column_bound_as_a_document_s_mention_is_the_party_s_phone():
    """The forensic contacts sheet: a Note keyed on the name and the number
    block, the block its `phoneMentioned`. A column of phones is the
    contact's phones, not what a note mentions."""
    from ftmap.plan.roles import bind_by_detector
    rows = [["Имя", "Абоненты"]]
    for i in range(10):
        # Mostly one labelled number per cell, as the sheet writes it; the
        # single-number detector the profile carries reads those.
        rows.append([f"Особа {i}", f"Mobile-: +38050123456{i}\nHome-: 044234567{i}"
                     if i < 2 else f"Mobile-: +38050123456{i}"])
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.xlsx", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    entities = [{"key": "c0", "schema": "Person", "keys": ["c0"]},
                {"key": "note", "schema": "Note", "keys": ["c0", "c1"]}]
    taken = [{"column": "c1", "prop": "Note:phoneMentioned", "entity": "note"}]
    out = bind_by_detector(entities, taken, profiles, "Person", CAT, frame=frame)
    assert [(c["column"], c["prop"], c["entity"]) for c, _ in out] == [
        ("c1", "Person:phone", "c0")]
    assert out[0][0]["replaces"] == taken


def test_a_header_that_spells_one_party_s_property_beats_a_binding_on_the_relation():
    """The roster, run cold: «Дата народження» on the Occupancy's `date`,
    «Звання» on its `endDate`. The header spells the person's own property;
    a bare word («дата») would spell nothing."""
    from ftmap.plan.roles import spelled_property_beats_relation
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c1", "ПІБ"), _p("c2", "Дата народження"), _p("c3", "Звання"),
                _p("c6", "Посада"), _p("c9", "Дата")]
    entities = [{"key": "c1", "schema": "Person", "keys": ["c1"]},
                {"key": "c6", "schema": "Position", "keys": ["c6"]}]
    taken = [{"column": "c2", "prop": "Occupancy:date", "entity": "c1_edge"},
             {"column": "c3", "prop": "Occupancy:endDate", "entity": "c1_edge"},
             {"column": "c9", "prop": "Occupancy:date", "entity": "c1_edge"},
             {"column": "c6", "prop": "Position:name", "entity": "c6"}]
    got = spelled_property_beats_relation(entities, taken, profiles, CAT,
                                          load_lexicon(None), {"c1_edge"})
    assert sorted((c["column"], c["prop"], c["entity"]) for c, _n in got) == [
        ("c2", "Person:birthDate", "c1"), ("c3", "Person:title", "c1")]
    by_col = {c["column"]: c["replaces"] for c, _n in got}
    assert by_col["c2"] == [taken[0]] and by_col["c3"] == [taken[1]]
    # The model's binding the value check refused still counts as the column
    # having been put on the link; nothing survives to replace.
    got = spelled_property_beats_relation(entities, [], profiles, CAT,
                                          load_lexicon(None), {"c1_edge"},
                                          attempted=taken[:2])
    assert sorted(c["column"] for c, _n in got) == ["c2", "c3"]
    assert all(c["replaces"] == [] for c, _n in got)
    # Two persons on the row: the header cannot say whose birth date.
    two = [*entities, {"key": "c4", "schema": "Person", "keys": ["c4"]}]
    assert spelled_property_beats_relation(two, taken, profiles, CAT,
                                           load_lexicon(None), {"c1_edge"}) == []


def test_a_column_varying_within_a_record_s_key_stays_where_it_is():
    """The enforcement register: the row's entity by the effective subject
    is the order Document, keyed on its number; the debtor's name, which
    varies within the debtor's redacted code, is not the order's."""
    from ftmap.plan.roles import rehome_varying
    profiles = [_p("c0", "DEBTOR_NAME"), _p("c2", "DEBTOR_CODE", distinct=0.2), _p("c9", "VP_ORDERNUM")]
    entities = [{"key": "debtor", "schema": "LegalEntity", "keys": ["c2"]},
                {"key": "order", "schema": "Document", "keys": ["c9"]}]
    bindings = [{"column": "c0", "prop": "LegalEntity:name", "entity": "debtor"},
                {"column": "c9", "prop": "Document:recordId", "entity": "order"}]
    assert rehome_varying(entities, bindings, profiles, "Document", CAT, 2,
                          lambda c, e: 9.0) == []


def test_a_level_word_declares_the_organisation_its_column_names_and_a_twin_header_is_declined():
    """The regiment: «Подр-е», «До роты/взвода», «До взвода/отд» — unit
    designations no name detector accepts — and a lower-cased «до
    взвода/отд» thirty-nine columns later, a working copy."""
    profiles = [_p("c0", "Ф.И.О.", detectors={"proper_name": 0.9}), _p("c1", "Подр-е"),
                _p("c2", "До роты/взвода"), _p("c3", "До взвода/отд"), _p("c4", "до взвода/отд"),
                _p("c5", "ВУС")]
    entities = [{"key": "p", "schema": "Person", "keys": ["c0"]}]
    text = {"c0": ["Коваленко Іван"], "c1": ["1 тб", "2 тб"], "c2": ["1 рота", "2 рота"],
            "c3": ["1 взвод", "2 взвод"], "c4": ["1 взвод"], "c5": ["1234567"]}
    ents, binds, notes = declare_parties(entities, [], profiles, CAT, values=lambda c: text[c])
    assert [(e["key"], e["schema"], e["keys"]) for e in ents] == [
        ("c1", "Organization", ["c1"]), ("c2", "Organization", ["c2"]), ("c3", "Organization", ["c3"])]
    assert all(b["prop"] == "Organization:name" for b in binds)
    assert any(col == "c4" and note.startswith("declined") and "written twice" in note
               for col, _k, note in notes)
    # A refused column — the sheet's machinery — declares nothing; and with
    # c3 refused too, c4 would be the first «до взвода/отд» and is declared.
    ents2, _b, _n = declare_parties(entities, [], profiles, CAT, values=lambda c: text[c],
                                    skip={"c1", "c2"})
    assert [e["key"] for e in ents2] == ["c3"]
    ents3, _b, _n = declare_parties(entities, [], profiles, CAT, values=lambda c: text[c],
                                    skip={"c1", "c2", "c3"})
    assert [e["key"] for e in ents3] == ["c4"]


def test_a_city_spelled_header_declares_an_address_and_a_code_column_does_not():
    from ftmap.plan.roles import declare_places
    from ftmap.vocab.shortlist import load_lexicon
    profiles = [_p("c0", "ПІБ"), _p("c1", "Населенный пункт"), _p("c2", "Місто"), _p("c3", "Код")]
    entities = [{"key": "p", "schema": "Person", "keys": ["c0"]}]
    bindings = [{"column": "c0", "prop": "Person:name", "entity": "p"},
                {"column": "c2", "prop": "Person:address", "entity": "p"}]
    text = {"c1": ["Київ", "Львів"], "c2": ["Одеса"], "c3": ["12345", "67890"]}
    ents, binds, notes = declare_places(entities, bindings, profiles, CAT, load_lexicon(None),
                                        lambda c: text[c])
    # c1 is unbound and spells the city; c2 is bound (and not declined) and stays.
    assert [(e["key"], e["schema"], e["keys"]) for e in ents] == [("place_c1", "Address", ["c1"])]
    assert binds == [{"column": "c1", "prop": "Address:city", "entity": "place_c1",
                      "why": "the header «Населенный пункт» spells Address:city exactly"}]
    # Declined by the model, a bound column is revisited; a refused one never.
    ents2, _b, _n = declare_places(entities, bindings, profiles, CAT, load_lexicon(None),
                                   lambda c: text[c], declined={"c2"})
    assert [e["keys"] for e in ents2] == [["c1"], ["c2"]]
    assert declare_places(entities, bindings, profiles, CAT, load_lexicon(None),
                          lambda c: text[c], skip={"c1"})[0] == []
