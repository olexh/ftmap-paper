# tests/test_groups.py
"""Prefix groups: the columns an export names as one thing's fields."""
from ftmap.plan.groups import group_of, prefix_groups, remainder
from ftmap.profile.columns import ColumnProfile


def _p(cid, header):
    return ColumnProfile(
        id=cid, index=int(cid[1:]), header=header, label=None, count=10,
        filled=10, fill_rate=1.0, distinct=10, distinct_ratio=1.0, min_len=3,
        max_len=30, shapes=[], detectors={}, samples=[])


def test_snake_upper_camel_and_dotted_headers_group_on_their_prefix():
    """Four conventions from the corpus: the enforcement register's
    `DEBTOR_*` and `ORG_*`, the corruption register's `CORRUPTIONER_*`, the
    land valuers' `qualCert*`, and a dotted export."""
    profiles = [_p("c0", "DEBTOR_NAME"), _p("c1", "DEBTOR_BIRTHDATE"), _p("c2", "DEBTOR_CODE"),
                _p("c3", "PUBLISHER"), _p("c4", "ORG_NAME"), _p("c5", "ORG_PHONE_NUM"),
                _p("c6", "qualCertNumber"), _p("c7", "qualCertIssueDate"),
                _p("c8", "qualCertIssuingInstitution"), _p("c9", "employmentCity"),
                _p("c10", "owner.name"), _p("c11", "owner.code")]
    groups = prefix_groups(profiles)
    assert groups == {"debtor": ["c0", "c1", "c2"], "org": ["c4", "c5"],
                      "qual cert": ["c6", "c7", "c8"], "owner": ["c10", "c11"]}
    assert group_of(groups, "c7") == "qual cert" and group_of(groups, "c3") is None
    assert remainder(profiles[8], "qual cert") == "issuing institution"
    assert remainder(profiles[5], "org") == "phone num"


def test_a_human_s_first_word_is_not_a_prefix():
    """«Дата початку дії» and «Дата закінчення дії» describe nothing
    together, and neither do two English headers with spaces."""
    profiles = [_p("c0", "Дата початку дії"), _p("c1", "Дата закінчення дії"),
                _p("c2", "Full name"), _p("c3", "Full address")]
    assert prefix_groups(profiles) == {}


def test_a_prefix_that_is_a_whole_header_is_not_a_group():
    """`party` beside `party_name` and `party_text`: the bare word is a
    member with nothing after the prefix, so the prefix shortens until every
    member keeps a field — here to nothing, so no group."""
    assert prefix_groups([_p("c0", "party"), _p("c1", "party_name")]) == {}
    # With two fields the group stands and the bare header stays outside it.
    groups = prefix_groups([_p("c0", "party_id"), _p("c1", "party_name"), _p("c2", "party_text")])
    assert groups == {"party": ["c0", "c1", "c2"]}


def test_the_prefix_is_the_longest_run_common_to_the_members():
    groups = prefix_groups([_p("c0", "CODEX_ARTICLES_LIST_CODEX_ART_NUMBER"),
                            _p("c1", "CODEX_ARTICLES_LIST_CODEX_ART_PART"),
                            _p("c2", "CODEX_NOTE")])
    assert groups == {"codex": ["c0", "c1", "c2"]}


def test_a_group_s_unbound_columns_bind_by_the_exact_spelling_of_their_field():
    """The corruption register's `CORRUPTIONER_LAST_NAME`, `_FIRST_NAME`,
    `_SURNAME` were the person's keys with no property. «last name» and
    «first name» are labels of one property each; «surname» spells none
    and stays for the model. A column keyed on by another entity is not
    touched; a role-keyed member is not the group's entity."""
    from ftmap.plan.groups import bind_group_members
    from ftmap.plan.roles import role_keyed
    from ftmap.vocab.catalogue import Catalogue
    from ftmap.vocab.shortlist import load_lexicon
    cat = Catalogue.load()
    profiles = [_p("c0", "CORRUPTIONER_LAST_NAME"), _p("c1", "CORRUPTIONER_FIRST_NAME"),
                _p("c2", "CORRUPTIONER_SURNAME"), _p("c3", "CORR_WORK_PLACE"),
                _p("c4", "CORR_WORK_POS")]
    entities = [{"key": "person", "schema": "Person", "keys": ["c0", "c1", "c2"]},
                {"key": "employer", "schema": "Organization", "keys": ["c3"]}]
    out = bind_group_members(entities, [], profiles, cat, load_lexicon(None), role_keyed)
    assert {(c["column"], c["prop"], c["entity"]) for c, _ in out} == {
        ("c0", "Person:lastName", "person"), ("c1", "Person:firstName", "person")}
    # The office's chief under the office's prefix: the group's entity is the
    # office (bound on `sti_name`), not the chief keyed on `sti_chief_name`.
    profiles = [_p("c6", "c_sti"), _p("c7", "sti_name"), _p("c8", "sti_chief_name"),
                _p("c9", "sti_phone")]
    entities = [{"key": "office", "schema": "PublicBody", "keys": ["c6"]},
                {"key": "chief", "schema": "Person", "keys": ["c8"]}]
    bindings = [{"column": "c7", "prop": "PublicBody:name", "entity": "office"}]
    out = bind_group_members(entities, bindings, profiles, cat, load_lexicon(None), role_keyed)
    assert {(c["column"], c["prop"], c["entity"]) for c, _ in out} == {
        ("c9", "PublicBody:phone", "office")}


def test_two_readings_of_one_thing_keyed_in_one_group_merge_on_the_shared_key():
    """The enforcement register's cold plan: a Person on `DEBTOR_NAME` +
    `DEBTOR_CODE`, another on `DEBTOR_NAME` + `DEBTOR_BIRTHDATE`. One
    debtor, keyed on the column both agree on; the other's bindings and
    edges follow. A LegalEntity beside a Person is not merged."""
    from ftmap.plan.groups import merge_group_twins
    from ftmap.plan.keys import key_identification
    from ftmap.plan.roles import role_keyed
    from dataclasses import replace
    profiles = [_p("c0", "DEBTOR_NAME"), _p("c1", "DEBTOR_BIRTHDATE"),
                replace(_p("c2", "DEBTOR_CODE"), fill_rate=0.08, filled=1), _p("c3", "PUBLISHER")]
    entities = [{"key": "a", "schema": "Person", "keys": ["c0", "c2"]},
                {"key": "b", "schema": "Person", "keys": ["c0", "c1"]},
                {"key": "issuer", "schema": "PublicBody", "keys": ["c3"]}]
    bindings = [{"column": "c2", "prop": "Person:idNumber", "entity": "a"},
                {"column": "c1", "prop": "Person:birthDate", "entity": "b"}]
    edges = [{"key": "d", "schema": "Debt", "source": "a", "target": "issuer"}]
    kept, notes = merge_group_twins(entities, bindings, edges, profiles, role_keyed,
                                    lambda e: key_identification(e, profiles))
    assert [(e["key"], e["keys"]) for e in kept] == [("b", ["c0"]), ("issuer", ["c3"])]
    assert {b["entity"] for b in bindings} == {"b"} and edges[0]["source"] == "b"
    assert "merged a into it" in notes[0][2]
    entities = [{"key": "a", "schema": "LegalEntity", "keys": ["c2"]},
                {"key": "b", "schema": "Person", "keys": ["c0"]}]
    kept, notes = merge_group_twins(entities, [], [], profiles, role_keyed,
                                    lambda e: key_identification(e, profiles))
    assert len(kept) == 2 and notes == []


def test_a_rule_s_entity_wins_a_merge_and_keeps_its_keys():
    """Боярка (`work-p7`): the address block keyed on street and number
    against the model's Address keyed on the street alone, in one
    `address*` group. The block wins, keys intact, and the model's
    bindings follow it."""
    from ftmap.plan.groups import merge_group_twins
    from ftmap.plan.keys import key_identification
    from ftmap.plan.roles import role_keyed
    profiles = [_p("c26", "addressPostName"), _p("c27", "addressThoroughfare"),
                _p("c28", "addressLocatorDesignator")]
    entities = [{"key": "c27", "schema": "Address", "keys": ["c27"]},
                {"key": "address_block", "schema": "Address", "keys": ["c27", "c28"]}]
    bindings = [{"column": "c26", "prop": "Address:city", "entity": "c27"}]
    kept, notes = merge_group_twins(entities, bindings, [], profiles, role_keyed,
                                    lambda e: key_identification(e, profiles),
                                    prefer={"address_block"})
    assert [(e["key"], e["keys"]) for e in kept] == [("address_block", ["c27", "c28"])]
    assert bindings[0]["entity"] == "address_block"
