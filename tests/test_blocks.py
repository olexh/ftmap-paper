# tests/test_blocks.py
"""The ПКМУ-835 address block, read as the Address it spells."""

from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.plan.blocks import ADDRESS_KEY, declare_address_block
from ftmap.plan.propose import Plan
from ftmap.plan.validate import validate
from ftmap.profile.columns import ColumnProfile, profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()


def _p(cid, header, filled=10):
    return ColumnProfile(
        id=cid, index=int(cid[1:]), header=header, label=None, count=10,
        filled=filled, fill_rate=filled / 10, distinct=5, distinct_ratio=0.5,
        min_len=3, max_len=30, shapes=[], detectors={}, samples=[])


def test_the_address_fields_of_the_layout_are_one_address():
    profiles = [_p("c2", "familyName"), _p("c22", "addressPostCode", 0),
                _p("c23", "addressAdminUnitL1"), _p("c24", "addressAdminUnitL2"),
                _p("c25", "addressAdminUnitL3"), _p("c27", "addressPostName"),
                _p("c28", "addressThoroughfare"), _p("c29", "addressLocatorDesignator"),
                _p("c31", "addressLocatorName"), _p("c32", "openingHours")]
    entities, bindings, notes = declare_address_block([], profiles)
    assert entities == [{"key": ADDRESS_KEY, "schema": "Address", "keys": ["c28", "c29"]}]
    assert {(b["column"], b["prop"]) for b in bindings} == {
        ("c23", "Address:country"), ("c24", "Address:region"),
        ("c27", "Address:city"), ("c28", "Address:street"),
        ("c29", "Address:full"), ("c31", "Address:summary")}
    declined = [n for c, k, n in notes if n.startswith("declined")]
    assert len(declined) == 1 and "c25" in declined[0]


def test_two_address_columns_are_not_a_block():
    profiles = [_p("c0", "name"), _p("c1", "addressPostName"), _p("c2", "address")]
    assert declare_address_block([], profiles) == ([], [], [])


def test_the_snake_case_variant_of_the_layout_reads_the_same():
    profiles = [_p("c1", "address_post_name"), _p("c2", "address_thoroughfare"),
                _p("c3", "address_locator_designator")]
    entities, bindings, _ = declare_address_block([], profiles)
    assert entities[0]["keys"] == ["c2", "c3"]
    assert {b["prop"] for b in bindings} == {"Address:city", "Address:street", "Address:full"}


def test_the_block_overrides_the_model_and_attaches_to_the_row_s_person():
    """Івано-Франківськ (`work-c7-cold`): seven address columns bound as
    `Person:address` strings, no Address declared. The rule declares it,
    keys it on street and number, and attaches it to the deputy."""
    frame = build_frame(Grid(rows=[
        ["familyName", "partyName", "addressAdminUnitL2", "addressPostName",
         "addressThoroughfare", "addressLocatorDesignator"],
        ["Коваленко", "ВО «Свобода»", "Івано-Франківська область", "Івано-Франківськ", "Сахарова", "32"],
        ["Шевченко", "ВО «Свобода»", "Івано-Франківська область", "Івано-Франківськ", "Василіянок", "28"],
        ["Бондаренко", "Батьківщина", "Івано-Франківська область", "Івано-Франківськ", "Сахарова", "32"]],
        sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    plan = Plan(subject="Person",
                entities=[{"key": "deputy", "schema": "Person", "keys": ["c0"]},
                          {"key": "party", "schema": "Organization", "keys": ["c1"]}],
                edges=[], shortlists={},
                bindings=[{"column": "c0", "prop": "Person:lastName", "entity": "deputy", "why": "x"},
                          {"column": "c1", "prop": "Organization:name", "entity": "party", "why": "x"},
                          {"column": "c3", "prop": "Person:address", "entity": "deputy", "why": "x"},
                          {"column": "c4", "prop": "Person:address", "entity": "deputy", "why": "x"}])
    v = validate(plan, profiles, CAT, CFG, frame)
    addr = [e for e in v.entities if e["key"] == ADDRESS_KEY]
    assert addr and addr[0]["keys"] == ["c4", "c5"]
    got = {(b["column"], b["prop"], b["entity"]) for b in v.bindings}
    assert ("c3", "Address:city", ADDRESS_KEY) in got
    assert ("c4", "Address:street", ADDRESS_KEY) in got
    assert not any(b["prop"] == "Person:address" for b in v.bindings)
    assert {"entity": "deputy", "prop": "Person:addressEntity", "target": ADDRESS_KEY} in v.attachments
    overridden = [d for d in v.decisions if "overridden by the address-block rule" in d.reason]
    assert {d.column for d in overridden} == {"c3", "c4"}


def test_a_purchase_row_with_a_supplier_is_a_contract_with_an_award():
    """The court's purchases (`work-c7-cold`): a supplier under «Учасник»,
    procurement headers, a link per row, and no Contract on the plan."""
    from ftmap.plan.blocks import CONTRACT_KEY, declare_contract
    from dataclasses import replace
    profiles = [_p("c0", "Найменування"), _p("c6", "Видпредметузакупівлі"),
                _p("c15", "Учасник"), _p("c17", "Пропозиція"),
                replace(_p("c19", "Посилання"), detectors={"url": 1.0}, distinct_ratio=1.0)]
    entities = [{"key": "c0", "schema": "PublicBody", "keys": ["c0"]},
                {"key": "c15", "schema": "LegalEntity", "keys": ["c15"]}]
    new, notes = declare_contract(entities, profiles)
    assert new == [{"key": CONTRACT_KEY, "schema": "Contract", "keys": ["c19"]}]
    assert "keyed on the link column c19" in notes[0][2]
    # No supplier role word, no contract.
    assert declare_contract(entities, [_p("c6", "Видпредметузакупівлі")]) == ([], [])


def test_the_award_takes_the_award_things_and_their_columns():
    frame = build_frame(Grid(rows=[
        ["процедура", "назва закупівлі", "Код по ДК 021:2015", "Сума, грн.", "з ким укладено договір"],
        ["відкриті торги", "папір", "30190000-7", "12000", "ТОВ «Ромашка»"],
        ["спрощена", "пальне", "09130000-9", "45000", "ПП «Зоря»"],
        ["відкриті торги", "меблі", "39100000-3", "9800", "ТОВ «Ромашка»"]],
        sheet="s", merges=[]), "/x.xlsx", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    plan = Plan(subject="ContractAward",
                entities=[{"key": "c2", "schema": "ContractAward", "keys": ["c2"]},
                          {"key": "c3", "schema": "ContractAward", "keys": ["c3"]},
                          {"key": "c4", "schema": "LegalEntity", "keys": ["c4"]}],
                edges=[], shortlists={},
                bindings=[{"column": "c2", "prop": "ContractAward:cpvCode", "entity": "c2", "why": "x"},
                          {"column": "c3", "prop": "ContractAward:amount", "entity": "c3", "why": "x"},
                          {"column": "c4", "prop": "LegalEntity:name", "entity": "c4", "why": "x"}])
    v = validate(plan, profiles, CAT, CFG, frame)
    from ftmap.plan.blocks import CONTRACT_KEY
    keys = {e["key"]: e for e in v.entities}
    # Declared keyed on nothing; the key proposal then keys it on the one
    # name bound to it, as it does for any keyless entity.
    assert CONTRACT_KEY in keys and keys[CONTRACT_KEY]["schema"] == "Contract"
    assert "c2" not in keys and "c3" not in keys, keys
    award = [e for e in v.edges if e["schema"] == "ContractAward"]
    # FollowTheMoney's ContractAward runs contract -> supplier.
    assert len(award) == 1 and award[0]["source"] == CONTRACT_KEY and award[0]["target"] == "c4"
    got = {(b["column"], b["prop"], b["entity"]) for b in v.bindings}
    assert ("c2", "ContractAward:cpvCode", award[0]["key"]) in got
    assert ("c3", "ContractAward:amount", award[0]["key"]) in got
    assert ("c0", "Contract:procedure", CONTRACT_KEY) in got
    assert ("c1", "Contract:name", CONTRACT_KEY) in got


def test_a_redacted_identifier_splits_the_debtor_into_the_coded_and_the_named():
    """The tax-debtor register: `tin_s` is `**********` on three quarters of
    the rows and an 8-digit ЄДРПОУ on the rest. The coded rows are
    organisations keyed on the code; the redacted rows are persons keyed
    on the name and selected on the redaction."""
    rows = [["tin_s", "name", "chief_name"]]
    people = ["Коваленко Іван Петрович", "Шевченко Ольга Іванівна", "Бондаренко Петро Ілліч",
              "Мельник Андрій Васильович", "Ткаченко Марія Олегівна", "Кравець Олег Юрійович"]
    rows += [["**********", name, ""] for name in people]
    # Real ЄДРПОУ shapes: the detector checks the control digit.
    rows += [[code, f"ТОВ «Ромашка {i}»", "Мельник Андрій"]
             for i, code in enumerate(("14360570", "00032112", "21560045"))]
    frame = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    profiles = profile_frame(frame, CFG)
    plan = Plan(subject="LegalEntity",
                entities=[{"key": "debtor", "schema": "LegalEntity", "keys": ["c0"]}],
                edges=[], shortlists={},
                bindings=[{"column": "c1", "prop": "LegalEntity:name", "entity": "debtor", "why": "x"},
                          {"column": "c0", "prop": "LegalEntity:registrationNumber", "entity": "debtor", "why": "x"}])
    v = validate(plan, profiles, CAT, CFG, frame)
    keys = {e["key"]: e for e in v.entities}
    assert keys["debtor"]["schema"] == "Organization" and keys["debtor"]["keys"] == ["c0"]
    assert "debtor_redacted" in keys, keys
    sib = keys["debtor_redacted"]
    assert sib["schema"] == "Person" and sib["keys"] == ["c1"] \
        and sib["filter"] == {"column": "c0", "value": "**********"}
    assert ("c1", "Person:name", "debtor_redacted") in {
        (b["column"], b["prop"], b["entity"]) for b in v.bindings}


def test_a_code_filled_on_a_few_rows_does_not_make_the_party_an_organisation():
    from ftmap.plan.blocks import specialise_coded_party
    from ftmap.profile.columns import ColumnProfile
    thin = ColumnProfile(id="c2", index=2, header="DEBTOR_CODE", label=None, count=100,
                         filled=8, fill_rate=0.08, distinct=8, distinct_ratio=1.0,
                         min_len=8, max_len=8, shapes=[], detectors={"edrpou": 0.99}, samples=[])
    e = {"key": "debtor", "schema": "LegalEntity", "keys": ["c0", "c2"]}
    assert specialise_coded_party([e], [_p("c0", "DEBTOR_NAME"), thin]) == []
    assert e["schema"] == "LegalEntity"


def _q(cid, header, distinct=1.0, fill=1.0, detectors=None):
    return ColumnProfile(
        id=cid, index=int(cid[1:]), header=header, label=None, count=10,
        filled=int(10 * fill), fill_rate=fill, distinct=int(10 * distinct),
        distinct_ratio=distinct, min_len=3, max_len=30, shapes=[],
        detectors=detectors or {}, samples=[])


def test_a_certificate_number_beside_a_party_is_an_identification():
    """The aircraft register keyed a License on «№ реєстраційного
    посвідчення» and the land valuers one on `qualCertNumber`
    (`work-c15-cold`); a License has no holder. The entity becomes the
    Identification, keyed where it was, and the column is its number."""
    from ftmap.plan.blocks import (IDENTIFICATION_KEY, certificate_column,
                                   declare_identification)
    profiles = [_q("c2", "Реєстраційний знак"), _q("c6", "№ реєстраційного посвідчення"),
                _q("c7", "Дата видачі", distinct=0.3, detectors={"date": 1.0}),
                _q("c9", "Статус сертифікату", distinct=0.1), _q("c11", "Власник", distinct=0.4)]
    assert certificate_column(profiles).id == "c6"
    entities = [{"key": "plane", "schema": "Vehicle", "keys": ["c2"]},
                {"key": "cert", "schema": "License", "keys": ["c6"]},
                {"key": "owner", "schema": "LegalEntity", "keys": ["c11"]}]
    new, bindings, notes = declare_identification(entities, profiles, CAT)
    assert new == [] and entities[1]["schema"] == "Identification"
    assert bindings == [{"column": "c6", "prop": "Identification:number", "entity": "cert",
                         "why": "«№ реєстраційного посвідчення» (c6) is the certificate's number"}]
    assert "re-schemed from License" in notes[0][2]
    # Nothing keyed on it: declared. The land valuers' camelCase form.
    profiles = [_q("c1", "fullName", detectors={"proper_name": 1.0}),
                _q("c3", "qualCertNumber"),
                _q("c2", "qualCertIssueDate", distinct=0.3, detectors={"date": 1.0})]
    new, bindings, _ = declare_identification(
        [{"key": "valuer", "schema": "Person", "keys": ["c1"]}], profiles, CAT)
    assert new == [{"key": IDENTIFICATION_KEY, "schema": "Identification", "keys": ["c3"]}]
    assert bindings[0]["entity"] == IDENTIFICATION_KEY
    # The row's asset keyed on the certificate number is not re-schemed.
    entities = [{"key": "plane", "schema": "Vehicle", "keys": ["c6"]}]
    new, bindings, notes = declare_identification(
        entities, [_q("c6", "№ посвідчення"), _q("c11", "Власник")], CAT)
    assert new == [] and bindings == [] and entities[0]["schema"] == "Vehicle"
    assert notes[0][2].startswith("declined")


def test_the_certificate_s_holder_is_the_row_s_party_or_the_asset_s_owner():
    from ftmap.plan.blocks import attach_identification_holder
    profiles = [_q("c1", "fullName"), _q("c3", "qualCertNumber")]
    entities = [{"key": "valuer", "schema": "Person", "keys": ["c1"]},
                {"key": "cert", "schema": "Identification", "keys": ["c3"]}]
    out, notes = attach_identification_holder(entities, [], [], profiles, "Person", CAT)
    assert out == [{"entity": "cert", "prop": "Identification:holder", "target": "valuer"}]
    # A vehicle's certificate: two parties, the owner of the row's asset holds it.
    profiles = [_q("c2", "Реєстраційний знак"), _q("c6", "№ посвідчення"),
                _q("c10", "Експлуатант"), _q("c11", "Власник")]
    entities = [{"key": "plane", "schema": "Vehicle", "keys": ["c2"]},
                {"key": "cert", "schema": "Identification", "keys": ["c6"]},
                {"key": "operator", "schema": "Company", "keys": ["c10"]},
                {"key": "owner", "schema": "LegalEntity", "keys": ["c11"]}]
    edges = [{"key": "own", "schema": "Ownership", "source": "owner", "target": "plane"}]
    out, notes = attach_identification_holder(entities, edges, [], profiles, "Vehicle", CAT)
    assert out == [{"entity": "cert", "prop": "Identification:holder", "target": "owner"}]
    # Without the ownership, two parties and nothing to choose by.
    out, notes = attach_identification_holder(entities, [], [], profiles, "Vehicle", CAT)
    assert out == [] and "2 parties" in notes[0][1]
