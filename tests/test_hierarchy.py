# tests/test_hierarchy.py
"""Organisations a sheet nests, read from their columns.

The case is the establishment table of `2026-09-05-manual-corpus.md`: a
serviceman's unit named in four columns, one Organization declared per
column, the person related to all four, and the fan gate dropping every
strand because a relation to every counterpart claims nothing. Values here
are invented in the sheet's shape — a section, a platoon, a unit — and every
assertion is about the shape of the values, not any value.
"""

from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.frame import column_values
from ftmap.io.tabular import Grid
from ftmap.plan.hierarchy import nest_organisations, nested, organisation_tree
from ftmap.plan.propose import Plan
from ftmap.plan.validate import validate
from ftmap.profile.columns import profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()

# ПІБ | Отделение | Рота | войсковая часть — every section in one platoon,
# every platoon in the one unit, and one row (the last) where the section is
# written under the wrong platoon, as a real sheet does.
ROWS = [["ПІБ", "Отделение", "Рота", "войсковая часть"],
        ["Коваленко Іван", "1 отд", "1 взвод", "в/ч 11111"],
        ["Шевченко Ольга", "1 отд", "1 взвод", "в/ч 11111"],
        ["Бондаренко Ігор", "2 отд", "1 взвод", "в/ч 11111"],
        ["Мельник Петро", "2 отд", "1 взвод", "в/ч 11111"],
        ["Ткаченко Ніна", "3 отд", "2 взвод", "в/ч 11111"],
        ["Гончар Олесь", "3 отд", "2 взвод", "в/ч 11111"],
        ["Кравець Марія", "4 отд", "2 взвод", "в/ч 11111"],
        ["Лисенко Юрій", "4 отд", "2 взвод", "в/ч 11111"],
        ["Поліщук Анна", "5 отд", "3 взвод", "в/ч 11111"],
        ["Савченко Дмитро", "5 отд", "3 взвод", "в/ч 11111"],
        ["Руденко Олена", "1 отд", "3 взвод", "в/ч 11111"]]


def _setup(rows=ROWS):
    f = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.xlsx", "0" * 64, "sid/s", CFG)
    cache = {}

    def values(col):
        if col not in cache:
            cache[col] = column_values(f, col)
        return cache[col]
    return f, profile_frame(f, CFG), values


ENTITIES = [{"key": "person", "schema": "Person", "keys": ["c0"]},
            {"key": "squad", "schema": "Organization", "keys": ["c1"]},
            {"key": "platoon", "schema": "Organization", "keys": ["c2"]},
            {"key": "unit", "schema": "Organization", "keys": ["c3"]}]


def test_a_section_nests_in_its_platoon_within_the_floor_and_not_the_other_way():
    _f, _p, values = _setup()
    # 10 of 11 rows carry the section's commonest platoon; 1 row does not.
    assert nested(values, "c1", "c2", 0.9) == (10, 11)
    assert nested(values, "c1", "c2", 0.95) is None
    # A platoon names several sections: the outer has fewer values, never more.
    assert nested(values, "c2", "c1", 0.5) is None
    assert nested(values, "c2", "c3", 0.9) == (11, 11)


def test_two_constants_nest_neither_way():
    rows = [["ПІБ", "Подразделение", "войсковая часть"],
            ["Коваленко Іван", "2 рота", "в/ч 11111"],
            ["Шевченко Ольга", "2 рота", "в/ч 11111"]]
    _f, _p, values = _setup(rows)
    assert nested(values, "c1", "c2", 0.9) is None
    assert nested(values, "c2", "c1", 0.9) is None


def test_the_tree_names_the_nearest_ancestor_as_parent():
    _f, profiles, values = _setup()
    parents, notes = organisation_tree(ENTITIES, values, CAT, 0.9, profiles)
    assert parents == {"squad": "platoon", "platoon": "unit"}
    assert any("nested in platoon" in n and "10 of the 11 rows" in n
               for _k, n in notes)


def test_a_person_s_relation_into_the_tree_keeps_the_leaf_and_the_parents_say_the_rest():
    edges = [{"key": "e1", "schema": "Employment", "source": "person", "target": "unit"},
             {"key": "e2", "schema": "Employment", "source": "person", "target": "squad"},
             {"key": "e3", "schema": "Employment", "source": "person", "target": "platoon"},
             {"key": "m1", "schema": "Membership", "source": "unit", "target": "squad"}]
    _f, profiles, values = _setup()
    kept, attachments, notes, _tree = nest_organisations(
        ENTITIES, edges, [], values, CAT, 0.9, profiles=profiles)
    assert [e["key"] for e in kept] == ["e2"]
    assert attachments == [
        {"entity": "squad", "prop": "Organization:parent", "target": "platoon"},
        {"entity": "platoon", "prop": "Organization:parent", "target": "unit"}]
    verdicts = {k: v for k, v, _n in notes}
    assert verdicts["e1"] == "rejected" and verdicts["e3"] == "rejected"
    assert verdicts["m1"] == "rejected"
    assert any("containment said as a relation" in n for _k, _v, n in notes)


def test_two_unrelated_organisations_are_left_to_the_fan_gate():
    """No nesting, no tree: the rule changes nothing, and the gate keeps its
    measured behaviour."""
    rows = [["ПІБ", "Партія", "Комісія"],
            ["Коваленко Іван", "Слуга Народу", "бюджетна"],
            ["Шевченко Ольга", "Батьківщина", "бюджетна"],
            ["Бондаренко Ігор", "Слуга Народу", "земельна"]]
    _f, profiles, values = _setup(rows)
    ents = [{"key": "p", "schema": "Person", "keys": ["c0"]},
            {"key": "party", "schema": "Organization", "keys": ["c1"]},
            {"key": "commission", "schema": "Organization", "keys": ["c2"]}]
    edges = [{"key": "m1", "schema": "Membership", "source": "p", "target": "party"},
             {"key": "m2", "schema": "Membership", "source": "p", "target": "commission"}]
    kept, attachments, notes, _tree = nest_organisations(
        ents, edges, [], values, CAT, 0.9, profiles=profiles)
    assert kept == edges and attachments == [] and notes == []


def test_values_that_nest_under_headers_that_name_no_level_do_not_nest():
    """The MPs (`test_validate`'s folded-parties case): every party of
    membership has one nominating party, so the values depend exactly as a
    section's on its platoon — and a nomination is not a containment. No
    level word heads either column; nothing nests, both Memberships stand."""
    rows = [["ПІБ", "party_text", "party_name"],
            ["Коваленко Іван", "Слуга Народу", "Безпартійний"],
            ["Шевченко Ольга", "Слуга Народу", "Слуга Народу"],
            ["Мельник Андрій", "Голос", "Голос"],
            ["Бойко Сергій", "Слуга Народу", "Безпартійний"]]
    _f, profiles, values = _setup(rows)
    ents = [{"key": "p", "schema": "Person", "keys": ["c0"]},
            {"key": "nominator", "schema": "Organization", "keys": ["c1"]},
            {"key": "party", "schema": "Organization", "keys": ["c2"]}]
    assert nested(values, "c2", "c1", 0.9) is not None
    parents, _notes = organisation_tree(ents, values, CAT, 0.9, profiles)
    assert parents == {}


def test_validate_keeps_the_leaf_relation_through_the_fan_gate_and_attaches_the_parents():
    """The whole path: three Employments from the person into the tree and a
    Membership from the unit to its section, as the model declared them on
    the establishment table. One Employment survives, the tree's parents are
    attached, and the fan gate has nothing to drop."""
    frame, profiles, _values = _setup()
    v = validate(Plan(
        subject="Person", entities=ENTITIES, shortlists={},
        edges=[{"key": "e1", "schema": "Employment", "source": "person", "target": "unit"},
               {"key": "e2", "schema": "Employment", "source": "person", "target": "squad"},
               {"key": "e3", "schema": "Employment", "source": "person", "target": "platoon"},
               {"key": "m1", "schema": "Membership", "source": "unit", "target": "squad"}],
        bindings=[{"column": "c0", "prop": "Person:name", "entity": "person", "why": "x"},
                  {"column": "c1", "prop": "Organization:name", "entity": "squad", "why": "x"},
                  {"column": "c2", "prop": "Organization:name", "entity": "platoon", "why": "x"},
                  {"column": "c3", "prop": "Organization:name", "entity": "unit", "why": "x"}],
    ), profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.edges] == ["e2"]
    assert {(a["entity"], a["prop"], a["target"]) for a in v.attachments} >= {
        ("squad", "Organization:parent", "platoon"),
        ("platoon", "Organization:parent", "unit")}
    assert not any("every counterpart" in d.reason for d in v.decisions)
    assert "nesting_floor" in CFG.as_manifest()["plan"]


def test_two_constants_above_the_tree_are_both_implied_by_the_leaf():
    """The unit and the company are one value each over the whole sheet:
    neither nests in the other, both are above every section, and a
    person's relation to either is implied by the one to the section — so
    the fan gate is left one strand, not two."""
    rows = [["ПІБ", "Отделение", "Подразделение", "войсковая часть"],
            ["Коваленко Іван", "1 отд", "2 рота", "в/ч 11111"],
            ["Шевченко Ольга", "1 отд", "2 рота", "в/ч 11111"],
            ["Бондаренко Ігор", "2 отд", "2 рота", "в/ч 11111"]]
    _f, profiles, values = _setup(rows)
    ents = [{"key": "person", "schema": "Person", "keys": ["c0"]},
            {"key": "squad", "schema": "Organization", "keys": ["c1"]},
            {"key": "company", "schema": "Organization", "keys": ["c2"]},
            {"key": "unit", "schema": "Organization", "keys": ["c3"]}]
    from ftmap.plan.hierarchy import organisation_ancestors
    assert organisation_ancestors(ents, values, CAT, 0.9, profiles) == {
        "squad": {"company", "unit"}}
    edges = [{"key": "e1", "schema": "Employment", "source": "person", "target": "unit"},
             {"key": "e2", "schema": "Employment", "source": "person", "target": "squad"},
             {"key": "e3", "schema": "Employment", "source": "person", "target": "company"}]
    kept, attachments, notes, _tree = nest_organisations(
        ents, edges, [], values, CAT, 0.9, profiles=profiles)
    assert [e["key"] for e in kept] == ["e2"]
    # The parent is the leftmost of the two constants; the other has none.
    assert attachments == [{"entity": "squad", "prop": "Organization:parent",
                            "target": "company"}]


def test_a_relative_level_is_keyed_on_its_path_under_the_nearest_coarser_level():
    """The regiment: «1 рота» in every battalion, «1 взвод» in every
    company. The company names one value of no level above it by value, so
    it is keyed on the subunit's key and itself, the platoon on the
    company's, coarse to fine by the lexicon's order of level words; the
    unit every level names is the root, and the subunit nests in it by
    value and keeps its own key."""
    rows = [["ПІБ", "В/Ч", "Подр-е", "До роты/взвода", "До взвода/отд"]]
    units = ["в/ч 11111", "в/ч 22222"]
    for u in range(2):
        for b in range(2):
            for r in range(2):
                for v in range(2):
                    rows.append([f"Особа {u}{b}{r}{v}", units[u], f"{u * 2 + b + 1} тб",
                                 f"{r + 1} рота", f"{v + 1} взвод"])
    _f, profiles, values = _setup(rows)
    ents = [{"key": "person", "schema": "Person", "keys": ["c0"]},
            {"key": "unit", "schema": "Organization", "keys": ["c1"]},
            {"key": "subunit", "schema": "Organization", "keys": ["c2"]},
            {"key": "company", "schema": "Organization", "keys": ["c3"]},
            {"key": "platoon", "schema": "Organization", "keys": ["c4"]}]
    edges = [{"key": "m1", "schema": "Membership", "source": "person", "target": "unit"},
             {"key": "m2", "schema": "Membership", "source": "person", "target": "platoon"}]
    kept, attachments, notes, tree = nest_organisations(
        ents, edges, [], values, CAT, 0.9, profiles=profiles)
    by_key = {e["key"]: e["keys"] for e in ents}
    assert by_key["subunit"] == ["c2"]
    assert by_key["company"] == ["c2", "c3"]
    assert by_key["platoon"] == ["c2", "c3", "c4"]
    assert tree == {"subunit": "unit", "company": "subunit", "platoon": "company"}
    assert [e["key"] for e in kept] == ["m2"]
    assert any("keyed on its path" in n for _k, _v, n in notes)
    assert {(a["entity"], a["target"]) for a in attachments} == {
        ("subunit", "unit"), ("company", "subunit"), ("platoon", "company")}
