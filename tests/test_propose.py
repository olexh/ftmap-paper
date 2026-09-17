from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.plan.prompt import BINDING_MARKER, EDGE_MARKER, SPLIT_MARKER
from ftmap.plan.propose import propose
from ftmap.plan.response_schema import UNMAPPED
from ftmap.plan.validate import validate
from ftmap.profile.columns import profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()


class FakeClient:
    """Answers structure once, edges at most once, then one binding batch per
    chunk. Distinguishes the three call kinds by their prompt text.

    Each item in `batches` is a raw grammar-shaped binding: `{"column",
    "binding", "why"}`, where `binding` is an `"entity|prop"` pair or
    `"unmapped"` — the same shape the real llama-server's grammar-constrained
    answer takes (Change 2), not the internal `Plan.bindings` shape `propose`
    produces after splitting it.
    """

    def __init__(self, structure, batches):
        self.structure = structure
        self.batches = list(batches)
        self.prompts: list[str] = []
        self.schemas: list[dict] = []
        self.max_tokens_seen: list[int | None] = []

    def complete(self, system, user, schema, max_tokens=None):
        self.prompts.append(user)
        self.schemas.append(schema)
        self.max_tokens_seen.append(max_tokens)
        if BINDING_MARKER in user:
            return {"bindings": self.batches.pop(0)}
        if EDGE_MARKER in user:
            return {"edges": self.structure.get("edges", [])}
        return {k: v for k, v in self.structure.items() if k != "edges"}


def _fp(rows):
    f = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64, "sid/s", CFG)
    return f, profile_frame(f, CFG)


def test_a_binding_chunk_too_big_for_the_context_is_split_and_retried():
    """The ship register's ДРСУ sheet, 2026-08-30: a six-entity roster with
    four `LegalEntity` multiplied the candidate pairs until one twenty-column
    binding prompt reached 33 506 tokens against the server's 32 768, the
    server answered 400 `exceed_context_size_error`, and the whole source
    FAILED — a run-level failure for a prompt-length problem the pipeline can
    solve itself. Half the columns is half the candidates: split the chunk,
    ask twice, keep every answer."""
    from ftmap.plan.client import ModelError

    class OverflowingClient(FakeClient):
        def __init__(self, structure, batches, fails_over):
            super().__init__(structure, batches)
            self.fails_over = fails_over

        def complete(self, system, user, schema, max_tokens=None):
            if (BINDING_MARKER in user
                    and user.count("\nc") > self.fails_over):
                self.prompts.append(user)
                raise ModelError(
                    'llama-server returned 400: {"error":{"type":'
                    '"exceed_context_size_error","n_prompt_tokens":33506}}')
            return super().complete(system, user, schema, max_tokens)

    frame, profiles = _fp([["ПІБ", "Дата народження", "Телефон", "Адреса"],
                           ["Коваленко Іван", "17.09.1980", "0501234567",
                            "м. Київ"]])
    client = OverflowingClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        # The four-column chunk overflows; each two-column half succeeds.
        [[{"column": "c0", "binding": "person|Person:name", "why": "x"},
          {"column": "c1", "binding": "person|Person:birthDate", "why": "x"}],
         [{"column": "c2", "binding": "person|Person:phone", "why": "x"},
          {"column": "c3", "binding": "person|Person:address", "why": "x"}]],
        fails_over=2,
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    by_col = {b["column"]: b["prop"] for b in plan.bindings}
    assert by_col["c0"] == "Person:name"
    assert by_col["c3"] == "Person:address"


def test_a_single_column_that_still_overflows_is_a_real_ceiling():
    """The terminal case of the split: a chunk of ONE column that overflows
    cannot be halved, so the original `ModelError` propagates — no further
    binding request is made, and the source fails at the runner rather than
    committing a partial plan. Splitting forever, or swallowing the error and
    shipping the columns that fit, would each be a silent loss."""
    import pytest

    from ftmap.plan.client import ModelError

    class AlwaysOverflowing(FakeClient):
        def complete(self, system, user, schema, max_tokens=None):
            if BINDING_MARKER in user:
                self.prompts.append(user)
                raise ModelError(
                    'llama-server returned 400: {"error":{"type":'
                    '"exceed_context_size_error","n_prompt_tokens":40000}}')
            return super().complete(system, user, schema, max_tokens)

    frame, profiles = _fp([["ПІБ", "Дата народження"],
                           ["Коваленко Іван", "17.09.1980"]])
    client = AlwaysOverflowing(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [],
    )
    with pytest.raises(ModelError) as err:
        propose(frame, profiles, CAT, CFG, client)
    assert "exceed_context_size" in str(err.value)
    # The two-column chunk was asked, split, and each single column asked
    # once; after the LAST one-column overflow nothing else was requested.
    binding_prompts = [p for p in client.prompts if BINDING_MARKER in p]
    assert 2 <= len(binding_prompts) <= 3
    assert binding_prompts[-1].count("\nc") == 1


def test_bloc_profiling_reads_only_the_live_columns(monkeypatch):
    """A wide sheet declares thousands of empty columns (one fixture declares
    Excel's full 16 384) and a bloc round re-profiles per bloc — so the
    re-profile must be over the LIVE columns alone, or the masking pass does
    full-width work per bloc for profiles it discards on the next line."""
    import ftmap.plan.propose as propose_mod
    from ftmap.plan.propose import _bloc_profiles
    from ftmap.profile.columns import profile_frame as real_profile_frame

    wide = [["name", "kind"] + [f"пусто{i}" for i in range(48)],
            ["Коваленко Іван", "Person"] + [""] * 48,
            ["ТОВ Ромашка", "Organization"] + [""] * 48]
    frame, profiles = _fp(wide)
    live = [p for p in profiles if p.filled > 0]
    assert len(live) == 2 and len(profiles) == 50

    seen_widths = []

    def spying(frame, cfg):
        seen_widths.append(len(frame.columns))
        return real_profile_frame(frame, cfg)

    monkeypatch.setattr("ftmap.profile.columns.profile_frame", spying)
    masked = _bloc_profiles(frame, CFG,
                            {"column": "c1", "value": "Person"}, live)
    assert seen_widths == [2]
    assert [p.id for p in masked] == [p.id for p in live]
    # And the mask really is the bloc's own rows: one Person row.
    assert masked[0].filled == 1


def test_two_calls_for_a_narrow_table():
    frame, profiles = _fp([["ПІБ", "Дата народження"], ["Коваленко Іван", "17.09.1980"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": "person|Person:birthDate", "why": "dates"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    # structure, then bindings. One entity, so no edge call.
    assert len(client.prompts) == 2
    assert plan.subject == "Person"
    assert {b["column"] for b in plan.bindings} == {"c0", "c1"}
    by_col = {b["column"]: b for b in plan.bindings}
    assert by_col["c0"]["prop"] == "Person:name" and by_col["c0"]["entity"] == "person"
    assert by_col["c1"]["prop"] == "Person:birthDate" and by_col["c1"]["entity"] == "person"
    assert plan.shortlists["c1"]


def test_wide_table_chunks_and_keeps_one_structure():
    headers = [f"Поле {i}" for i in range(45)]
    frame, profiles = _fp([headers, [f"v{i}" for i in range(45)]])
    batches = []
    for start in range(0, 45, CFG.chunk_size):
        ids = [f"c{i}" for i in range(start, min(start + CFG.chunk_size, 45))]
        batches.append([{"column": c, "binding": "unmapped", "why": "x"} for c in ids])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": []}],
         "edges": []},
        batches,
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert len(client.prompts) == 1 + 3  # one entity, so no edge call
    assert len(plan.bindings) == 45


def test_max_tokens_is_sized_to_the_chunk_not_a_flat_ceiling():
    """max_tokens was accepted by client.complete() but never actually passed
    by any caller, so every call fell back to the flat cfg.max_tokens
    regardless of how many columns it covered. It is now proportional to the
    chunk: the last, smaller chunk of a wide table must ask for less."""
    headers = [f"Поле {i}" for i in range(25)]
    frame, profiles = _fp([headers, [f"v{i}" for i in range(25)]])
    batches = []
    for start in range(0, 25, CFG.chunk_size):
        ids = [f"c{i}" for i in range(start, min(start + CFG.chunk_size, 25))]
        batches.append([{"column": c, "binding": "unmapped", "why": "x"} for c in ids])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": []}],
         "edges": []},
        batches,
    )
    propose(frame, profiles, CAT, CFG, client)
    binding_calls = [mt for prompt, mt in zip(client.prompts, client.max_tokens_seen)
                     if BINDING_MARKER in prompt]
    assert len(binding_calls) == 2  # 20 columns, then 5
    full_chunk, last_chunk = binding_calls
    assert full_chunk == CFG.max_tokens  # a full chunk gets the whole budget
    assert last_chunk == round(CFG.max_tokens * 5 / CFG.chunk_size)
    assert last_chunk < full_chunk


def test_a_wrong_subject_no_longer_caps_the_source():
    """The structure call answered `Audio` for a deputies table on the live
    server, three separate times. The shortlist is a rank-interleaved union of
    the scoped and unrestricted rankings, so the right candidate survives a
    wrong subject with no threshold and no decision."""
    frame, profiles = _fp([["ПІБ", "Дата народження", "Телефон"],
                           ["Коваленко Іван Петрович", "17.09.1980", "0501234567"]])
    client = FakeClient(
        {"subject": "Audio",
         "entities": [{"key": "thing", "schema": "Audio", "keys": []}],
         "edges": []},
        [[{"column": f"c{i}", "binding": "unmapped", "why": "x"}
          for i in range(3)]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    # THE PROPERTY SURVIVES, under whatever spelling. «ПІБ» is a spelling of
    # `Thing:name`, which every Thing inherits — so the scoped side spells it
    # `Audio:name` at 1.0 and the one slot per name is that one; the
    # unrestricted side's `Person:name` is the same property. What a wrong
    # subject must not do is take `name` off the list, and it does not.
    assert any(q.split(":", 1)[1] == "name" for q in plan.shortlists["c0"])


def test_scoping_follows_declared_entities_not_the_subject_string():
    """Change 1: the scoped ranking is built from what call 1 actually
    DECLARED as entities, not from `neighbours(subject)`. A `subject` that
    disagrees with the declared entity — the structure call answered `Audio`
    for a deputies table on the live server, three separate times — reaches
    the ranking only by BEING DECLARED: a subject that contradicts the one
    entity on the plan is promoted to an entity of its own, because neither
    answer is trustworthy enough to overrule the other (see `plan/subject.py`).
    So the scoped side spans exactly `Person` and `Vessel`, and nothing else:
    `neighbours(Vessel)` would have widened it to ten schemata, and it widens
    it to none. The wrong subject still cannot cap the source — the Person
    candidates it competes with are all still offered."""
    from ftmap.vocab.shortlist import shortlist_scored

    frame, profiles = _fp([["ПІБ", "Дата народження", "Телефон"],
                           ["Коваленко Іван Петрович", "17.09.1980", "0501234567"]])
    client = FakeClient(
        {"subject": "Vessel",  # deliberately wrong, and a second reading now
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": f"c{i}", "binding": "unmapped", "why": "x"}
          for i in range(3)]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e["schema"] for e in plan.entities] == ["Person", "Vessel"]
    for p in profiles:
        scoped = shortlist_scored(p, CAT, CFG, schemas=["Person", "Vessel"])
        if scoped:
            assert plan.shortlists[p.id][0] == scoped[0][0], p.id
        person_only = shortlist_scored(p, CAT, CFG, schemas=["Person"])
        if person_only:
            assert person_only[0][0] in plan.shortlists[p.id], p.id


def test_the_shortlist_is_still_bounded_after_the_union():
    frame, profiles = _fp([["ПІБ", "Дата народження", "Телефон"],
                           ["Коваленко Іван Петрович", "17.09.1980", "0501234567"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": f"c{i}", "binding": "unmapped", "why": "x"}
          for i in range(3)]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    for cid, qs in plan.shortlists.items():
        assert len(qs) <= CFG.shortlist_size, cid
        assert len(qs) == len(set(qs)), cid


def test_empty_columns_are_reported_without_costing_a_model_call():
    """Measured on the private corpus: 58% of its columns hold no value at all,
    and one sheet declares 2 575 columns of which 32 are live. Planning over
    them would be 129 binding calls where 2 would do."""
    frame, profiles = _fp([["ПІБ", "Порожня", "Дата народження"],
                           ["Коваленко Іван Петрович", None, "17.09.1980"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "x"},
          {"column": "c2", "binding": "person|Person:birthDate", "why": "x"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    empty = [b for b in plan.bindings if b["column"] == "c1"][0]
    assert empty["prop"] == "unmapped"
    assert empty["why"] == "the column holds no value"
    # Still reported: coverage is over all three columns, not over the two live.
    assert len(plan.bindings) == 3
    assert "c1" not in [line.split(":")[0] for line in client.prompts[-1].splitlines()
                        if line[:1] == "c"]


def test_edges_are_asked_for_separately_and_only_when_possible():
    """One entity cannot be connected to anything, so the call is skipped — and
    so is a pair of entities no FtM edge can relate, which would otherwise be
    asked a question with an empty enum for an answer."""
    frame, profiles = _fp([["ПІБ", "Фракція"],
                           ["Коваленко Іван Петрович", "Слуга народу"]])
    two = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "party", "schema": "Organization", "keys": ["c1"]}],
         "edges": [{"key": "member", "edge": "Membership|person|party"}]},
        [[{"column": f"c{i}", "binding": "unmapped", "why": "x"}
          for i in range(2)]],
    )
    plan = propose(frame, profiles, CAT, CFG, two)
    assert len(two.prompts) == 3  # structure, edges, bindings
    assert plan.edges[0]["source"] == "person"

    one = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": f"c{i}", "binding": "unmapped", "why": "x"}
          for i in range(2)]],
    )
    plan = propose(frame, profiles, CAT, CFG, one)
    assert len(one.prompts) == 2  # no edge call
    assert plan.edges == []


def test_a_source_whose_columns_are_all_empty_costs_no_model_call():
    """Distinct from a sheet with no columns at all: here the columns exist and
    must each be reported, but none can be bound."""
    frame, profiles = _fp([["А", "Б", "В"], [None, None, None]])

    class Boom:
        prompts: list[str] = []

        def complete(self, *a, **kw):
            raise AssertionError("no live column, so nothing to ask about")

    plan = propose(frame, profiles, CAT, CFG, Boom())
    assert len(plan.bindings) == 3
    assert {b["why"] for b in plan.bindings} == {"the column holds no value"}


def test_a_colliding_edge_key_is_renamed_not_lost():
    """Measured on two real deputies files: the model reused one key for a
    Person entity and its Membership edges on both, so every edge was rejected
    and the run produced zero Membership entities — the relationship half of
    the output, empty. Renaming loses nothing."""
    frame, profiles = _fp([["ПІБ", "Фракція"],
                           ["Коваленко Іван Петрович", "Слуга народу"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "party", "schema": "Organization", "keys": ["c1"]}],
         "edges": [{"key": "person", "edge": "Membership|person|party"}]},
        [[{"column": f"c{i}", "binding": "unmapped", "why": "x"}
          for i in range(2)]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.edges[0]["key"] == "person_edge"
    assert plan.edges[0]["source"] == "person"
    assert {e["key"] for e in plan.entities} == {"person", "party"}


def test_an_empty_sheet_costs_no_model_call():
    frame, profiles = _fp([[]])

    class Boom:
        def complete(self, *a, **kw):
            raise AssertionError("an empty sheet must not reach the model")

    plan = propose(frame, profiles, CAT, CFG, Boom())
    assert plan.bindings == []
    assert plan.shortlists == {}


def test_missing_binding_becomes_an_explicit_unmapped():
    frame, profiles = _fp([["ПІБ", "Дата народження"], ["Коваленко Іван", "17.09.1980"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    missing = [b for b in plan.bindings if b["column"] == "c1"][0]
    assert missing["prop"] == "unmapped"
    assert "no answer" in missing["why"]


# --- Change 2: propose() splits the grammar's "entity|prop" pair back apart ---


def test_propose_splits_the_grammar_pair_back_into_prop_and_entity():
    frame, profiles = _fp([["ПІБ", "Фракція"],
                           ["Коваленко Іван Петрович", "Слуга народу"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "party", "schema": "Organization", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": "party|Organization:name", "why": "org"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    by_col = {b["column"]: b for b in plan.bindings}
    assert by_col["c0"] == {"column": "c0", "prop": "Person:name",
                            "entity": "person", "why": "names"}
    assert by_col["c1"] == {"column": "c1", "prop": "Organization:name",
                            "entity": "party", "why": "org"}


def test_a_column_with_no_valid_pair_is_unmapped_without_a_model_call(monkeypatch):
    """Change 1 narrows retrieval, but a column can still end up with a
    shortlist none of the declared entities can carry — narrower retrieval is
    not a guarantee of usability, only an improvement in its odds. Such a
    column must still be reported, honestly, and without spending a model
    call whose only possible answer is 'unmapped' — deciding it here is
    strictly better than asking a question with one legal answer."""
    import ftmap.plan.propose as propose_mod

    def fake_shortlist(profile, cat, cfg, subject=None, schemas=None):
        # c0's only candidate belongs to a schema nobody declared; c1's does.
        return ["Vessel:flag"] if profile.id == "c0" else ["Person:name"]

    monkeypatch.setattr(propose_mod, "shortlist", fake_shortlist)

    frame, profiles = _fp([["А", "ПІБ"], ["1", "Коваленко Іван"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": []}],
         "edges": []},
        [[{"column": "c1", "binding": "person|Person:name", "why": "names"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)

    by_col = {b["column"]: b for b in plan.bindings}
    assert by_col["c0"]["prop"] == "unmapped"
    assert "no declared entity" in by_col["c0"]["why"]
    assert by_col["c1"]["prop"] == "Person:name"

    # c0 never reached the model: the one binding prompt only names c1.
    binding_prompts = [p for p in client.prompts if BINDING_MARKER in p]
    assert len(binding_prompts) == 1
    assert "c0:" not in binding_prompts[0]
    assert "c1:" in binding_prompts[0]


def test_an_entity_typed_property_is_never_offered_for_a_column():
    """A CELL IS NOT AN ENTITY REFERENCE, so no column may be offered a
    property that only a reference can satisfy.

    `carries` asks whether the schema HAS the property, which is necessary and
    not sufficient. `Vehicle:addressEntity` is carried by Vehicle and is
    satisfied only by the id of an Address entity — a hash the engine computes
    from the key prefix, the plan key and the key columns, never a string a
    source printed. Measured on the МВС vehicle registry
    (`reestrtz_2026_sample.csv`, 39 607 rows): its `DEP` column of
    service-centre labels was bound to `Vehicle:addressEntity`, and
    followthemoney's own entity validator refused all 39 607 values — every
    one recorded as a reject, none emitted. On the private corpus that reason
    was the largest single reject class.

    Read off the GRAMMAR rather than off the shortlist or the pair list: the
    enum is what the model can actually produce, and it is the only artefact
    that a later change to retrieval cannot quietly widen.
    """
    frame, profiles = _fp([["DEP", "MODEL"],
                           ["ТСЦ 8045", "MDX"],
                           ["ТСЦ 6141", "LEAF"]])
    client = FakeClient(
        {"subject": "Vehicle",
         "entities": [{"key": "vehicle", "schema": "Vehicle", "keys": []},
                      {"key": "dep", "schema": "Organization", "keys": []}],
         "edges": []},
        [[{"column": "c0", "binding": "unmapped", "why": "x"},
          {"column": "c1", "binding": "vehicle|Vehicle:model", "why": "models"}]],
    )
    propose(frame, profiles, CAT, CFG, client)

    grammar = [s for p, s in zip(client.prompts, client.schemas)
               if BINDING_MARKER in p][0]
    enums = [b["properties"]["binding"]["enum"]
             for b in grammar["properties"]["bindings"]["prefixItems"]]
    offered = {pair for enum in enums for pair in enum if pair != UNMAPPED}
    assert offered, "every column was offered nothing but `unmapped`"
    # The exact pairing the model took on the real registry.
    assert "vehicle|Vehicle:addressEntity" not in offered
    assert [q for q in offered
            if CAT.prop(q.split("|", 1)[1]).type_name == "entity"] == []
    # NARROWED, NOT EMPTIED: the DEP column is still offered properties it
    # could satisfy, so the column is decided on evidence rather than by
    # having nothing left to choose from.
    assert [q for q in enums[0] if q != UNMAPPED]
    # And the prompt describes the same question the grammar asks.
    binding_prompt = [p for p in client.prompts if BINDING_MARKER in p][0]
    assert "addressEntity" not in binding_prompt


def test_two_entities_declaring_one_key_both_reach_the_grammar():
    """The whole run `ua_war_sanctions.targets.simple.csv` lost. The model
    declared `id -> Person` AND `id -> Organization`; `propose` built
    `declared` with a dict comprehension, so Person vanished, every candidate
    the grammar offered read `id (Organization) -> Organization:*`, and
    `validate` — which renames a colliding entity key instead of merging it,
    and therefore still had `id` meaning Person — rejected the model's
    faithful answers with "schema Person does not carry Organization:name".
    0 mapped columns, 0 statements, 11 242 property-less entities from 5 621
    rows. The grammar was sound and built from the wrong entity table.
    """
    frame, profiles = _fp([["ПІБ", "Організація"],
                           ["Коваленко Іван Петрович", "ТОВ Ромашка"],
                           ["Шевченко Ольга Ігорівна", "ПАТ Мрія"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "id", "schema": "Person", "keys": ["c0"]},
                      {"key": "id", "schema": "Organization", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "id|Person:name", "why": "people"},
          {"column": "c1", "binding": "id_2|Organization:name", "why": "orgs"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)

    # BOTH ENTITIES SURVIVE, under distinct keys, and the rename is on record.
    assert [e["key"] for e in plan.entities] == ["id", "id_2"]
    assert [e["schema"] for e in plan.entities] == ["Person", "Organization"]
    assert plan.key_renames == [{"key": "id", "renamed_to": "id_2",
                                 "schema": "Organization"}]

    # ...and both reach the model: the closed enum offers Person pairs, which
    # under the old `declared` it could not, and the prompt names both.
    binding_prompt = [p for p in client.prompts if BINDING_MARKER in p][0]
    assert "  id (Person), identified by c0 (ПІБ)" in binding_prompt
    assert "  id_2 (Organization), identified by c1 (Організація)" in binding_prompt
    enums = [b["properties"]["binding"]["enum"]
             for b in client.schemas[-1]["properties"]["bindings"]["prefixItems"]]
    assert "id|Person:name" in enums[0]
    assert "id_2|Organization:name" in enums[1]

    # ...and nothing the model then answered is rejected for a schema mismatch,
    # which is the disagreement this collision used to cause.
    v = validate(plan, profiles, CAT, CFG, frame)
    assert [d for d in v.decisions if "does not carry" in d.reason] == []
    assert {b["prop"] for b in v.bindings} == {"Person:name", "Organization:name"}
    assert {b["entity"] for b in v.bindings} == {"id", "id_2"}
    # The rename is recorded once, by the stage that writes decisions.jsonl,
    # even though it was resolved a stage earlier.
    renamed = [d for d in v.decisions if "two entities declared the key" in d.reason]
    assert len(renamed) == 1 and renamed[0].entity == "id"


def test_the_edge_call_sees_both_colliding_entities_as_endpoints():
    """Passing the entities as bare keys hid the duplicate in a list the same
    way `declared` hid it in a dict: the enum offered one key twice and one
    entity not at all, so an edge between the two things the row describes was
    not expressible. The combinations now carry each entity's schema too, so
    the collision cannot flatten them."""
    frame, profiles = _fp([["ПІБ", "Організація"],
                           ["Коваленко Іван Петрович", "ТОВ Ромашка"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "id", "schema": "Person", "keys": ["c0"]},
                      {"key": "id", "schema": "Organization", "keys": ["c1"]}],
         "edges": [{"key": "member", "edge": "Membership|id|id_2"}]},
        [[{"column": "c0", "binding": "id|Person:name", "why": "people"},
          {"column": "c1", "binding": "id_2|Organization:name", "why": "orgs"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    edge_schema_sent = client.schemas[[BINDING_MARKER not in p and EDGE_MARKER in p
                                       for p in client.prompts].index(True)]
    combos = (edge_schema_sent["properties"]["edges"]["items"]
              ["properties"]["edge"]["enum"])
    assert "Membership|id|id_2" in combos
    # Both keys appear as endpoints; neither is hidden by the collision.
    assert {c.split("|")[1] for c in combos} | {c.split("|")[2] for c in combos} \
        == {"id", "id_2"}
    edge_prompt_text = [p for p in client.prompts if EDGE_MARKER in p][0]
    assert "id_2  schema Organization" in edge_prompt_text
    # The endpoint range is satisfied because the endpoints are the schemata
    # the model declared, not whichever one survived the collision.
    v = validate(plan, profiles, CAT, CFG, frame)
    assert [e["key"] for e in v.edges] == ["member"]


def test_the_structure_grammar_closes_keys_to_the_live_column_ids():
    """`keys` names columns, so it is an enum of the columns this frame has —
    and of the LIVE ones only: a column with no values cannot identify
    anything, and offering it would be offering a key the key-quality check
    must then reject."""
    frame, profiles = _fp([["id", "ПІБ", "Порожня"],
                           ["ua-1", "Коваленко Іван Петрович", None],
                           ["ua-2", "Шевченко Ольга Ігорівна", None]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "unmapped", "why": "identifier"},
          {"column": "c1", "binding": "person|Person:name", "why": "names"}]],
    )
    propose(frame, profiles, CAT, CFG, client)
    structure_sent = client.schemas[0]
    keys = structure_sent["properties"]["entities"]["items"]["properties"]["keys"]
    assert keys["items"]["enum"] == ["c0", "c1"]  # c2 is empty, so not offered


def test_a_key_column_named_by_its_header_is_read_as_that_column():
    """The measured failure on `ua_war_sanctions.targets.simple.csv`: the model
    answered `"keys": ["id"]` — the HEADER of `c0` — for both declared
    entities. `validate` rejected every key with "key column does not exist",
    each entity fell back to the row ordinal, and the run emitted 5 621 Person
    plus 5 621 Organization, one of each per row, deduplicating nothing.

    The grammar now makes that answer unproduceable; this is the reading side,
    for a client the grammar does not constrain."""
    frame, profiles = _fp([["id", "ПІБ"],
                           ["ua-1", "Коваленко Іван Петрович"],
                           ["ua-2", "Шевченко Ольга Ігорівна"],
                           ["ua-3", "Мельник Петро Іванович"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["id"]}],
         "edges": []},
        [[{"column": "c0", "binding": "unmapped", "why": "identifier"},
          {"column": "c1", "binding": "person|Person:name", "why": "names"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.entities[0]["keys"] == ["c0"]
    assert plan.key_columns_resolved == [
        {"key": "person", "given": "id", "column": "c0"}]

    # ...and the key survives validation, which is the whole point: the entity
    # is keyed on its identifier, not on the row ordinal.
    v = validate(plan, profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == ["c0"]
    assert [d for d in v.decisions if d.reason == "key column does not exist"] == []
    # The repair is on record, once, in the file that records decisions.
    read = [d for d in v.decisions if "is the header of exactly one column" in d.reason]
    assert len(read) == 1
    assert read[0].column == "c0" and read[0].entity == "person"


def test_a_header_shared_by_two_columns_is_not_guessed_at():
    """A header naming two columns is not a reference. Keying the entity on
    whichever one happened to come first would be a silent wrong answer, so the
    key stays as given and is rejected as the non-existent column it is."""
    frame, profiles = _fp([["код", "код", "ПІБ"],
                           ["1", "a", "Коваленко Іван Петрович"],
                           ["2", "b", "Шевченко Ольга Ігорівна"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["код"]}],
         "edges": []},
        [[{"column": "c0", "binding": "unmapped", "why": "x"},
          {"column": "c1", "binding": "unmapped", "why": "x"},
          {"column": "c2", "binding": "person|Person:name", "why": "names"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.entities[0]["keys"] == ["код"]
    assert plan.key_columns_resolved == []
    v = validate(plan, profiles, CAT, CFG, frame)
    assert v.entities[0]["keys"] == []
    assert [d.reason for d in v.decisions if d.column == "код"] == [
        "key column does not exist"]


def test_a_lone_entity_contradicting_the_subject_does_not_own_the_candidates():
    """Measured on the МВС vehicle registry `reestrtz_2026_sample.csv`
    (39 607 rows, 17 columns). The structure call answered `{"subject":
    "Vehicle", "entities": [{"key": "c6", "schema": "Organization", "keys":
    ["c6"]}]}` — one question answered two ways in one call — and everything
    downstream is built from the ENTITY, so every (entity, property) pair the
    grammar offered for all 17 columns was `Organization:*`. The model then
    refused all 17 ("Values are VINs (Vehicle Identification Numbers), not
    Organization IDs"): 0 mapped columns, 0 statements from 39 607 rows,
    39 406 `Organization` entities with `properties: {}`.

    With one entity declared, subject and schema ARE the same question, so the
    subject is declared as an entity too and the candidate list can no longer
    come from one unchecked answer."""
    # «Назва» rather than the registry's `BRAND`: a header no spelling matches
    # retrieves nothing once the lexical floor is in (2026-09-03), and the
    # test needs one column both readings can claim — `name`, which
    # Organization and Vehicle both carry.
    frame, profiles = _fp([["VIN", "Назва", "MODEL"],
                           ["Y6DA69700A0000199", "ЗАЗ", "SENS"],
                           ["XW8ZZZ61ZDG000123", "SKODA", "OCTAVIA"]])
    client = FakeClient(
        {"subject": "Vehicle",
         "entities": [{"key": "c0", "schema": "Organization", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "unmapped", "why": "x"},
          {"column": "c1", "binding": "unmapped", "why": "x"},
          {"column": "c2", "binding": "vehicle|Vehicle:model", "why": "models"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)

    # BOTH READINGS ARE ON THE PLAN, the promoted one keyed like the entity it
    # was declared beside, and the DECLARED subject is untouched because it is
    # what made the disagreement visible at all.
    assert [(e["key"], e["schema"]) for e in plan.entities] == [
        ("c0", "Organization"), ("vehicle", "Vehicle")]
    assert plan.entities[1]["keys"] == ["c0"]
    assert plan.subject == "Vehicle"
    assert plan.subject_contradictions == [
        {"key": "c0", "declared": "Organization", "subject": "Vehicle",
         "added": "vehicle"}]
    # The pair is two readings of one row, not two things on it, so nothing is
    # asked about what connects them: structure, then bindings, no edge call.
    assert [p for p in client.prompts if EDGE_MARKER in p] == []

    # THE CANDIDATE LIST THE MODEL WAS ACTUALLY GIVEN, read off the grammar
    # itself rather than off the plan: the vehicle's own properties are in it,
    # which is what all 17 columns were refused for want of.
    binding_grammar = [s for p, s in zip(client.prompts, client.schemas)
                       if BINDING_MARKER in p][0]
    enums = [b["properties"]["binding"]["enum"]
             for b in binding_grammar["properties"]["bindings"]["prefixItems"]]
    offered = {pair for enum in enums for pair in enum if pair != "unmapped"}
    assert offered, "every column was offered nothing but `unmapped`"
    assert [q for q in offered if q.startswith("vehicle|Vehicle:")]
    assert any("vehicle|Vehicle:model" in enum for enum in enums)
    # ...and the entity the model DID declare is still offered, because which
    # of the two readings is right is exactly what is not being decided here.
    assert [q for q in offered if q.startswith("c0|")]
    binding_prompt = [p for p in client.prompts if BINDING_MARKER in p][0]
    assert "  c0 (Organization), identified by c0 (VIN)" in binding_prompt
    assert "  vehicle (Vehicle), identified by c0 (VIN)" in binding_prompt

    # ...and the answer survives validation, which under the contradiction it
    # could not: a Vehicle property on an entity declared Organization is
    # exactly what `_structural_check` rejects.
    v = validate(plan, profiles, CAT, CFG, frame)
    assert [d for d in v.decisions if "does not carry" in d.reason] == []
    assert [(b["entity"], b["prop"]) for b in v.bindings] == [
        ("vehicle", "Vehicle:model")]

    # THE READING THE EVIDENCE NEVER USED DOES NOT REACH THE OUTPUT. This is
    # where the 39 406 property-less `Organization` entities came from.
    assert [e["key"] for e in v.entities] == ["vehicle"]
    dropped = [d for d in v.decisions if "no accepted binding" in d.reason]
    assert len(dropped) == 1 and dropped[0].entity == "c0"

    # The promotion is recorded once, by the stage that writes decisions.jsonl,
    # even though it was resolved a stage earlier.
    said = [d for d in v.decisions if "its only declared entity was" in d.reason]
    assert len(said) == 1
    assert said[0].entity == "vehicle" and said[0].prop == "Vehicle"
    assert "Organization" in said[0].reason and said[0].verdict == "accepted"


def test_the_declared_entity_survives_a_contradicting_subject_it_outweighs():
    """The other direction of the same failure, and the reason no winner is
    picked: the structure call answered subject `Audio` for a deputies table on
    the live server three separate times, with the entities right. The promoted
    reading is offered and simply loses — the columns bind to the entity the
    model declared, and the unused half is dropped instead of being written as
    one property-less entity per row."""
    frame, profiles = _fp([["ПІБ", "Дата народження"],
                           ["Коваленко Іван Петрович", "17.09.1980"],
                           ["Шевченко Ольга Ігорівна", "01.02.1990"]])
    client = FakeClient(
        {"subject": "Audio",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": "person|Person:birthDate", "why": "dates"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e["schema"] for e in plan.entities] == ["Person", "Audio"]
    v = validate(plan, profiles, CAT, CFG, frame)
    assert {b["prop"] for b in v.bindings} == {"Person:name", "Person:birthDate"}
    assert [e["key"] for e in v.entities] == ["person"]
    dropped = [d for d in v.decisions if "no accepted binding" in d.reason]
    assert len(dropped) == 1 and dropped[0].entity == "audio"


def test_a_subject_that_refines_the_declared_entity_is_not_a_contradiction():
    """`Company` under a declared `Organization` is one answer refined, not two
    answers. The entity stands alone, because a second reading here would spend
    half the candidate slots restating the first one."""
    frame, profiles = _fp([["Назва", "ЄДРПОУ"],
                           ["ТОВ Ромашка", "12345678"],
                           ["ПАТ Мрія", "87654321"]])
    client = FakeClient(
        {"subject": "Company",
         "entities": [{"key": "org", "schema": "Organization", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "org|Organization:name", "why": "names"},
          {"column": "c1", "binding": "unmapped", "why": "x"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e["schema"] for e in plan.entities] == ["Organization"]
    assert plan.subject_contradictions == []


def test_a_multi_entity_plan_keeps_a_subject_no_declared_entity_has():
    """A row holding a person, an organization and the membership between them
    is mainly about none of the three on its own, so a subject matching no
    declared entity's schema is the ORDINARY multi-entity reading and not a
    contradiction. Promoting it here would add a third entity to a
    decomposition that is already right."""
    frame, profiles = _fp([["ПІБ", "Організація"],
                           ["Коваленко Іван Петрович", "ТОВ Ромашка"],
                           ["Шевченко Ольга Ігорівна", "ПАТ Мрія"]])
    client = FakeClient(
        {"subject": "Membership",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Organization", "keys": ["c1"]}],
         "edges": [{"key": "member", "schema": "Membership",
                    "source": "org", "target": "person"}]},
        [[{"column": "c0", "binding": "person|Person:name", "why": "people"},
          {"column": "c1", "binding": "org|Organization:name", "why": "orgs"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e["schema"] for e in plan.entities] == ["Person", "Organization"]
    assert plan.subject_contradictions == []
    v = validate(plan, profiles, CAT, CFG, frame)
    assert [d for d in v.decisions if "its only declared entity was" in d.reason] == []
    assert {b["prop"] for b in v.bindings} == {"Person:name", "Organization:name"}


def test_a_keyless_entity_still_says_which_column_it_came_from():
    """Measured on the ship register: four of five entities reached the edge
    call as bare "no key column", though their headers are `ПІБ`, `Назва (юр)`,
    `Назва юр. (фрахтувальник)` — an owner and a charterer, each written twice.
    Unable to tell them apart the model related the owner-person to the
    owner-company and produced six Directorship and Employment edges while
    `Ownership|owner|vessel` sat on the offered list unchosen."""
    from ftmap.plan.prompt import edge_prompt

    frame, profiles = _fp([["Назва судна", "ПІБ", "Назва (юр)"],
                           ["Нептун", "Коваленко Іван", "ТОВ Ромашка"]])
    entities = [{"key": "c0", "schema": "Vessel", "keys": ["c0"]},
                {"key": "c1", "schema": "Person", "keys": []},
                {"key": "c2", "schema": "Organization", "keys": []}]
    text = edge_prompt(entities, profiles)
    assert "c1  schema Person  no key column; named after c1 (ПІБ)" in text
    assert "c2  schema Organization  no key column; named after c2 (Назва (юр))" in text
    # A key column, where one survived, still wins over the fallback.
    assert "c0  schema Vessel  identified by c0 (Назва судна)" in text


def test_an_entity_named_after_nothing_says_so_rather_than_inventing():
    from ftmap.plan.prompt import edge_prompt

    frame, profiles = _fp([["Назва судна", "ПІБ"], ["Нептун", "Коваленко Іван"]])
    text = edge_prompt([{"key": "vessel", "schema": "Vessel", "keys": []},
                        {"key": "owner", "schema": "Person", "keys": []}], profiles)
    assert "vessel  schema Vessel  no key column" in text
    assert "named after" not in text


ROW_KIND_ROWS = [["name", "schema"],
                 ["Коваленко Іван Петрович", "Person"],
                 ["ТОВ Ромашка", "Organization"],
                 ["Шевченко Ольга Іванівна", "Person"],
                 ["ПрАТ Мрія", "Organization"]]


class SplitClient(FakeClient):
    """A `FakeClient` that also answers the row-selection question."""

    def __init__(self, structure, batches, selections):
        super().__init__(structure, batches)
        self.selections = selections

    def complete(self, system, user, schema, max_tokens=None):
        if SPLIT_MARKER in user:
            self.prompts.append(user)
            self.schemas.append(schema)
            self.max_tokens_seen.append(max_tokens)
            return {"selections": self.selections}
        return super().complete(system, user, schema, max_tokens)


def test_two_entities_on_one_key_are_asked_which_rows_each_is_built_from():
    """The polymorphic table, and the ONLY shape this question is asked of.

    `us_ofac_sdn` declares five schemata keyed on one column; without a row
    selection each is built from every row and the file emits 20 054 of each
    from 20 079 rows, 976 of which are the wallets one of them claims all of.
    """
    frame, profiles = _fp(ROW_KIND_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Organization", "keys": ["c0"]}],
         "edges": []},
        # One batch per bloc round — the selections turn the binding stage
        # into a round per bloc (see the per-bloc test below).
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": UNMAPPED, "why": "the kind, not data"}],
         [{"column": "c0", "binding": "org|Organization:name", "why": "names"},
          {"column": "c1", "binding": UNMAPPED, "why": "the kind, not data"}]],
        [{"entity": "person", "filter": "c1=Person", "why": "the person rows"},
         {"entity": "org", "filter": "c1=Organization", "why": "the rest"}],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e.get("filter") for e in plan.entities] == [
        {"column": "c1", "value": "Person"},
        {"column": "c1", "value": "Organization"}]


def test_a_polymorphic_source_is_asked_per_bloc_with_its_own_rows():
    """The four refusals in `2026-08-20-negative-results.md` share one
    trigger: the binding call sees a column whose samples mix kinds, and
    every phrasing that explained the mixing made the refusal more
    coherent. So the question changes instead — one binding call per bloc,
    the roster holding that bloc alone, the samples drawn from that bloc's
    own rows. From inside the call the table is monomorphic, which is the
    shape the model answers well. See
    the per-bloc binding design note (2026-08-31)."""
    frame, profiles = _fp(ROW_KIND_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Organization", "keys": ["c0"]}],
         "edges": []},
        # One batch per bloc round, in declaration order.
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": UNMAPPED, "why": "the kind column"}],
         [{"column": "c0", "binding": "org|Organization:name", "why": "names"},
          {"column": "c1", "binding": UNMAPPED, "why": "the kind column"}]],
        [{"entity": "person", "filter": "c1=Person", "why": "the person rows"},
         {"entity": "org", "filter": "c1=Organization", "why": "the rest"}],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    binding_prompts = [p for p in client.prompts if BINDING_MARKER in p]
    assert len(binding_prompts) == 2
    person_call, org_call = binding_prompts
    # The bloc's roster holds the bloc, not its siblings.
    assert "person (Person)" in person_call
    assert "org (Organization)" not in person_call
    assert "person (Person)" not in org_call
    # And the samples are the bloc's own rows.
    assert "Коваленко Іван Петрович" in person_call
    assert "ТОВ Ромашка" not in person_call
    assert "ТОВ Ромашка" in org_call
    assert "Коваленко Іван Петрович" not in org_call
    # Both blocs' answers survive side by side — column -> bindings.
    got = {(b["column"], b["entity"]): b["prop"] for b in plan.bindings
           if b.get("prop") and b["prop"] != UNMAPPED}
    assert got[("c0", "person")] == "Person:name"
    assert got[("c0", "org")] == "Organization:name"


def test_a_kind_column_spelled_in_ftm_names_derives_the_blocs_without_a_model():
    """OFAC writes `Person`, `Organization`, `Vessel` — FollowTheMoney's own
    vocabulary — in its schema column, and the model's structure call reads
    the file as one Person over 20 079 mixed rows whenever the lineage
    lands cautious. Nothing about the blocs is the model's to guess there:
    when exactly one party is declared and a kind column's every value IS a
    concrete FtM schema name, the blocs are derived — one entity per value,
    the party's keys, filtered on that value — and the split call is not
    spent. The guard is strict on purpose: a status column («чинна»,
    «анульована») names no schema, so the licences register that the
    wide-filter arm destroyed can never qualify."""
    frame, profiles = _fp(ROW_KIND_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": UNMAPPED, "why": "the kind column"}],
         [{"column": "c0", "binding": "person_2|Organization:name", "why": "names"},
          {"column": "c1", "binding": UNMAPPED, "why": "the kind column"}]],
        [],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert not any(SPLIT_MARKER in p for p in client.prompts)
    got = {(e["schema"], (e.get("filter") or {}).get("value"))
           for e in plan.entities}
    assert got == {("Person", "Person"), ("Organization", "Organization")}
    assert all(e["keys"] == ["c0"] for e in plan.entities)
    # And the binding stage ran per bloc.
    assert len([p for p in client.prompts if BINDING_MARKER in p]) == 2


DERIVE_KEY_ROWS = [["id", "name", "schema"],
                   ["NK-1001", "Коваленко Іван", "Person"],
                   ["NK-1002", "ТОВ Ромашка", "Organization"],
                   ["NK-1003", "Коваленко Іван", "Person"],
                   ["NK-1004", "ПрАТ Мрія", "Organization"]]


def test_the_blocs_are_derived_whatever_parties_the_model_declared():
    """Shown the schema glossary (`work-x7`, 2026-09-03), the model read OFAC
    as a Person keyed on the name and a second Person keyed on the aliases —
    two unfiltered parties — and the derivation, gated on exactly one,
    stepped aside: 20 054 Persons, every bloc missing, every designation
    with them. The file's own kind column outranks how many readings of the
    protagonist the model wrote: the first lends its key, the rest are
    replaced, things that are not parties stay."""
    frame, profiles = _fp(DERIVE_KEY_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c1"]},
                      {"key": "other", "schema": "Person", "keys": ["c1"]},
                      {"key": "note", "schema": "Note", "keys": []}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:idNumber", "why": "x"},
          {"column": "c1", "binding": "person|Person:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}],
         [{"column": "c0", "binding": "person_2|Organization:idNumber", "why": "x"},
          {"column": "c1", "binding": "person_2|Organization:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}]],
        [],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert not any(SPLIT_MARKER in p for p in client.prompts)
    got = [(e["schema"], (e.get("filter") or {}).get("value"), e["keys"])
           for e in plan.entities]
    assert ("Person", "Person", ["c0"]) in got
    assert ("Organization", "Organization", ["c0"]) in got
    assert not any(e["key"] == "other" for e in plan.entities)
    assert any(e["schema"] == "Note" for e in plan.entities)


def test_a_row_selection_the_model_wrote_is_not_second_guessed():
    """When the structure call already filtered its entities on the kind
    column, the question is answered; derivation does not re-ask it."""
    frame, profiles = _fp(DERIVE_KEY_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "p", "schema": "Person", "keys": ["c0"],
                       "filter": {"column": "c2", "value": "Person"}},
                      {"key": "o", "schema": "Organization", "keys": ["c0"],
                       "filter": {"column": "c2", "value": "Organization"}}],
         "edges": []},
        [[{"column": "c0", "binding": "p|Person:idNumber", "why": "x"},
          {"column": "c1", "binding": "p|Person:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}],
         [{"column": "c0", "binding": "o|Organization:idNumber", "why": "x"},
          {"column": "c1", "binding": "o|Organization:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}]],
        [],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e["key"] for e in plan.entities] == ["p", "o"]


def test_a_derived_bloc_keys_on_the_file_s_own_identifier():
    """The key clause bought +6 keys agreeing and was refused a THIRD time
    (`work-full-keyc`): told to prefer a code, the model declared the
    id-keyed blocs itself, bypassed derivation, and dropped the Sanction —
    every designation went with it. The keys the clause bought are had
    deterministically instead, exactly where derivation already fires: a
    derived bloc keys on the most distinct filled column (ties to the
    leftmost), which on both polymorphic files is the id column the etalons
    key on — and the land-valuers trap that refused the general key-upgrade
    rule can never reach this path, because no certificate register spells
    its kinds in FollowTheMoney vocabulary. The party's own keys stay only
    when no column qualifies."""
    frame, profiles = _fp(DERIVE_KEY_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:idNumber", "why": "x"},
          {"column": "c1", "binding": "person|Person:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}],
         [{"column": "c0", "binding": "person_2|Organization:idNumber", "why": "x"},
          {"column": "c1", "binding": "person_2|Organization:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}]],
        [],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert all(e["keys"] == ["c0"] for e in plan.entities), \
        [(e["key"], e["keys"]) for e in plan.entities]


def test_a_plan_whose_entities_do_not_share_a_key_is_never_asked():
    """A question asked everywhere is answered everywhere. Measured
    2026-08-28: offered on every entity, the model filtered the transport
    licences on their status column and emitted 0 of 71 306, and kept 2 820 of
    24 650 court decisions. Two entities that key on DIFFERENT columns are two
    different things in one row, not two readings of one thing."""
    # A kind column in the register's OWN words, not FollowTheMoney's — so
    # the derivation above cannot fire and the split call alone is in play,
    # which is what this test is about.
    frame, profiles = _fp([["name", "вид"],
                           ["Коваленко Іван Петрович", "фізична особа"],
                           ["ТОВ Ромашка", "юридична особа"],
                           ["Шевченко Ольга Іванівна", "фізична особа"],
                           ["ПрАТ Мрія", "юридична особа"]])
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "kind", "schema": "Organization", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": "kind|Organization:name", "why": "x"}]],
        [],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [e.get("filter") for e in plan.entities] == [None, None]
    assert not any(SPLIT_MARKER in p for p in client.prompts)


def test_a_subject_no_declared_entity_supports_is_replaced_by_the_roster_s():
    """The subject is one answer and the roster is another, and when they
    disagree completely the roster is the one with evidence behind it.

    `resolve_subject_contradiction` handles exactly one entity — with two or
    more, a subject naming neither is "the ordinary multi-entity reading". It
    is not, when NO declared entity is related to it: the subject is then a
    schema the plan itself says the row is not.

    Measured on `us_ofac_sdn` 2026-08-29: the structure call declared Person,
    Organization, Vessel, Airplane and CryptoWallet — the five blocs the
    etalon asks for — and a subject of `Audio`. The binding prompt names the
    subject, every one of the sixteen columns came back `unmapped`, all five
    entities were dropped for carrying no property, and the source was
    declined outright: 6 correct bindings and 1 matched entity became 0.
    """
    frame, profiles = _fp([["ПІБ", "Компанія"],
                           ["Коваленко Іван Петрович", "ТОВ Альфа"],
                           ["Шевченко Ольга Іванівна", "ПрАТ Бета"]])
    client = FakeClient(
        {"subject": "Audio",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Organization", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": "org|Organization:name", "why": "orgs"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.subject == "Person"
    binding_prompt = [p for p in client.prompts if BINDING_MARKER in p][0]
    assert "subject: Person" in binding_prompt
    assert plan.subject_replaced == {"declared": "Audio", "used": "Person"}


def test_a_subject_one_declared_entity_supports_is_left_alone():
    frame, profiles = _fp([["ПІБ", "Компанія"],
                           ["Коваленко Іван Петрович", "ТОВ Альфа"],
                           ["Шевченко Ольга Іванівна", "ПрАТ Бета"]])
    client = FakeClient(
        {"subject": "LegalEntity",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Organization", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "names"},
          {"column": "c1", "binding": "org|Organization:name", "why": "orgs"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.subject == "LegalEntity" and plan.subject_replaced is None


def test_one_relation_declared_three_times_is_one_edge():
    """Showing the edge call three records made it answer three times.

    Measured 2026-08-29: with records in the prompt the state-enterprise
    register declared `Ownership` between the same two entities three times
    over, the court procurement three `Representation`, the ICIJ addresses
    three `Documentation` — one per record shown. Every duplicate scores as an
    edge the etalon does not ask for, and one of those triples is a relation
    it does.

    An edge is (schema, source, target). Declaring it twice says nothing the
    first one did not.
    """
    frame, profiles = _fp([["ПІБ", "Компанія"],
                           ["Коваленко Іван Петрович", "ТОВ Альфа"],
                           ["Шевченко Ольга Іванівна", "ПрАТ Бета"]])
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Company", "keys": ["c1"]}],
         "edges": [{"key": "own", "edge": "Ownership|person|org"},
                   {"key": "own2", "edge": "Ownership|person|org"},
                   {"key": "own3", "edge": "Ownership|person|org"}]},
        [[{"column": "c0", "binding": "person|Person:name", "why": "x"},
          {"column": "c1", "binding": "org|Company:name", "why": "y"}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert [(e["schema"], e["source"], e["target"]) for e in plan.edges] == [
        ("Ownership", "person", "org")]


def test_a_relational_subject_supported_by_an_edge_is_not_replaced():
    """`Ownership` names what the row states, not what any single entity is;
    support checked against the entity roster alone could never see that, so
    a correct relational subject was clobbered by declaration order."""
    frame, profiles = _fp([["ПІБ", "Компанія"],
                           ["Коваленко Іван", "ТОВ Ромашка"],
                           ["Шевченко Ольга", "ТОВ Барвінок"]])
    client = FakeClient(
        {"subject": "Ownership",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "company", "schema": "Company", "keys": ["c1"]}],
         "edges": [{"key": "own", "edge": "Ownership|person|company"}]},
        [[{"column": "c0", "binding": "person|Person:name", "why": ""},
          {"column": "c1", "binding": "company|Company:name", "why": ""}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.subject == "Ownership"
    assert plan.subject_replaced is None


def test_a_replaced_subject_follows_the_key_evidence_not_declaration_order():
    """When the subject really is unsupported, its replacement is the entity
    whose key identifies rows — not whichever entity the model listed first.
    The office recurs; the person is one per row; the row is the person."""
    frame, profiles = _fp([["Орган", "ПІБ"],
                           ["ДПС", "Коваленко Іван"],
                           ["ДПС", "Шевченко Ольга"],
                           ["ДПС", "Мельник Петро"]])
    client = FakeClient(
        {"subject": "Audio",
         "entities": [{"key": "office", "schema": "Organization", "keys": ["c0"]},
                      {"key": "person", "schema": "Person", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "office|Organization:name", "why": ""},
          {"column": "c1", "binding": "person|Person:name", "why": ""}]],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert plan.subject == "Person"
    assert plan.subject_replaced == {"declared": "Audio", "used": "Person"}


KIND_THING_ROWS = [["id", "name", "schema"],
                   ["NK-1001", "Коваленко Іван", "Person"],
                   ["NK-1002", "Аврора", "Vessel"],
                   ["NK-1003", "Шевченко Ольга", "Person"],
                   ["NK-1004", "Нептун", "Vessel"]]


def test_an_unfiltered_thing_of_a_kind_the_column_names_is_its_own_bloc():
    """`ua_war_sanctions` (`work-x10b`): beside the four blocs the kind
    column derives, the model had declared an unfiltered `Vessel` on the
    same key. Not a party, so the derivation left it standing; its key
    collided with the derived `c0_3`; the LegalEntity bloc's binding round
    was built from a roster in which `c0_3` meant Vessel, and the 1 370-row
    bloc came out with no binding and was dropped as empty. A thing of a
    schema the kind column itself names is that bloc without its row
    selection — absorbed, recorded — and a derived key never collides with
    an entity that stays."""
    frame, profiles = _fp(KIND_THING_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c1"]},
                      {"key": "ship", "schema": "Vessel", "keys": ["c1"]},
                      {"key": "person_2", "schema": "Note", "keys": []}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:idNumber", "why": "x"},
          {"column": "c1", "binding": "person|Person:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}],
         [{"column": "c0", "binding": "person_3|Vessel:imoNumber", "why": "x"},
          {"column": "c1", "binding": "person_3|Vessel:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"}]],
        [],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    keys = [e["key"] for e in plan.entities]
    assert len(keys) == len(set(keys)), keys
    got = {e["key"]: (e["schema"], (e.get("filter") or {}).get("value"))
           for e in plan.entities}
    assert got == {"person": ("Person", "Person"),
                   "person_3": ("Vessel", "Vessel"),
                   "person_2": ("Note", None)}
    assert plan.absorbed == [{"key": "ship", "schema": "Vessel", "into": "person_3"}]
    # Every bloc round was asked about a roster in which each key meant one
    # schema: the Vessel bloc's round names person_3 as the Vessel.
    assert any("person_3" in p and "Vessel" in p for p in client.prompts)


SANCTION_ROWS = [["id", "name", "schema", "program"],
                 ["NK-1001", "Коваленко Іван", "Person", "UA-WS"],
                 ["NK-1002", "ТОВ Ромашка", "Organization", "UA-WS"],
                 ["NK-1003", "Шевченко Ольга", "Person", "UA-WS"],
                 ["NK-1004", "ПрАТ Мрія", "Organization", "UA-WS"]]


def test_two_intervals_on_one_key_are_not_asked_which_rows_they_are_built_from():
    """OFAC (`work-x11`): a Sanction and an Identification both keyed on the
    target's name are two facts about the row, not two readings of what the
    row is. Asked anyway, the model gave the Sanction to the Person rows
    and the Identification to the Organization rows, and every designation
    of an organisation, vessel or aircraft went unattached. The question is
    asked of Things; an Interval keeps the whole table."""
    frame, profiles = _fp(SANCTION_ROWS)
    client = SplitClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]},
                      {"key": "org", "schema": "Organization", "keys": ["c0"]},
                      {"key": "sanction", "schema": "Sanction", "keys": ["c1"]},
                      {"key": "ident", "schema": "Identification", "keys": ["c1"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:idNumber", "why": "x"},
          {"column": "c1", "binding": "person|Person:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"},
          {"column": "c3", "binding": "sanction|Sanction:programId", "why": "x"}],
         [{"column": "c0", "binding": "person_2|Organization:idNumber", "why": "x"},
          {"column": "c1", "binding": "person_2|Organization:name", "why": "x"},
          {"column": "c2", "binding": UNMAPPED, "why": "kind"},
          {"column": "c3", "binding": "sanction|Sanction:programId", "why": "x"}]],
        [{"entity": "sanction", "filter": "c2=Person", "why": "would be wrong"},
         {"entity": "ident", "filter": "c2=Organization", "why": "would be wrong"}],
    )
    plan = propose(frame, profiles, CAT, CFG, client)
    assert not any(SPLIT_MARKER in p for p in client.prompts)
    filters = {e["key"]: e.get("filter") for e in plan.entities}
    assert filters["sanction"] is None and filters["ident"] is None
    assert filters["person"] == {"column": "c2", "value": "Person"}


def test_a_headerless_column_with_almost_nothing_in_it_is_stray_cells_not_a_column():
    """The establishment workbook's third sheet, formatted out to Excel's last
    column: 16 384 columns, 12 headed, 7 139 with a value or a few in 2 398
    rows, every one listed to the structure call — 350 884 tokens against a
    32 768-token context (`2026-09-05-manual-corpus.md` §5). Such a column
    is answered without a call; a headed column, however sparse, is not."""
    rows = [["ПІБ", "Дата народження", None, "Примітка"]]
    for i in range(300):
        rows.append([f"Особа {i}", "17.09.1980", "x" if i == 7 else None,
                     "x" if i == 3 else None])
    frame, profiles = _fp(rows)
    client = FakeClient(
        {"subject": "Person",
         "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
         "edges": []},
        [[{"column": "c0", "binding": "person|Person:name", "why": "x"},
          {"column": "c1", "binding": "person|Person:birthDate", "why": "x"},
          {"column": "c3", "binding": "unmapped", "why": "x"}]])
    plan = propose(frame, profiles, CAT, CFG, client)
    structure = client.prompts[0]
    assert "\nc2:" not in structure and "\nc3:" in structure
    by_col = {b["column"]: b for b in plan.bindings}
    assert by_col["c2"]["prop"] == "unmapped"
    assert "stray cells" in by_col["c2"]["why"] and "1 of 300" in by_col["c2"]["why"]
    assert "stray" not in by_col["c3"]["why"]
    assert "stray_column_fill" in CFG.as_manifest()["plan"]
