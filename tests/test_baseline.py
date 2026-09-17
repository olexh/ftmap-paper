# tests/test_baseline.py
"""The binding-baseline arms: what each switch changes, and only that.

`docs/measurements/2026-09-10-binding-baselines.md` compares three binding
policies on the structure the control produced. These tests pin the rules
the arms are defined by — the heuristic's two tie rules, the catalogue's
shape, the replay gate — and the two refusals that keep a comparison from
being about something else: a heuristic run that could reach model repair,
and a report over a source set the protocol did not freeze.
"""

import dataclasses
import importlib.util
import json
import pathlib

import httpx
import pytest

from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import Grid
from ftmap.pipeline import _StageClock
from ftmap.plan.candidates import (candidate_pairs, catalogue_pairs,
                                   heuristic_choice)
from ftmap.plan.client import LlamaClient, ModelError, stage_of
from ftmap.plan.prompt import (BINDING_MARKER, REPAIR_MARKER, STRUCTURE_MARKER,
                               repair_prompt)
from ftmap.plan.propose import Plan, propose
from ftmap.plan.response_schema import UNMAPPED, valid_pairs
from ftmap.plan.validate import validate
from ftmap.profile.columns import profile_frame
from ftmap.vocab.catalogue import Catalogue

CFG = Config.load(None)
CAT = Catalogue.load()
HEURISTIC = dataclasses.replace(CFG, binding_mode="heuristic")
STATIC = dataclasses.replace(CFG, binding_mode="static")

REPO = pathlib.Path(__file__).resolve().parent.parent


def _fp(rows):
    f = build_frame(Grid(rows=rows, sheet="s", merges=[]), "/x.csv", "0" * 64,
                    "sid/s", CFG)
    return f, profile_frame(f, CFG)


def _report_module():
    spec = importlib.util.spec_from_file_location(
        "baseline_report", REPO / "tools" / "baseline_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_config_rejects_an_unknown_binding_mode(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text('[plan]\nbinding_mode = "lexical"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="binding_mode 'lexical'"):
        Config.load(str(p))
    p.write_text('[plan]\nbinding_mode = "heuristic-first"\n', encoding="utf-8")
    assert Config.load(str(p)).binding_mode == "heuristic-first"


def test_config_rejects_an_unknown_live_stage(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text('[model]\nlive_stages = ["binding", "bindings"]\n',
                 encoding="utf-8")
    with pytest.raises(ValueError, match="bindings"):
        Config.load(str(p))


def test_both_keys_are_echoed_into_the_manifest(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text('[plan]\nbinding_mode = "static"\n'
                 '[model]\nlive_stages = ["binding", "repair"]\n',
                 encoding="utf-8")
    m = Config.load(str(p)).as_manifest()
    assert m["plan"]["binding_mode"] == "static"
    assert m["model"]["live_stages"] == ["binding", "repair"]


# --------------------------------------------------------------------------
# The replay gate and the stage counters
# --------------------------------------------------------------------------

def _recording_transport(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "m", "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "choices": [{"message": {"content": '{"ok": true}'}}]})
    return httpx.MockTransport(handler)


def test_a_replay_fails_on_a_cache_miss_before_any_request(tmp_path):
    """`[model] live_stages = []` is the replay. A prompt the cache does not
    hold must raise here, not reach the server: a replay that generates one
    answer is a live run under a replay's name."""
    seen: list[dict] = []
    cfg = dataclasses.replace(CFG, model_revision="r", model_identity="m",
                              live_stages=())
    c = LlamaClient(cfg, cache_path=str(tmp_path / "prompts.jsonl"),
                    transport=_recording_transport(seen))
    with pytest.raises(ModelError, match="cache miss on the structure stage"):
        c.complete("sys", f"columns:\nc0: x\n{STRUCTURE_MARKER}", {"type": "object"})
    assert seen == []
    assert c.stages["structure"] == {"attempts": 1, "hits": 0, "generated": 0,
                                     "failures": 1, "prompt_tokens": 0,
                                     "completion_tokens": 0, "discarded": 0}
    c.close()


def test_a_permitted_stage_generates_and_the_rest_stay_closed(tmp_path):
    seen: list[dict] = []
    cfg = dataclasses.replace(CFG, model_revision="r", model_identity="m",
                              live_stages=("binding",))
    c = LlamaClient(cfg, cache_path=str(tmp_path / "prompts.jsonl"),
                    transport=_recording_transport(seen))
    assert c.complete("sys", f"{BINDING_MARKER}\nc0: x", {"type": "object"}) == {"ok": True}
    assert len(seen) == 1
    assert c.stages["binding"]["generated"] == 1
    assert c.stages["binding"]["prompt_tokens"] == 7
    # A repair prompt is a binding prompt with the repair marker, and it is
    # its own stage — closed here.
    with pytest.raises(ModelError, match="repair stage"):
        c.complete("sys", f"{REPAIR_MARKER}\n{BINDING_MARKER}", {"type": "object"})
    assert len(seen) == 1
    # And a hit on a closed stage is fine: it is what a replay is.
    assert c.complete("sys", f"{BINDING_MARKER}\nc0: x", {"type": "object"}) == {"ok": True}
    assert c.stages["binding"]["hits"] == 1
    c.close()


def test_the_stage_is_read_off_the_prompt_and_repair_comes_first():
    assert stage_of(repair_prompt([{"column": "c0", "prop": "Person:name",
                                    "reason": "x", "rejected": []}])) == "repair"
    assert stage_of(f"{STRUCTURE_MARKER}") == "structure"
    assert stage_of("user") == "other"


def test_the_stage_clock_records_deltas_and_a_mode():
    class Client:
        def __init__(self):
            self.stages = {"binding": {"attempts": 2, "hits": 2, "generated": 0,
                                       "failures": 0, "prompt_tokens": 0,
                                       "completion_tokens": 0, "discarded": 0}}
    client = Client()
    clock = _StageClock(client)
    clock.start("propose")
    client.stages["binding"]["attempts"] += 1
    client.stages["binding"]["hits"] += 1
    clock.stop()
    row = clock.stages["propose"]
    assert row["mode"] == "replay"
    assert row["calls"] == {"binding": {"attempts": 1, "hits": 1, "generated": 0,
                                        "failures": 0, "prompt_tokens": 0,
                                        "completion_tokens": 0, "discarded": 0}}
    clock.start("validate")
    clock.stop()
    assert clock.stages["validate"]["mode"] == "none"
    assert clock.stages["validate"]["calls"] == {}


# --------------------------------------------------------------------------
# Arm B: the heuristic
# --------------------------------------------------------------------------

def test_b_a_tie_between_distinct_properties_is_unmapped():
    """A date column with a header the lexicon does not know: every date
    property scores the type bonus and nothing else, so `birthDate` and
    `deathDate` tie, and the header cannot choose. Tie rule 1."""
    _, profiles = _fp([["xq", "ПІБ"], ["17.09.1980", "Коваленко Іван"],
                       ["01.02.1990", "Шевченко Ольга"]])
    pairs = ["person|Person:birthDate", "person|Person:deathDate"]
    pair, why = heuristic_choice(profiles[0], pairs, CAT, CFG)
    assert pair is None
    assert "2 distinct properties tie" in why


def test_b_prime_breaks_the_same_tie_by_offered_order():
    """`heuristic-first` is the heuristic with tie rule 1 inverted: the same
    tie that `heuristic` answers `unmapped` takes the first offered pair.
    Everything else — the scores, tie rule 2, the no-candidate answer — is
    the same code path."""
    _, profiles = _fp([["xq", "ПІБ"], ["17.09.1980", "Коваленко Іван"],
                       ["01.02.1990", "Шевченко Ольга"]])
    pairs = ["person|Person:deathDate", "person|Person:birthDate"]
    assert heuristic_choice(profiles[0], pairs, CAT, CFG)[0] is None
    pair, why = heuristic_choice(profiles[0], pairs, CAT, CFG, first_on_tie=True)
    assert pair == "person|Person:deathDate"
    assert "first of 2 distinct properties tying" in why
    # No candidate above zero is still `unmapped` under either.
    assert heuristic_choice(profiles[0], ["person|Person:nationality"], CAT, CFG,
                            first_on_tie=True)[0] is None
    # And the switch reaches it through propose.
    frame, profiles = _fp([["xq"], ["17.09.1980"], ["01.02.1990"]])

    class StructureOnly:
        def complete(self, system, user, schema, max_tokens=None):
            assert BINDING_MARKER not in user
            return {"subject": "Person",
                    "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}]}

    first = dataclasses.replace(CFG, binding_mode="heuristic-first")
    plan = propose(frame, profiles, CAT, first, StructureOnly())
    assert plan.bindings[0]["prop"] != UNMAPPED
    assert propose(frame, profiles, CAT, HEURISTIC, StructureOnly()).bindings[0]["prop"] == UNMAPPED


def test_b_one_property_on_several_targets_takes_the_first_offered():
    """`Person:name` on the owner and on the charterer score the same — the
    score reads the property and the column, not the entity — and the first
    in `valid_pairs` order wins, which is `round_declared` order. Tie rule 2,
    and the order it rests on is asserted, not assumed."""
    _, profiles = _fp([["ПІБ"], ["Коваленко Іван"], ["Шевченко Ольга"]])
    declared = {"charterer": "Person", "owner": "Person"}
    pairs = valid_pairs(["Person:name"], declared, CAT)
    assert pairs == ["charterer|Person:name", "owner|Person:name"]
    pair, why = heuristic_choice(profiles[0], pairs, CAT, CFG)
    assert pair == "charterer|Person:name"
    assert "first of 2 targets" in why
    # Declare them the other way round and the other one wins: the rule is
    # the order, not the name.
    swapped = valid_pairs(["Person:name"], {"owner": "Person", "charterer": "Person"}, CAT)
    assert heuristic_choice(profiles[0], swapped, CAT, CFG)[0] == "owner|Person:name"


def test_b_scores_an_inherited_property_under_the_entity_s_own_spelling():
    """Retrieval returned `Organization:name`; the declared supplier is a
    `LegalEntity`, which does not descend from Organization, so `valid_pairs`
    re-spells the candidate as `LegalEntity:name`. The heuristic scores the
    pair it was offered — `LegalEntity:name` inherits Thing's spellings, so
    «Назва» still matches — and binds it, without literal qname equality."""
    _, profiles = _fp([["Назва"], ["ТОВ Ромашка"], ["ПрАТ Мрія"]])
    pairs = valid_pairs(["Organization:name"], {"supplier": "LegalEntity"}, CAT)
    assert pairs == ["supplier|LegalEntity:name"]
    pair, why = heuristic_choice(profiles[0], pairs, CAT, CFG)
    assert pair == "supplier|LegalEntity:name"
    assert why.startswith("heuristic: score")


def test_b_asks_the_model_nothing_at_the_binding_stage():
    frame, profiles = _fp([["ПІБ", "Дата народження"],
                           ["Коваленко Іван", "17.09.1980"],
                           ["Шевченко Ольга", "01.02.1990"]])

    class StructureOnly:
        def __init__(self):
            self.prompts = []

        def complete(self, system, user, schema, max_tokens=None):
            self.prompts.append(user)
            if BINDING_MARKER in user:
                raise AssertionError("the heuristic arm asked the model to bind")
            return {"subject": "Person",
                    "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}]}

    client = StructureOnly()
    plan = propose(frame, profiles, CAT, HEURISTIC, client)
    assert len(client.prompts) == 1
    got = {b["column"]: b for b in plan.bindings}
    assert got["c0"]["prop"] == "Person:name"
    assert got["c1"]["prop"] == "Person:birthDate"
    assert all(b["why"].startswith("heuristic:") for b in plan.bindings)
    # The inputs the report compares are on the plan, pairs included.
    assert plan.rounds[0]["declared"] == [["person", "Person"]]
    assert plan.rounds[0]["columns"] == plan.rounds[0]["askable"] == ["c0", "c1"]


def test_b_cannot_reach_model_repair():
    """A phone column bound as a date fails the value check. Under the
    method that is a repair prompt; under the heuristic arm the model is
    never asked, whatever client the caller passed — the deterministic
    rejection stands and the column ends unmapped."""
    frame, profiles = _fp([["Дата"], ["79253902086"], ["79114585963"]])

    class Boom:
        def complete(self, *a, **kw):
            raise AssertionError("the heuristic arm reached model repair")

    plan = Plan(subject="Person",
                entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
                edges=[], bindings=[{"column": "c0", "prop": "Person:birthDate",
                                     "entity": "person", "why": "x"}],
                shortlists={})
    v = validate(plan, profiles, CAT, HEURISTIC, frame, client=Boom())
    assert v.bindings == []
    assert not any(d.decided_by == "model-repair" for d in v.decisions)
    assert any(d.verdict == "rejected" and d.column == "c0" for d in v.decisions)


def test_an_attachment_on_an_entity_dissolved_into_an_edge_goes_with_it():
    """Found by arm B-prime on the procurement plan: a rule attached
    `ContractAward:supplier` to a `ContractAward` declared as a thing, a
    later rule dissolved that thing into the row's `ContractAward` edge and
    re-homed its bindings — and the attachment kept pointing at the
    dissolved key, which `compile_mapping` then could not find. Every
    attachment must name declared keys when `validate` returns, and the
    plan must compile."""
    from ftmap.plan.compile import compile_mapping

    frame, profiles = _fp([
        ["місяць", "процедура", "назва закупівлі", "Код по ДК 021:2015",
         "Сума, грн.", "з ким укладено договір"],
        ["січень", "відкриті торги", "Папір офісний", "30190000-7", "12000",
         "ТОВ Ромашка"],
        ["лютий", "відкриті торги", "Картриджі", "30125100-2", "8400",
         "ПрАТ Мрія"],
        ["березень", "спрощена", "Послуги зв'язку", "64210000-1", "3100",
         "ТОВ Ромашка"],
    ])

    class Structure:
        def complete(self, system, user, schema, max_tokens=None):
            assert BINDING_MARKER not in user
            return {"subject": "ContractAward", "entities": [
                {"key": "c2", "keys": ["c2"], "schema": "ContractAward"},
                {"key": "c3", "keys": ["c3"], "schema": "ContractAward"},
                {"key": "c4", "keys": ["c4"], "schema": "ContractAward"},
                {"key": "c5", "keys": ["c5"], "schema": "LegalEntity"},
                {"key": "c0", "keys": ["c0"], "schema": "Event"},
                {"key": "c1", "keys": ["c1"], "schema": "ContractAward"}]}

    first = dataclasses.replace(CFG, binding_mode="heuristic-first")
    plan = propose(frame, profiles, CAT, first, Structure())
    v = validate(plan, profiles, CAT, first, frame)
    declared = {e["key"] for e in v.entities} | {e["key"] for e in v.edges}
    for a in v.attachments:
        assert a["entity"] in declared and a["target"] in declared, a
    compile_mapping(v, frame, "/tmp/x.csv")


# --------------------------------------------------------------------------
# Arm H: the structure without a model
# --------------------------------------------------------------------------

def test_config_rejects_an_unknown_structure_mode(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text('[plan]\nstructure_mode = "vote"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="structure_mode 'vote'"):
        Config.load(str(p))
    p.write_text('[plan]\nstructure_mode = "heuristic"\n', encoding="utf-8")
    cfg = Config.load(str(p))
    assert cfg.structure_mode == "heuristic"
    assert cfg.as_manifest()["plan"]["structure_mode"] == "heuristic"


H_ROWS = [["ПІБ", "Дата народження", "Код ЄДРПОУ", "Назва підприємства"],
          ["Коваленко Іван Петрович", "17.09.1980", "00032112", "ТОВ Ромашка"],
          ["Шевченко Ольга Іванівна", "01.02.1990", "32855961", "ПрАТ Мрія"],
          ["Бондаренко Петро Сергійович", "12.12.1975", "14360570", "ДП Світанок"]]


def test_the_subject_vote_is_independent_of_shortlist_size():
    """The vote is the max over a schema's FULL bindable property set, never
    off `shortlist_scored`, which truncates to `shortlist_size`. Three sizes,
    one of them the whole catalogue, must give a byte-identical vote."""
    from ftmap.plan.candidates import subject_vote
    _, profiles = _fp(H_ROWS)
    votes = [subject_vote(profiles, CAT, dataclasses.replace(CFG, shortlist_size=n))
             for n in (12, 24, 2641)]
    assert votes[0] == votes[1] == votes[2]
    assert votes[0]["Person"] > 0


def test_the_structure_rule_declares_one_row_entity_keyed_on_identifiers():
    from ftmap.plan.candidates import (ROW_KEY, STRUCTURE_EXCLUDED,
                                       heuristic_structure, identifier_keys)
    _, profiles = _fp(H_ROWS)
    answer, note = heuristic_structure(profiles, CAT, CFG)
    assert answer["entities"] == [{"key": ROW_KEY, "schema": answer["subject"],
                                   "keys": ["c2"]}]
    assert identifier_keys(profiles) == ["c2"]
    assert answer["subject"] not in STRUCTURE_EXCLUDED
    assert note["no_signal"] is False and note["subject"] == answer["subject"]
    # A tie, or no signal at all, falls to `concrete_schemata` order.
    _, blank = _fp([["xq", "zz"], ["a", "b"], ["c", "d"]])
    answer, note = heuristic_structure(blank, CAT, CFG)
    order = [s for s in CAT.concrete_schemata() if s not in STRUCTURE_EXCLUDED]
    assert note["no_signal"] is True
    assert answer["subject"] == order[0] == "Address"


def test_the_no_model_arm_makes_no_call_of_any_kind():
    """`structure_mode = "heuristic"` with `binding_mode = "heuristic"`: the
    structure, split, edge, binding and repair calls are all unreachable,
    and the rule's answer still takes the model's path — key resolution,
    bloc derivation, the validator — to a plan that compiles."""
    from ftmap.plan.compile import compile_mapping

    class Boom:
        def complete(self, *a, **kw):
            raise AssertionError("the no-model arm asked the model")

    frame, profiles = _fp(H_ROWS)
    cfg = dataclasses.replace(CFG, structure_mode="heuristic", binding_mode="heuristic")
    plan = propose(frame, profiles, CAT, cfg, Boom())
    assert plan.structure_rule["subject"] == plan.subject
    assert [e["key"] for e in plan.entities] == ["row"]
    assert plan.edges == []
    assert all(b["why"].startswith("heuristic:") or b["prop"] == UNMAPPED
               for b in plan.bindings)
    v = validate(plan, profiles, CAT, cfg, frame, client=Boom())
    compile_mapping(v, frame, "/tmp/x.csv")
    # And the polymorphic derivation still runs on the rule's one entity.
    frame, profiles = _fp(ROW_KIND_ROWS)
    plan = propose(frame, profiles, CAT, cfg, Boom())
    got = {(e["schema"], (e.get("filter") or {}).get("value")) for e in plan.entities}
    assert {("Person", "Person"), ("Organization", "Organization")} <= got


# --------------------------------------------------------------------------
# Arm C: the shared catalogue
# --------------------------------------------------------------------------

def test_c_catalogue_covers_every_declared_object_including_relationships():
    declared = {"person": "Person", "party": "Organization",
                "member": "Membership"}
    pairs = catalogue_pairs(declared, CAT)
    assert "member|Membership:role" in pairs
    assert "person|Person:birthDate" in pairs
    assert "party|Organization:registrationNumber" in pairs
    # Entity-typed properties are not bindable and are not offered.
    assert "member|Membership:organization" not in pairs
    assert not any(CAT.is_entity_reference(p.partition("|")[2]) for p in pairs)
    # Grouped by object in declared order, like `valid_pairs`.
    owners = [p.partition("|")[0] for p in pairs]
    assert owners == sorted(owners, key=list(declared).index)
    # Every column of a round gets this same list; the shortlist is ignored.
    assert candidate_pairs("static", ["Person:name"], declared, CAT) == pairs
    assert candidate_pairs("model", ["Person:name"], declared, CAT) \
        == valid_pairs(["Person:name"], declared, CAT)


ROW_KIND_ROWS = [["name", "schema"],
                 ["Коваленко Іван Петрович", "Person"],
                 ["ТОВ Ромашка", "Organization"],
                 ["Шевченко Ольга Іванівна", "Person"],
                 ["ПрАТ Мрія", "Organization"]]


class _BlocClient:
    """Structure once, then one binding batch per round, all `unmapped`."""

    def __init__(self):
        self.prompts = []
        self.schemas = []

    def complete(self, system, user, schema, max_tokens=None):
        self.prompts.append(user)
        self.schemas.append(schema)
        if BINDING_MARKER in user:
            cols = [line.split(":", 1)[0] for line in user.splitlines()
                    if line.startswith("c") and ":" in line
                    and not line.startswith("candidates")]
            return {"bindings": [{"column": c, "binding": UNMAPPED, "why": "x"}
                                 for c in dict.fromkeys(cols)]}
        return {"subject": "Person",
                "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}]}


def test_c_preserves_the_blocs_and_offers_each_round_its_own_catalogue():
    """A kind column spelled in FtM names derives two blocs, so the binding
    stage runs two rounds with different rosters. The static arm keeps both
    rounds and gives each its round's catalogue — the Person bloc's columns
    never see an Organization property — rather than one source-wide list.
    The rosters and askable columns are the control's, exactly."""
    frame, profiles = _fp(ROW_KIND_ROWS)
    control = propose(frame, profiles, CAT, CFG, _BlocClient())
    static = _BlocClient()
    plan = propose(frame, profiles, CAT, STATIC, static)
    assert len(plan.rounds) == len(control.rounds) == 2
    for got, want in zip(plan.rounds, control.rounds):
        assert got["declared"] == want["declared"]
        assert got["filter"] == want["filter"] == "c1"
        assert got["columns"] == want["columns"] == ["c0", "c1"]
        # A superset of what the control offered, per column — and the kind
        # column, which the control answered `unmapped` without a call for
        # want of a candidate, is askable under the catalogue. Retrieval
        # decides which columns are asked at all; removing it removes that.
        assert want["askable"] == ["c0"]
        assert got["askable"] == ["c0", "c1"]
        for col, pairs in want["pairs"].items():
            assert set(pairs) <= set(got["pairs"][col])
    person_round, org_round = plan.rounds
    assert {p.partition("|")[0] for pairs in person_round["pairs"].values()
            for p in pairs} == {"person"}
    assert {p.partition("|")[0] for pairs in org_round["pairs"].values()
            for p in pairs} == {"person_2"}
    assert person_round["pairs"]["c0"] == catalogue_pairs(
        dict(person_round["declared"]), CAT)
    # The prompt still lists candidates per column, in the control's format,
    # and the grammar offers the same list the prompt shows.
    binding_prompts = [p for p in static.prompts if BINDING_MARKER in p]
    assert len(binding_prompts) == 2
    assert "  candidates: person (Person) -> Person:" in binding_prompts[0]
    grammar = [s for s in static.schemas if "bindings" in s.get("properties", {})]
    offered = grammar[0]["properties"]["bindings"]["prefixItems"][0]["properties"]["binding"]["enum"]
    assert offered == [*person_round["pairs"]["c0"], UNMAPPED]


def test_c_repair_round_offers_the_catalogue_not_the_shortlist():
    frame, profiles = _fp([["Дата"], ["79253902086"], ["79114585963"]])
    seen = []

    class Repairer:
        def complete(self, system, user, schema, max_tokens=None):
            seen.append(schema)
            return {"bindings": [{"column": "c0", "binding": UNMAPPED, "why": "x"}]}

    plan = Plan(subject="Person",
                entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
                edges=[], bindings=[{"column": "c0", "prop": "Person:birthDate",
                                     "entity": "person", "why": "x"}],
                shortlists={"c0": ["Person:birthDate"]})
    validate(plan, profiles, CAT, STATIC, frame, client=Repairer())
    assert len(seen) == 1
    enum = seen[0]["properties"]["bindings"]["prefixItems"][0]["properties"]["binding"]["enum"]
    assert "person|Person:phone" in enum
    assert enum[:-1] == catalogue_pairs({"person": "Person"}, CAT)


# --------------------------------------------------------------------------
# The report's gate
# --------------------------------------------------------------------------

def test_the_report_refuses_a_mismatched_source_set():
    report = _report_module()
    expected = {"a#", "b#", "c#"}
    with pytest.raises(report.BaselineRefused, match="fixture does not name"):
        report.check_source_set(expected, {"a#", "b#", "d#"}, set(), "X")
    with pytest.raises(report.BaselineRefused, match="neither scored nor"):
        report.check_source_set(expected, {"a#", "b#"}, set(), "X")
    # Two runs over the same incomplete subset are still refused.
    with pytest.raises(report.BaselineRefused):
        report.check_source_set(expected, {"a#"}, set(), "X")
    # A source the run itself recorded as failed stays in the denominator.
    assert report.check_source_set(expected, {"a#", "b#"}, {"c#"}, "X") == {"c#"}
    assert report.check_source_set(expected, expected, set(), "X") == set()


def test_the_identity_check_compares_pairs_per_round_in_order():
    """A dict keyed by column alone let a later round overwrite an earlier
    one, so a changed first-round list passed as identical. The check keys
    on (round index, column) and compares ordered lists."""
    report = _report_module()

    def arm(name, mode, pairs):
        return {"name": name, "report": {"config": {"plan": {"binding_mode": mode}}},
                "sources": {"s#": {"failed": False, "profile_sha": "p",
                                   "inputs": {"rounds": []}, "pairs": pairs}}}

    control = arm("A", "model", {(0, "c0"): ["person|Person:name"],
                                 (1, "c0"): ["org|Organization:name"]})
    same = arm("B", "heuristic", {(0, "c0"): ["person|Person:name"],
                                  (1, "c0"): ["org|Organization:name"]})
    assert report.check_identical_inputs([control, same], "A")
    first_round_changed = arm("B", "heuristic",
                              {(0, "c0"): ["person|Person:alias"],
                               (1, "c0"): ["org|Organization:name"]})
    with pytest.raises(report.BaselineRefused, match="round 0 c0"):
        report.check_identical_inputs([control, first_round_changed], "A")
    control2 = arm("A", "model", {(0, "c0"): ["person|Person:name", "person|Person:alias"]})
    reordered = arm("B", "heuristic", {(0, "c0"): ["person|Person:alias", "person|Person:name"]})
    with pytest.raises(report.BaselineRefused, match="content or order"):
        report.check_identical_inputs([control2, reordered], "A")
    # A static arm must be a superset per (round, column), first round included.
    catalogue = arm("C", "static", {(0, "c0"): ["person|Person:alias"],
                                    (1, "c0"): ["org|Organization:name", "org|Organization:alias"]})
    with pytest.raises(report.BaselineRefused, match="round 0 c0"):
        report.check_identical_inputs([control, catalogue], "A")


def _etalon(entities, edges, columns):
    from ftmap.etalon.document import parse
    return parse({"etalon": 2,
                  "source": {"path": "x.csv", "sha256": "0" * 64, "sheet": ""},
                  "subject": {"answer": "Person"},
                  "entities": entities, "edges": edges, "columns": columns}, CAT)


def test_availability_folds_a_column_over_its_rounds_and_counts_it_once():
    """A column asked in two rounds is one column: offered if offered in
    any round, reachable if an acceptable property is among the declared
    objects' full property sets in any round. Three shapes: unreachable
    (no declared object carries the property), reachable but not offered
    (the shortlist cut it), and offered — in the second round only."""
    report = _report_module()
    doc = _etalon(
        [{"key": "person", "schema": "Person", "keys": ["c0"]},
         {"key": "org", "schema": "LegalEntity", "keys": ["c3"]},
         # The etalon declares the vessel; the arm's rounds never do.
         {"key": "vessel", "schema": "Vessel", "keys": ["c2"]}],
        [],
        [{"id": "c0", "header": "ПІБ", "role": "key+property", "answer": "Person:name"},
         {"id": "c1", "header": "дата", "role": "property", "answer": "Person:birthDate"},
         {"id": "c2", "header": "IMO", "role": "property", "answer": "Vessel:imoNumber"},
         {"id": "c3", "header": "назва", "role": "key+property",
          "answer": "LegalEntity:name"}])
    plan = {"rounds": [
        {"declared": [["person", "Person"]], "filter": "c9",
         "columns": ["c0", "c1", "c2"], "askable": ["c0", "c1", "c2"],
         "pairs": {"c0": ["person|Person:alias"],
                   "c1": ["person|Person:alias"],
                   "c2": ["person|Person:alias"]}},
        {"declared": [["org", "LegalEntity"]], "filter": "c9",
         "columns": ["c0", "c1", "c2", "c3"], "askable": ["c0", "c3"],
         "pairs": {"c0": ["org|LegalEntity:name"],
                   "c3": ["org|LegalEntity:name"]}},
    ]}
    offered = report.availability(doc, plan, CAT)
    within = report.reachable(doc, plan, CAT)
    # c0: `LegalEntity:name` is `Person:name` under `_same_property` (Person
    # descends from LegalEntity) — one property, one column, counted once,
    # offered in the second round only.
    assert offered == {"c0": True, "c1": False, "c2": False, "c3": True}
    # c1: Person carries birthDate, the shortlist cut it. c2: nothing
    # declared carries an IMO number.
    assert within == {"c0": True, "c1": True, "c2": False, "c3": True}
    assert len(offered) == 4 == len(within)
    # The chooser buckets, as observations.
    assert report.classify(True, ["Person:alias"], False) == "acceptable"
    assert report.classify(False, ["Person:name"], True) \
        == "chooser acceptable, removed or replaced by post-processing"
    assert report.classify(False, ["Person:alias"], False) == "chooser answered a different pair"
    assert report.classify(False, [], False) == "chooser answered unmapped"


def test_a_failed_source_counts_its_producible_entities_and_edges_as_missing():
    """The scorer's rule for a source that produced nothing: every
    producible entity is expected and missing; every edge with two
    producible endpoints, link or property, is expected and missing; an
    unproducible entity and the edge on it are outside both denominators."""
    report = _report_module()
    doc = _etalon(
        [{"key": "plane", "schema": "Airplane", "keys": ["c0"]},
         {"key": "operator", "schema": "LegalEntity", "keys": ["c1"]},
         {"key": "buyer", "schema": "LegalEntity", "producible": False,
          "why": "in a title cell"}],
        [{"key": "operated_by", "kind": "property", "prop": "Airplane:operator",
          "source": "plane", "target": "operator"},
         {"key": "owned", "schema": "Ownership", "source": "operator", "target": "plane"},
         {"key": "bought", "schema": "Ownership", "source": "buyer", "target": "plane"}],
        [{"id": "c0", "header": "борт", "role": "key+property",
          "answer": "Airplane:registrationNumber"},
         {"id": "c1", "header": "експлуатант", "role": "key+property",
          "answer": "LegalEntity:name"}])
    label = "x.csv"
    arm = {"name": "X", "report": {"config": {"plan": {}}},
           "sources": {label: {"failed": True, "score": None}}}
    lines = report.render_layers([arm], {label: doc})
    total = next(l for l in lines if l.startswith("| X |"))
    cells = [c.strip() for c in total.strip("|").split("|")]
    assert cells[1] == "0 / 0 / 1"      # subject: wrong
    assert cells[2] == "0 / 2"          # entities matched of expected
    assert cells[3] == "2 / 0 / 0"      # missing / extra / empty
    assert cells[6] == "0 / 2"          # relations found of expected
    assert cells[7] == "2 / 0 / 0"      # missing / extra / endpoints wrong


def test_the_frozen_fixture_matches_the_etalons_on_disk():
    report = _report_module()
    fixture = report.load_fixture()
    assert len(fixture["etalons"]) == 19
    report.check_etalons(str(REPO / "docs" / "measurements" / "etalon"), fixture)


# --------------------------------------------------------------------------
# The 2026-09-16 arms: no grammar (G), property-only pairing (P), no
# deterministic refinement (D)
# --------------------------------------------------------------------------

def test_config_refuses_an_unknown_refinement_mode(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text('[plan]\nrefinement = "some"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="refinement"):
        Config.load(str(p))
    p.write_text('[plan]\nrefinement = "none"\n[model]\ngrammar = false\n',
                 encoding="utf-8")
    cfg = Config.load(str(p))
    assert cfg.refinement == "none" and cfg.grammar is False
    assert cfg.as_manifest()["plan"]["refinement"] == "none"
    assert cfg.as_manifest()["model"]["grammar"] is False


def test_property_offers_drop_the_participant_and_the_rule_assigns_it():
    """Arm P: the model is offered `Person:name` once, not `officer|Person:name`
    and `office|PublicBody:name`; the answer goes to the first declared
    thing that can carry it, in declaration order."""
    from ftmap.plan.candidates import assign_property, property_offers
    declared = {"officer": "Person", "office": "PublicBody"}
    pairs = valid_pairs(["Person:name", "Thing:name", "PublicBody:jurisdiction",
                         "Person:birthDate"], declared, CAT)
    offers = property_offers(pairs)
    assert offers == candidate_pairs("property", ["Person:name", "Thing:name",
                                                  "PublicBody:jurisdiction",
                                                  "Person:birthDate"], declared, CAT)
    assert all("|" not in o for o in offers)
    assert offers == ["Person:name", "Person:jurisdiction", "Person:birthDate"]
    assert assign_property("Person:name", declared, CAT) == "officer|Person:name"
    assert assign_property("PublicBody:name", declared, CAT) == "officer|Person:name"
    # `jurisdiction` is a Thing property, so the officer takes it first;
    # a registration number is nobody's but the office's.
    assert assign_property("PublicBody:jurisdiction", declared, CAT) \
        == "officer|Person:jurisdiction"
    # A Person is a LegalEntity too, so only a property the first entity
    # cannot carry reaches the second.
    assert assign_property("Vessel:tonnage", declared, CAT) is None
    assert assign_property("Vessel:tonnage", {"owner": "Person", "ship": "Vessel"}, CAT) \
        == "ship|Vessel:tonnage"
    assert assign_property(UNMAPPED, declared, CAT) is None
    # A pair that already names its participant passes through.
    assert assign_property("office|PublicBody:name", declared, CAT) == "office|PublicBody:name"


def test_grammar_off_is_in_the_cache_key_only_when_off():
    from ftmap.plan.client import _key
    on = _key(CFG, "s", "u", {"type": "object"}, 10, None)
    assert on == _key(dataclasses.replace(CFG, grammar=True), "s", "u",
                      {"type": "object"}, 10, None)
    assert on != _key(dataclasses.replace(CFG, grammar=False), "s", "u",
                      {"type": "object"}, 10, None)


def test_a_free_answer_is_pruned_to_what_the_grammar_would_have_allowed():
    from ftmap.plan.admissible import Inadmissible, prune, strip_fences
    from ftmap.plan.response_schema import binding_schema, structure_schema
    declared = {"e": "Person"}
    schema = binding_schema(["c0", "c1"], {"c0": ["Person:name"],
                                            "c1": ["Person:birthDate"]}, declared, CAT)
    answer = {"bindings": [
        {"column": "c1", "binding": "e|Person:birthDate", "why": "x" * 200},
        {"column": "c0", "binding": "e|Person:nationality", "why": "not offered"},
        {"column": "c9", "binding": "unmapped", "why": "no such column"},
    ], "extra": 1}
    got, discarded = prune(answer, schema)
    # The out-of-enum binding and the unknown column are gone, the order the
    # model chose is kept, the long `why` is cut, the stray key dropped.
    assert [b["column"] for b in got["bindings"]] == ["c1"]
    assert len(got["bindings"][0]["why"]) == 160
    assert "extra" not in got and discarded == 3
    st = structure_schema(CAT, ["c0", "c1"])
    got, discarded = prune({"subject": "Person", "entities": [
        {"key": "a", "schema": "Person", "keys": ["c0", "id"]},
        {"key": "b", "schema": "NoSuchSchema", "keys": ["c1"]}]}, st)
    assert [e["key"] for e in got["entities"]] == ["a"]
    assert got["entities"][0]["keys"] == ["c0"] and discarded == 2
    with pytest.raises(Inadmissible):
        prune({"subject": "NoSuchSchema", "entities": []}, st)
    # A leaf the schema constrains by pattern is one dropped entry, not a
    # failed answer: the edge key `"5"` of the court procurement file.
    from ftmap.plan.response_schema import edge_schema
    ents = [{"key": "a", "schema": "Person", "keys": ["c0"]},
            {"key": "b", "schema": "Organization", "keys": ["c1"]}]
    es = edge_schema(CAT, ents, ["Membership|a|b"])
    got, discarded = prune({"edges": [{"key": "5", "edge": "Membership|a|b"},
                                      {"key": "m", "edge": "Membership|a|b"}]}, es)
    assert [e["key"] for e in got["edges"]] == ["m"] and discarded == 1
    with pytest.raises(Inadmissible):
        prune({"bindings": "not a list"}, schema)
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_no_refinement_runs_the_checks_and_none_of_the_additive_rules():
    """Arm D on the redacted-debtor fixture of `test_blocks`: under the
    method the rule splits the debtor into the coded organisation and the
    named person; under `refinement = "none"` the model's plan passes the
    checks as declared and nothing is added."""
    rows = [["tin_s", "name", "chief_name"]]
    people = ["Коваленко Іван Петрович", "Шевченко Ольга Іванівна", "Бондаренко Петро Ілліч",
              "Мельник Андрій Васильович", "Ткаченко Марія Олегівна", "Кравець Олег Юрійович"]
    rows += [["**********", name, ""] for name in people]
    rows += [[code, f"ТОВ «Ромашка {i}»", "Мельник Андрій"]
             for i, code in enumerate(("14360570", "00032112", "21560045"))]
    frame, profiles = _fp(rows)

    def plan():
        # Fresh each time: the coded-party rule specialises the entity in place.
        return Plan(subject="LegalEntity",
                    entities=[{"key": "debtor", "schema": "LegalEntity", "keys": ["c0"]}],
                    edges=[], shortlists={},
                    bindings=[{"column": "c1", "prop": "LegalEntity:name", "entity": "debtor", "why": "x"},
                              {"column": "c0", "prop": "LegalEntity:registrationNumber", "entity": "debtor", "why": "x"}])
    with_rules = validate(plan(), profiles, CAT, CFG, frame)
    assert {"debtor", "debtor_redacted"} <= {e["key"] for e in with_rules.entities}
    none = validate(plan(), profiles, CAT, dataclasses.replace(CFG, refinement="none"), frame)
    assert [(e["key"], e["schema"], e["keys"]) for e in none.entities] \
        == [("debtor", "LegalEntity", ["c0"])]
    assert {(b["column"], b["prop"]) for b in none.bindings} \
        == {("c1", "LegalEntity:name"), ("c0", "LegalEntity:registrationNumber")}
    assert not any(d.verdict == "accepted" and d.column is None and d.entity
                   for d in none.decisions if "declared" in d.reason), \
        [d for d in none.decisions if "declared" in d.reason]
