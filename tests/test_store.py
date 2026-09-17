# tests/test_store.py
import json
import os
import pathlib

import pytest

from ftmap.review.store import DecisionStore
from ftmap.workspace import create_workspace, open_workspace


def _fixture(tmp_path):
    """A real committed generation, built the way the pipeline builds one.

    Hand-laying files under `<out>/<safe_id>/` used to be enough, and it is
    exactly the shape the manifest replaced: that layout is a LEGACY workspace
    now, readable and read-only, so a store test that wrote into one would be
    testing a path production no longer takes.
    """
    workspace = create_workspace(str(tmp_path / "work"))
    d = pathlib.Path(workspace.staging_dir(label="sid__s"))
    (d / "summary.json").write_text(json.dumps({
        "safe_id": "sid__s", "path": "/corpus/a.csv", "subject": "Person",
        "columns": {"total": 2, "mapped": 1, "unmapped": 1},
        "entities": {"Person": 3},
    }), encoding="utf-8")
    (d / "profile.json").write_text(json.dumps([
        {"id": "c0", "header": "ПІБ", "label": None, "group": "Судновласник",
         "fill_rate": 1.0,
         "distinct_ratio": 1.0, "shapes": [["CCCC CCCC", 1.0]],
         "detectors": {}, "samples": ["Коваленко Іван"], "index": 0,
         "count": 3, "filled": 3, "distinct": 3, "min_len": 5, "max_len": 20},
        {"id": "c1", "header": "№ з/п", "label": None, "fill_rate": 1.0,
         "distinct_ratio": 1.0, "shapes": [["d", 1.0]],
         "detectors": {"numeric": 1.0}, "samples": ["1"], "index": 1,
         "count": 3, "filled": 3, "distinct": 3, "min_len": 1, "max_len": 1},
    ]), encoding="utf-8")
    (d / "plan.validated.json").write_text(json.dumps({
        "subject": "Person",
        "entities": [{"key": "person", "schema": "Person", "keys": ["c0"]}],
        "edges": [],
        "bindings": [{"column": "c0", "prop": "Person:name", "entity": "person",
                      "type_name": "name", "why": "names"}],
        "decisions": [
            {"column": "c0", "prop": "Person:name", "entity": "person",
             "verdict": "accepted", "reason": "names", "decided_by": "model"},
            {"column": "c1", "prop": None, "entity": None, "verdict": "unmapped",
             "reason": "row ordinal", "decided_by": "model"},
        ],
    }), encoding="utf-8")
    (d / "plan.json").write_text(json.dumps({
        "subject": "Person", "entities": [], "edges": [], "bindings": [],
        "shortlists": {"c0": ["Person:name", "Thing:name"], "c1": []},
    }), encoding="utf-8")
    workspace.commit_source("sid__s", str(d), {"source_id": "sid/s"})
    return str(tmp_path / "work")


def test_lists_sources_with_their_counts(tmp_path):
    """`summaries()` is the one read the API's rows are built from.

    There was a second, narrower projection — `list_sources` — whose `reviewed`
    field was `bool(overrides)`, the answer `review_state`'s own docstring calls
    wrong in both directions. No dashboard read it; only these tests did, which
    is the whole reason it survived. Review state is asserted through
    `review_state` below, which is what the contract actually publishes.
    """
    s = DecisionStore(_fixture(tmp_path))
    assert s.source_ids() == ["sid__s"]
    summaries = s.summaries()
    assert summaries["sid__s"]["columns"]["mapped"] == 1
    assert s.review_state("sid__s")["state"] == "pending"


def test_cards_join_profile_plan_and_candidates(tmp_path):
    s = DecisionStore(_fixture(tmp_path))
    cards = {c["column"]: c for c in s.source("sid__s")["cards"]}
    assert cards["c0"]["header"] == "ПІБ"
    assert cards["c0"]["group"] == "Судновласник"
    assert cards["c0"]["prop"] == "Person:name"
    assert cards["c0"]["candidates"] == ["Person:name", "Thing:name"]
    assert cards["c1"]["verdict"] == "unmapped"
    assert cards["c1"]["samples"] == ["1"]


def test_an_override_wins_and_survives_a_restart(tmp_path):
    out = _fixture(tmp_path)
    DecisionStore(out).record("sid__s", "c1", "Person:position", "person", "посада")
    fresh = DecisionStore(out)
    assert fresh.overrides("sid__s")["c1"]["prop"] == "Person:position"
    card = {c["column"]: c for c in fresh.source("sid__s")["cards"]}["c1"]
    assert card["prop"] == "Person:position"
    assert card["decided_by"] == "analyst"
    # And the row's own count of it moves, which is what the review app reads.
    assert fresh.override_revision("sid__s") == 1


def test_last_write_wins(tmp_path):
    out = _fixture(tmp_path)
    s = DecisionStore(out)
    s.record("sid__s", "c1", "Person:position", "person", "first")
    s.record("sid__s", "c1", "unmapped", None, "second thought")
    assert s.overrides("sid__s")["c1"]["prop"] == "unmapped"
    with open(open_workspace(out).decisions_path("sid__s"),
              encoding="utf-8") as fh:
        assert len(fh.readlines()) == 2


def test_clear_removes_the_override(tmp_path):
    out = _fixture(tmp_path)
    s = DecisionStore(out)
    s.record("sid__s", "c1", "Person:position", "person", "x")
    s.clear("sid__s", "c1")
    assert "c1" not in s.overrides("sid__s")


def test_an_uncommitted_tail_is_ignored_rather_than_read(tmp_path):
    """A process killed between the write and the manifest replacement leaves
    a line nobody finished making. The manifest records how many BYTES the
    workspace stands behind, so the tail is invisible rather than half-read."""
    out = _fixture(tmp_path)
    store = DecisionStore(out)
    store.record("sid__s", "c0", "Person:name", "person", "committed")
    assert list(store.overrides("sid__s")) == ["c0"]

    with open(open_workspace(out).decisions_path("sid__s"), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "override", "column": "c1",
                             "prop": "Person:birthDate"}) + "\n")
    assert list(store.overrides("sid__s")) == ["c0"], "an uncommitted tail was read"
    assert store.override_revision("sid__s") == 1


def test_an_edited_committed_prefix_is_refused(tmp_path):
    """Append-only means append-only. Bytes changing underneath the committed
    prefix means something other than this program edited an analyst's
    record, and reading it as if nothing happened is the one response that
    cannot be right."""
    from ftmap.review.store import DecisionLogCorrupt

    out = _fixture(tmp_path)
    store = DecisionStore(out)
    store.record("sid__s", "c0", "Person:name", "person", "as written")

    path = open_workspace(out).decisions_path("sid__s")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace("as written", "as EDITED!"))

    with pytest.raises(DecisionLogCorrupt):
        store.overrides("sid__s")


def test_approval_and_override_are_independent_projections(tmp_path):
    """R7. Neither can be inferred from the other, and the single `reviewed`
    boolean they replaced was wrong in both directions."""
    out = _fixture(tmp_path)
    store = DecisionStore(out)

    assert store.review_state("sid__s")["state"] == "pending"
    store.approve("sid__s")
    assert store.review_state("sid__s")["state"] == "approved"
    assert store.overrides("sid__s") == {}, "approval invented an override"

    store.record("sid__s", "c0", "Person:name", "person", "x")
    assert store.review_state("sid__s")["state"] == "stale"

    store.reopen("sid__s")
    state = store.review_state("sid__s")
    assert state["state"] == "pending"
    assert store.overrides("sid__s"), "reopen removed the override"


def test_an_approval_only_event_does_not_advance_the_override_revision(tmp_path):
    """Otherwise approving a source would make it stale against itself, and
    the analyst would be sent to rerun output nothing had changed."""
    out = _fixture(tmp_path)
    store = DecisionStore(out)
    store.approve("sid__s")
    store.reopen("sid__s")
    assert store.override_revision("sid__s") == 0
