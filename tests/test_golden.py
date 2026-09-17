# tests/test_golden.py
"""The whole pipeline over the tracked fixtures, byte for byte.

See `tests/golden.py` for what the snapshot holds and why. This module is two
questions: does the run reproduce itself, and does it still match what was
reviewed and committed.
"""

import json
import re

import pytest

from golden import GOLDEN, compare, run_golden, snapshot


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("golden"))
    run_golden(out)
    return out


def test_two_identical_scripted_runs_write_the_same_bytes(tmp_path):
    """Determinism first, because a golden over a nondeterministic pipeline is
    a test that fails at random and gets deleted."""
    first, second = str(tmp_path / "a"), str(tmp_path / "b")
    run_golden(first)
    run_golden(second)
    assert snapshot(first) == snapshot(second)


def test_the_fixture_corpus_matches_the_committed_golden(golden_run):
    live, committed = compare(golden_run)
    if live != committed:
        moved = sorted(set(live) ^ set(committed)) or sorted(
            name for name in live if live[name] != committed.get(name))
        pytest.fail(
            "the fixture corpus no longer produces the committed bytes.\n"
            f"artefacts that differ: {moved}\n"
            "If this change is intended, regenerate with\n"
            "  FTMAP_GOLDEN_UPDATE=1 uv run --frozen --offline pytest "
            "tests/test_golden.py\n"
            "and review the diff — it is the unit's evidence, not a chore.")


def test_the_golden_corpus_exercises_the_paths_it_claims_to(golden_run):
    """A golden is only an oracle for what it actually contains.

    Each assertion here names a path some later unit changes. Without them the
    corpus could quietly stop covering multi-column keys or edges — and would
    then keep passing through the very unit that breaks them.
    """
    committed = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # A mapping now lives INSIDE the generation it describes — U3 moved it
    # out of a shared `mappings/` directory, which is what let a failed run
    # leave one behind pointing at output that was rolled back.
    mappings = {name: entry["content"] for name, entry in committed.items()
                if name.endswith("mapping.yml")}
    assert len(mappings) >= 3, "fewer mapping documents than fixtures"

    # U2 re-encodes multi-column keys into one synthetic length-prefixed
    # field and leaves single-column keys alone. Both cases have to be in here
    # for its golden diff — and every later unit's byte-equivalence claim — to
    # mean anything.
    #
    # This assertion used to count `keys:` entries and expect one entity with
    # more than one. That is exactly what U2 stopped producing, and the check
    # failing was the intended signal rather than a broken test: a multi-key
    # entity now names ONE field, and which one it is has become the thing
    # worth asserting.
    documents = "\n".join(mappings.values())
    assert "__ck_" in documents, "no multi-column key survived into a mapping"
    assert re.search(r"- c\d+__key", documents), "no single-column key"

    ids = [entry for name, entry in committed.items()
           if name.endswith("entities.ftm.json") and "entity_ids" in entry]
    assert ids, "no entity ids recorded"
    schemas = {schema for entry in ids for schema, _ in entry["entity_ids"]}
    assert len(schemas) > 1, f"only one schema emitted: {schemas}"

    # EVERY FIXTURE, NOT WHICHEVER ONES SURVIVED. The first version of this
    # corpus covered three of five and said nothing: the scripted client took
    # the first offered candidate, on two sources that was a property the
    # ontology has deprecated, `EntityMapping.bind()` warns about one, this
    # suite promotes warnings to errors, and `run_corpus` recorded the
    # resulting exception as a source failure. The two lost were the
    # two-header-row and title-row workbooks — the defect classes the fixture
    # set exists for. A partial oracle that reports itself green is worse than
    # no oracle.
    run = next(entry["content"] for name, entry in committed.items()
               if name.endswith("run.json"))
    assert run["sources"] == 5, f"{run['sources']} of 5 fixtures ran"
    assert run["failed"] == 0, run["failures"]
    assert run["statements"] > 1000, run["statements"]
