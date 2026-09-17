# tests/test_boundary.py
"""The value boundary and coverage honesty, over a real run.

`ScriptedClient` never calls the live model — it answers deterministically so
these tests run fast and offline, but the source file is a real fixture
(`deputies_popolo.csv`, see `tests/fixtures/README.md`) so the values crossing
the boundary are the values the pipeline was actually built to carry, not
synthetic stand-ins.

`ScriptedClient` maps every column to "unmapped": these tests check the
boundary and the coverage arithmetic, not the model's judgment, so a client
that always abstains is the simplest one that produces a full run to inspect.
"""

from __future__ import annotations

import json
import os

import pytest

from ftmap.config import Config
from ftmap.pipeline import run_corpus
from ftmap.plan.prompt import BINDING_MARKER
from ftmap.vocab.catalogue import Catalogue
from ftmap.workspace import open_workspace

CFG = Config.load(None)
CAT = Catalogue.load()
FIX = os.path.join(os.path.dirname(__file__), "fixtures")


class ScriptedClient:
    calls = 0
    cache_hits = 0

    def complete(self, system, user, schema, max_tokens=None):
        if BINDING_MARKER not in user:
            return {"subject": "Person",
                    "entities": [{"key": "person", "schema": "Person", "keys": []}],
                    "edges": []}
        ids = [line.split(":", 1)[0].strip() for line in user.splitlines()
               if line and line[0] == "c" and ":" in line]
        return {"bindings": [{"column": c, "binding": "unmapped", "why": "x"}
                             for c in ids]}


def test_a_non_loopback_model_endpoint_aborts_the_run(tmp_path):
    p = tmp_path / "ftmap.toml"
    p.write_text("[model]\nurl = 'http://10.0.0.5:8080'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="loopback"):
        Config.load(str(p))


def test_summary_and_run_report_carry_no_cell_value(tmp_path):
    out = str(tmp_path / "work")
    run_corpus(os.path.join(FIX, "deputies_popolo.csv"), CFG, CAT,
               ScriptedClient(), out)
    needles = {"Шаправський", "Ганаба", "1990-09-11", "bucha-rada"}
    for dirpath, _dirs, files in os.walk(out):
        for name in files:
            if name not in ("summary.json", "run.json"):
                continue
            with open(os.path.join(dirpath, name), encoding="utf-8") as fh:
                blob = fh.read()
            for needle in needles:
                assert needle not in blob, f"{name} leaked {needle!r}"


def test_value_bearing_artefacts_are_owner_only(tmp_path):
    out = str(tmp_path / "work")
    run_corpus(os.path.join(FIX, "deputies_popolo.csv"), CFG, CAT,
               ScriptedClient(), out)
    private = ("normalized.csv", "entities.ftm.json", "statements.csv",
               "rejects.jsonl")
    for dirpath, _dirs, files in os.walk(out):
        for name in files:
            if name in private:
                mode = oct(os.stat(os.path.join(dirpath, name)).st_mode)[-3:]
                assert mode == "600", f"{name} is {mode}"


class NoEntitySurvives(ScriptedClient):
    """Declines every column like `ScriptedClient`, but declares a subject its
    only entity contradicts, so `validate` drops both readings and the plan
    reaches `compile_mapping` with no entity at all."""

    def complete(self, system, user, schema, max_tokens=None):
        if BINDING_MARKER not in user:
            return {"subject": "Vehicle",
                    "entities": [{"key": "person", "schema": "Person",
                                  "keys": []}]}
        return super().complete(system, user, schema, max_tokens)


def _nothing_to_declare(tmp_path) -> str:
    """A table no rule can read anything into: coded headers, figures and
    dates, no name-shaped value, no role word. Three columns, thirty rows."""
    p = tmp_path / "codes.csv"
    lines = ["Код,Показник,Дата"]
    for i in range(30):
        lines.append(f"Δ-{i + 61},{7700 + i * 31},2019-04-{(i % 28) + 1:02d}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_a_source_that_produces_no_entity_is_declined_and_still_counted(tmp_path):
    """The same run as above, one step worse: no entity survives either, so
    the compiled mapping is `{"entities": {}}` and followthemoney refuses it.
    Before this, the source raised out of `execute` and was recorded as a
    failure — which writes no summary, so its columns and its values left the
    corpus totals with it and the run reported coverage over the sources that
    happened to succeed. Measured on the private corpus: 20 of 244 sources.
    """
    out = str(tmp_path / "work")
    # NOT THE DEPUTIES FIXTURE ANY MORE. Since 2026-09-04 a column headed
    # «партія» over party names is declared a party by rule whatever the
    # model says (`plan.roles.declare_parties`), so that file now produces
    # an entity under a client that maps nothing — which is the rule doing
    # its job, not this test's subject. A table of codes and figures under
    # headers no lexicon names is what "produces nothing" needs.
    src = _nothing_to_declare(tmp_path)
    report = run_corpus(src, CFG, CAT, NoEntitySurvives(), out)
    assert report["failed"] == 0
    assert report["sources"] == 1 and report["declined"] == 1

    corpus = report["coverage"]
    assert corpus["columns"] == report["columns"]["total"] > 0
    assert corpus["values"] > 0
    assert corpus["declined"] == corpus["values"]
    assert corpus["offered"] == 0 and corpus["unaccounted"] == 0

    with open(os.path.join(open_workspace(out).source_dir(report["summaries"][0]),
                           "summary.json"),
              encoding="utf-8") as fh:
        blob = fh.read()
    assert json.loads(blob)["declined"]
    for needle in ("7731", "2019-04-02", "Δ-91"):
        assert needle not in blob


def test_coverage_is_reported_over_all_columns_not_survivors(tmp_path):
    """`ScriptedClient` declines every column, so this is a whole run that
    produces nothing — the reestrtz shape, offline.

    The assertions used to stop at `columns.mapped`, which is a count of
    answers and not an accounting. `coverage` was `{}`: no column, therefore
    no column that failed to add up, therefore a run that read as fully
    reconciled while emitting not one statement. Every column the source held
    now has a row, and `coverage_totals` states what it held against what came
    out — in `summary.json` per source, and folded over the corpus in
    `run.json`, which is the one the operator sees.
    """
    out = str(tmp_path / "work")
    report = run_corpus(_nothing_to_declare(tmp_path), CFG, CAT,
                        ScriptedClient(), out)
    # The scripted client maps nothing, so mapped must be 0 and total must not be.
    assert report["columns"]["mapped"] == 0
    assert report["columns"]["total"] > 0

    assert report["statements"] == 0
    corpus = report["coverage"]
    assert corpus["columns"] == report["columns"]["total"]
    assert corpus["statuses"] == {"bound": 0, "structural": 0, "dropped": 0,
                                  "undecided": 0,
                                  "unmapped": report["columns"]["total"]}
    # The source held values and none of them became a statement. Declining is
    # correct behaviour, so they are declined rather than unaccounted — but the
    # run cannot be read as having produced anything.
    assert corpus["values"] > 0
    assert corpus["offered"] == 0
    assert corpus["declined"] == corpus["values"]
    assert corpus["unaccounted"] == 0

    with open(os.path.join(open_workspace(out).source_dir(report["summaries"][0]),
                           "summary.json"),
              encoding="utf-8") as fh:
        summary = json.load(fh)
    assert len(summary["coverage"]) == summary["columns"]["total"]
    assert summary["coverage_totals"]["values"] == corpus["values"]


def test_the_input_snapshot_store_is_owner_only_and_never_published(tmp_path):
    """It holds COMPLETE COPIES of the corpus — the most value-bearing thing
    under a workspace, and not a deliverable at all.

    Every other artefact under `work*/` is a derivation that leaks values by
    accident (`profile.json` carries five per column, the model's `why` was
    measured to quote them in 13 of 148 bindings). This one carries them by
    construction, in full, and is here so that fact is stated in the same
    place as the rest of the boundary rather than left to the module that
    happens to create it.
    """
    from ftmap.fsutil import SnapshotStore

    out = str(tmp_path / "work")
    run_corpus(os.path.join(FIX, "deputies_popolo.csv"), CFG, CAT,
               ScriptedClient(), out)

    store = os.path.join(out, SnapshotStore.DIRNAME)
    assert os.path.isdir(store), "no snapshot was taken"
    found = 0
    for dirpath, _dirs, files in os.walk(store):
        for name in files:
            found += 1
            mode = oct(os.stat(os.path.join(dirpath, name)).st_mode)[-3:]
            assert mode == "600", f"{name} is {mode}"
    assert found == 1

    # And it is not named by the published manifest, which is world readable.
    with open(os.path.join(open_workspace(out).aggregate_dir(), "run.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert SnapshotStore.DIRNAME not in json.dumps(manifest)


@pytest.mark.parametrize("workspace", ["work", "out", "результати"])
def test_snapshot_and_cache_paths_are_ignored_whatever_the_workspace_is_called(
        workspace):
    """The `work*/` glob is the guard, and it only fires on a conventional name.

    `.gitignore`'s own comment records that the by-name list was wrong twice
    because it named the artefacts someone happened to think of. A workspace
    at `out/` or `результати/` is not exotic — `--out` takes any path — and a
    snapshot store under one holds the entire corpus verbatim.
    """
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (f"{workspace}/.snapshots/ab/abcdef.xlsx",
                      f"{workspace}/.staging/deadbeef/normalized.csv",
                      f"{workspace}/.derived/gen/abc-1234/index.sqlite",
                      f"{workspace}/decisions/abc-1234.jsonl",
                      f"{workspace}/sources/abc-1234/gen/mapping.yml",
                      f"{workspace}/sources/abc-1234/gen/statements.csv",
                      f"{workspace}/aggregates/gen/entities.ftm.json",
                      f"{workspace}/prompts.jsonl",
                      # `failures.jsonl` was the one artefact the code chmods
                      # 0600 that this list did not name — the by-name list
                      # being wrong a third time, in the same way, for the
                      # same reason: it named the artefacts someone thought of.
                      f"{workspace}/aggregates/gen/failures.jsonl",
                      f"{workspace}/sources/abc-1234/gen/failures.jsonl"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", candidate],
            cwd=repo, capture_output=True)
        assert result.returncode == 0, f"{candidate} is not ignored"


def test_the_workspace_manifest_is_published_and_carries_no_cell_value(tmp_path):
    """It is the registry every reader goes through, so it is world readable —
    and that is a claim about its contents, checked here rather than assumed.

    It holds source ids, sheet names, digests, generation names, run ids and
    artefact hashes. A sheet name is source text and is allowed out by the same
    rule headers are; a cell value is not, and nothing in the manifest is
    derived from one.
    """
    from ftmap.workspace import MANIFEST, open_workspace

    out = str(tmp_path / "work")
    run_corpus(os.path.join(FIX, "deputies_popolo.csv"), CFG, CAT,
               ScriptedClient(), out)

    path = os.path.join(out, MANIFEST)
    assert oct(os.stat(path).st_mode)[-3:] == "644"
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for value in _fixture_values():
        assert value not in text, value

    # The committed aggregate's own manifest is published for the same reason
    # and under the same claim.
    report = os.path.join(open_workspace(out).aggregate_dir(), "run.json")
    assert oct(os.stat(report).st_mode)[-3:] == "644"


def _fixture_values() -> list[str]:
    """Distinct cell values long enough to be recognisable, from the fixture
    this module runs over. Read from the file rather than hard-coded, so the
    check follows the fixture if it is ever replaced."""
    import csv

    values = set()
    with open(os.path.join(FIX, "deputies_popolo.csv"), encoding="utf-8") as fh:
        for index, row in enumerate(csv.reader(fh)):
            if index == 0:
                continue  # headers may leave; that is the documented rule
            values.update(cell.strip() for cell in row if len(cell.strip()) > 15)
    return sorted(values)
