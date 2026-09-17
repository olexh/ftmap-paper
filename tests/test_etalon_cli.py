"""`ftmap etalon` as automation sees it: exit status and stderr.

The 2026-08-31 follow-up review's P1: the command printed that an invalid
ledger was not eligible for publication and then exited 0, so any script
checking `$?` published it anyway. And its P2: an archived version-1 summary
scored with no visible marker, reading exactly like a vouched evaluator-2
report.
"""

import json

from typer.testing import CliRunner

from ftmap.cli import app

runner = CliRunner()

SHA = "ab" * 32

ETALON = f"""\
etalon: 2
source:
  path: x.csv
  sha256: {SHA}
  sheet: ""
subject:
  answer: Person
  accept: []
produces_nothing: >-
  a fixture about the ledger gate, not about mapping
columns:
  - id: c0
    header: id
    role: not-data
    why: a fixture column
entities: []
edges: []
"""


def _workspace(tmp_path, coverage_totals):
    """A legacy-scan workspace holding one summary for the etalon's bytes."""
    out = tmp_path / "out"
    (out / "src1").mkdir(parents=True)
    summary = {
        "source": {"sha256": SHA, "sheet": ""},
        "subject": None, "subject_declared": None,
        "structure": {"entities": [], "edges": [], "attachments": [],
                      "bindings": {}},
        "coverage": {}, "entities": {},
    }
    if coverage_totals is not None:
        summary["coverage_totals"] = coverage_totals
        summary["claims"] = []
        summary["coverage_schema"] = 2
    (out / "src1" / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    etalon = tmp_path / "gold.etalon.yaml"
    etalon.write_text(ETALON, encoding="utf-8")
    return str(etalon), str(out)


def test_an_invalid_ledger_fails_the_command(tmp_path):
    etalon, out = _workspace(tmp_path, {"accounting_valid": False})
    result = runner.invoke(app, ["etalon", etalon, "--out", out])
    assert result.exit_code == 3
    assert "LEDGER INVALID" in result.stderr
    assert "NOT eligible for publication" in result.stderr
    # The scores are still printed — an invalid run is worth diagnosing.
    assert "PRODUCES-NOTHING" in result.stdout


def test_a_legacy_summary_is_named_unvouched_and_does_not_fail(tmp_path):
    etalon, out = _workspace(tmp_path, None)
    result = runner.invoke(app, ["etalon", etalon, "--out", out])
    assert result.exit_code == 0
    assert "LEDGER UNVOUCHED" in result.stderr
    assert "schema-2 rerun" in result.stderr
    assert "UNVOUCHED" in result.stdout  # the markdown carries it too


def test_a_valid_schema_2_run_stays_quietly_successful(tmp_path):
    etalon, out = _workspace(tmp_path, {"accounting_valid": True})
    result = runner.invoke(app, ["etalon", etalon, "--out", out])
    assert result.exit_code == 0
    assert "LEDGER" not in result.stderr
    assert "UNVOUCHED" not in result.stdout
