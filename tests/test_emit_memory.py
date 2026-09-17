# tests/test_emit_memory.py
"""What emitting a million rows costs, and which part of it this unit owns.

The ceiling is in `docs/measurements/2026-08-25-scale-ceilings.md` §3, written
before `execute` was split — and it is **not met**. That is recorded here as a
strict `xfail` rather than adjusted, because the note's own rule says a ceiling
may never be raised by the unit whose optimization it grades.

What the measurement showed, at 1 000 000 rows on the pinned machine:

    build the CSV                    22 MB
    + read it into a Frame        1 107 MB     <- deferred: CSV materialization
    + normalize_frame             1 167 MB
    + emission, collecting        2 922 MB
    + emission, streaming         1 814 MB

Re-measured 2026-09-04, after `_read_csv` stopped holding the file's text and
a `StringIO` copy of it at the same time:

    read it into a Frame            702 MB     <- was 1 090 on the same machine
    emission, streaming           2 084 MB     <- was 2 174

The registered 600 MB bounds the TOTAL, and the total is still nowhere near it.
What changed is the ATTRIBUTION. `io/tabular._read_csv` still materializes the
whole table — the plan defers that explicitly ("CSV input still materializes a
complete table... bounded CSV ingestion is follow-up work") — but it no longer
also holds a second copy while doing so, and the frame is now a third of the
peak rather than half of it. Emission is the larger term today. No change to
either alone brings the total under 600.

What U8 does own is the difference between those last two lines: **1 108 MB**,
the statements and rejects that used to be accumulated in lists until the source
was finished. That is what `test_streaming_removes_the_statement_and_reject_lists`
gates, and it is a before/after on the same fixture in the same process rather
than a number chosen after the fact.

Marked `slow` and deselected by default; `scripts/verify.sh` runs them. The
million-row fixture takes about 80 s per measurement.
"""

import pytest

from bench import measure

pytestmark = pytest.mark.slow

#: Registered 2026-08-25 in `2026-08-25-scale-ceilings.md` §3. NOT MET — see
#: the module docstring and `2026-08-25-streaming-emission.md` §3.
EMISSION_RSS_MB = 600

#: What U8 actually controls: the statements and the rejects. Measured
#: 2 922 MB collecting against 1 814 MB streaming at a million rows, so a
#: third is the conservative form of a 38 % saving.
MIN_SAVING = 0.30

ROWS = 1_000_000

_COMMON = """
from emit_probe import build
from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import read_source
from ftmap.plan.compile import compile_mapping, normalize_frame
from ftmap.plan.validate import ValidatedPlan
from ftmap.vocab.catalogue import Catalogue

path = build(DIRECTORY, ROWS)
cfg, cat = Config.load(None), Catalogue.load()
frame = build_frame(read_source(path)[0], path, "a" * 64, "aaaaaaaaaaaa/", cfg)
vplan = ValidatedPlan(
    subject="Person",
    entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}], edges=[],
    bindings=[{"column": f"c{i}", "prop": q, "entity": "person",
               "type_name": t, "why": ""}
              for i, (q, t) in enumerate([("Person:idNumber", "identifier"),
                                          ("Person:name", "name"),
                                          ("Person:position", "string"),
                                          ("Person:notes", "text")])],
    decisions=[])
normalized = DIRECTORY + "/n.csv"
normalize_frame(frame, vplan, normalized)
mapping = compile_mapping(vplan, frame, normalized)
"""

_COLLECTING = _COMMON + """
from ftmap.build.emit import summarise, write_entities, write_statements
from ftmap.build.execute import execute
from ftmap.normalize.report import write_rejects

entities, statements, rejects = execute(mapping, vplan, frame, cat, "b", {})
write_statements(statements, DIRECTORY + "/s.csv")
write_rejects(rejects, DIRECTORY + "/r.jsonl")
write_entities(entities, DIRECTORY + "/e.json")
summarise(frame, vplan, entities, statements, rejects, cat, cfg, {})
result = {"statements": len(statements), "entities": len(entities)}
"""

_STREAMING = _COMMON + """
from ftmap.build.emit import (Tally, statement_writer, summarise_tally,
                              write_entities)
from ftmap.build.execute import execute_into
from ftmap.normalize.report import reject_writer

tally = Tally()
with reject_writer(DIRECTORY + "/r.jsonl") as write_reject, \\
        statement_writer(DIRECTORY + "/s.csv") as write_statement:
    def keep_reject(x):
        tally.add_reject(x)
        write_reject(x)

    def keep_statement(x):
        tally.add_statement(x, cat)
        write_statement(x)

    entities = execute_into(mapping, vplan, frame, cat, "b", {},
                            on_statement=keep_statement, on_reject=keep_reject)
for entity in entities:
    tally.add_entity(entity)
write_entities(entities, DIRECTORY + "/e.json")
summarise_tally(frame, vplan, tally, cat, cfg, {})
result = {"statements": tally.statements, "entities": len(entities)}
"""


def _run(body: str, directory, rows: int):
    return measure(f"DIRECTORY = {str(directory)!r}\nROWS = {rows}\n" + body,
                   timeout=2400)


def test_streaming_removes_the_statement_and_reject_lists(tmp_path):
    """R20, as a before/after on one fixture rather than as an absolute.

    A million rows, four bound columns, every value emitted: four million
    statements. Held in a list, at roughly 250 bytes a `Statement`, that alone
    is a gigabyte — which is what this removes, and what the two measurements
    below differ by.

    The counter assertions matter as much as the memory one: a streaming path
    that dropped statements would pass a memory gate triumphantly.
    """
    collecting = _run(_COLLECTING, tmp_path / "collect", ROWS)
    streaming = _run(_STREAMING, tmp_path / "stream", ROWS)

    assert streaming.result == collecting.result, (
        "the two paths produced different output")
    assert streaming.result["statements"] == ROWS * 4
    assert streaming.result["entities"] == ROWS

    saved = 1 - streaming.peak_mb / collecting.peak_mb
    assert saved >= MIN_SAVING, (
        f"collecting={collecting}  streaming={streaming}  saved {saved:.0%}, "
        f"expected at least {MIN_SAVING:.0%}")


@pytest.mark.xfail(strict=True, reason=(
    "The registered 600 MB bounds the TOTAL, and 1 107 MB of it is the CSV "
    "frame — `io/tabular._read_csv` materializes the whole table, which the "
    "plan defers. No change to emission can bring the total under 600 while "
    "that is true. Strict, so this turns red the day bounded CSV ingestion "
    "lands and the ceiling becomes reachable. See "
    "docs/measurements/2026-08-25-streaming-emission.md §3."))
def test_the_registered_total_ceiling(tmp_path):
    """Kept and failing, rather than adjusted.

    `2026-08-25-scale-ceilings.md` §5: a ceiling may be raised only by a dated
    note saying what measurement forced it, and never by the unit whose
    optimization it grades. This is that unit. So the gate stays as registered
    and stays red, with the attribution in the module docstring saying which
    stage owns the miss.
    """
    streaming = _run(_STREAMING, tmp_path / "total", ROWS)
    assert streaming.peak_mb < EMISSION_RSS_MB, str(streaming)


#: The frame's share of the streaming peak, measured 2026-09-04 at 0.34
#: (702 MB of 2 084). Asserted as a BAND, because both directions are a real
#: regression and each is somebody's mistake: above the ceiling means CSV
#: materialization came back, below the floor means emission ballooned.
FRAME_SHARE_FLOOR = 0.25
FRAME_SHARE_CEILING = 0.45


def test_where_the_peak_is_spent_is_what_the_docstring_says(tmp_path):
    """The attribution, asserted so the account above cannot go stale.

    It did go stale once, and this is what said so: the assertion used to read
    `frame > streaming / 2` and it went red on 2026-09-04 — not because
    emission grew, but because `_read_csv` stopped holding the file's text AND
    a `StringIO` copy of it, which took the frame from 1 090 MB to 702 MB. The
    test was doing its job; the claim it guarded had simply stopped being true.

    A band rather than a floor, for that reason. A one-sided assertion cannot
    tell "the stage this unit does not own got cheaper" from "the stage it does
    own got worse", and on the numbers above it happened to pass by 3 MB.
    """
    frame_only = measure(f"""
from emit_probe import build
from ftmap.config import Config
from ftmap.io.layout import build_frame
from ftmap.io.tabular import read_source

path = build({str(tmp_path / "frame")!r}, {ROWS})
frame = build_frame(read_source(path)[0], path, "a" * 64, "aaaaaaaaaaaa/",
                    Config.load(None))
result = {{"rows": len(frame.rows)}}
""", timeout=2400)
    streaming = _run(_STREAMING, tmp_path / "full", ROWS)

    assert frame_only.result["rows"] == ROWS
    share = frame_only.peak_mb / streaming.peak_mb
    assert FRAME_SHARE_FLOOR <= share <= FRAME_SHARE_CEILING, (
        f"frame={frame_only}  full={streaming}  share={share:.2f}, expected "
        f"{FRAME_SHARE_FLOOR}-{FRAME_SHARE_CEILING}. Above the ceiling, CSV "
        "materialization is back; below the floor, emission grew. Either way "
        "the attribution in this module's docstring is out of date — measure, "
        "then update the docstring AND this band in one dated note.")
