# tests/emit_probe.py
"""A synthetic source of N rows, run through the streaming emission path.

Built rather than committed, and built inside the measured subprocess: a
million-row CSV is 60 MB, and generating it in the parent would move the cost
out of the number the benchmark is trying to report.

WHAT THE FIXTURE IS SHAPED FOR. Four bound columns and a distinct key per row,
so every row emits four statements and one entity — four million statements and
one million entities at the full size. That is deliberately the WORST case for
the deferred disk-backed merge: nothing deduplicates, so the entity state is as
large as this pipeline can make it. A fixture where rows collapsed into a few
entities would flatter the measurement.

Named `emit_probe` rather than `test_emit_probe` so pytest does not collect it.
"""

from __future__ import annotations

import csv
import os
import sys


def build(directory: str, rows: int) -> str:
    """One CSV, four columns, every key distinct."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "big.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["код", "ПІБ", "посада", "підрозділ"])
        for i in range(rows):
            writer.writerow([f"UA{i:09d}", f"Коваленко Іван {i}",
                             "депутат", f"фракція {i % 97}"])
    return path


def main() -> dict:
    directory, rows = sys.argv[1], int(sys.argv[2])
    path = build(directory, rows)

    from ftmap.build.emit import Tally, statement_writer
    from ftmap.build.execute import execute_into
    from ftmap.config import Config
    from ftmap.io.layout import build_frame
    from ftmap.io.tabular import read_source
    from ftmap.normalize.report import reject_writer
    from ftmap.plan.compile import compile_mapping, normalize_frame
    from ftmap.plan.validate import ValidatedPlan
    from ftmap.vocab.catalogue import Catalogue

    cfg, cat = Config.load(None), Catalogue.load()
    frame = build_frame(read_source(path)[0], path, "a" * 64,
                        "aaaaaaaaaaaa/", cfg)
    vplan = ValidatedPlan(
        subject="Person",
        entities=[{"key": "person", "schema": "Person", "keys": ["c0"]}],
        edges=[],
        bindings=[
            {"column": "c0", "prop": "Person:idNumber", "entity": "person",
             "type_name": "identifier", "why": ""},
            {"column": "c1", "prop": "Person:name", "entity": "person",
             "type_name": "name", "why": ""},
            {"column": "c2", "prop": "Person:position", "entity": "person",
             "type_name": "string", "why": ""},
            {"column": "c3", "prop": "Person:notes", "entity": "person",
             "type_name": "text", "why": ""},
        ],
        decisions=[])

    normalized = os.path.join(directory, "normalized.csv")
    tally = Tally()
    for reject in normalize_frame(frame, vplan, normalized):
        tally.add_reject(reject)
    mapping = compile_mapping(vplan, frame, normalized)

    with reject_writer(os.path.join(directory, "rejects.jsonl")) as write_reject, \
            statement_writer(os.path.join(directory, "statements.csv")) as write_statement:
        def keep_reject(reject):
            tally.add_reject(reject)
            write_reject(reject)

        def keep_statement(statement):
            tally.add_statement(statement, cat)
            write_statement(statement)

        entities = execute_into(mapping, vplan, frame, cat, "bench",
                                {}, on_statement=keep_statement,
                                on_reject=keep_reject)

    return {"rows": rows,
            "statements": tally.statements,
            "entities": len(entities),
            "rejects": tally.rejects.total,
            # Four bound columns, every value accepted, one entity per row.
            "expected_statements": rows * 4,
            "expected_entities": rows}


if __name__ == "__main__":
    print(main())
