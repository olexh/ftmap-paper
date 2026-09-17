"""Rejection reporting."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

from ftmap.fsutil import json_line, open_private


def write_rejects(rejects: list[dict], path: str) -> None:
    with reject_writer(path) as write:
        for r in rejects:
            write(r)


@contextmanager
def reject_writer(path: str):
    """A sink that writes each reject as it is produced."""
    with open_private(path) as fh:
        yield lambda reject: fh.write(json_line(reject))


class RejectTally:
    """The five counters the reject block and the claim ledger need, one reject
    at a time.
    """

    __slots__ = ("total", "by_reason", "by_column", "by_stage", "by_claim")

    def __init__(self) -> None:
        self.total = 0
        self.by_reason: Counter = Counter()
        self.by_column: Counter = Counter()
        self.by_stage: Counter = Counter()
        self.by_claim: Counter = Counter()

    def add(self, reject: dict) -> None:
        self.total += 1
        self.by_reason[reject["reason"]] += 1
        self.by_column[reject["column"]] += 1
        stage = reject.get("stage", "normalize")
        self.by_stage[stage] += 1
        if stage == "emit":
            self.by_claim[(reject["column"], reject["prop"],
                           reject.get("entity"))] += 1

    def to_dict(self) -> dict:
        return {"total": self.total,
                "by_reason": dict(self.by_reason.most_common()),
                "by_column": dict(self.by_column.most_common()),
                "by_stage": dict(self.by_stage.most_common())}

