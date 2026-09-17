"""One verdict on an analyst's binding, asked in both places it is needed."""

from __future__ import annotations

from dataclasses import dataclass

from ftmap.plan.response_schema import UNMAPPED

__all__ = ["UNMAPPED", "Verdict", "judge"]


@dataclass(frozen=True)
class Verdict:
    """What one candidate binding is, before anything durable happens."""

    accepted: bool
    column: str
    prop: str | None
    entity: str | None
    reason: str

    @property
    def clears(self) -> bool:
        """Whether this verdict removes a binding rather than making one."""
        return self.accepted and self.prop is None


def declared_keys(entities: list[dict], edges: list[dict]) -> dict[str, dict]:
    """Everything a binding may attach to: parties AND relationships."""
    declared = {e["key"]: e for e in entities}
    declared.update({e["key"]: e for e in edges})
    return declared


def judge(column: str, prop: str | None, entity: str | None,
          declared: dict[str, dict], cat, why: str = "") -> Verdict:
    """Accept or reject one analyst binding, with the reason either way."""
    from ftmap.plan.validate import _structural_check

    if not prop or prop == UNMAPPED:
        return Verdict(True, column, None, None, why or "analyst")

    if cat.prop(prop) is None:
        return Verdict(False, column, prop, entity,
                       "not a property in the ontology")

    resolved = entity or next(iter(declared), None)
    if resolved is None:
        return Verdict(False, column, prop, None,
                       "no entity was declared for this override to attach to")

    structural = _structural_check(
        {"column": column, "prop": prop, "entity": resolved}, cat, declared)
    if structural is not None:
        return Verdict(False, column, prop, resolved, structural.reason)
    return Verdict(True, column, prop, resolved, why or "analyst")
