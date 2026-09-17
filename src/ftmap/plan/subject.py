"""The structure call answers one question twice, and the two may disagree."""

from __future__ import annotations

import re


def contradiction_reason(repair: dict) -> str:
    """The one wording, so `decisions.jsonl` reads the same whichever stage
    resolved the contradiction."""
    return (f"the plan contradicted itself: subject {repair['subject']!r} but "
            f"its only declared entity was {repair['declared']!r}, which is "
            f"neither {repair['subject']!r} nor a schema it inherits from; "
            f"{repair['subject']!r} is declared as {repair['added']!r} too, so "
            "the binding call chooses between the two readings with the values "
            "in front of it")


def unused_reading_reason(repair: dict, key: str) -> str:
    """Why one half of a contradicted pair is not in the output."""
    other = repair["added"] if key == repair["key"] else repair["key"]
    return (f"{key!r} is one of two readings of a row the plan contradicted "
            f"itself about ({other!r} is the other); no accepted binding and no "
            "surviving edge used it, so it is dropped rather than written as "
            "an entity per row with no properties")


def _fresh_key(subject: str, taken: set[str]) -> str:
    """A key for the promoted subject, in the form `structure_schema`'s own
    `key` pattern allows (`^[a-z][a-z0-9_]{0,23}$`), and distinct from every
    key already in the plan's one namespace — see `plan/keys.py` for what a
    collision costs."""
    base = re.sub(r"[^a-z0-9_]", "", subject.lower())[:24] or "subject"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base[:22]}_{n}", n + 1
    return candidate


def resolve_subject_contradiction(cat, subject, entities) -> tuple[list[dict], list[dict]]:
    """Return the entities with a contradicting subject declared as one of
    them, plus what was added.
    """
    if not subject or len(entities) != 1:
        return entities, []
    entity = entities[0]
    declared = entity.get("schema")
    if not declared or not cat.is_concrete(subject):
        return entities, []
    if cat.related_to(declared, subject):
        return entities, []
    added = _fresh_key(subject, {str(entity.get("key"))})
    promoted = {"key": added, "schema": subject,
                "keys": list(entity.get("keys") or [])}
    return ([entity, promoted],
            [{"key": entity.get("key"), "declared": declared,
              "subject": subject, "added": added}])
