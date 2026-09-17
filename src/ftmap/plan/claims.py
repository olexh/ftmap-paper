"""One (entity, property) is read from ONE column, and which one is decided
here rather than by the order a dict happened to be filled in.
"""

from __future__ import annotations

MERGEABLE_TYPES = frozenset({"name", "country", "address", "phone", "email",
                             "url", "language", "topic"})


def mergeable(type_name: str | None) -> bool:
    """Whether a second column may feed this property beside the first."""
    return type_name in MERGEABLE_TYPES


def claim(binding: dict) -> tuple[str, str]:
    """The thing at most one column may hold: an entity's own property."""
    return binding["entity"], binding["prop"].split(":", 1)[1]


def column_order(column: str) -> int:
    """Column ids are `c<n>`, so source order is numeric, not lexical — `c10`
    comes after `c9`.
    """
    return int(column[1:])


_order = column_order


def resolve_claims(bindings: list[dict], strength, *,
                   merge: bool = True) -> tuple[list[dict], list[tuple[dict, dict]]]:
    """Keep one binding per claim — or every binding, where the property is a
    set; return the kept ones and what they displaced.
    """
    winners: dict[tuple[str, str], dict] = {}
    for b in bindings:
        c = claim(b)
        best = winners.get(c)
        if best is None or (-strength(b), _order(b["column"])) < (-strength(best), _order(best["column"])):
            winners[c] = b
    kept: list[dict] = []
    displaced: list[tuple[dict, dict]] = []
    for b in bindings:
        winner = winners[claim(b)]
        if b is winner or (merge and mergeable(b.get("type_name"))):
            kept.append(b)
        else:
            displaced.append((b, winner))
    return kept, displaced


def shared_claims(kept: list[dict]) -> list[tuple[dict, dict]]:
    """Each binding that shares a claim with an earlier column, paired with
    the earliest column's binding — so the caller can record, per column,
    that the property is fed by more than one."""
    first: dict[tuple[str, str], dict] = {}
    out: list[tuple[dict, dict]] = []
    for b in sorted(kept, key=lambda b: _order(b["column"])):
        c = claim(b)
        if c in first:
            out.append((b, first[c]))
        else:
            first[c] = b
    return out


def share_reason(binding: dict, first: dict) -> str:
    """Why this column keeps a property another column already holds."""
    return (f"{first['column']} also binds {binding['prop']}; a "
            f"{binding.get('type_name')} is a facet followthemoney holds "
            f"several of, so both columns feed the property and every "
            f"value keeps the cell it came from")


def decline_reason(loser: dict, winner: dict, loser_n: float, winner_n: float) -> str:
    """Why this column was declined, in terms of the column that took the
    property. Value-free: it names columns, a property and two counts."""
    prop, col = loser["prop"], winner["column"]
    lead = (f"{col} already binds {prop}, and followthemoney keeps one set of "
            f"values per property, so a second column on it would mix two "
            f"different facts into one field")
    if round(winner_n) == round(loser_n):
        return (f"{lead}; the two contribute equally many values "
                f"({round(winner_n)}), so the earlier column in the source "
                f"keeps the property and this one is declined")
    return (f"{lead}; {col} contributes {round(winner_n)} values against this "
            f"column's {round(loser_n)}, so this one is declined")


def override_decline_reason(loser: dict, winner: dict) -> str:
    """The same rule when an analyst is one of the claimants."""
    return (f"the analyst bound {loser['prop']} to {winner['column']}, and "
            f"followthemoney keeps one set of values per property, so this "
            f"column is declined rather than merged into it")
