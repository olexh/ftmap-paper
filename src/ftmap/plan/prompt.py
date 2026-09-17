"""Prompt construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ftmap.io.frame import Frame, column_values
from ftmap.profile.columns import ColumnProfile
from ftmap.vocab.catalogue import Catalogue

STRUCTURE_MARKER = "Say what one row of this table describes"
EDGE_MARKER = "The entities one row of this table describes"
BINDING_MARKER = "declared entities:"
SPLIT_MARKER = "These entities are identified by the same column"
REPAIR_MARKER = "These assignments were rejected"

STRUCTURE_SYSTEM = """\
You read one table and say what its rows describe, using the FollowTheMoney \
ontology.

Headers and values are usually Ukrainian or Russian. Judge them as written. Do \
not translate them, and do not answer about a translation.

A row often describes more than one thing at once, and usually more than you \
expect. Go column by column and ask who or what each name belongs to: a debtor \
and the office that registered the debt are two parties, not one, and the \
person heading each of them is a third and a fourth. Declare one entity per \
party. Two columns holding names are two parties unless they hold the SAME \
name — a natural person copied into a legal-name column is one counterparty \
written twice, and that is one entity.

NOT EVERYTHING A ROW DESCRIBES IS A PARTY. A row is often about a thing the \
parties share, and that thing is an entity too: a contract and the award that \
settled it, a licence or a certificate with its own number and dates, a court \
case, a debt, a vessel, an address. Declare those on the same rule — one \
entity per thing the row names, identified by whatever names it.

You are not asked about the connections between them here; that comes next, \
once the entities have names.

Choose the most specific schema the values actually support. A subschema \
carries properties AND relationships its parent does not, and choosing the \
general one throws them away silently: a Company can be owned and an \
Organization cannot, so a register of state enterprises declared as \
Organization has no way left to say who controls them. An Airplane has a serial \
number and a Vehicle does not. Choose the broader schema only when the column \
genuinely holds more than one kind of thing — a list mixing companies and sole \
traders is LegalEntity, and that is the right answer.

Choose key columns that identify the entity within this file. A row ordinal is \
not a key. If nothing identifies it, leave keys empty and the pipeline will use \
the row number.

Answer only in the required JSON form."""

BINDING_SYSTEM = """\
You assign one FollowTheMoney property, on one declared entity, to each column \
of a table.

Headers and values are usually Ukrainian or Russian. Judge them as written. Do \
not translate them, and do not answer about a translation.

For each column you are given a closed candidate list. Every candidate already \
names BOTH the declared entity and the property together, for example \
"person (Person) -> Person:name" — you are choosing the whole pairing, not a \
property alone. Choose from that list or answer "unmapped". "unmapped" is the \
right answer for a row ordinal, an internal flag, an empty column, or anything \
you are not sure about. A wrong pairing is more expensive than an abstention.

A header can name what the cell CONTAINS, or the ROLE the cell plays. One \
column holding an organization's name may be there to say what that \
organization is called; another may be there to attach that organization to \
the person the row is about. Judge the content, and pick the candidate whose \
entity matches which one the value belongs to.

Answer only in the required JSON form."""


def _fmt_shapes(profile: ColumnProfile) -> str:
    return ", ".join(f"{s} {r:.0%}" for s, r in profile.shapes[:3]) or "-"


_PROMPT_HIDDEN_DETECTORS = frozenset({"org_name"})


def _fmt_detectors(profile: ColumnProfile) -> str:
    top = sorted(((k, v) for k, v in profile.detectors.items()
                  if k not in _PROMPT_HIDDEN_DETECTORS), key=lambda kv: -kv[1])[:3]
    return ", ".join(f"{k} {v:.0%}" for k, v in top) or "-"


SPLIT_SYSTEM = """\
You are given entities that one table declares on the SAME identifying column, with different FollowTheMoney schemata, and the columns whose values say what kind of thing each row is.

A table like this holds several kinds of row and says which is which. Each entity should be built from the rows of its own kind: say which value of which column selects them. If an entity really is built from every row — it is not one kind among several — answer "none" for it.

Answer only with entries from the list you are given, and only in the required JSON form."""


def split_prompt(entities: list[dict], profiles: list[ColumnProfile],
                 kinds: list[KindColumn]) -> str:
    """Which rows each of these entities is built from."""
    by_id = {p.id: p for p in profiles}
    lines = ["These entities are identified by the same column, "
             "so the table holds more than one kind of row:", ""]
    for e in entities:
        cols = ", ".join(f"{c} ({by_id[c].header or '(no header)'})"
                         if c in by_id else c
                         for c in (e.get("keys") or [])) or "the row ordinal"
        lines.append(f"  {e['key']} ({e['schema']}), identified by {cols}")
    lines += ["", "columns that say what KIND of thing each row is:"]
    for kind in kinds:
        shown = ", ".join(f"{value} ({n})" for value, n in kind.counts)
        lines.append(f"  {kind.column} ({kind.header or 'no header'}): {shown}")
    lines += ["", "Say which rows each entity is built from."]
    result = "\n".join(lines)
    assert SPLIT_MARKER in result
    return result


def _entity_identity(keys: list[str] | None, key: str | None,
                     by_id: dict[str, ColumnProfile]) -> str:
    """The phrase that says which column identifies an entity."""
    cols = ", ".join(f"{c} ({by_id[c].header or '(no header)'})"
                     for c in (keys or []) if c in by_id)
    if cols:
        return f"identified by {cols}"
    if key in by_id:
        return (f"no key column; named after {key} "
                f"({by_id[key].header or '(no header)'})")
    return "no key column"


EDGE_SYSTEM = """\
You are given the entities that one row of a table describes, each with the \
columns it is identified by. Say which pairs are connected and by what kind of \
relationship, using the FollowTheMoney ontology.

Two entities on one row are often connected, and the table usually says how: a \
person beside an organization may work for it, own it, or merely be its \
contact. Read the columns each entity comes from and choose the relationship \
the table states. If the row shows an owner and a thing owned, that is \
ownership, not employment.

You are given every relationship the ontology permits between these entities. \
Answer only with entries from that list. Answer only in the required JSON \
form."""


def edge_prompt(entities: list[dict], profiles: list[ColumnProfile],
                combinations: list[str] | None = None) -> str:
    """What each entity IS, not just what it is called."""
    by_id = {p.id: p for p in profiles}
    lines = ["The entities one row of this table describes:", ""]
    for e in entities:
        what = _entity_identity(e.get("keys"), e.get("key"), by_id)
        lines.append(f"  {e['key']}  schema {e['schema']}  {what}")
        source_col = next((c for c in e.get("keys") or [] if c in by_id),
                          e.get("key") if e.get("key") in by_id else None)
        if source_col and by_id[source_col].samples:
            lines.append("    values: "
                         + " | ".join(by_id[source_col].samples[:3]))
    if combinations:
        lines += ["", "The relationships FollowTheMoney allows between these, "
                      "written Schema|source|target:", ""]
        lines += [f"  {c}" for c in combinations]
    lines += ["", "Which pairs are connected, and by what relationship? "
                  "Answer with entries from the list above."]
    result = "\n".join(lines)
    assert EDGE_MARKER in result
    return result


@dataclass(frozen=True)
class KindColumn:
    """A column whose values say what KIND of thing each row is."""
    column: str
    header: str | None
    counts: list[tuple[str, int]]


def kind_columns(frame: Frame, profiles: list[ColumnProfile],
                 cfg) -> list[KindColumn]:
    """The columns a row filter could be written against."""
    out: list[KindColumn] = []
    for p in profiles:
        if p.fill_rate < cfg.kind_min_fill:
            continue
        counts = Counter(v for v in column_values(frame, p.id) if v)
        if not 2 <= len(counts) <= cfg.kind_max_values:
            continue
        if len(counts) >= sum(counts.values()):
            continue
        out.append(KindColumn(p.id, p.header,
                              counts.most_common(cfg.kind_max_values)))
    return out


def structure_prompt(frame: Frame, profiles: list[ColumnProfile],
                     cat: Catalogue | None = None) -> str:
    lines = [
        f"file: {frame.path.rsplit('/', 1)[-1]}",
        f"sheet: {frame.sheet or '-'}",
        f"rows: {len(frame.rows)}",
        "",
        "columns:",
    ]
    for p in profiles:
        label = f" | label: {p.label}" if p.label else ""
        lines.append(
            f"{p.id}: {p.header or '(no header)'}{label} | filled {p.fill_rate:.0%}"
            + (f" ({p.sentinel_share:.0%} of it a null placeholder)"
               if p.sentinel_share > 0.2 else "")
            + f" | distinct {p.distinct_ratio:.0%}"
            + f" | shape {_fmt_shapes(p)} | looks like {_fmt_detectors(p)}"
        )
        lines.append("  values: " + (" | ".join(p.samples) if p.samples else "-"))
    if cat is not None:
        lines += ["", "what these schemata mean, in the ontology's own words:"]
        for info in cat.schemata():
            text = " ".join((info.description or "").split())[:120]
            lines.append(f"  {info.name}: {info.label}"
                         + (f" — {text}" if text else ""))
    lines += [
        "",
        "Say what one row of this table describes: the subject schema, every "
        "entity a row carries, and the key columns of each. Connections "
        "between them are asked for separately.",
    ]
    result = "\n".join(lines)
    assert STRUCTURE_MARKER in result
    return result


def binding_prompt(
    profiles: list[ColumnProfile],
    pairs: dict[str, list[str]],
    subject: str | None,
    declared: dict[str, str],
    cat: Catalogue,
    edges: list[dict] | None = None,
    *,
    entities: list[dict],
    all_profiles: list[ColumnProfile] | None = None,
) -> str:
    """`pairs` is each column's closed candidate list, already the
    `"entity|prop"` strings the grammar offers — not the bare property
    shortlist. The entity a property would attach to is explicit in the text,
    `person (Person) -> Person:name`, not left for the model to infer from the
    declared-entities line: prompt wording changes what the model declares, and
    showing a property without its entity describes a question the grammar no
    longer asks.
    """
    edge_keys = {e["key"] for e in edges} if edges else set()
    links = ", ".join(f"{k} ({v})" for k, v in declared.items()
                      if k in edge_keys)

    by_id = {p.id: p for p in (all_profiles or profiles)}
    by_key = {e.get("key"): e for e in entities}

    def _identified(key: str) -> str:
        e = by_key.get(key)
        return ", " + _entity_identity((e or {}).get("keys"), key, by_id)

    party_keys = [k for k in declared if k not in edge_keys]
    if party_keys:
        roster = ["declared entities:"] + [
            f"  {k} ({declared[k]}){_identified(k)}" for k in party_keys]
    else:
        roster = ["declared entities: none"]
    lines = [f"subject: {subject}", *roster]
    if links:
        lines.append(f"declared relationships: {links}")
        lines.append(
            "A relationship is not a party. Bind a column to one only when the "
            "column describes the RELATIONSHIP itself — when it started, what "
            "share it covers, what role it confers. A column holding a party's "
            "name, code or address belongs to that party, never to the link "
            "between them.")
    lines.append("")
    for p in profiles:
        label = f" | label: {p.label}" if p.label else ""
        lines.append(f"{p.id}: {p.header or '(no header)'}{label}")
        lines.append(f"  filled {p.fill_rate:.0%}"
                     + (f" ({p.sentinel_share:.0%} of it a null placeholder)"
                        if p.sentinel_share > 0.2 else "")
                     + f" | distinct {p.distinct_ratio:.0%}"
                     f" | shape {_fmt_shapes(p)} | looks like {_fmt_detectors(p)}")
        lines.append("  values: " + (" | ".join(p.samples) if p.samples else "-"))
        cands = []
        for pair in pairs.get(p.id, []):
            entity_key, sep, qname = pair.partition("|")
            if not sep:
                entity_key, qname = None, entity_key
            info = cat.prop(qname)
            qtext = f"{qname} ({info.label})" if info else qname
            if entity_key is None:
                cands.append(qtext)
                continue
            schema = declared.get(entity_key, "?")
            cands.append(f"{entity_key} ({schema}) -> {qtext}")
        lines.append("  candidates: " + ("; ".join(cands) if cands else "none"))
        lines.append("")
    seen: dict[str, str] = {}
    for column_pairs in pairs.values():
        for pair in column_pairs:
            qname = pair.rpartition("|")[2]
            if qname in seen:
                continue
            info = cat.prop(qname)
            text = (info.description or "").strip() if info else ""
            if text and info and text != info.label:
                seen[qname] = " ".join(text.split())[:120]
    if seen:
        lines.append("what these properties mean, in the ontology's own words:")
        for qname in sorted(seen):
            lines.append(f"  {qname}: {seen[qname]}")
        lines.append("")
    lines.append('Assign one candidate or "unmapped" to every column listed above.')
    result = "\n".join(lines)
    assert BINDING_MARKER in result
    return result


def repair_prompt(failures: list[dict]) -> str:
    """Ask again about the columns whose first answer did not survive."""
    lines = [
        "These assignments were rejected. Reconsider only these columns.",
        "",
    ]
    for f in failures:
        lines.append(f"{f['column']}: you chose {f['prop']}")
        lines.append(f"  rejected by: {f['reason']}")
        if f.get("rejected"):
            lines.append("  the column's values did not pass the check that "
                         "property makes of a value — its type's validator, or "
                         "the format the property declares on top of it")
            lines.append("  values it rejected: " + " | ".join(f["rejected"]))
        else:
            lines.append("  no value was at fault: the column's own header "
                         "states what it measures, and that contradicts what "
                         "the property means. A different property of the same "
                         "type will be refused for the same reason.")
        lines.append("")
    lines.append('Answer again for these columns only. "unmapped" is acceptable.')
    result = "\n".join(lines)
    assert REPAIR_MARKER in result
    return result
