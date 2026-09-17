"""One etalon file: its shape, and everything checkable about it."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from ftmap.vocab.catalogue import Catalogue

VERSION = 2

ROLES = {
    "property",
    "key",
    "key+property",
    "unmappable",
    "not-data",
}
CARRIES_PROPERTY = {"property", "key+property"}
IS_KEY = {"key", "key+property"}
NO_HOME = {"unmappable", "not-data"}


class EtalonError(ValueError):
    """The etalon file is wrong. Not the pipeline."""


def _related(a: str, b: str, cat) -> bool:
    """One schema is the other, or descends from it."""
    return a == b or cat.is_descendant(a, b) or cat.is_descendant(b, a)


@dataclass(frozen=True)
class Column:
    id: str
    header: str | None
    role: str
    answer: str | None
    accept: tuple[str, ...] = ()
    accept_decline: bool = False
    accept_binding: tuple[str, ...] = ()
    caveat: str = ""
    why: str = ""

    @property
    def carries_property(self) -> bool:
        return self.role in CARRIES_PROPERTY

    @property
    def is_key(self) -> bool:
        return self.role in IS_KEY

    @property
    def has_home(self) -> bool:
        return self.role not in NO_HOME

    def better_than_declining(self, qname: str) -> bool:
        return qname in self.accept_binding


ROW, TABLE = "row", "table"
SCOPES = {ROW, TABLE}


@dataclass(frozen=True)
class Entity:
    key: str
    schema: str
    keys: tuple[str, ...]
    accept: tuple[str, ...] = ()
    scope: str = ROW
    producible: bool = True
    instances: int | None = None
    rows: tuple[str, str] | None = None
    why: str = ""

    columns: tuple[str, ...] = ()

    def allows_related(self, schema: str, cat) -> bool:
        """As `allows`, but a subschema of an accepted answer is accepted too.
        """
        return (_related(schema, self.schema, cat)
                or any(_related(schema, a, cat) for a in self.accept))


LINK, PROPERTY = "link", "property"
EDGE_KINDS = {LINK, PROPERTY}


@dataclass(frozen=True)
class Edge:
    key: str
    schema: str
    source: str
    target: str
    kind: str = LINK
    prop: str | None = None
    accept: tuple[str, ...] = ()
    why: str = ""


@dataclass(frozen=True)
class Etalon:
    path: str
    sha256: str
    sheet: str
    subject: str
    columns: tuple[Column, ...]
    entities: tuple[Entity, ...]
    edges: tuple[Edge, ...]
    subject_accept: tuple[str, ...] = ()
    produces_nothing: str = ""
    notes: str = ""
    file: str | None = None

    def allows_subject(self, schema: str | None) -> bool:
        return schema == self.subject or schema in self.subject_accept


_REQUIRED = ("source", "subject", "columns", "entities", "edges")


def load(path: str, cat: Catalogue | None = None) -> Etalon:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse(raw, cat, file=path)


def parse(raw: object, cat: Catalogue | None = None,
          file: str | None = None) -> Etalon:
    cat = cat or Catalogue.load()
    if not isinstance(raw, dict):
        raise EtalonError("an etalon is a mapping at the top level")
    version = raw.get("etalon")
    if version != VERSION:
        raise EtalonError(
            f"etalon version {version!r}; this loader reads {VERSION}. "
            f"Version 1 had no subject, no entity keys and no edges, so a "
            f"version-1 file cannot be scored on three of the four layers."
        )
    for key in _REQUIRED:
        if key not in raw:
            raise EtalonError(
                f"missing {key!r}. Every question is answered explicitly; an "
                f"absent key is an unasked question, not an empty answer."
            )
    source = raw["source"] or {}
    if not source.get("sha256"):
        raise EtalonError(
            "source.sha256 is required: an etalon is an answer about a "
            "specific sequence of bytes, and a path is not one."
        )

    produces_nothing = str(raw.get("produces_nothing", "") or "")
    columns = _columns(raw["columns"], cat)
    entities = _entities(raw["entities"], columns, cat, produces_nothing)
    edges = _edges(raw["edges"], entities, cat)
    subject = raw["subject"]
    accept: tuple[str, ...] = ()
    if isinstance(subject, dict):
        accept = tuple(subject.get("accept") or ())
        subject = subject.get("answer")
    for schema in (subject, *accept):
        if not cat.is_concrete(schema):
            raise EtalonError(f"subject {schema!r} is not a concrete FtM schema")

    sole_claimant: dict[str, str] = {}
    for column in columns:
        if not column.answer:
            continue
        schema = column.answer.split(":", 1)[0]
        claimants = [e for e in entities if _related(schema, e.schema, cat)]
        if len(claimants) == 1:
            sole_claimant[column.id] = claimants[0].key

    def _owned(entity: Entity) -> tuple[str, ...]:
        own = list(entity.keys)
        own += [cid for cid, key in sole_claimant.items()
                if key == entity.key and cid not in own]
        return tuple(dict.fromkeys(own))

    entities = tuple(Entity(**{**e.__dict__, "columns": _owned(e)})
                     for e in entities)
    _cross_check(columns, entities, edges, cat)
    return Etalon(
        path=source.get("path", ""),
        sha256=str(source["sha256"]),
        sheet=str(source.get("sheet", "") or ""),
        subject=subject,
        subject_accept=accept,
        columns=columns,
        entities=entities,
        edges=edges,
        produces_nothing=produces_nothing,
        notes=raw.get("notes", "") or "",
        file=file,
    )


def _columns(raw: object, cat: Catalogue) -> tuple[Column, ...]:
    if not isinstance(raw, list) or not raw:
        raise EtalonError("columns must be a non-empty list")
    out: list[Column] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        cid = str(item.get("id", ""))
        if not (cid.startswith("c") and cid[1:].isdigit()):
            raise EtalonError(f"column {i}: id {cid!r} is not of the form c<n>")
        if cid in seen:
            raise EtalonError(f"column {cid} is declared twice")
        seen.add(cid)
        role = item.get("role")
        if role not in ROLES:
            raise EtalonError(
                f"{cid}: role {role!r} is not one of {sorted(ROLES)}")
        answer = item.get("answer")
        accept = tuple(item.get("accept") or ())
        if role in CARRIES_PROPERTY and not answer:
            raise EtalonError(f"{cid}: role {role!r} needs an answer qname")
        if role not in CARRIES_PROPERTY and answer:
            raise EtalonError(
                f"{cid}: role {role!r} carries no property, so answer "
                f"{answer!r} cannot be scored; use role 'property' or drop it")
        better = tuple(item.get("accept_binding") or ())
        if better and role in CARRIES_PROPERTY:
            raise EtalonError(
                f"{cid}: `accept_binding` is for a column the etalon DECLINES. "
                f"This one carries a property, so a rival answer belongs in "
                f"`accept`.")
        for qname in filter(None, (answer, *accept, *better)):
            _check_qname(cid, qname, cat)
        out.append(Column(id=cid, header=item.get("header"), role=role,
                          answer=answer, accept=accept,
                          accept_decline=bool(item.get("accept_decline")),
                          accept_binding=better,
                          caveat=item.get("caveat", "") or "",
                          why=item.get("why", "") or ""))
    return tuple(out)


def _check_qname(where: str, qname: str, cat: Catalogue) -> None:
    """The check that would have caught `Vehicle:serialNumber` in phase 1."""
    if ":" not in qname:
        raise EtalonError(f"{where}: {qname!r} is not a Schema:property qname")
    schema = qname.split(":", 1)[0]
    if not cat.is_concrete(schema):
        raise EtalonError(f"{where}: {schema!r} is not a concrete FtM schema")
    if cat.prop(qname) is None:
        raise EtalonError(
            f"{where}: {schema} does not carry {qname.split(':', 1)[1]!r}. "
            f"Check whether a subschema declares it — an answer naming a "
            f"property the schema lacks scores the pipeline against an "
            f"ontology that does not exist."
        )


def _entities(raw: object, columns: tuple[Column, ...], cat: Catalogue,
              produces_nothing: str = "") -> tuple[Entity, ...]:
    if not isinstance(raw, list):
        raise EtalonError("entities must be a list")
    if not raw:
        if not produces_nothing:
            raise EtalonError(
                "entities is empty. A table that should produce no entity is a "
                "real answer — a codelist, a legend, a lookup consumed during "
                "another sheet's ingest — but it is a CLAIM, so say it: set "
                "`produces_nothing:` to the reason. An empty list with no "
                "reason is an omission.")
        if any(c.has_home for c in columns):
            raise EtalonError(
                "produces_nothing says this table yields no entity, but a "
                "column is answered with a property home. One of the two is "
                "wrong.")
        return ()
    if produces_nothing:
        raise EtalonError(
            "produces_nothing and a non-empty entities list contradict each "
            "other")
    ids = {c.id for c in columns}
    out: list[Entity] = []
    seen: set[str] = set()
    for item in raw:
        key = str(item.get("key", ""))
        if not key:
            raise EtalonError("every etalon entity needs a key")
        if key in seen:
            raise EtalonError(f"entity key {key!r} is declared twice")
        seen.add(key)
        schema = item.get("schema")
        accept = tuple(item.get("accept") or ())
        for name in (schema, *accept):
            if not cat.is_concrete(name):
                raise EtalonError(
                    f"entity {key}: {name!r} is not a concrete FtM schema")
        scope = item.get("scope", ROW)
        if scope not in SCOPES:
            raise EtalonError(
                f"entity {key}: scope {scope!r} is not one of {sorted(SCOPES)}")
        producible = item.get("producible", True)
        keys = tuple(item.get("keys") or ())
        if not producible and keys:
            raise EtalonError(
                f"entity {key}: producible: false says no column can produce "
                f"this entity, so it cannot also be keyed on {list(keys)}")
        if "keys" not in item and producible:
            raise EtalonError(
                f"entity {key}: `keys` is required. An entity with no key "
                f"emits one un-deduplicated copy per row — 30 courts where "
                f"there is one — and phase 1 could not see that because it "
                f"never asked. Write `keys: []` to mean the row ordinal is "
                f"the only identity available.")
        for cid in keys:
            if cid not in ids:
                raise EtalonError(
                    f"entity {key}: key column {cid!r} is not declared")
        selector = item.get("rows")
        rows_sel: tuple[str, str] | None = None
        if selector is not None:
            if (not isinstance(selector, dict)
                    or set(selector) != {"column", "value"}):
                raise EtalonError(
                    f"entity {key}: rows must be a mapping of `column` and "
                    f"`value`, naming the rows this entity is built from")
            if selector["column"] not in ids:
                raise EtalonError(
                    f"entity {key}: rows column {selector['column']!r} is not "
                    f"declared")
            rows_sel = (str(selector["column"]), str(selector["value"]))
        instances = item.get("instances")
        if instances is not None and (not isinstance(instances, int)
                                      or isinstance(instances, bool)
                                      or instances < 1):
            raise EtalonError(
                f"entity {key}: instances must be a positive whole number of "
                f"instances the source holds, not {instances!r}")
        out.append(Entity(key=key, schema=schema, keys=keys, accept=accept,
                          scope=scope, producible=bool(producible),
                          instances=instances, rows=rows_sel,
                          why=item.get("why", "") or ""))
    return tuple(out)


def _edges(raw: object, entities: tuple[Entity, ...],
           cat: Catalogue) -> tuple[Edge, ...]:
    if not isinstance(raw, list):
        raise EtalonError("edges must be a list; write `edges: []` for none")
    by_schema = {e.schema: e for e in cat.edges()}
    keys = {e.key: e for e in entities}
    out: list[Edge] = []
    for item in raw:
        key = str(item.get("key", ""))
        kind = item.get("kind", LINK)
        if kind not in EDGE_KINDS:
            raise EtalonError(f"edge {key}: kind {kind!r} is not one of "
                              f"{sorted(EDGE_KINDS)}")
        for role in ("source", "target"):
            if item.get(role) not in keys:
                raise EtalonError(
                    f"edge {key}: {role} {item.get(role)!r} is not a declared "
                    f"entity")
        schema = item.get("schema")
        prop = item.get("prop")
        if kind == PROPERTY:
            if not prop:
                raise EtalonError(
                    f"edge {key}: kind 'property' needs the qname that carries "
                    f"the reference, e.g. prop: Airplane:operator")
            _check_qname(f"edge {key}", prop, cat)
            if not cat.is_entity_reference(prop):
                raise EtalonError(
                    f"edge {key}: {prop} is not entity-typed, so it is a value "
                    f"a cell can hold and belongs in `columns`, not here")
            if schema and schema != prop.split(":", 1)[0]:
                raise EtalonError(
                    f"edge {key}: schema {schema!r} does not own {prop}")
            schema = prop.split(":", 1)[0]
            src = keys[item["source"]].schema
            if not cat.is_descendant(src, schema) and src != schema:
                raise EtalonError(
                    f"edge {key}: {prop} is declared on {schema}, but the "
                    f"source entity is {src}")
        else:
            info = by_schema.get(schema)
            if info is None:
                raise EtalonError(
                    f"edge {key}: {schema!r} is not an edge schema in this "
                    f"ontology. FollowTheMoney expresses a relation only where "
                    f"it declares one — `CourtCase:court` is a string, so no "
                    f"CourtCase-to-PublicBody edge exists to expect. If the "
                    f"relation is an entity-typed property, say kind: property.")
            for role in ("source", "target"):
                want = (info.source_range if role == "source"
                        else info.target_range)
                got = keys[item[role]].schema
                if not cat.is_descendant(got, want):
                    raise EtalonError(
                        f"edge {key}: {role} is {got}, but {schema}'s {role} "
                        f"ranges on {want}")
        accept = tuple(item.get("accept") or ())
        if accept and kind != LINK:
            raise EtalonError(
                f"edge {key}: `accept` is for a link edge; a property edge "
                f"is one qname and another qname is another claim")
        for other in accept:
            alt = by_schema.get(other)
            if alt is None:
                raise EtalonError(
                    f"edge {key}: accepted {other!r} is not an edge schema in "
                    f"this ontology")
            for role in ("source", "target"):
                want = (alt.source_range if role == "source"
                        else alt.target_range)
                got = keys[item[role]].schema
                if not cat.is_descendant(got, want):
                    raise EtalonError(
                        f"edge {key}: accepts {other}, but its {role} is {got} "
                        f"and {other}'s {role} ranges on {want}; an accepted "
                        f"reading the endpoints cannot satisfy is not a reading")
        out.append(Edge(key=key, schema=schema, source=item["source"],
                        target=item["target"], kind=kind, prop=prop,
                        accept=accept, why=item.get("why", "") or ""))
    return tuple(out)


def _cross_check(columns: tuple[Column, ...], entities: tuple[Entity, ...],
                 edges: tuple[Edge, ...], cat: Catalogue) -> None:
    """The answers have to agree with each other."""
    schemata = ({e.schema for e in entities if e.producible}
                | {s for e in entities if e.producible for s in e.accept}
                | {g.schema for g in edges if g.kind == LINK})
    for col in columns:
        if not col.carries_property:
            continue
        schema = col.answer.split(":", 1)[0]
        if not any(cat.is_descendant(s, schema) or cat.is_descendant(schema, s)
                   for s in schemata):
            raise EtalonError(
                f"{col.id}: answered {col.answer} but no declared entity is a "
                f"{schema} or a relative of one. Declare the entity that "
                f"carries it, or the property has nowhere to live.")
    declared_keys = {cid for e in entities for cid in e.keys}
    for col in columns:
        if col.is_key and col.id not in declared_keys:
            raise EtalonError(
                f"{col.id}: role {col.role!r} says it identifies the row, but "
                f"no declared entity keys on it")
