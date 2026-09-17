"""The property vocabulary, read from the installed FollowTheMoney model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from followthemoney import model as ftm_model


@dataclass(frozen=True)
class PropInfo:
    qname: str
    schema: str
    name: str
    label: str
    description: str
    type_name: str
    matchable: bool
    format: str | None = None
    range_schema: str | None = None
    deprecated: bool = False


@dataclass(frozen=True)
class EdgeInfo:
    schema: str
    source_prop: str
    target_prop: str
    source_range: str
    target_range: str


@dataclass(frozen=True)
class SchemaInfo:
    """A concrete schema as the ontology describes it — the name the grammar
    offers, the label a person reads, and the one-line description that is
    the only evidence a prompt can give about what the name means."""
    name: str
    label: str
    description: str


class Catalogue:
    def __init__(self) -> None:
        self._by_schema: dict[str, dict[str, PropInfo]] = {}
        self._by_qname: dict[str, PropInfo] = {}
        self._concrete: list[str] = []
        self._schema_info: dict[str, SchemaInfo] = {}
        self._edges: list[EdgeInfo] = []
        self._ancestors: dict[str, set[str]] = {}
        self._entity_property_ranges: dict[str, set[str]] = {}

        for name, schema in ftm_model.schemata.items():
            props: dict[str, PropInfo] = {}
            ranges: set[str] = set()
            for pname, prop in schema.properties.items():
                if prop.stub:
                    continue
                info = PropInfo(
                    qname=f"{name}:{pname}",
                    schema=name,
                    name=pname,
                    label=prop.label or pname,
                    description=prop.description or "",
                    type_name=prop.type.name,
                    matchable=bool(prop.matchable),
                    format=prop.format or None,
                    range_schema=(prop.range.name
                                  if prop.type.name == "entity" and prop.range
                                  else None),
                    deprecated=bool(getattr(prop, "deprecated", False)),
                )
                props[pname] = info
                self._by_qname[info.qname] = info
                if prop.type.name == "entity" and prop.range is not None:
                    ranges.add(prop.range.name)
            self._by_schema[name] = props
            self._entity_property_ranges[name] = ranges
            if not schema.abstract:
                self._concrete.append(name)
                self._schema_info[name] = SchemaInfo(
                    name=name, label=schema.label or name,
                    description=schema.description or "")
            self._ancestors[name] = {s.name for s in schema.schemata}
            if schema.edge and schema.edge_source and schema.edge_target:
                src = schema.properties[schema.edge_source]
                tgt = schema.properties[schema.edge_target]
                self._edges.append(
                    EdgeInfo(
                        schema=name,
                        source_prop=schema.edge_source,
                        target_prop=schema.edge_target,
                        source_range=src.range.name if src.range else "Thing",
                        target_range=tgt.range.name if tgt.range else "Thing",
                    )
                )
        self._concrete.sort()
        self._edges.sort(key=lambda e: e.schema)

        self._concrete_names = tuple(self._concrete)
        self._concrete_set = frozenset(self._concrete)
        self._schemata = tuple(self._schema_info[name] for name in self._concrete)
        self._all_props = tuple(self._by_qname.values())
        self._edges_view = tuple(self._edges)
        self._edge_index = MappingProxyType({e.schema: e for e in self._edges})
        self._edge_schemata = frozenset(e.schema for e in self._edges)
        self._props_view = {name: MappingProxyType(props)
                            for name, props in self._by_schema.items()}
        self._no_props: Mapping[str, PropInfo] = MappingProxyType({})

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "Catalogue":
        return cls()

    def concrete_schemata(self) -> list[str]:
        """The 64 schemata the grammar may name, alphabetically."""
        return list(self._concrete_names)

    def is_concrete(self, schema: str) -> bool:
        """Whether `schema` is one FollowTheMoney instantiates."""
        return schema in self._concrete_set

    def schemata(self) -> tuple[SchemaInfo, ...]:
        """Every concrete schema with its label and description, in the same
        order `concrete_schemata` offers them to the grammar."""
        return self._schemata

    def properties_of(self, schema: str) -> Mapping[str, PropInfo]:
        """The properties `schema` carries, keyed by bare name. Read-only."""
        return self._props_view.get(schema, self._no_props)

    def prop(self, qname: str) -> PropInfo | None:
        return self._by_qname.get(qname)

    def all_props(self) -> tuple[PropInfo, ...]:
        return self._all_props

    def edges(self) -> tuple[EdgeInfo, ...]:
        return self._edges_view

    def edge_index(self) -> Mapping[str, EdgeInfo]:
        """Edge schema name to its endpoint declaration. Read-only."""
        return self._edge_index

    def edge_schemata(self) -> frozenset[str]:
        """The names of the schemata FollowTheMoney declares as edges."""
        return self._edge_schemata

    def is_descendant(self, child: str, parent: str) -> bool:
        """True when `child` is `parent` or inherits from it."""
        return parent in self._ancestors.get(child, set())

    def related_to(self, a: str, b: str) -> bool:
        """True when `a` and `b` stand on one line of descent, either way up.
        """
        return self.is_descendant(a, b) or self.is_descendant(b, a)

    def is_party(self, schema: str) -> bool:
        """True when `schema` is a thing that can be a party — a `LegalEntity`.
        """
        return self.is_descendant(schema, "LegalEntity")

    def carries(self, schema: str, qname: str) -> bool:
        """True when an entity declared as `schema` can hold `qname`."""
        info = self.prop(qname)
        return info is not None and self.is_descendant(schema, info.schema)

    def declared_on(self, qname: str) -> str | None:
        """The schema that DECLARES the property behind `qname`, which is not
        the schema in the qname: `Sanction:publisher` is `Interval`'s, inherited.
        None for a qname the ontology does not know."""
        schema, _, name = qname.partition(":")
        sc = ftm_model.get(schema)
        prop = sc.get(name) if sc is not None else None
        return None if prop is None else prop.schema.name

    def is_abstract(self, schema: str) -> bool:
        """Whether FollowTheMoney marks the schema abstract — `Interval`,
        `Thing`, `Value`: the roots no entity is ever an instance of."""
        sc = ftm_model.get(schema)
        return bool(sc is not None and sc.abstract)

    def is_entity_reference(self, qname: str) -> bool:
        """True when `qname` is satisfied by a REFERENCE to another entity
        rather than by a value.
        """
        info = self.prop(qname)
        return info is not None and info.type_name == "entity"

    def bindable(self, schema: str, qname: str) -> bool:
        """True when a COLUMN may be bound to `qname` on an entity declared as
        `schema`.
        """
        return self.carries(schema, qname) and not self.is_entity_reference(qname)

    def entity_property_ranges(self, schema: str) -> set[str]:
        """The schemata reachable through `schema`'s own entity-typed
        properties.
        """
        return set(self._entity_property_ranges.get(schema, set()))
