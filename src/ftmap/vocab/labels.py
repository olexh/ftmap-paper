"""Ukrainian labels for FollowTheMoney properties."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from importlib import resources

from ftmap.vocab.catalogue import Catalogue


@lru_cache(maxsize=1)
def load_label_entries() -> dict[str, dict]:
    """`labels_uk.json`, metadata keys dropped. Parsed once, read two ways."""
    raw = json.loads(
        resources.files("ftmap.vocab").joinpath("labels_uk.json").read_text("utf-8")
    )
    return {k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict)}


class Labels:
    def __init__(self, entries: dict[str, dict]) -> None:
        self._entries = entries
        by_name: dict[str, list[str]] = defaultdict(list)
        for qname in entries:
            by_name[qname.split(":", 1)[1]].append(qname)
        self._by_name = dict(by_name)
        self._unambiguous = {
            name: qnames[0]
            for name, qnames in by_name.items()
            if len({entries[q]["label"] for q in qnames}) == 1
        }
        self._cat = Catalogue.load()
        self._resolved: dict[str, dict | None] = {}

    @classmethod
    def load(cls) -> "Labels":
        return cls(load_label_entries())

    def _resolve(self, qname: str) -> dict | None:
        try:
            return self._resolved[qname]
        except KeyError:
            pass
        entry = self._resolve_uncached(qname)
        self._resolved[qname] = entry
        return entry

    def _resolve_uncached(self, qname: str) -> dict | None:
        if qname in self._entries:
            return self._entries[qname]
        if ":" not in qname or qname.startswith("_"):
            return None
        schema, name = qname.split(":", 1)
        if self._cat.prop(qname) is None:
            return None
        ancestors = [q for q in self._by_name.get(name, ())
                     if self._cat.is_descendant(schema, q.split(":", 1)[0])]
        if ancestors:
            ancestors.sort(key=lambda q: sum(
                1 for other in ancestors
                if self._cat.is_descendant(q.split(":", 1)[0], other.split(":", 1)[0])
            ), reverse=True)
            return self._entries[ancestors[0]]
        fallback = self._unambiguous.get(name)
        return self._entries[fallback] if fallback else None

    def label_uk(self, qname: str) -> str | None:
        entry = self._resolve(qname)
        return entry.get("label") if entry else None

    def hint(self, qname: str) -> str | None:
        entry = self._resolve(qname)
        return entry.get("hint") if entry else None
