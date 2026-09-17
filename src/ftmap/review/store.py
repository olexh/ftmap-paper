"""Analyst decisions, and the cards the app renders."""

from __future__ import annotations

import hashlib
import json
import os

from ftmap.build.emit import UNFALSIFIABLE_TYPES
from ftmap.fsutil import (PRIVATE, json_line, now_iso, open_private, sync_dir,
                          sync_file)
from ftmap.workspace import (PLAN_JSON, PLAN_VALIDATED_JSON, PROFILE_JSON,
                             SUMMARY, UnknownSource)


class DecisionLogCorrupt(RuntimeError):
    """The committed prefix of a decision log no longer hashes to what the
    manifest recorded. Append-only means append-only."""


class ApprovalRefused(RuntimeError):
    """An approval was asked for output that predates the decisions."""


def _read_artefact(path: str, default: list | dict) -> list | dict:
    """One artefact of a committed generation, or `default` if there is none.
    """
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def is_unfalsifiable(info) -> bool:
    """Whether a property's TYPE accepts any string, so no evidential check
    could ever refuse a value bound to it.
    """
    return bool(info) and info.type_name in UNFALSIFIABLE_TYPES


def mark_unfalsifiable(cards: list[dict], cat) -> list[dict]:
    """Flag every card whose bound property's type accepts any string."""
    for card in cards:
        card["unfalsifiable"] = is_unfalsifiable(
            cat.prop(card["prop"]) if card.get("prop") else None)
    return cards


class DecisionStore:
    """Reads the artefacts one source directory holds, and appends decisions.
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir

    @property
    def workspace(self):
        """Re-opened on every access, deliberately."""
        from ftmap.workspace import open_workspace

        return open_workspace(self.out_dir)

    def _open(self, workspace=None):
        """The workspace the CALLER already opened, or a fresh one."""
        return self.workspace if workspace is None else workspace

    def _dir(self, safe_id: str, workspace=None) -> str:
        """The manifest decides, not `os.path.join`."""
        try:
            return self._open(workspace).source_dir(safe_id)
        except UnknownSource:
            return os.path.join(self.out_dir, "\x00absent", safe_id)

    def source_ids(self, workspace=None) -> list[str]:
        """Every source this workspace publishes, in manifest order."""
        try:
            return self._open(workspace).source_ids()
        except FileNotFoundError:
            return []

    def events(self, safe_id: str, workspace=None) -> list[dict]:
        """The committed decision events for one source, in order."""
        workspace = self._open(workspace)
        path = workspace.decisions_path(safe_id)
        if not os.path.exists(path):
            return []
        committed = (workspace.manifest.get("decisions") or {}).get(safe_id)
        if committed is None:
            return []
        with open(path, "rb") as fh:
            raw = fh.read()
        prefix = raw[:committed["bytes"]]
        digest = hashlib.sha256(prefix).hexdigest()
        if digest != committed["sha256"]:
            raise DecisionLogCorrupt(
                f"{path}: the committed prefix hashes to {digest[:12]}, "
                f"not the {committed['sha256'][:12]} recorded for it")
        raw = prefix
        out = []
        for line in raw.decode("utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        return out

    @staticmethod
    def _overrides_of(events: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for rec in events:
            if rec.get("kind") not in (None, "override", "clear"):
                continue
            if rec.get("cleared") or rec.get("kind") == "clear":
                out.pop(rec["column"], None)
            else:
                out[rec["column"]] = rec
        return out

    @staticmethod
    def _revision_of(events: list[dict]) -> int:
        return sum(1 for rec in events
                   if rec.get("kind") in (None, "override", "clear"))

    @staticmethod
    def _approval_of(events: list[dict]) -> dict | None:
        """The approval still standing, or None. A `reopen` withdraws it."""
        approved_at = None
        for rec in events:
            if rec.get("kind") == "approve":
                approved_at = rec
            elif rec.get("kind") == "reopen":
                approved_at = None
        return approved_at

    @classmethod
    def _review_state_of(cls, events: list[dict], entry) -> dict:
        approved_at = cls._approval_of(events)
        generation = entry.generation if entry else ""
        revision = cls._revision_of(events)
        if approved_at is None:
            state = "pending"
        elif (approved_at.get("generation") == generation
              and approved_at.get("override_revision") == revision):
            state = "approved"
        else:
            state = "stale"
        return {"state": state, "override_revision": revision,
                "generation": generation,
                "approved_generation": (approved_at or {}).get("generation"),
                "approved_override_revision":
                    (approved_at or {}).get("override_revision")}

    def overrides(self, safe_id: str, workspace=None) -> dict[str, dict]:
        """The current override per column: last write wins, tombstones drop."""
        return self._overrides_of(self.events(safe_id, workspace))

    def override_revision(self, safe_id: str, workspace=None) -> int:
        """How many MAPPING-AFFECTING decisions this source has committed."""
        return self._revision_of(self.events(safe_id, workspace))

    def review_state(self, safe_id: str, workspace=None) -> dict:
        """Approval and override as two independent projections."""
        workspace = self._open(workspace)
        return self._review_state_of(self.events(safe_id, workspace),
                                     workspace.entry(safe_id))

    def review(self, safe_id: str, workspace=None) -> dict:
        """Everything one dashboard row needs about one source's review, from
        ONE read of its decision log and ONE look at the manifest.
        """
        workspace = self._open(workspace)
        events = self.events(safe_id, workspace)
        entry = workspace.entry(safe_id)
        return {"overrides": self._overrides_of(events),
                "review_state": self._review_state_of(events, entry),
                "applied": entry.override_revision if entry else 0}

    def record(self, safe_id: str, column: str, prop: str | None,
               entity: str | None, why: str) -> None:
        """Append the analyst's decision for one column. Outranks the model."""
        self._append(safe_id, {
            "kind": "override",
            "column": column, "prop": prop, "entity": entity, "why": why,
            "decided_by": "analyst", "at": now_iso(),
        })

    def clear(self, safe_id: str, column: str) -> None:
        """Remove the override on read, without deleting its history."""
        self._append(safe_id, {
            "kind": "clear",
            "column": column, "cleared": True,
            "at": now_iso(),
        })

    def approve(self, safe_id: str) -> dict:
        """Record that an analyst stands by THIS generation of this source."""
        workspace = self.workspace
        entry = workspace.entry(safe_id)
        if entry is None:
            raise UnknownSource(safe_id)
        revision = self.override_revision(safe_id, workspace)
        if entry.override_revision != revision:
            raise ApprovalRefused(
                f"{safe_id} was last run against override revision "
                f"{entry.override_revision} and its decisions are now at "
                f"{revision}. Rerun the source first — approving now would "
                "record agreement with output that predates the decision.")
        self._append(safe_id, {
            "kind": "approve", "generation": entry.generation,
            "override_revision": revision,
            "at": now_iso(),
        })
        return self.review_state(safe_id)

    def reopen(self, safe_id: str) -> dict:
        """Withdraw approval explicitly, without manufacturing an override."""
        self._append(safe_id, {
            "kind": "reopen", "at": now_iso()})
        return self.review_state(safe_id)

    def _append(self, safe_id: str, record: dict) -> None:
        """Append durably, then commit the new length to the manifest."""
        workspace = self.workspace
        workspace.require_writable()
        path = workspace.decisions_path(safe_id)
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with workspace.lock(f"decisions:{safe_id}"):
            new_file = not os.path.exists(path)
            with open_private(path, PRIVATE, append=True) as fh:
                fh.write(json_line(record))
                fh.flush()
                sync_file(fh.fileno())
                length = os.fstat(fh.fileno()).st_size
            if new_file:
                sync_dir(directory)
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read(length)).hexdigest()
            workspace.commit_decisions(safe_id, length, digest)

    def summary(self, safe_id: str, workspace=None) -> dict:
        """One source's whole `summary.json`, or `{}` if it has none."""
        return _read_artefact(
            os.path.join(self._dir(safe_id, workspace), SUMMARY), {})

    def summaries(self, workspace=None) -> dict[str, dict]:
        """Every source's whole `summary.json`, keyed by safe_id."""
        workspace = self._open(workspace)
        return {safe_id: self.summary(safe_id, workspace)
                for safe_id in self.source_ids(workspace)}

    def source(self, safe_id: str, workspace=None) -> dict:
        """Join profile, validated plan and overrides into one card per column."""
        workspace = self._open(workspace)
        d = self._dir(safe_id, workspace)
        summary = _read_artefact(os.path.join(d, SUMMARY), {})
        profiles = _read_artefact(os.path.join(d, PROFILE_JSON), [])
        vplan = _read_artefact(os.path.join(d, PLAN_VALIDATED_JSON), {})
        plan = _read_artefact(os.path.join(d, PLAN_JSON), {})
        overrides = self.overrides(safe_id, workspace)

        bindings = {}
        for b in vplan.get("bindings", []):
            if b["column"] not in bindings or bindings[b["column"]].get("replica"):
                bindings[b["column"]] = b
        decisions = {}
        for dec in vplan.get("decisions", []):
            if dec.get("column") is not None:
                decisions[dec["column"]] = dec
        shortlists = plan.get("shortlists", {})

        cards = []
        for p in profiles:
            col = p["id"]
            binding = bindings.get(col, {})
            dec = decisions.get(col, {})
            card = {
                "column": col,
                "header": p.get("header"),
                "label": p.get("label"),
                "group": p.get("group"),
                "fill_rate": p.get("fill_rate"),
                "distinct_ratio": p.get("distinct_ratio"),
                "shapes": p.get("shapes", []),
                "samples": p.get("samples", []),
                "detectors": p.get("detectors", {}),
                "prop": binding.get("prop"),
                "entity": binding.get("entity"),
                "why": binding.get("why") or dec.get("reason", ""),
                "verdict": dec.get("verdict", "unmapped"),
                "decided_by": dec.get("decided_by", "model"),
                "candidates": shortlists.get(col, []),
            }
            if col in overrides:
                ov = overrides[col]
                card["prop"] = ov.get("prop")
                card["entity"] = ov.get("entity")
                card["why"] = ov.get("why", "")
                card["decided_by"] = "analyst"
                card["verdict"] = ("unmapped" if ov.get("prop") in (None, "unmapped")
                                   else "accepted")
            cards.append(card)

        return {
            "safe_id": safe_id,
            "summary": summary,
            "entities": vplan.get("entities", []),
            "edges": vplan.get("edges", []),
            "cards": cards,
        }
