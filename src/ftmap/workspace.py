"""What a workspace contains, decided by its manifest rather than by `ls`."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from ftmap.fsutil import (PRIVATE, PUBLISHED, SnapshotStore, dump_json,
                          durable_write, now_iso, sync_dir, sync_file)

MANIFEST = "workspace.json"
LEGACY_MANIFEST = "run.json"
SOURCES = "sources"
AGGREGATES = "aggregates"
DECISIONS = "decisions"
MANIFEST_LOCK = "workspace"
STAGING = ".staging"
DERIVED = ".derived"
LOCKS = ".locks"

FORMAT = 2

SUMMARY = "summary.json"
MAPPING = "mapping.yml"
PROFILE_JSON = "profile.json"
PLAN_JSON = "plan.json"
PLAN_VALIDATED_JSON = "plan.validated.json"
DECISIONS_JSONL = "decisions.jsonl"
NORMALIZED_CSV = "normalized.csv"
REJECTS_JSONL = "rejects.jsonl"
ENTITIES_JSON = "entities.ftm.json"
STATEMENTS_CSV = "statements.csv"
FAILURES_JSONL = "failures.jsonl"
RUN_JSON = "run.json"

_NOT_A_SOURCE = re.compile(r"^\.|\.rollback-|^_")


class WorkspaceError(RuntimeError):
    """Base for every refusal this module makes."""


class LegacyWorkspace(WorkspaceError):
    """A durable write was attempted against retained evidence."""


class GenerationConflict(WorkspaceError):
    """Another writer holds this workspace, or committed under us."""


class UnknownSource(WorkspaceError):
    """A safe_id the manifest does not name."""


def new_generation_id() -> str:
    """Sortable, unique, and readable in that order."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}Z-{uuid.uuid4().hex[:8]}"


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return default


@dataclass(frozen=True)
class SourceEntry:
    """One committed source, as the manifest names it."""

    safe_id: str
    generation: str
    path: str
    source_id: str = ""
    sha256: str = ""
    sheet: str = ""
    origin: str = ""
    run_id: str = ""
    committed_at: str = ""
    override_revision: int = 0
    approved_generation: str = ""
    approved_override_revision: int = -1


class Workspace:
    """The manifest, and every path that follows from it."""

    def __init__(self, root: str, mode: str, manifest: dict) -> None:
        self.root = os.path.abspath(root)
        self.mode = mode
        self.manifest = manifest
        self._held: set[str] = set()
        self._scanned: list[str] | None = None

    @property
    def writable(self) -> bool:
        return self.mode == "generation"

    def require_writable(self) -> None:
        if self.writable:
            return
        raise LegacyWorkspace(
            f"{self.root} is a {self.mode} workspace and is retained evidence: "
            "it is the corpus a published measurement was read from, and "
            "rewriting it in place would destroy what it is kept for. Run "
            f"`ftmap upgrade {self.root} <new-destination>` to build a "
            "generation-aware copy, and write there.")

    def source_ids(self) -> list[str]:
        """Every source this workspace currently publishes, in order."""
        if self.mode == "generation":
            return sorted(self.manifest.get("sources", {}))
        if self.mode == "legacy-manifest":
            return list(self.manifest.get("summaries", []))
        if self._scanned is None:
            names = []
            for name in sorted(os.listdir(self.root)):
                if _NOT_A_SOURCE.search(name):
                    continue
                if os.path.exists(os.path.join(self.root, name, SUMMARY)):
                    names.append(name)
            self._scanned = names
        return list(self._scanned)

    def entry(self, safe_id: str) -> SourceEntry | None:
        """The manifest's record for one source, or None if it has none."""
        if self.mode != "generation":
            return None
        raw = self.manifest.get("sources", {}).get(safe_id)
        if raw is None:
            return None
        known = SourceEntry.__dataclass_fields__
        return SourceEntry(safe_id=safe_id,
                           **{k: v for k, v in raw.items() if k in known})

    def source_dir(self, safe_id: str) -> str:
        """Where `safe_id`'s CURRENT committed artefacts are."""
        if self.mode == "generation":
            entry = self.entry(safe_id)
            if entry is None:
                raise UnknownSource(
                    f"{safe_id!r} is not named by {self.root}/{MANIFEST}")
            return os.path.join(self.root, entry.path)
        if safe_id not in self.source_ids():
            raise UnknownSource(f"{safe_id!r} is not a source of {self.root}")
        return os.path.join(self.root, safe_id)

    def mapping_path(self, safe_id: str) -> str:
        """The mapping document, INSIDE the source's own generation."""
        if self.mode == "generation":
            return os.path.join(self.source_dir(safe_id), MAPPING)
        return os.path.join(self.root, "mappings", safe_id + ".yml")

    def decisions_path(self, safe_id: str) -> str:
        """Analyst decisions, OUTSIDE every generation."""
        if self.mode == "generation":
            return os.path.join(self.root, DECISIONS, safe_id + ".jsonl")
        return os.path.join(self.root, safe_id, "overrides.jsonl")

    def aggregate_dir(self) -> str | None:
        if self.mode != "generation":
            return self.root
        aggregate = self.manifest.get("aggregate")
        if not aggregate:
            return None
        return os.path.join(self.root, aggregate["path"])

    def report(self) -> dict:
        """The corpus manifest a reader means by `run.json`."""
        directory = self.aggregate_dir()
        if directory is None:
            return {}
        return _read_json(os.path.join(directory, LEGACY_MANIFEST), {})

    def stale_sources(self) -> list[str]:
        """Sources committed since the aggregate the workspace publishes."""
        if self.mode != "generation":
            return []
        aggregate = self.manifest.get("aggregate") or {}
        covered = aggregate.get("sources") or {}
        stale = [name for name, raw in self.manifest.get("sources", {}).items()
                 if covered.get(name) != raw.get("generation")]
        return sorted(stale)

    def publication_grade(self) -> bool:
        """Whether this workspace may be described as a publication artefact.
        """
        aggregate = self.manifest.get("aggregate")
        if not aggregate or aggregate.get("incomplete"):
            return False
        if self.report().get("accounting_valid") is not True:
            return False
        return not self.stale_sources()

    def staging_dir(self, label: str = "") -> str:
        """A path nobody can discover a source at."""
        name = f"{uuid.uuid4().hex}{'-' + label if label else ''}"
        path = os.path.join(self.root, STAGING, name)
        os.makedirs(path, mode=0o700, exist_ok=True)
        return path

    def commit_source(self, safe_id: str, staged: str, entry: dict,
                      generation: str | None = None) -> str:
        """Rename a complete staged directory into its generation, then name
        it.
        """
        self.require_writable()
        generation = generation or new_generation_id()
        relative = os.path.join(SOURCES, safe_id, generation)
        final = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        _sync_tree(staged)
        os.rename(staged, final)
        sync_dir(os.path.dirname(final))
        sync_dir(os.path.join(self.root, SOURCES))

        with self._updating_manifest() as manifest:
            manifest.setdefault("sources", {})[safe_id] = {
                **entry, "generation": generation, "path": relative,
                "committed_at": now_iso(),
            }
        return final

    def commit_aggregate(self, staged: str, report: dict, *,
                         incomplete: bool = False,
                         failures: list | None = None) -> str:
        """The same protocol, for the corpus-wide exports."""
        self.require_writable()
        generation = new_generation_id()
        relative = os.path.join(AGGREGATES, generation)
        final = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        _sync_tree(staged)
        os.rename(staged, final)
        sync_dir(os.path.dirname(final))

        with self._updating_manifest() as manifest:
            manifest["aggregate"] = {
                "generation": generation,
                "path": relative,
                "sources": {name: raw["generation"]
                            for name, raw in manifest.get("sources", {}).items()
                            if name in set(report.get("summaries") or [])},
                "incomplete": bool(incomplete),
                "failures": failures or [],
                "committed_at": now_iso(),
            }
        return final

    def commit_decisions(self, safe_id: str, length: int, digest: str) -> None:
        """Name how much of a decision log the workspace stands behind."""
        self.require_writable()
        with self._updating_manifest() as manifest:
            manifest.setdefault("decisions", {})[safe_id] = {
                "bytes": length, "sha256": digest,
                "at": now_iso()}

    @contextmanager
    def _updating_manifest(self) -> Iterator[dict]:
        """The read-modify-write of `workspace.json`, serialized under ONE
        lock.
        """
        with self.lock(MANIFEST_LOCK):
            manifest = self._load_for_update()
            yield manifest
            self._commit_manifest(manifest)

    def _load_for_update(self) -> dict:
        """Re-read from disk before every commit."""
        manifest = _read_json(os.path.join(self.root, MANIFEST), None)
        if manifest is None:
            manifest = _blank_manifest()
        return manifest

    def _commit_manifest(self, manifest: dict) -> None:
        manifest["format"] = FORMAT
        manifest["updated_at"] = now_iso()
        path = os.path.join(self.root, MANIFEST)
        with durable_write(path, mode=PUBLISHED, text=True) as fh:
            dump_json(manifest, fh)
        self.manifest = manifest

    @contextmanager
    def lock(self, name: str = "workspace") -> Iterator[None]:
        """One writer at a time, refused rather than queued."""
        if name in self._held:
            yield
            return
        with file_lock(os.path.join(self.root, LOCKS), name, self.root):
            self._held.add(name)
            try:
                yield
            finally:
                self._held.discard(name)

    def active(self) -> list[str]:
        """Which locks are held BY A LIVE PROCESS. Prune refuses while any is.
        """
        directory = os.path.join(self.root, LOCKS)
        if not os.path.isdir(directory):
            return []
        return sorted(name for name in os.listdir(directory)
                      if name.endswith(".lock")
                      and _alive(_read_json(os.path.join(directory, name),
                                            {}).get("pid")))


def _take(path: str, token: str) -> None:
    """Publish a fully-written lock file at `path`, atomically."""
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".lock-")
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "token": token,
                                 "at": now_iso()}
                                ).encode("utf-8"))
        os.fchmod(fd, PRIVATE)
        os.close(fd)
        fd = -1
        try:
            os.link(tmp, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(path) from None
            raise
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


@contextmanager
def file_lock(directory: str, name: str, described_as: str) -> Iterator[None]:
    """One holder at a time for `name` under `directory`, refused not queued.
    """
    os.makedirs(directory, mode=0o700, exist_ok=True)
    path = os.path.join(directory, name.replace("/", "_").replace(":", "_")
                        + ".lock")
    token = uuid.uuid4().hex
    try:
        _take(path, token)
    except FileExistsError:
        holder = _read_json(path, {})
        if _alive(holder.get("pid")):
            raise GenerationConflict(
                f"{described_as} is being written by pid {holder.get('pid')} "
                f"since {holder.get('at')} ({name}). Two writers would "
                "each commit a manifest that never saw the other's work; "
                "wait for it to finish.") from None
        claimed = f"{path}.stale-{uuid.uuid4().hex}"
        try:
            os.rename(path, claimed)
        except OSError:
            raise GenerationConflict(
                f"{described_as} was locked by a dead pid "
                f"{holder.get('pid')} and another writer claimed it "
                "first; retry.") from None
        try:
            os.unlink(claimed)
        except FileNotFoundError:
            pass
        try:
            _take(path, token)
        except FileExistsError:
            raise GenerationConflict(
                f"{described_as} was locked by a dead pid "
                f"{holder.get('pid')} and another writer claimed it "
                "first; retry.") from None
    try:
        yield
    finally:
        if _read_json(path, {}).get("token") == token:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _alive(pid) -> bool:
    """Whether a recorded lock holder is still running."""
    if pid is None:
        return False
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sync_tree(path: str) -> None:
    """Flush every staged file and directory before the rename that publishes.
    """
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            try:
                sync_file(fd)
            finally:
                os.close(fd)
        sync_dir(dirpath)


def _blank_manifest() -> dict:
    return {"format": FORMAT, "sources": {}, "aggregate": None,
            "decisions": {},
            "created_at": now_iso()}


def create_workspace(root: str) -> Workspace:
    """Make a new generation-mode workspace, or open an existing one."""
    os.makedirs(root, exist_ok=True)
    if os.path.exists(os.path.join(root, MANIFEST)):
        return open_workspace(root)
    existing = open_workspace(root)
    if existing.source_ids():
        raise LegacyWorkspace(
            f"{existing.root} already holds a {existing.mode} run with "
            f"{len(existing.source_ids())} sources. Writing a {MANIFEST} into "
            "it would claim a layout its artefacts do not have. Run "
            f"`ftmap upgrade {existing.root} <new-destination>` instead.")
    workspace = Workspace(root, "generation", _blank_manifest())
    workspace._commit_manifest(workspace.manifest)
    return workspace


def open_workspace(root: str) -> Workspace:
    """Decide which of the three modes `root` is, and read it accordingly."""
    absolute = os.path.abspath(root)
    manifest = _read_json(os.path.join(absolute, MANIFEST), None)
    if manifest is not None:
        version = manifest.get("format", 1)
        if version > FORMAT:
            raise WorkspaceError(
                f"{absolute}/{MANIFEST} is format {version}; this ftmap "
                f"understands {FORMAT}. A newer version wrote it, and reading "
                "it as if it were older is how a manifest stops meaning what "
                "it says.")
        return Workspace(absolute, "generation", manifest)

    legacy = _read_json(os.path.join(absolute, LEGACY_MANIFEST), None)
    if legacy is not None:
        return Workspace(absolute, "legacy-manifest", legacy)
    return Workspace(absolute, "legacy-scan", {})


def upgrade(source_root: str, dest_root: str) -> Workspace:
    """Copy a legacy workspace into a new generation-mode one."""
    old = open_workspace(source_root)
    if old.mode == "generation":
        raise WorkspaceError(f"{old.root} is already generation-aware")
    if os.path.exists(dest_root) and os.listdir(dest_root):
        raise WorkspaceError(
            f"{dest_root} exists and is not empty; upgrade writes a new "
            "workspace and will not merge into one that already holds data")

    new = create_workspace(dest_root)
    with new.lock("upgrade"):
        for safe_id in old.source_ids():
            staged = new.staging_dir(label="upgrade")
            shutil.copytree(old.source_dir(safe_id), staged,
                            dirs_exist_ok=True)
            legacy_mapping = os.path.join(old.root, "mappings", safe_id + ".yml")
            if os.path.exists(legacy_mapping):
                shutil.copy2(legacy_mapping, os.path.join(staged, MAPPING))
            carried = os.path.join(staged, "overrides.jsonl")
            summary = _read_json(os.path.join(staged, SUMMARY), {})
            if os.path.exists(carried):
                target = new.decisions_path(safe_id)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(carried, target)
                os.chmod(target, PRIVATE)
                os.unlink(carried)
            new.commit_source(safe_id, staged, {
                "source_id": (summary.get("source") or {}).get("id", ""),
                "sha256": (summary.get("source") or {}).get("sha256", ""),
                "sheet": (summary.get("source") or {}).get("sheet", ""),
                "origin": summary.get("path", ""),
                "run_id": summary.get("run_id", ""),
            })
    return open_workspace(dest_root)


def _tree_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


class Retention:
    """What a workspace could reclaim, and what it must not."""

    KEEP = 2

    def __init__(self, workspace: "Workspace") -> None:
        self.workspace = workspace

    def _generations(self, kind: str) -> dict[str, list[str]]:
        """Every generation on disk, per source (or the single aggregate key).
        """
        root = os.path.join(self.workspace.root, kind)
        if not os.path.isdir(root):
            return {}
        if kind == AGGREGATES:
            return {"": sorted(os.listdir(root))}
        return {name: sorted(os.listdir(os.path.join(root, name)))
                for name in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, name))}

    def _on_disk(self) -> dict[str, dict[str, list[str]]]:
        """Both scans, once. `reclaimable` asked for them six times — twice
        itself, twice inside `_keep`, and twice more inside the second `_keep`
        `_referenced_snapshots` was making — and each is a full `listdir` of
        every source directory."""
        return {kind: self._generations(kind)
                for kind in (SOURCES, AGGREGATES)}

    def _keep(self, on_disk: dict | None = None) -> set[str]:
        """Absolute paths nothing may remove."""
        on_disk = self._on_disk() if on_disk is None else on_disk
        keep: set[str] = set()
        manifest = self.workspace.manifest
        for safe_id, generations in on_disk[SOURCES].items():
            current = (manifest.get("sources", {}).get(safe_id) or {}).get(
                "generation")
            ordered = [g for g in sorted(generations, reverse=True)
                       if g != current]
            for generation in ([current] if current else []) + \
                    ordered[:self.KEEP - 1]:
                keep.add(os.path.join(self.workspace.root, SOURCES, safe_id,
                                      generation))
        aggregate = (manifest.get("aggregate") or {}).get("generation")
        ordered = [g for g in sorted(on_disk[AGGREGATES].get("", []),
                                     reverse=True) if g != aggregate]
        for generation in ([aggregate] if aggregate else []) + \
                ordered[:self.KEEP - 1]:
            keep.add(os.path.join(self.workspace.root, AGGREGATES, generation))
        return keep

    def _referenced_snapshots(self, keep: set[str] | None = None) -> set[str]:
        """Snapshot files any RETAINED generation still needs."""
        keep = self._keep() if keep is None else keep
        referenced: set[str] = set()
        store = os.path.join(self.workspace.root, SnapshotStore.DIRNAME)
        for directory in keep:
            summary = _read_json(os.path.join(directory, SUMMARY), {})
            digest = (summary.get("source") or {}).get("sha256")
            if not digest:
                continue
            prefix = os.path.join(store, digest[:2])
            if os.path.isdir(prefix):
                referenced.update(os.path.join(prefix, name)
                                  for name in os.listdir(prefix)
                                  if name.startswith(digest))
        return referenced

    def reclaimable(self) -> dict:
        """What prune WOULD remove, without removing it."""
        on_disk = self._on_disk()
        keep = self._keep(on_disk)
        generations, snapshots, staging = [], [], []
        for kind in (SOURCES, AGGREGATES):
            for name, found in on_disk[kind].items():
                base = os.path.join(self.workspace.root, kind, name)
                for generation in found:
                    path = os.path.join(base, generation)
                    if path not in keep and os.path.isdir(path):
                        generations.append(path)

        store = os.path.join(self.workspace.root, SnapshotStore.DIRNAME)
        referenced = self._referenced_snapshots(keep)
        if os.path.isdir(store):
            for prefix in sorted(os.listdir(store)):
                directory = os.path.join(store, prefix)
                if not os.path.isdir(directory):
                    continue
                for name in sorted(os.listdir(directory)):
                    path = os.path.join(directory, name)
                    if path not in referenced:
                        snapshots.append(path)

        staging_root = os.path.join(self.workspace.root, STAGING)
        if os.path.isdir(staging_root):
            staging = [os.path.join(staging_root, name)
                       for name in sorted(os.listdir(staging_root))]

        derived = []
        derived_root = os.path.join(self.workspace.root, DERIVED)
        if os.path.isdir(derived_root):
            kept_generations = {os.path.basename(p) for p in keep}
            derived = [os.path.join(derived_root, name)
                       for name in sorted(os.listdir(derived_root))
                       if name not in kept_generations]

        return {
            "generations": sorted(generations),
            "snapshots": sorted(snapshots),
            "staging": sorted(staging),
            "derived": sorted(derived),
            "bytes": sum(_tree_bytes(p)
                         for p in generations + staging + derived)
            + sum(os.path.getsize(p) for p in snapshots
                  if os.path.exists(p)),
        }

    def prune(self) -> dict:
        """Remove exactly what `reclaimable` named, under the exclusive lock.
        """
        self.workspace.require_writable()
        with self.workspace.lock("prune"):
            held = [name for name in self.workspace.active()
                    if name != "prune.lock"]
            if held:
                raise GenerationConflict(
                    f"{self.workspace.root} is in use ({', '.join(held)}). "
                    "Pruning while a reader holds a superseded generation "
                    "would fail its request with a truncated file. Stop the "
                    "server and any running command first.")
            plan = self.reclaimable()
            for path in plan["generations"] + plan["staging"] + plan["derived"]:
                shutil.rmtree(path, ignore_errors=True)
            for path in plan["snapshots"]:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            return plan
