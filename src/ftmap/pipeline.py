"""One source through every stage, and a corpus through every source."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone

import followthemoney

from ftmap import __version__ as FTMAP_VERSION
from ftmap.build.emit import (Tally, bound_evidence, column_order,
                              merge_totals, statement_writer, summarise_tally,
                              write_decisions, write_entities)
from ftmap.build.execute import execute_into
from ftmap.fsutil import (PRIVATE, PUBLISHED, SnapshotStore, dump_json,
                          durable_write, json_line, now_iso, open_private,
                          sha256_file)
from ftmap.inventory import SourceRef, inventory
from ftmap.io.frame import Frame
from ftmap.io.layout import build_frame
from ftmap.io.tabular import read_source
from ftmap.normalize.report import reject_writer
from ftmap.plan.compile import (compile_queries, ensure_mappable_root,
                                normalize_frame, unmappable_reason,
                                write_mapping)
from ftmap.plan.propose import propose
from ftmap.plan.validate import Decision, ValidatedPlan, validate
from ftmap.profile.columns import profile_frame
from ftmap.workspace import (DECISIONS_JSONL, ENTITIES_JSON,
                             FAILURES_JSONL, MAPPING, NORMALIZED_CSV,
                             PLAN_JSON, PLAN_VALIDATED_JSON,
                             PROFILE_JSON, REJECTS_JSONL, RUN_JSON,
                             SOURCES, STATEMENTS_CSV, SUMMARY, Workspace,
                             create_workspace, new_generation_id,
                             open_workspace)


class IncompleteGeneration(RuntimeError):
    """A staged generation failed its checks and was never committed."""


def model_provenance(client) -> dict[str, str]:
    """Where this run's answers came from, or an explicit statement that the
    question does not apply.
    """
    provenance = getattr(client, "model_provenance", None)
    if callable(provenance):
        return provenance()
    return {"revision": "not_applicable", "producing_model": "not_applicable"}


def prompt_digests() -> dict[str, str]:
    """A short digest of each system prompt, for the run manifest."""
    from ftmap.plan.prompt import BINDING_SYSTEM, EDGE_SYSTEM, STRUCTURE_SYSTEM

    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        for name, text in (("structure", STRUCTURE_SYSTEM),
                           ("edge", EDGE_SYSTEM),
                           ("binding", BINDING_SYSTEM))
    }


def _write_json(obj, path: str, private: bool = True) -> None:
    """Private by default, because the exceptions are the short list."""
    with durable_write(path, mode=PRIVATE if private else PUBLISHED,
                       text=True) as fh:
        dump_json(obj, fh, newline=False)


class _StageClock:
    """What each stage of one source cost, and whether the model was involved.
    """

    def __init__(self, client) -> None:
        self._client = client
        self.stages: dict[str, dict] = {}

    def _snapshot(self) -> dict | None:
        stages = getattr(self._client, "stages", None)
        return copy.deepcopy(stages) if isinstance(stages, dict) else None

    def start(self, name: str) -> None:
        self._name = name
        self._t0 = time.monotonic()
        self._before = self._snapshot()

    def stop(self) -> None:
        row: dict = {"duration_seconds": round(time.monotonic() - self._t0, 3)}
        after = self._snapshot()
        if after is not None and self._before is not None:
            calls = {}
            for stage, counters in after.items():
                delta = {k: counters[k] - self._before[stage][k]
                         for k in counters}
                if delta["attempts"] or delta["failures"]:
                    calls[stage] = delta
            row["calls"] = calls
            generated = sum(d["generated"] for d in calls.values())
            attempts = sum(d["attempts"] for d in calls.values())
            row["mode"] = ("live" if generated else
                           "replay" if attempts else "none")
        self.stages[self._name] = row


class WorkbookCache:
    """Parse a workbook once per run, not once per sheet."""

    __slots__ = ("_path", "_grids")

    def __init__(self) -> None:
        self._path: str | None = None
        self._grids: list | None = None

    def grids(self, path: str) -> list:
        if path != self._path:
            self._path, self._grids = None, None
            self._grids = read_source(path)
            self._path = path
        return self._grids


def _staged_artifacts(staged: str) -> dict[str, str]:
    """SHA-256 of everything about to be committed, for the manifest."""
    out = {}
    for name in sorted(os.listdir(staged)):
        path = os.path.join(staged, name)
        if not os.path.isfile(path):
            continue
        out[name] = sha256_file(path)
    return out


REQUIRED = (PROFILE_JSON, PLAN_JSON, PLAN_VALIDATED_JSON, DECISIONS_JSONL,
            NORMALIZED_CSV, REJECTS_JSONL, ENTITIES_JSON, STATEMENTS_CSV,
            SUMMARY)


def _verify_staged(staged: str, safe_id: str, run_id: str,
                   declined: bool) -> dict[str, str]:
    """The gate between "written" and "committed"."""
    missing = [name for name in REQUIRED
               if not os.path.exists(os.path.join(staged, name))]
    if missing:
        raise IncompleteGeneration(
            f"{safe_id}: staged generation is missing {missing}")

    mapping = os.path.join(staged, MAPPING)
    if declined and os.path.exists(mapping):
        raise IncompleteGeneration(
            f"{safe_id}: the plan declared no entity, so there is no mapping "
            "to publish — but one is staged")
    if not declined and not os.path.exists(mapping):
        raise IncompleteGeneration(f"{safe_id}: no mapping document staged")

    with open(os.path.join(staged, SUMMARY), encoding="utf-8") as fh:
        summary = json.load(fh)
    if summary.get("safe_id") != safe_id or summary.get("run_id") != run_id:
        raise IncompleteGeneration(
            f"{safe_id}: staged summary says it belongs to "
            f"{summary.get('safe_id')!r} of run {summary.get('run_id')!r}")
    return _staged_artifacts(staged)


def run_source(ref: SourceRef, cfg, cat, client, run_id: str, out_dir: str,
               overrides: dict[str, dict] | None = None,
               cache: WorkbookCache | None = None,
               workspace: Workspace | None = None,
               override_revision: int = 0) -> dict:
    """One source, staged whole and committed atomically."""
    ws = workspace or create_workspace(out_dir)
    ws.require_writable()
    ensure_mappable_root(ws.root)
    if ref.snapshot is not None:
        SnapshotStore(ws.root).verify(ref.sha256, ref.ext)
    read_from = ref.read_path
    grids = cache.grids(read_from) if cache is not None else read_source(read_from)
    grid = grids[ref.sheet_index]
    frame = build_frame(grid, ref.path, ref.sha256, ref.source_id, cfg)

    generation = new_generation_id()
    staged = ws.staging_dir(label=ref.safe_id)
    final_dir = os.path.join(ws.root, SOURCES, ref.safe_id, generation)
    mapping_path = os.path.join(staged, MAPPING)

    clock = _StageClock(client)
    try:
        clock.start("profile")
        profiles = profile_frame(frame, cfg)
        _write_json([p.to_dict() for p in profiles],
                    os.path.join(staged, PROFILE_JSON))
        clock.stop()

        clock.start("propose")
        plan = propose(frame, profiles, cat, cfg, client)
        _write_json(plan.to_dict(), os.path.join(staged, PLAN_JSON))
        clock.stop()

        clock.start("validate")
        vplan = validate(plan, profiles, cat, cfg, frame, client=client)
        clock.stop()
        clock.start("emit")

        decided_by = {dec.column: dec.decided_by for dec in vplan.decisions}
        if overrides:
            vplan = _apply_overrides(vplan, overrides, cat)
            decided_by.update({c: "analyst" for c in overrides})
        declined = unmappable_reason(vplan)
        if declined is not None:
            vplan.decisions.append(
                Decision(None, None, None, "declined", declined, "rule"))

        _write_json(vplan.to_dict(), os.path.join(staged, PLAN_VALIDATED_JSON))
        write_decisions(vplan.decisions, os.path.join(staged, DECISIONS_JSONL))

        staged_csv = os.path.join(staged, NORMALIZED_CSV)
        tally = Tally()
        rejects_path = os.path.join(staged, REJECTS_JSONL)
        statements_path = os.path.join(staged, STATEMENTS_CSV)
        entities_path = os.path.join(staged, ENTITIES_JSON)
        evidence = bound_evidence(frame, vplan)

        with reject_writer(rejects_path) as write_reject, \
                statement_writer(statements_path) as write_statement:
            def keep_reject(reject):
                tally.add_reject(reject)
                write_reject(reject)

            def keep_statement(statement):
                tally.add_statement(statement, cat)
                write_statement(statement)

            for reject in normalize_frame(frame, vplan, staged_csv, evidence):
                keep_reject(reject)
            entities = _emit(final_dir, frame, vplan, cat, run_id, declined,
                             decided_by, staged_csv, mapping_path,
                             keep_statement, keep_reject, evidence)
        for entity in entities:
            tally.add_entity(entity)
        write_entities(entities, entities_path)
        clock.stop()

        summary = summarise_tally(
            frame, vplan, tally, cat, cfg,
            {"run_id": run_id, "safe_id": ref.safe_id, "path": ref.path,
             "mapping": (os.path.join(final_dir, MAPPING)
                         if declined is None else None)},
            declined=declined, evidence=evidence)
        summary["stages"] = clock.stages
        _write_json(summary, os.path.join(staged, SUMMARY),
                    private=False)

        artifacts = _verify_staged(staged, ref.safe_id, run_id,
                                   declined is not None)
        with ws.lock(f"source:{ref.safe_id}"):
            ws.commit_source(ref.safe_id, staged, {
                "source_id": ref.source_id,
                "sha256": ref.sha256,
                "sheet": ref.sheet,
                "origin": ref.path,
                "run_id": run_id,
                "artifacts": artifacts,
                "override_revision": override_revision,
            }, generation=generation)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise

    return summary


def _emit(final_dir: str, frame: Frame, vplan: ValidatedPlan, cat,
          run_id: str, declined: str | None, decided_by: dict[str, str],
          staged_csv: str, mapping_path: str, on_statement, on_reject,
          evidence: dict | None = None) -> list[dict]:
    """Compile, publish the mapping, and run the engine into the sinks."""
    if declined is not None:
        return []

    queries = compile_queries(vplan, frame,
                              os.path.join(final_dir, NORMALIZED_CSV))
    write_mapping(queries, mapping_path, frame.source_id)

    staged_url = "file://" + os.path.abspath(staged_csv)
    return execute_into(
        [{**query, "csv_url": staged_url} for query in queries],
        vplan, frame, cat, run_id, decided_by,
        on_statement=on_statement, on_reject=on_reject, evidence=evidence)

def _apply_overrides(vplan, overrides: dict[str, dict], cat):
    """Analyst decisions outrank the model — but not physics."""
    from ftmap.plan.claims import override_decline_reason, resolve_claims
    from ftmap.review.decide import declared_keys, judge

    declared = declared_keys(vplan.entities, vplan.edges)

    bindings = [b for b in vplan.bindings if b["column"] not in overrides]
    for column, ov in overrides.items():
        verdict = judge(column, ov.get("prop"), ov.get("entity"), declared,
                        cat, ov.get("why", ""))
        if not verdict.accepted:
            vplan.decisions.append(Decision(column, verdict.prop,
                                            verdict.entity, "rejected",
                                            verdict.reason, "analyst"))
            continue
        if verdict.clears:
            vplan.decisions.append(Decision(column, None, None, "unmapped",
                                            verdict.reason, "analyst"))
            continue
        bindings.append({"column": column, "prop": verdict.prop,
                         "entity": verdict.entity,
                         "type_name": cat.prop(verdict.prop).type_name,
                         "why": verdict.reason})
        vplan.decisions.append(Decision(column, verdict.prop, verdict.entity,
                                        "accepted", verdict.reason, "analyst"))
    bindings, displaced = resolve_claims(
        bindings, lambda b: 1.0 if b["column"] in overrides else 0.0,
        merge=False)
    for loser, winner in displaced:
        vplan.decisions.append(Decision(
            loser["column"], loser["prop"], loser["entity"], "unmapped",
            override_decline_reason(loser, winner), "analyst"))
    vplan.bindings = sorted(bindings, key=lambda b: column_order(b["column"]))
    return vplan


def _failure_code(exc: Exception) -> str:
    """A stable identifier for WHERE an exception was raised, not what its
    message says. The message carries the model's own words — `client.py`'s
    `ModelError` embeds up to 200 characters of the raw answer, whose free-text
    `why` was measured to quote real cell values — so `run.json`, published at
    0644, must never hold it. Two failures from one line get one code, so the
    manifest can say how many sources failed for the same reason without a byte
    of the reason; the exception text and traceback go only to
    `failures.jsonl`, mode 0600.
    """
    tb = exc.__traceback__
    frame = traceback.extract_tb(tb)[-1] if tb else None
    site = f"{os.path.basename(frame.filename)}:{frame.lineno}" if frame else "?"
    return hashlib.sha256(f"{type(exc).__name__}:{site}".encode()).hexdigest()[:8]


def select_sources(refs: list, only: list[str] | None) -> list:
    """The sources an `--only` asks for: every ref whose path, sheet or source
    id contains one of the given substrings, in inventory order.
    """
    if not only:
        return refs
    return [r for r in refs
            if any(needle in r.path or needle in (r.sheet or "")
                   or needle in r.source_id for needle in only)]


def run_corpus(root: str, cfg, cat, client, out_dir: str, limit: int | None = None,
               on_progress=None, stop=None, only: list[str] | None = None) -> dict:
    """Every source under `root`, through every stage."""
    ensure_mappable_root(out_dir)
    ws = create_workspace(out_dir)
    from ftmap.review.store import DecisionStore

    store = DecisionStore(ws.root)
    all_refs = inventory(root, snapshots=SnapshotStore(ws.root))
    duplicates = [{"path": r.path, "sheet": r.sheet,
                   "duplicate_of": r.duplicate_of}
                  for r in all_refs if r.duplicate_of]
    refs = [r for r in all_refs if r.duplicate_of is None]
    refs = select_sources(refs, only)
    if limit:
        refs = refs[:limit]
    started_at = datetime.now(timezone.utc)
    run_id = (f"{os.path.basename(os.path.abspath(root))}-"
             f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}")

    summaries: list[dict] = []
    failures: list[dict] = []
    cache = WorkbookCache()
    cancelled = False
    with ws.lock():
        for index, ref in enumerate(refs):
            if stop is not None and stop():
                cancelled = True
                break
            event = {"safe_id": ref.safe_id, "path": ref.path,
                     "sheet": ref.sheet, "index": index, "total": len(refs)}
            if on_progress is not None:
                on_progress({"kind": "source_started", **event})
            try:
                summaries.append(run_source(
                    ref, cfg, cat, client, run_id, ws.root, cache=cache,
                    workspace=ws,
                    overrides=store.overrides(ref.safe_id),
                    override_revision=store.override_revision(ref.safe_id)))
            except Exception as exc:
                failures.append({"path": ref.path, "sheet": ref.sheet,
                                 "error_type": type(exc).__name__,
                                 "code": _failure_code(exc),
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc(limit=4)})
                if on_progress is not None:
                    on_progress({"kind": "source_failed", **event})
            else:
                if on_progress is not None:
                    on_progress({"kind": "source_done", **event})

        return _commit_aggregate(
            ws, root, cfg, client, run_id, started_at, summaries, failures,
            duplicates, cancelled)


def _commit_aggregate(ws, root, cfg, client, run_id, started_at, summaries,
                      failures: list[dict], duplicates: list[dict],
                      cancelled) -> dict:
    """Merge every committed source into one aggregate generation."""
    staged = ws.staging_dir(label="aggregate")

    if failures:
        with open_private(os.path.join(staged, FAILURES_JSONL)) as fh:
            for f in failures:
                fh.write(json_line(f))

    merged_entities = os.path.join(staged, ENTITIES_JSON)
    merged_statements = os.path.join(staged, STATEMENTS_CSV)
    seen: set[str] = set()
    with open_private(merged_entities) as out:
        for s in summaries:
            path = os.path.join(ws.source_dir(s["safe_id"]), ENTITIES_JSON)
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    eid = json.loads(line)["id"]
                    if eid in seen:
                        continue
                    seen.add(eid)
                    out.write(line)

    header_written = False
    with open_private(merged_statements) as out:
        for s in summaries:
            path = os.path.join(ws.source_dir(s["safe_id"]), STATEMENTS_CSV)
            with open(path, encoding="utf-8") as fh:
                head = fh.readline()
                if not header_written:
                    out.write(head)
                    header_written = True
                for line in fh:
                    out.write(line)

    declined = [s for s in summaries if s.get("declined")]

    entity_counts: Counter = Counter()
    column_total = column_mapped = statement_total = 0
    for s in summaries:
        entity_counts.update(s["entities"])
        column_total += s["columns"]["total"]
        column_mapped += s["columns"]["mapped"]
        statement_total += s["statements"]

    report = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": now_iso(),
        "ftmap_version": FTMAP_VERSION,
        "followthemoney_version": followthemoney.__version__,
        "model_id": model_provenance(client)["producing_model"],
        "model_revision": model_provenance(client)["revision"],
        "model_server": model_provenance(client).get("server", "not_applicable"),
        "prompts": prompt_digests(),
        "root": os.path.abspath(root),
        "sources": len(summaries),
        "declined": len(declined),
        "declined_sources": [{"path": s["path"],
                              "sheet": (s.get("source") or {}).get("sheet"),
                              "safe_id": s["safe_id"]} for s in declined],
        "failed": len(failures),
        "cancelled": cancelled,
        "failures": [{"path": f["path"], "sheet": f["sheet"],
                      "error_type": f["error_type"], "code": f["code"]}
                     for f in failures],
        "duplicates": len(duplicates),
        "duplicate_sources": [dict(d) for d in duplicates],
        "columns": {"total": column_total, "mapped": column_mapped},
        "statements": statement_total,
        "coverage": (corpus_coverage :=
                     merge_totals([s["coverage_totals"] for s in summaries])),
        "coverage_schema": max([s.get("coverage_schema", 1)
                                for s in summaries], default=1),
        "accounting_valid": corpus_coverage["accounting_valid"],
        "entities": dict(entity_counts),
        "model_calls": getattr(client, "calls", None),
        "cache_hits": getattr(client, "cache_hits", None),
        "stages": copy.deepcopy(getattr(client, "stages", None)),
        "config": cfg.as_manifest(),
        "summaries": [s["safe_id"] for s in summaries],
        "incomplete": bool(cancelled or failures),
    }
    _write_json(report, os.path.join(staged, RUN_JSON), private=False)
    ws.commit_aggregate(staged, report,
                        incomplete=bool(cancelled or failures),
                        failures=report["failures"])
    return report


class _CarriedProvenance:
    """The previous run's model provenance, replayed into a republished
    aggregate.
    """

    def __init__(self, prior: dict):
        self._prior = prior
        self.calls = prior.get("model_calls")
        self.cache_hits = prior.get("cache_hits")

    def model_provenance(self) -> dict[str, str]:
        return {"revision": self._prior.get("model_revision") or "unknown",
                "producing_model": self._prior.get("model_id") or "unknown",
                "server": self._prior.get("model_server") or "unknown"}


class PublishRefused(RuntimeError):
    """A corpus publish was asked for while the workspace disagrees with itself."""


def publish_corpus(out_dir: str, cfg, cat, store=None) -> dict:
    """Rebuild the corpus exports from the CURRENT committed source
    generations.
    """
    from ftmap.review.store import DecisionStore

    ws = open_workspace(out_dir)
    ws.require_writable()
    store = store or DecisionStore(out_dir)

    lagging = []
    for safe_id in ws.source_ids():
        entry = ws.entry(safe_id)
        committed = store.override_revision(safe_id)
        if entry is not None and entry.override_revision != committed:
            lagging.append((safe_id, entry.override_revision, committed))
    if lagging:
        raise PublishRefused(
            "these sources have decisions their committed output does not "
            "reflect; rerun each one first: "
            + ", ".join(f"{name} (ran against revision {applied}, log is at "
                        f"{current})" for name, applied, current in lagging))

    summaries = []
    for safe_id in ws.source_ids():
        path = os.path.join(ws.source_dir(safe_id), SUMMARY)
        with open(path, encoding="utf-8") as fh:
            summaries.append(json.load(fh))

    prior = ws.report()
    with ws.lock():
        report = _commit_aggregate(
            ws, prior.get("root") or out_dir, cfg, _CarriedProvenance(prior),
            f"publish-{new_generation_id()}", datetime.now(timezone.utc),
            summaries, prior.get("failures") or [],
            prior.get("duplicate_sources") or [],
            cancelled=bool(prior.get("cancelled")))
    return report
