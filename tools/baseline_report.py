"""The binding-baseline arms, side by side, on the scorer's own terms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from ftmap.etalon import load as load_etalon
from ftmap.etalon.score import (AGREE, DEFENSIBLE, _label, _same_property,
                                find_summary, score as score_source)
from ftmap.plan.candidates import catalogue_pairs
from ftmap.plan.response_schema import UNMAPPED
from ftmap.vocab.catalogue import Catalogue
from ftmap.workspace import (PLAN_JSON, PLAN_VALIDATED_JSON,
                             open_workspace)

FIXTURE = os.path.join(HERE, "baseline_sources.json")


class BaselineRefused(RuntimeError):
    """The inputs are not the experiment this report is for."""


def load_fixture(path: str = FIXTURE) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_etalons(etalon_dir: str, fixture: dict) -> None:
    """Every frozen etalon present, with the frozen bytes; nothing else."""
    want = {e["file"]: e["sha256"] for e in fixture["etalons"]}
    have = sorted(f for f in os.listdir(etalon_dir) if f.endswith(".etalon.yaml"))
    if set(have) != set(want):
        raise BaselineRefused(
            f"{etalon_dir} holds {len(have)} etalons, the fixture names "
            f"{len(want)}; extra {sorted(set(have) - set(want))}, missing "
            f"{sorted(set(want) - set(have))}")
    for name, sha in want.items():
        with open(os.path.join(etalon_dir, name), "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != sha:
            raise BaselineRefused(f"{name} has sha256 {got[:12]}…, the fixture "
                                  f"froze {sha[:12]}…")


def check_source_set(expected: set[str], scored: set[str],
                     failed: set[str], arm: str) -> set[str]:
    """The scored labels must be the expected ones, less those the run
    itself recorded as failed. Returns the failed-and-expected labels."""
    extra = scored - expected
    if extra:
        raise BaselineRefused(f"{arm}: scored {sorted(extra)}, which the "
                              "fixture does not name")
    missing = expected - scored
    unexplained = missing - failed
    if unexplained:
        raise BaselineRefused(
            f"{arm}: {sorted(unexplained)} neither scored nor recorded as a "
            "failure in run.json; an incomplete run is not this experiment")
    return missing


def _failed_labels(report: dict, etalons: dict) -> set[str]:
    """Labels of the expected sources `run.json` lists under `failures`."""
    out = set()
    for f in report.get("failures") or []:
        for label, doc in etalons.items():
            if os.path.abspath(f.get("path", "")) == os.path.abspath(doc.path) \
                    and (f.get("sheet") or "") == (doc.sheet or ""):
                out.add(label)
    return out


def _artefact(ws, summary: dict, name: str) -> dict:
    path = os.path.join(ws.source_dir(summary["safe_id"]), name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _plan_of(ws, summary: dict) -> dict:
    return _artefact(ws, summary, PLAN_JSON)


def pairs_by_round(plan: dict) -> dict[tuple[int, str], list[str]]:
    """`(round index, column)` to the ORDERED pair list that round offered
    the column. Keyed per round, because a dict keyed by column alone let a
    later round overwrite an earlier one and a changed first-round list
    pass as identical."""
    return {(i, col): list(pairs)
            for i, r in enumerate(plan.get("rounds") or [])
            for col, pairs in (r.get("pairs") or {}).items()}


def binding_inputs(plan: dict) -> dict:
    """What the binding stage was given, less the pairs: what must be equal
    across arms for the comparison to be about binding policy."""
    return {
        "subject": plan.get("subject"),
        "entities": plan.get("entities"),
        "edges": plan.get("edges"),
        "shortlists": plan.get("shortlists"),
        "rounds": [{k: r.get(k) for k in ("declared", "filter", "columns")}
                   for r in plan.get("rounds") or []],
    }


def offered_pairs(plan: dict) -> dict[str, set[str]]:
    """Column id to every qname any round offered it."""
    out: dict[str, set[str]] = {}
    for r in plan.get("rounds") or []:
        for col, pairs in (r.get("pairs") or {}).items():
            out.setdefault(col, set()).update(p.partition("|")[2] for p in pairs)
    return out


def availability(doc, plan: dict, cat) -> dict[str, bool]:
    """Per mappable column: was the etalon's answer, under `_same_property`
    against the answer and its `accept` list, among the qnames offered?"""
    offered = offered_pairs(plan)
    out = {}
    for col in doc.columns:
        if not col.carries_property:
            continue
        wanted = [col.answer, *col.accept]
        out[col.id] = any(_same_property(q, w, cat)
                          for q in offered.get(col.id, ())
                          for w in wanted)
    return out


def reachable(doc, plan: dict, cat) -> dict[str, bool]:
    """Per mappable column, BEFORE retrieval truncates: was an acceptable
    property among every bindable property of the objects declared for the
    column's round — `catalogue_pairs` over the round's `round_declared`, the
    list arm C offered. An observation about where the answer went missing, not
    a cause: a reachable property can still be cut by the shortlist, and an
    offered one lost later.
    """
    by_col: dict[str, set[str]] = {}
    for r in plan.get("rounds") or []:
        declared = dict(r.get("declared") or [])
        qnames = {pair.partition("|")[2] for pair in catalogue_pairs(declared, cat)}
        for col in r.get("columns") or []:
            by_col.setdefault(col, set()).update(qnames)
    out = {}
    for col in doc.columns:
        if not col.carries_property:
            continue
        wanted = [col.answer, *col.accept]
        out[col.id] = any(_same_property(q, w, cat)
                          for q in by_col.get(col.id, ())
                          for w in wanted)
    return out


def _acceptable(qname: str | None, col, cat) -> bool:
    return bool(qname) and any(_same_property(qname, w, cat)
                               for w in (col.answer, *col.accept))


def classify(final_ok: bool, answered: list[str], chooser_ok: bool) -> str:
    """What happened on one column, as an observation: the final binding is
    acceptable; or the chooser's acceptable answer was removed or replaced
    by post-processing; or the chooser answered a different pair; or the
    chooser answered `unmapped` (nothing at all)."""
    if final_ok:
        return "acceptable"
    if chooser_ok:
        return "chooser acceptable, removed or replaced by post-processing"
    if answered:
        return "chooser answered a different pair"
    return "chooser answered unmapped"


def column_join(doc, summary: dict, plan: dict, validated: dict,
                cat) -> list[dict]:
    """One record per mappable column: offered?, final acceptable?, and what
    the chooser did. `verdict` and `final` are the scorer's own, from
    re-scoring the summary; `chooser` is `plan.json`'s binding for the
    column, which is the chooser's answer before `validate` runs."""
    offered = availability(doc, plan, cat)
    within_reach = reachable(doc, plan, cat)
    rows = {r.id: r for r in score_source(doc, summary, cat).columns}
    chooser: dict[str, list[str]] = {}
    for b in plan.get("bindings") or []:
        if b.get("prop") and b["prop"] != UNMAPPED:
            chooser.setdefault(b["column"], []).append(b["prop"])
    out = []
    for col in doc.columns:
        if not col.carries_property:
            continue
        row = rows.get(col.id)
        final_ok = bool(row and row.pipeline and row.verdict in (AGREE, DEFENSIBLE))
        answered = chooser.get(col.id, [])
        chooser_ok = any(_acceptable(q, col, cat) for q in answered)
        what = classify(final_ok, answered, chooser_ok)
        out.append({"column": col.id, "offered": offered[col.id],
                    "reachable": within_reach[col.id],
                    "final": final_ok, "what": what,
                    "chooser_acceptable": chooser_ok,
                    "decline_defensible": bool(col.accept_decline)})
    return out


def heuristic_tally(plan: dict) -> dict[str, int]:
    """What a heuristic arm's `why` strings say it did on this source: how
    many columns the rule bound, how many of those on a lexical score of
    zero (type bonus alone), how many it withheld on a tie between distinct
    properties, and how many for want of any candidate above zero."""
    t = {"bound": 0, "bound_lexical_zero": 0, "tie_distinct": 0,
         "tie_targets": 0, "tie_distinct_taken": 0, "nothing_above_zero": 0}
    for b in plan.get("bindings") or []:
        why = b.get("why") or ""
        if not why.startswith("heuristic:"):
            continue
        if b.get("prop") == UNMAPPED:
            if "distinct properties tie" in why:
                t["tie_distinct"] += 1
            elif "above zero" in why:
                t["nothing_above_zero"] += 1
            continue
        t["bound"] += 1
        if "lexical 0.00" in why:
            t["bound_lexical_zero"] += 1
        if "distinct properties tying" in why:
            t["tie_distinct_taken"] += 1
        elif "targets offering it" in why:
            t["tie_targets"] += 1
    return t


def read_arm(name: str, out_dir: str, etalons: dict, cat) -> dict:
    with open(os.path.join(out_dir, "score.json"), encoding="utf-8") as fh:
        scores = json.load(fh)
    ws = open_workspace(out_dir)
    report = ws.report()
    failed = check_source_set(set(etalons), set(scores),
                              _failed_labels(report, etalons), name)
    sources: dict[str, dict] = {}
    for label, doc in etalons.items():
        if label in failed:
            sources[label] = {"failed": True, "score": None}
            continue
        summary = find_summary(doc, out_dir)
        plan = _plan_of(ws, summary)
        validated = _artefact(ws, summary, PLAN_VALIDATED_JSON)
        stages = summary.get("stages") or {}
        sources[label] = {
            "failed": False,
            "score": scores[label],
            "safe_id": summary["safe_id"],
            "join": column_join(doc, summary, plan, validated, cat),
            "heuristic": heuristic_tally(plan),
            "structure_rule": plan.get("structure_rule"),
            "initial": {"subject": plan.get("subject"),
                        "entities": plan.get("entities"), "edges": plan.get("edges")},
            "final": {"subject_effective": summary.get("subject"),
                      "entities": validated.get("entities"),
                      "edges": validated.get("edges")},
            "profile_sha": ((ws.manifest.get("sources") or {})
                            .get(summary["safe_id"], {})
                            .get("artifacts") or {}).get("profile.json"),
            "inputs": binding_inputs(plan),
            "pairs": pairs_by_round(plan),
            "available": availability(doc, plan, cat),
            "stages": stages,
        }
    provenance = {}
    arm_json = os.path.join(out_dir, "arm.json")
    if os.path.exists(arm_json):
        with open(arm_json, encoding="utf-8") as fh:
            provenance = json.load(fh)
    return {"name": name, "dir": out_dir, "report": report, "sources": sources,
            "provenance": provenance}


def check_identical_inputs(arms: list[dict], control: str) -> list[str]:
    """Every arm's binding inputs equal the control's, per source; pairs
    equal for a `heuristic` arm, a superset for a `static` arm. Returns the
    lines to print; raises on the first disagreement."""
    base = next(a for a in arms if a["name"] == control)
    lines = []
    for arm in arms:
        if arm is base:
            continue
        plan_cfg = (arm["report"].get("config") or {}).get("plan") or {}
        mode = plan_cfg.get("binding_mode")
        if plan_cfg.get("structure_mode") == "heuristic":
            lines.append(f"{arm['name']} (structure {plan_cfg['structure_mode']}, "
                         f"binding {mode}): the binding-input identity check does "
                         "not apply — its structure is the rule's, not the "
                         "control's; compared end to end on all four layers")
            continue
        checked = 0
        for label, src in arm["sources"].items():
            ctrl = base["sources"][label]
            if src["failed"] or ctrl["failed"]:
                continue
            if src["profile_sha"] != ctrl["profile_sha"]:
                raise BaselineRefused(f"{arm['name']}: {label}: profile.json differs "
                                      "from the control's")
            if src["inputs"] != ctrl["inputs"]:
                raise BaselineRefused(f"{arm['name']}: {label}: the binding inputs "
                                      "(subject, entities, edges, shortlists, "
                                      "rounds) differ from the control's")
            if mode != "static" and set(src["pairs"]) != set(ctrl["pairs"]):
                raise BaselineRefused(f"{arm['name']}: {label}: the askable "
                                      "(round, column) set differs from the control's")
            if not set(ctrl["pairs"]) <= set(src["pairs"]):
                raise BaselineRefused(f"{arm['name']}: {label}: a (round, column) "
                                      "the control asked about was not asked")
            for key, pairs in ctrl["pairs"].items():
                got = src["pairs"][key]
                if mode == "static":
                    if not set(pairs) <= set(got):
                        raise BaselineRefused(f"{arm['name']}: {label}: round {key[0]} "
                                              f"{key[1]}: the catalogue omits a pair "
                                              "the control offered")
                elif got != pairs:
                    raise BaselineRefused(f"{arm['name']}: {label}: round {key[0]} "
                                          f"{key[1]}: offered pairs differ from the "
                                          "control's, in content or order")
            checked += 1
        lines.append(f"{arm['name']} ({mode}): binding inputs identical to "
                     f"{control} on {checked} sources"
                     + ("; offered pairs a superset of the control's, every "
                        "live column askable"
                        if mode == "static"
                        else "; askable columns and offered pairs identical"))
    return lines


def _stage_calls(stages: dict) -> tuple[int, int, int]:
    """(generated, hits, failures) over propose and validate."""
    g = h = f = 0
    for name in ("propose", "validate"):
        for row in (stages.get(name) or {}).get("calls", {}).values():
            g += row["generated"]
            h += row["hits"]
            f += row["failures"]
    return g, h, f


def _planning(stages: dict) -> tuple[float, str]:
    seconds = sum((stages.get(n) or {}).get("duration_seconds", 0.0)
                  for n in ("propose", "validate"))
    modes = {(stages.get(n) or {}).get("mode") for n in ("propose", "validate")}
    mode = ("live" if "live" in modes else
            "replay" if "replay" in modes else "none")
    return seconds, mode


def source_row(src: dict, doc) -> dict:
    mappable = sum(1 for c in doc.columns if c.carries_property)
    if src["failed"]:
        return {"failed": True, "bound": 0, "correct": 0, "wrong": 0,
                "missed": mappable, "key_only": 0, "mappable": mappable,
                "no_home": sum(1 for c in doc.columns if not c.carries_property),
                "stretched": 0, "produced": 0, "extra": 0,
                "offered": 0, "generated": 0, "hits": 0, "failures": 1,
                "seconds": 0.0, "mode": "failed"}
    c = src["score"]["columns"]
    h = src["score"]["no_home"]
    g, hits, f = _stage_calls(src["stages"])
    seconds, mode = _planning(src["stages"])
    return {"failed": False, "bound": c["bound"], "correct": c["correct"],
            "wrong": c["wrong"], "missed": c["missed"],
            "key_only": c["key_only"], "mappable": c["with_a_property_home"],
            "no_home": h["total"], "stretched": h["stretched"],
            "produced": c["bindings_produced"],
            "extra": c["extra_sibling_claims"],
            "offered": sum(1 for v in src["available"].values() if v),
            "generated": g, "hits": hits, "failures": f,
            "seconds": seconds, "mode": mode}


def totals(rows: dict[str, dict]) -> dict:
    keys = ("bound", "correct", "wrong", "missed", "key_only", "mappable",
            "no_home", "stretched", "produced", "extra", "offered",
            "generated", "hits", "failures", "seconds")
    out = {k: sum(r[k] for r in rows.values()) for k in keys}
    out["failed"] = sum(1 for r in rows.values() if r["failed"])
    modes = {r["mode"] for r in rows.values() if not r["failed"]}
    out["mode"] = "+".join(sorted(modes)) or "none"
    out["seconds"] = round(out["seconds"], 1)
    return out


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.3f}" if den else f"{num}/{den}"


def _short(label: str) -> str:
    base = label.rsplit("/", 1)[-1]
    name, _, sheet = base.partition("#")
    name = name.rsplit(".", 1)[0]
    for suffix in ("_2026-08-20", "_2023q3", "_2026_every250th", "_2020-12",
                   "_2026-09-02_every200th", "_2026-09-04", "_2026-09-03",
                   "_2026-04-01", "_2022-02-23_every40th", "_2024-08-02",
                   "_2026-07-01", "_2020-01-31"):
        name = name.replace(suffix, "")
    return f"{name}#{sheet}" if sheet else name


def layer_row(score: dict | None) -> dict:
    """The subject, entity and edge layers of one scored source, in the
    scorer's own names; zeros for a failed source."""
    if score is None:
        return {"subject": "FAILED", "subject_effective": "FAILED",
                "e_expected": 0, "e_matched": 0, "e_missing": 0, "e_extra": 0,
                "e_empty": 0, "keys_agree": 0, "keys_differ": 0, "keys_empty": 0,
                "inst_checked": 0, "inst_agree": 0,
                "g_expected": 0, "g_found": 0, "g_missing": 0, "g_extra": 0,
                "g_wrong": 0}
    e, g = score["entities"], score["edges"]
    return {"subject": score.get("subject"),
            "subject_effective": score.get("subject_effective"),
            "e_expected": e["expected"], "e_matched": e["matched"],
            "e_missing": e["missing"], "e_extra": e["extra"],
            "e_empty": e.get("empty", 0),
            "keys_agree": e["keys_agree"], "keys_differ": e["keys_differ"],
            "keys_empty": e["keys_empty"],
            "inst_checked": e.get("instances_checked", 0),
            "inst_agree": e.get("instances_agree", 0),
            "g_expected": g["expected"], "g_found": g["found"],
            "g_missing": g["missing"], "g_extra": g["extra"],
            "g_wrong": g["endpoints_wrong"]}


def _short_verdict(v: str | None) -> str:
    return {"SUBJECT-AGREE": "agree", "SUBJECT-ACCEPTABLE": "acceptable",
            "SUBJECT-WRONG": "wrong"}.get(v or "", v or "-")


def render_layers(arms: list[dict], etalons: dict) -> list[str]:
    out = ["### The other three layers, per arm",
           "",
           "Effective row schema, entity roles with their keys and instance "
           "counts, and relations — the scorer's own names, each layer over "
           "its own denominator. A failed source counts as wrong on the "
           "subject and contributes nothing found.",
           "",
           "| arm | subject: agree / acceptable / wrong | entities: matched of expected | missing / extra / empty | keys agree / differ / empty | instances within tolerance of checked | relations: found of expected | missing / extra / endpoints wrong |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    per_arm = {}
    for arm in arms:
        rows = {label: layer_row(src["score"]) for label, src in arm["sources"].items()}
        per_arm[arm["name"]] = rows
        subj = {"agree": 0, "acceptable": 0, "wrong": 0}
        for r in rows.values():
            subj[_short_verdict(r["subject_effective"]) if r["subject_effective"] != "FAILED" else "wrong"] += 1
        t = {k: sum(r[k] for r in rows.values()) for k in rows[next(iter(rows))]
             if k not in ("subject", "subject_effective")}
        for label, src in arm["sources"].items():
            if src["failed"]:
                doc = etalons[label]
                producible = {e.key for e in doc.entities if e.producible}
                t["e_expected"] += len(producible)
                t["e_missing"] += len(producible)
                edges = sum(1 for g in doc.edges
                            if g.source in producible and g.target in producible)
                t["g_expected"] += edges
                t["g_missing"] += edges
        out.append(f"| {arm['name']} | {subj['agree']} / {subj['acceptable']} / {subj['wrong']} | "
                   f"{t['e_matched']} / {t['e_expected']} | {t['e_missing']} / {t['e_extra']} / {t['e_empty']} | "
                   f"{t['keys_agree']} / {t['keys_differ']} / {t['keys_empty']} | "
                   f"{t['inst_agree']} / {t['inst_checked']} | {t['g_found']} / {t['g_expected']} | "
                   f"{t['g_missing']} / {t['g_extra']} / {t['g_wrong']} |")
    out.append("")
    for arm in arms:
        out.append(f"#### {arm['name']}, the other three layers per source")
        out.append("")
        out.append("| source | effective schema | entities matched / expected | missing / extra | keys agree / differ / empty | instances ok / checked | relations found / expected | missing / extra / wrong endpoints |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for label in etalons:
            r = per_arm[arm["name"]][label]
            if r["subject"] == "FAILED":
                out.append(f"| `{_short(label)}` | FAILED | | | | | | |")
                continue
            out.append(f"| `{_short(label)}` | {_short_verdict(r['subject_effective'])} | "
                       f"{r['e_matched']} / {r['e_expected']} | {r['e_missing']} / {r['e_extra']} | "
                       f"{r['keys_agree']} / {r['keys_differ']} / {r['keys_empty']} | "
                       f"{r['inst_agree']} / {r['inst_checked']} | {r['g_found']} / {r['g_expected']} | "
                       f"{r['g_missing']} / {r['g_extra']} / {r['g_wrong']} |")
        out.append("")
    return out


def render_structures(arms: list[dict], etalons: dict) -> list[str]:
    """For a heuristic-structure arm: what the rule declared beside what
    left `validate`, per source."""
    out = []
    for arm in arms:
        plan_cfg = (arm["report"].get("config") or {}).get("plan") or {}
        if plan_cfg.get("structure_mode") != "heuristic":
            continue
        out.append(f"### {arm['name']}: initial structure (the rule's) and final structure (after validate), per source")
        out.append("")
        out.append("| source | rule: subject (vote, signal) | rule: keys | after propose: entities | final: subject effective | final: entities (schema, keys, filter) | final: edges |")
        out.append("|---|---|---|---|---|---|---|")
        for label in etalons:
            src = arm["sources"][label]
            if src["failed"]:
                out.append(f"| `{_short(label)}` | FAILED | | | | | |")
                continue
            rule = src.get("structure_rule") or {}
            init = src.get("initial") or {}
            fin = src.get("final") or {}
            ents = "; ".join(f"{e['key']} ({e['schema']}, keys {','.join(e.get('keys') or []) or '-'}"
                             f"{', filter ' + e['filter']['column'] if e.get('filter') else ''})"
                             for e in init.get("entities") or [])
            fents = "; ".join(f"{e['key']} ({e['schema']}, keys {','.join(e.get('keys') or []) or '-'}"
                              f"{', filter ' + e['filter']['column'] if e.get('filter') else ''})"
                              for e in fin.get("entities") or [])
            fedges = "; ".join(f"{g['schema']} {g['source']}→{g['target']}" for g in fin.get("edges") or []) or "-"
            out.append(f"| `{_short(label)}` | {rule.get('subject')} ({rule.get('vote')}, "
                       f"{'no signal' if rule.get('no_signal') else 'signal'}"
                       f"{', tie of ' + str(len(rule['tied'])) if len(rule.get('tied') or []) > 1 else ''}) | "
                       f"{','.join(rule.get('keys') or []) or '-'} | {ents or '-'} | "
                       f"{fin.get('subject_effective') or '-'} | {fents or '-'} | {fedges} |")
        out.append("")
    return out


def render(arms: list[dict], etalons: dict, control: str,
           identity_lines: list[str]) -> str:
    out: list[str] = []
    rows = {a["name"]: {label: source_row(a["sources"][label], etalons[label])
                        for label in etalons} for a in arms}
    tots = {name: totals(r) for name, r in rows.items()}

    out.append("## Checks")
    out.append("")
    out.append(f"- etalons: {len(etalons)} files, hashes as frozen in "
               f"`tools/baseline_sources.json`")
    for a in arms:
        rep = a["report"]
        cfg = rep.get("config") or {}
        out.append(f"- {a['name']}: `{a['dir']}` — run `{rep.get('run_id')}`, "
                   f"binding_mode `{(cfg.get('plan') or {}).get('binding_mode')}`, "
                   f"live_stages `{(cfg.get('model') or {}).get('live_stages')}`, "
                   f"model `{rep.get('model_id')}` server `{rep.get('model_server')}`, "
                   f"sources {rep.get('sources')}, failed {rep.get('failed')}, "
                   f"calls {rep.get('model_calls')} generated / "
                   f"{rep.get('cache_hits')} hits")
        prov = a.get("provenance") or {}
        if prov:
            out.append(f"  - code `{prov.get('code_root')}` at `{prov.get('code_head')}`"
                       + (" (dirty)" if prov.get("code_dirty") else "")
                       + f", data `{prov.get('data_root')}`")
        invalid = [label for label, src in a["sources"].items()
                   if not src["failed"] and src["score"].get("accounting_valid") is False]
        if invalid:
            out.append(f"  - LEDGER INVALID on {len(invalid)} source(s): "
                       + ", ".join(f"`{_short(l)}`" for l in invalid)
                       + " — not publication-eligible under the gate in cli.py")
        stages = rep.get("stages") or {}
        parts = [f"{s}: {r['generated']}g/{r['hits']}h/{r['failures']}f"
                 for s, r in stages.items() if r["attempts"] or r["failures"]]
        out.append(f"  - per stage: {', '.join(parts) or 'no model attempts'}")
    for line in identity_lines:
        out.append(f"- {line}")
    out.append("")

    out.append("## Totals over the 19 sources")
    out.append("")
    out.append("| arm | failed | bound | acceptable_of_bound | acceptable_of_mappable | wrong | missed | key_only | stretched (no home) | claims / extra_sibling | answer offered (of mappable) | generated / hits / failures | planning s (mode) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, t in tots.items():
        arm = next(a for a in arms if a["name"] == name)
        flag = sum(1 for src in arm["sources"].values()
                   if not src["failed"] and src["score"].get("accounting_valid") is False)
        out.append(f"| {name}{' (ledger invalid on %d)' % flag if flag else ''} | "
                   f"{t['failed']} | {t['bound']} | "
                   f"{_rate(t['correct'], t['bound'])} | "
                   f"{_rate(t['correct'], t['mappable'])} | {t['wrong']} | "
                   f"{t['missed']} | {t['key_only']} | "
                   f"{t['stretched']} / {t['no_home']} | "
                   f"{t['produced']} / {t['extra']} | "
                   f"{_rate(t['offered'], t['mappable'])} | "
                   f"{t['generated']} / {t['hits']} / {t['failures']} | "
                   f"{t['seconds']} ({t['mode']}) |")
    out.append("")
    out.append("Call counts and seconds are summed over the scored sources; a "
               "failed source contributes one failure here and its own calls "
               "appear only in its run.json line under Checks.")
    out.append("")
    out.extend(render_join(arms, etalons))
    out.extend(render_layers(arms, etalons))
    out.extend(render_structures(arms, etalons))

    for a in arms:
        name = a["name"]
        out.append(f"## {name} per source")
        out.append("")
        out.append("| source | bound | correct | wrong | missed | key_only | mappable | stretched / no home | claims / extra | answer offered | generated / hits / failures | planning s (mode) |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label in etalons:
            r = rows[name][label]
            if r["failed"]:
                out.append(f"| `{_short(label)}` | FAILED | | | | | {r['mappable']} | | | | | |")
                continue
            out.append(f"| `{_short(label)}` | {r['bound']} | {r['correct']} | "
                       f"{r['wrong']} | {r['missed']} | {r['key_only']} | "
                       f"{r['mappable']} | {r['stretched']} / {r['no_home']} | "
                       f"{r['produced']} / {r['extra']} | "
                       f"{r['offered']} / {r['mappable']} | "
                       f"{r['generated']} / {r['hits']} / {r['failures']} | "
                       f"{r['seconds']:.1f} ({r['mode']}) |")
        out.append("")

    for a in arms:
        name = a["name"]
        if name == control:
            continue
        out.append(f"## {control} versus {name}, per source (delta = {name} − {control})")
        out.append("")
        out.append("| source | bound | correct | wrong | missed | stretched | answer offered |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for label in etalons:
            b, x = rows[control][label], rows[name][label]
            if x["failed"]:
                out.append(f"| `{_short(label)}` | FAILED | −{b['correct']} | | | | |")
                continue
            cells = [f"{x[k] - b[k]:+d}" if x[k] != b[k] else "0"
                     for k in ("bound", "correct", "wrong", "missed",
                               "stretched", "offered")]
            out.append(f"| `{_short(label)}` | " + " | ".join(cells) + " |")
        tb, tx = tots[control], tots[name]
        out.append("| **total** | " + " | ".join(
            f"{tx[k] - tb[k]:+d}" for k in ("bound", "correct", "wrong", "missed",
                                           "stretched", "offered")) + " |")
        out.append("")
    return "\n".join(out)


def join_totals(arm: dict) -> dict:
    """The 2×2 and its buckets over the arm's scored sources, plus the
    mappable columns of its failed sources as `not run`."""
    cells = {"offered_ok": 0, "offered_not": 0, "not_offered_ok": 0,
             "not_offered_not": 0}
    buckets = {"chooser answered a different pair": 0,
               "chooser answered unmapped": 0,
               "chooser answered unmapped, decline pre-registered as defensible": 0,
               "chooser acceptable, removed or replaced by post-processing": 0}
    added_not_offered = 0
    chooser_ok_not_offered = 0
    not_run = 0
    reach = {"not_offered_reachable": 0, "not_offered_unreachable": 0,
             "offered_reachable": 0}
    heuristic = {"bound": 0, "bound_lexical_zero": 0, "tie_distinct": 0,
                 "tie_targets": 0, "tie_distinct_taken": 0, "nothing_above_zero": 0}
    for label, src in arm["sources"].items():
        if src["failed"]:
            not_run += sum(1 for c in arm["_etalons"][label].columns
                           if c.carries_property)
            continue
        for k in heuristic:
            heuristic[k] += src["heuristic"][k]
        for j in src["join"]:
            if j["offered"]:
                reach["offered_reachable"] += 1
            elif j["reachable"]:
                reach["not_offered_reachable"] += 1
            else:
                reach["not_offered_unreachable"] += 1
            if j["offered"] and j["final"]:
                cells["offered_ok"] += 1
            elif j["offered"]:
                cells["offered_not"] += 1
                if j["what"] == "chooser answered unmapped" and j["decline_defensible"]:
                    buckets["chooser answered unmapped, decline pre-registered as defensible"] += 1
                else:
                    buckets[j["what"]] += 1
            elif j["final"]:
                cells["not_offered_ok"] += 1
                added_not_offered += 1
                chooser_ok_not_offered += j["chooser_acceptable"]
            else:
                cells["not_offered_not"] += 1
    split = {"not_offered_not_reachable": 0, "not_offered_not_unreachable": 0}
    for label, src in arm["sources"].items():
        if src["failed"]:
            continue
        for j in src["join"]:
            if not j["offered"] and not j["final"]:
                split["not_offered_not_reachable" if j["reachable"]
                      else "not_offered_not_unreachable"] += 1
    return {"cells": cells, "buckets": buckets, "not_run": not_run,
            "added_not_offered": added_not_offered,
            "chooser_ok_not_offered": chooser_ok_not_offered,
            "heuristic": heuristic, "reach": reach, "split": split}


def render_join(arms: list[dict], etalons: dict) -> list[str]:
    out = ["### Candidate availability against the final verdict, per column",
           "",
           "Per mappable column of every scored source: was an acceptable "
           "property among the pairs the column was offered, and is the "
           "column's final best binding acceptable. The not-offered-not-"
           "acceptable cell is split by whether an acceptable property was "
           "among every bindable property of the objects declared for the "
           "column's rounds (`catalogue_pairs`). Then what the chooser did "
           "on the columns that were offered the answer and did not end "
           "acceptable, read from `plan.json` against `plan.validated.json`. "
           "Every category is an observation about where the answer went "
           "missing, not a cause. A column in several rounds is counted once: "
           "offered if offered in any round, reachable if reachable in any. "
           "A failed source's mappable columns are `not run` and in no cell.",
           "",
           "| arm | offered, acceptable | offered, not acceptable | not offered, acceptable | not offered, not acceptable | — of which reachable among the declared objects' properties / not reachable | not run | offered-not-acceptable: different pair / unmapped / unmapped, decline defensible / removed by post-processing | not-offered-acceptable: added by post-processing |",
           "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for arm in arms:
        arm["_etalons"] = etalons
        t = join_totals(arm)
        c, b, sp = t["cells"], t["buckets"], t["split"]
        out.append(f"| {arm['name']} | {c['offered_ok']} | {c['offered_not']} | "
                   f"{c['not_offered_ok']} | {c['not_offered_not']} | "
                   f"{sp['not_offered_not_reachable']} / {sp['not_offered_not_unreachable']} | "
                   f"{t['not_run']} | "
                   f"{b['chooser answered a different pair']} / "
                   f"{b['chooser answered unmapped']} / "
                   f"{b['chooser answered unmapped, decline pre-registered as defensible']} / "
                   f"{b['chooser acceptable, removed or replaced by post-processing']} | "
                   f"{t['added_not_offered']}"
                   + (f" ({t['chooser_ok_not_offered']} chooser-acceptable?!)"
                      if t["chooser_ok_not_offered"] else "") + " |")
    out.append("")
    out.append("The same columns before retrieval truncates: of the mappable "
               "columns NOT offered an acceptable property, how many had one "
               "among every bindable property of the objects declared for "
               "their round (`catalogue_pairs`, the list arm C offered) and "
               "how many had none. Observations about where the answer went "
               "missing — a reachable property can still be cut by the "
               "shortlist, an offered one lost later — not causes.")
    out.append("")
    out.append("| arm | offered (acceptable among the shortlist pairs) | not offered, reachable among the declared objects' properties | not offered, no declared object carries it |")
    out.append("|---|---:|---:|---:|")
    for arm in arms:
        r = join_totals(arm)["reach"]
        out.append(f"| {arm['name']} | {r['offered_reachable']} | "
                   f"{r['not_offered_reachable']} | {r['not_offered_unreachable']} |")
    out.append("")
    heur = [(a["name"], join_totals(a)["heuristic"]) for a in arms
            if ((a["report"].get("config") or {}).get("plan") or {})
            .get("binding_mode", "").startswith("heuristic")]
    if heur:
        out.append("Heuristic arms, from the rule's own `why` strings over the "
                   "scored sources: columns the rule bound, of which on a "
                   "lexical score of zero (type bonus alone), settled by tie "
                   "rule 2 (first of several targets), settled by tie rule 1 "
                   "inverted (first of several distinct properties), withheld "
                   "on a tie between distinct properties, withheld for no "
                   "candidate above zero.")
        out.append("")
        out.append("| arm | bound by the rule | lexical zero | first of several targets | first of tying distinct properties | withheld: distinct tie | withheld: nothing above zero |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, h in heur:
            out.append(f"| {name} | {h['bound']} | {h['bound_lexical_zero']} | "
                       f"{h['tie_targets']} | {h['tie_distinct_taken']} | "
                       f"{h['tie_distinct']} | {h['nothing_above_zero']} |")
        out.append("")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--etalons", required=True)
    ap.add_argument("--control", default="A")
    ap.add_argument("arms", nargs="+", help="name=dir, e.g. A=work-baseline-A")
    args = ap.parse_args(argv)

    fixture = load_fixture()
    check_etalons(args.etalons, fixture)
    cat = Catalogue.load()
    etalons: dict[str, object] = {}
    for e in fixture["etalons"]:
        doc = load_etalon(os.path.join(args.etalons, e["file"]), cat)
        label = _label(doc)
        if label != e["source"]:
            raise BaselineRefused(f"{e['file']} labels {label!r}, the fixture "
                                  f"froze {e['source']!r}")
        etalons[label] = doc

    arms = []
    for spec in args.arms:
        name, _, out_dir = spec.partition("=")
        if not out_dir:
            ap.error(f"{spec!r} is not name=dir")
        arms.append(read_arm(name, out_dir, etalons, cat))
    if not any(a["name"] == args.control for a in arms):
        ap.error(f"no arm named {args.control!r} to serve as control")
    identity = check_identical_inputs(arms, args.control)
    print(render(arms, etalons, args.control, identity))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaselineRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
