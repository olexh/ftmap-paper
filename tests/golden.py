# tests/golden.py
"""A deterministic run over the tracked fixtures, and a snapshot of what it wrote.

THE ORACLE FOR EVERY LATER UNIT. Most of this remediation changes how artefacts
are written — identity encoding, generation commits, streaming emission,
workbook reading — and each of those is supposed to leave the OUTPUT alone. A
per-module unit test cannot say that; it asserts the shape its own author
thought of. This runs the whole pipeline over the five tracked fixtures with a
scripted model and records a digest of every byte it produced, so a change that
was meant to be invisible and was not shows up as a diff in one committed file.

`tests/golden/corpus.json` is that file. Regenerate it with

    FTMAP_GOLDEN_UPDATE=1 uv run --frozen --offline pytest tests/test_golden.py

and REVIEW THE DIFF — it is the artefact, not a nuisance. A unit that changes it
must say in its measurement note which lines moved and why. Regenerating to make
a red test green is the one use this file cannot survive.

The client is scripted rather than live for the obvious reason and one less
obvious one: the goldens must be reproducible on a laptop with no model, or the
oracle is only available to whoever has the weights.

WHAT THE SNAPSHOT KEEPS. Value-free artefacts — `summary.json`, `run.json`, the
mapping documents — are kept whole, because that is where a readable diff lives.
Value-bearing ones are kept as a SHA-256 over their bytes: `profile.json` carries
up to five real cell values per column by construction and `decisions.jsonl`
carries the model's free-text `why`, and neither belongs in a tracked file. The
digest still proves byte-equivalence, which is what the gate asks for.

Entity ids are the exception that is kept in full. They are what U2's key
encoding changes, they are one-way hashes of key material rather than the
material itself, and the whole point of that unit's golden review is being able
to see WHICH ids moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from followthemoney import model as ftm_model

from ftmap.config import Config
from ftmap.fsutil import SnapshotStore
from ftmap.pipeline import run_corpus
from ftmap.plan.prompt import (BINDING_MARKER, EDGE_MARKER, REPAIR_MARKER,
                               STRUCTURE_MARKER)
from ftmap.plan.response_schema import UNMAPPED
from ftmap.vocab.catalogue import Catalogue

REPO = Path(__file__).resolve().parent.parent
FIXTURES = str(Path(__file__).parent / "fixtures")
GOLDEN = Path(__file__).parent / "golden" / "corpus.json"

# `c12: Прізвище | filled 100% | distinct 87% | ...` in either prompt.
_COLUMN = re.compile(r"^(c\d+): ")
_DISTINCT = re.compile(r"distinct (\d+)%")
_FILLED = re.compile(r"filled (\d+)%")
# `  person (Person) -> Person:name (Name)` inside a candidates line.
_CANDIDATE = re.compile(r"^(\w+) \([^)]+\) -> ([\w:]+)")

# Artefacts whose content is safe to keep in a tracked file, because the
# boundary tests already assert they carry no cell value. Everything else is
# digested. `run.json` is handled separately — it needs normalizing first.
_PUBLISHABLE = {"summary.json", "run.json", "workspace.json"}

# Fields that differ between two identical runs and say nothing about
# correctness. Replaced rather than dropped, so the snapshot still records that
# the field was written at all.
# Maps whose KEYS are the claim and whose values are digests of volatile bytes.
_HASHED = ("artifacts",)

_VOLATILE = ("run_id", "started_at", "finished_at", "root", "path",
             "ftmap_version", "followthemoney_version", "duration_seconds",
             "committed_at", "created_at", "updated_at")


# `20260824T201250779967Z-9ee16598` — the sortable stamp plus a short random
# suffix `workspace.new_generation_id` produces. Volatile by construction, and
# it appears in every committed path, in `summary.json`'s `mapping` field, and
# in the manifest.
_GENERATION = re.compile(r"\d{8}T\d{12}Z-[0-9a-f]{8}")


def _scrub(text: str, out_dir: str, run_id: str | None) -> str:
    """Replace what is different about THIS run, and nothing else.

    Substituted rather than blanked, for the same reason `_VOLATILE` keeps its
    keys: `<out>/mappings/x.yml` and `<out>/gen/01/x.yml` are a real difference
    that U3 is expected to produce, and a snapshot that erased the whole path
    would call the change invisible. Only the parts that cannot be the same on
    another machine — the workspace directory, the checkout, the run id — are
    replaced, and each with a distinct marker so the diff still reads.

    `statements.csv` carries `run_id` on every row and the absolute source path
    on every row, so this is what makes the digest of a value-bearing artefact
    mean "the same bytes" rather than "the same machine, twice".
    """
    text = text.replace(out_dir, "<out>").replace(str(REPO), "<repo>")
    text = _GENERATION.sub("<gen>", text)
    return text.replace(run_id, "<run>") if run_id else text


class GoldenClient:
    """First candidate, every time, and one multi-column key on purpose.

    Deterministic by construction: it reads the prompt and applies a rule, so
    two runs over the same fixtures give the same answers without a recorded
    transcript to drift out of date.

    It declares a SECOND entity keyed on TWO columns wherever the table is wide
    enough. That is not decoration — a single-column key and a multi-column key
    are compiled by different paths, U2 changes only the second, and a golden
    corpus holding only single-key entities would call that change invisible.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, user, schema, max_tokens=None):
        self.calls += 1
        if STRUCTURE_MARKER in user:
            return self._structure(user)
        if EDGE_MARKER in user:
            return self._edge(user)
        if REPAIR_MARKER in user:
            # A repair round is asked only about assignments already rejected.
            # Answering `unmapped` accepts the rejection, which keeps the
            # golden's second round deterministic without inventing a binding.
            return {"bindings": [{"column": c, "binding": UNMAPPED,
                                  "why": "golden"}
                                 for c in _columns(user)]}
        return self._binding(user)

    def _structure(self, user: str) -> dict:
        ranked = _ranked_columns(user)
        if not ranked:
            return {"subject": None, "entities": [], "edges": []}
        entities = [{"key": "person", "schema": "Person", "keys": [ranked[0]]}]
        if len(ranked) >= 3:
            entities.append({"key": "org", "schema": "Organization",
                             "keys": sorted(ranked[1:3])})
        return {"subject": "Person", "entities": entities, "edges": []}

    def _edge(self, user: str) -> dict:
        offered = _combinations(user)
        return {"edges": [{"edge": offered[0]}] if offered else []}

    def _binding(self, user: str) -> dict:
        bindings = []
        candidates: dict[str, str] = {}
        column = None
        for line in user.splitlines():
            match = _COLUMN.match(line)
            if match:
                column = match.group(1)
            elif column and line.startswith("  candidates: "):
                candidates[column] = line.removeprefix("  candidates: ")
                bindings.append({
                    "column": column,
                    "binding": _first_live_candidate(
                        line.removeprefix("  candidates: ")),
                    "why": "golden: first offered candidate",
                })
                column = None
        assert BINDING_MARKER in user
        # ONE BINDING MUST LAND ON THE SECOND ENTITY, or the corpus stops
        # covering the multi-column key it declares. `validate` now drops an
        # entity no column is bound to — the right rule, and it took `org`
        # with it, because a first-candidate client answers with the subject's
        # own properties on every column. The LAST column offering an `org`
        # candidate is moved to it: deterministic, one binding, and it leaves
        # every other column on the first-candidate rule this client is.
        if not any(b["binding"].startswith("org|") for b in bindings):
            for binding in reversed(bindings):
                offered = candidates.get(binding["column"], "")
                for candidate in offered.split(";"):
                    match = _CANDIDATE.match(candidate.strip())
                    if match is None or match.group(1) != "org":
                        continue
                    binding["binding"] = f"org|{match.group(2)}"
                    binding["why"] = "golden: keeps the second entity alive"
                    break
                else:
                    continue
                break
        return {"bindings": bindings}


def _first_live_candidate(offered: str) -> str:
    """The first candidate the ontology has not deprecated.

    NOT A PREFERENCE — A DEPRECATED PROPERTY KILLS THE SOURCE. FollowTheMoney's
    `EntityMapping.bind()` emits a `DeprecationWarning` for one, and this
    suite promotes every warning to an error, so `execute` raised and
    `run_corpus` recorded a source failure. Two of the five fixtures died that
    way and the golden corpus quietly became an oracle over three: the
    workbooks with two header rows and with a title row, which is to say the
    two defect classes it exists to cover.

    A first-candidate rule is still what the client does; this only skips the
    candidates that cannot be answered at all.
    """
    for candidate in offered.split(";"):
        match = _CANDIDATE.match(candidate.strip())
        if match is None:
            continue
        schema_name, _, prop_name = match.group(2).partition(":")
        schema = ftm_model.get(schema_name)
        prop = schema.get(prop_name) if schema else None
        if prop is not None and not prop.deprecated:
            return f"{match.group(1)}|{match.group(2)}"
    return UNMAPPED


def _columns(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines()
            if (m := _COLUMN.match(line))]


def _ranked_columns(structure_prompt: str) -> list[str]:
    """Column ids, most key-like first.

    `validate` drops an entity whose key columns are too sparse or too
    repetitive to identify anything, so picking keys at random would make the
    golden's entity set depend on which fixture happened to have a good column
    in position 0. Ranking by distinctness then fill — the two numbers the
    structure prompt already prints, and the two `key_distinct_floor` and
    `key_fill_floor` actually test — picks a key the validator will accept
    whenever the table has one at all.
    """
    scored: list[tuple[int, int, str]] = []
    for line in structure_prompt.splitlines():
        match = _COLUMN.match(line)
        if not match:
            continue
        distinct = _DISTINCT.search(line)
        filled = _FILLED.search(line)
        scored.append((-int(distinct.group(1)) if distinct else 0,
                       -int(filled.group(1)) if filled else 0,
                       match.group(1)))
    scored.sort()
    return [column for _, _, column in scored]


def _combinations(edge_prompt: str) -> list[str]:
    """The `Schema|source|target` lines the edge prompt offers, in order."""
    return [line.strip() for line in edge_prompt.splitlines()
            if line.startswith("  ") and line.count("|") == 2]


def run_golden(out_dir: str) -> dict:
    """The five tracked fixtures through the whole pipeline, scripted."""
    return run_corpus(FIXTURES, Config.load(None), Catalogue.load(),
                      GoldenClient(), out_dir)


def _normalize(obj):
    """Blank the fields that differ between two identical runs.

    Replaced with a marker rather than deleted: a snapshot that simply dropped
    `run_id` could not tell "the field is volatile" from "the field stopped
    being written", and the second is exactly the regression a manifest change
    could introduce.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in sorted(obj.items()):
            if k in _VOLATILE:
                out[k] = "<volatile>"
            elif k in _HASHED:
                # KEYS KEPT, VALUES BLANKED. The manifest records a SHA-256
                # per committed artefact, over the real file — which carries
                # the workspace path, the run id and the generation id, all of
                # which differ between two identical runs. Which artefacts got
                # a hash is the reviewable fact; the hash itself is checked
                # against the file it describes by `test_boundary`, not here.
                out[k] = {name: "<sha256>" for name in sorted(v)}
            else:
                out[k] = _normalize(v)
        return out
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def _entity_ids(path: Path) -> list[list[str]]:
    """`[schema, id]` for every emitted entity, sorted.

    Sorted rather than in file order because emission order is U8's business
    and identity is U2's; a golden that conflated them would fail the wrong
    unit. File order is still covered — `entities.ftm.json` is digested whole.
    """
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entity = json.loads(line)
                out.append([entity.get("schema"), entity.get("id")])
    return sorted(out)


def _run_id(out_dir: str) -> str | None:
    from ftmap.workspace import open_workspace

    return open_workspace(out_dir).report().get("run_id")


def snapshot(out_dir: str) -> dict:
    """Every file the run wrote, scrubbed, digested and ordered."""
    root = Path(out_dir)
    run_id = _run_id(out_dir)
    files: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = _GENERATION.sub("<gen>", str(path.relative_to(root)))
        mode = oct(path.stat().st_mode & 0o777)
        if name.startswith(SnapshotStore.DIRNAME + os.sep):
            # THE INPUT SNAPSHOTS, RECORDED BY NAME AND NOT BY CONTENT. The
            # name IS the digest, so listing it asserts the bytes exactly; the
            # bytes themselves are complete copies of the corpus and have no
            # business in a tracked file. Reading them would also mean decoding
            # a binary workbook as UTF-8, which is how this branch came to be
            # written.
            files[name] = {"mode": mode, "bytes": path.stat().st_size,
                           "content_addressed": True}
            continue
        raw = path.read_bytes()
        try:
            text = _scrub(raw.decode("utf-8"), out_dir, run_id)
        except UnicodeDecodeError:
            # No artefact should reach this, but a snapshot that silently
            # stopped covering one would be worse than one that says so.
            files[name] = {"mode": mode, "bytes": len(raw), "binary": True,
                           "sha256": hashlib.sha256(raw).hexdigest()}
            continue
        entry: dict[str, object] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
            "mode": mode,
        }
        if path.name in _PUBLISHABLE:
            entry["content"] = _normalize(json.loads(text))
            # The digest of a file holding a timestamp is itself volatile.
            del entry["sha256"], entry["bytes"]
        elif path.suffix == ".yml":
            # The mapping document names columns, entity keys and properties
            # and holds no cell value. It is also the artefact `ftm map` is run
            # against standalone, so a change to it is a change to the claim
            # that a run reproduces without a model.
            entry["content"] = text
            del entry["sha256"], entry["bytes"]
        elif path.name.endswith("entities.ftm.json"):
            entry["entity_ids"] = _entity_ids(path)
        files[name] = entry
    return files


def compare(out_dir: str) -> tuple[dict, dict]:
    """The live snapshot and the committed one, ready to assert on."""
    live = snapshot(out_dir)
    if os.environ.get("FTMAP_GOLDEN_UPDATE"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(
            json.dumps(live, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return live, json.loads(GOLDEN.read_text(encoding="utf-8"))
