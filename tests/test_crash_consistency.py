# tests/test_crash_consistency.py
"""What a workspace looks like when the process does not get to finish.

The outcome these guard is invisible in a healthy run and identical in every
broken one until someone reads the workspace afterwards. So they are written
two ways: as an OBSERVED SEQUENCE of filesystem calls, where the ordering is
the whole property, and as a real subprocess killed with SIGKILL at a chosen
point, where nothing about the interpreter's shutdown can be relied on to tidy
up.

`SIGKILL` and not `SIGTERM`: a terminate is delivered as a Python exception
and every `finally` in the program runs. That is worth testing too — it is
what Ctrl-C does — but it is not a crash. A power cut runs no `finally`.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from ftmap.config import Config
from ftmap.inventory import inventory
from ftmap.pipeline import run_corpus, run_source
from ftmap.plan.prompt import BINDING_MARKER
from ftmap.vocab.catalogue import Catalogue
from ftmap.workspace import create_workspace, open_workspace

CFG = Config.load(None)
CAT = Catalogue.load()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ScriptedClient:
    calls = 0
    cache_hits = 0

    def complete(self, system, user, schema, max_tokens=None):
        if BINDING_MARKER not in user:
            return {"subject": "Person",
                    "entities": [{"key": "person", "schema": "Person",
                                  "keys": ["c0"]}],
                    "edges": []}
        return {"bindings": [
            {"column": line.split(":", 1)[0], "binding": "person|Person:name",
             "why": "x"}
            for line in user.splitlines()
            if line[:1] == "c" and ":" in line
            and line.split(":", 1)[0][1:].isdigit()]}


def _bytes(directory: str) -> dict[str, bytes]:
    """Every file in a committed generation, by name.

    Read through `pathlib` rather than a bare `open(...).read()`: this suite
    promotes `ResourceWarning` to an error, and a leaked handle here fails
    whichever unrelated test next triggers a garbage collection.
    """
    import pathlib as _pathlib

    return {name: (_pathlib.Path(directory) / name).read_bytes()
            for name in sorted(os.listdir(directory))}


def _corpus(tmp_path, names=("a.csv", "b.csv")):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index, name in enumerate(names):
        (corpus / name).write_text(
            f"ПІБ\nКоваленко Іван {index}\n", encoding="utf-8")
    return str(corpus)


# ------------------------------------------------------- interrupted in-process


@pytest.mark.parametrize("stage", ["profile.json", "plan.validated.json",
                                   "normalized.csv", "statements.csv",
                                   "summary.json"])
def test_an_interrupt_at_each_staged_write_leaves_the_previous_generation(
        tmp_path, monkeypatch, stage):
    """Covers AE1.

    `KeyboardInterrupt` after each artefact a source writes. The committed
    generation must still be the previous one, whole, and no staging directory
    may have become visible as a source.
    """
    root = _corpus(tmp_path, names=("a.csv",))
    out = str(tmp_path / "work")
    refs = inventory(root)
    good = run_source(refs[0], CFG, CAT, ScriptedClient(), "run1", out)
    before = open_workspace(out).source_dir(good["safe_id"])
    contents = _bytes(before)

    import ftmap.pipeline as pipeline

    real_write_json = pipeline._write_json
    real_statement_writer = pipeline.statement_writer

    def interrupt_at(path):
        if os.path.basename(path) == stage:
            raise KeyboardInterrupt

    def guarded_json(obj, path, private=True):
        interrupt_at(path)
        return real_write_json(obj, path, private=private)

    def guarded_statements(path):
        # Interrupt as the SINK IS OPENED, which is the moment the old
        # `write_statements(statements, path)` was called. Emission streams
        # now, so there is no later call to intercept — and that is the point:
        # a sink that has already written half a file into staging is exactly
        # the state the committed generation must be unaffected by.
        interrupt_at(path)
        return real_statement_writer(path)

    monkeypatch.setattr(pipeline, "_write_json", guarded_json)
    monkeypatch.setattr(pipeline, "statement_writer", guarded_statements)
    if stage == "normalized.csv":
        monkeypatch.setattr(
            pipeline, "normalize_frame",
            lambda frame, vplan, path, evidence=None: interrupt_at(path))

    with pytest.raises(KeyboardInterrupt):
        run_source(refs[0], CFG, CAT, ScriptedClient(), "run2", out)

    workspace = open_workspace(out)
    assert workspace.source_ids() == [good["safe_id"]]
    after = workspace.source_dir(good["safe_id"])
    assert after == before
    assert _bytes(after) == contents
    staging = os.path.join(out, ".staging")
    assert not os.path.isdir(staging) or os.listdir(staging) == []


def test_a_staged_generation_that_fails_its_checks_is_never_committed(tmp_path):
    """The gate between "written" and "committed".

    A generation is immutable once named, so the verification before the
    rename is the last moment anything can be checked at all. A summary that
    belongs to another source or another run is the kind of mix-up that is
    silent afterwards and obvious here.
    """
    from ftmap.pipeline import IncompleteGeneration, _verify_staged

    workspace = create_workspace(str(tmp_path / "w"))
    staged = workspace.staging_dir()
    for name in ("profile.json", "plan.json", "plan.validated.json",
                 "decisions.jsonl", "normalized.csv", "rejects.jsonl",
                 "entities.ftm.json", "statements.csv"):
        with open(os.path.join(staged, name), "w", encoding="utf-8"):
            pass

    with pytest.raises(IncompleteGeneration) as excinfo:
        _verify_staged(staged, "s-1", "run1", declined=False)
    assert "summary.json" in str(excinfo.value)

    with open(os.path.join(staged, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"safe_id": "SOMEONE-ELSE", "run_id": "run1"}, fh)
    with open(os.path.join(staged, "mapping.yml"), "w", encoding="utf-8") as fh:
        fh.write("x: 1")
    with pytest.raises(IncompleteGeneration) as excinfo:
        _verify_staged(staged, "s-1", "run1", declined=False)
    assert "SOMEONE-ELSE" in str(excinfo.value)


def test_a_mapping_staged_beside_a_declined_summary_is_refused(tmp_path):
    """The specific inconsistency that used to outlive a failed run: a YAML
    describing entities beside a summary saying none survived."""
    from ftmap.pipeline import IncompleteGeneration, _verify_staged

    workspace = create_workspace(str(tmp_path / "w"))
    staged = workspace.staging_dir()
    for name in ("profile.json", "plan.json", "plan.validated.json",
                 "decisions.jsonl", "normalized.csv", "rejects.jsonl",
                 "entities.ftm.json", "statements.csv", "mapping.yml"):
        with open(os.path.join(staged, name), "w", encoding="utf-8"):
            pass
    with open(os.path.join(staged, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"safe_id": "s-1", "run_id": "run1"}, fh)

    with pytest.raises(IncompleteGeneration):
        _verify_staged(staged, "s-1", "run1", declined=True)
    # ...and the same directory passes when the plan really produced one.
    assert _verify_staged(staged, "s-1", "run1", declined=False)


# --------------------------------------------------------------- hard exit


_KILL_SCRIPT = textwrap.dedent('''
    import os, signal, sys, time
    sys.path.insert(0, {repo!r} + "/src")
    from ftmap.config import Config
    from ftmap.inventory import inventory
    from ftmap.pipeline import run_source
    from ftmap.plan.prompt import BINDING_MARKER
    from ftmap.vocab.catalogue import Catalogue
    import ftmap.pipeline as pipeline
    import ftmap.workspace as workspace

    class C:
        calls = cache_hits = 0
        def complete(self, system, user, schema, max_tokens=None):
            if BINDING_MARKER not in user:
                return {{"subject": "Person", "entities": [
                    {{"key": "person", "schema": "Person", "keys": ["c0"]}}],
                    "edges": []}}
            return {{"bindings": [
                {{"column": l.split(":", 1)[0],
                  "binding": "person|Person:name", "why": "x"}}
                for l in user.splitlines()
                if l[:1] == "c" and ":" in l and l.split(":", 1)[0][1:].isdigit()]}}

    # Kill the process at the chosen point with SIGKILL, which runs no
    # `finally`, no `atexit`, and no interpreter shutdown — the only faithful
    # stand-in for a power cut that a test can arrange.
    WHEN = {when!r}
    real_rename = os.rename
    real_commit = workspace.Workspace._commit_manifest

    def rename(a, b):
        if WHEN == "before-manifest" and "sources" in str(b):
            real_rename(a, b)
            os.kill(os.getpid(), signal.SIGKILL)
        return real_rename(a, b)

    def commit(self, manifest):
        result = real_commit(self, manifest)
        if WHEN == "after-manifest":
            os.kill(os.getpid(), signal.SIGKILL)
        return result

    os.rename = rename
    workspace.Workspace._commit_manifest = commit

    refs = inventory({corpus!r})
    run_source(refs[0], Config.load(None), Catalogue.load(), C(), "run2",
               {out!r})
    print("NOT KILLED", file=sys.stderr)
''')


@pytest.mark.parametrize("when", ["before-manifest", "after-manifest"])
def test_a_hard_exit_exposes_a_complete_old_or_a_complete_new_generation(
        tmp_path, when):
    """SIGKILL on either side of the manifest replacement.

    Before it, the new generation directory exists in full and is named by
    nothing, so a reader sees the old one — whole. After it, the reader sees
    the new one — whole. There is no third outcome, and no point at which a
    reader can see files from both.
    """
    root = _corpus(tmp_path, names=("a.csv",))
    out = str(tmp_path / "work")
    refs = inventory(root)
    first = run_source(refs[0], CFG, CAT, ScriptedClient(), "run1", out)
    safe_id = first["safe_id"]
    before_dir = open_workspace(out).source_dir(safe_id)

    script = _KILL_SCRIPT.format(repo=REPO, corpus=root, out=out, when=when)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    assert result.returncode == -signal.SIGKILL, (result.returncode,
                                                  result.stderr[-2000:])

    workspace = open_workspace(out)
    assert workspace.source_ids() == [safe_id]
    current = workspace.source_dir(safe_id)
    if when == "before-manifest":
        assert current == before_dir
    else:
        assert current != before_dir
    # Whichever it is, it is COMPLETE — the manifest never names a directory
    # that is missing an artefact a reader needs.
    for name in ("profile.json", "plan.json", "plan.validated.json",
                 "decisions.jsonl", "normalized.csv", "rejects.jsonl",
                 "entities.ftm.json", "statements.csv", "summary.json"):
        assert os.path.exists(os.path.join(current, name)), name
    with open(os.path.join(current, "summary.json"), encoding="utf-8") as fh:
        assert json.load(fh)["safe_id"] == safe_id


def test_a_hard_exit_mid_corpus_keeps_the_sources_that_committed(tmp_path):
    """Covers AE9, the crash half.

    A corpus run commits each source as it finishes. A process killed halfway
    leaves exactly the finished ones, each whole, with no aggregate — and the
    next run over the same workspace redoes only what it lost.
    """
    root = _corpus(tmp_path, names=("a.csv", "b.csv", "c.csv"))
    out = str(tmp_path / "work")
    script = textwrap.dedent(f'''
        import os, signal, sys
        sys.path.insert(0, {REPO!r} + "/src")
        from ftmap.config import Config
        from ftmap.pipeline import run_corpus
        from ftmap.plan.prompt import BINDING_MARKER
        from ftmap.vocab.catalogue import Catalogue

        class C:
            calls = cache_hits = 0
            def complete(self, system, user, schema, max_tokens=None):
                if BINDING_MARKER not in user:
                    return {{"subject": "Person", "entities": [
                        {{"key": "person", "schema": "Person",
                          "keys": ["c0"]}}], "edges": []}}
                return {{"bindings": [
                    {{"column": l.split(":", 1)[0],
                      "binding": "person|Person:name", "why": "x"}}
                    for l in user.splitlines()
                    if l[:1] == "c" and ":" in l
                    and l.split(":", 1)[0][1:].isdigit()]}}

        seen = []
        def progress(event):
            if event["kind"] == "source_done":
                seen.append(event["safe_id"])
                if len(seen) == 2:
                    os.kill(os.getpid(), signal.SIGKILL)

        run_corpus({root!r}, Config.load(None), Catalogue.load(), C(),
                   {out!r}, on_progress=progress)
    ''')
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    assert result.returncode == -signal.SIGKILL, result.stderr[-2000:]

    workspace = open_workspace(out)
    assert len(workspace.source_ids()) == 2
    # No aggregate: the run never reached the point of building one, and a
    # partial merge is not something that can be inferred later.
    assert workspace.aggregate_dir() is None
    assert not workspace.publication_grade()

    # The next run finishes the job, and only then is it publication-grade.
    report = run_corpus(root, CFG, CAT, ScriptedClient(), out)
    assert report["sources"] == 3
    assert not report["incomplete"]
    assert open_workspace(out).publication_grade()


# ------------------------------------------------------------- concurrency


def test_two_writers_over_one_workspace_cannot_interleave(tmp_path):
    """Covers the concurrency half of R2.

    Two processes each commit a manifest; interleaved, the second replaces the
    first's entry with a document that never saw it. One is accepted, the
    other is refused, and NEITHER committed generation is destroyed.
    """
    root = _corpus(tmp_path, names=("a.csv",))
    out = str(tmp_path / "work")
    run_corpus(root, CFG, CAT, ScriptedClient(), out)
    workspace = open_workspace(out)
    safe_id = workspace.source_ids()[0]
    before = workspace.source_dir(safe_id)

    from ftmap.workspace import GenerationConflict

    holder = open_workspace(out)
    with holder.lock():
        with pytest.raises(GenerationConflict):
            run_corpus(root, CFG, CAT, ScriptedClient(), out)

    assert open_workspace(out).source_dir(safe_id) == before
    assert os.path.exists(os.path.join(before, "summary.json"))


def test_the_manifest_is_replaced_last_and_after_the_generation_is_durable(
        tmp_path, monkeypatch):
    """The commit protocol, observed as a sequence.

    Renaming a directory whose contents are still in the page cache publishes
    a name that, after a power cut, can point at files shorter than they were
    written. And replacing the manifest before that rename would name a
    directory that is not there yet. Both are invisible in a healthy run,
    which is why this asserts the ORDER rather than the outcome.
    """
    import ftmap.fsutil as fsutil
    import ftmap.workspace as workspace_module

    events: list[str] = []
    real_rename, real_sync_dir = os.rename, fsutil.sync_dir
    real_sync_file = fsutil.sync_file
    real_durable = fsutil.durable_write

    monkeypatch.setattr(workspace_module.os, "rename",
                        lambda a, b: (events.append("rename-generation"),
                                      real_rename(a, b))[1])
    monkeypatch.setattr(workspace_module, "sync_dir",
                        lambda p: (events.append("sync-parent"),
                                   real_sync_dir(p))[1])
    monkeypatch.setattr(fsutil, "sync_file",
                        lambda fd: (events.append("sync-file"),
                                    real_sync_file(fd))[1])

    def watched(path, **kw):
        if os.path.basename(path) == "workspace.json":
            events.append("replace-manifest")
        return real_durable(path, **kw)

    monkeypatch.setattr(workspace_module, "durable_write", watched)

    root = _corpus(tmp_path, names=("a.csv",))
    out = str(tmp_path / "work")
    run_source(inventory(root)[0], CFG, CAT, ScriptedClient(), "run1", out)

    # Every staged file is synchronized before the rename that publishes them.
    rename = events.index("rename-generation")
    assert "sync-file" in events[:rename]
    # The parent directory is synchronized after the rename, so the rename
    # itself is durable and not merely the bytes it exposes.
    assert "sync-parent" in events[rename:]
    # And the manifest — the only thing that makes any of it visible — is
    # replaced after all of it. The FIRST manifest write in this list is
    # `create_workspace` writing an empty one, which is why the last is what
    # matters.
    assert len(events) - 1 - events[::-1].index("replace-manifest") > rename
