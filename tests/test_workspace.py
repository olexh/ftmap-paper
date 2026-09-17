# tests/test_workspace.py
"""The manifest as the registry, and the three ways to read a workspace.

Every test here is about something a directory listing gets wrong. The
filesystem knows what exists; it does not know what is FINISHED, and four
measured defects came from asking it anyway.
"""

import json
import os

import pytest

from ftmap.workspace import (FORMAT, GenerationConflict, LegacyWorkspace,
                             UnknownSource, Workspace, WorkspaceError,
                             create_workspace, new_generation_id,
                             open_workspace, upgrade)


def _stage(workspace, **files):
    staged = workspace.staging_dir()
    for name, body in files.items():
        with open(os.path.join(staged, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return staged


def _commit(workspace, safe_id, **files):
    files.setdefault("summary.json", json.dumps({"safe_id": safe_id}))
    return workspace.commit_source(safe_id, _stage(workspace, **files),
                                   {"source_id": safe_id})


def test_a_new_workspace_starts_empty_and_is_generation_mode(tmp_path):
    workspace = create_workspace(str(tmp_path / "w"))
    assert workspace.mode == "generation"
    assert workspace.writable
    assert workspace.source_ids() == []
    assert workspace.manifest["format"] == FORMAT


def test_the_manifest_is_the_registry_not_the_directory_listing(tmp_path):
    """The defect this module replaces, in one assertion.

    A source directory from an earlier corpus, a staging path, and a
    `.rollback-<pid>` backup all sit under the workspace and all can carry a
    `summary.json`. None of them is a source of THIS run, and a scan cannot
    tell the difference.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    _commit(workspace, "real-abcd1234")

    root = tmp_path / "w"
    for stranger in ("leftover-99999999", "real-abcd1234.rollback-4242"):
        (root / stranger).mkdir()
        (root / stranger / "summary.json").write_text("{}", encoding="utf-8")
    _stage(workspace, **{"summary.json": "{}"})  # an in-flight attempt

    assert open_workspace(str(root)).source_ids() == ["real-abcd1234"]


def test_an_unknown_source_raises_rather_than_returning_a_plausible_path(
        tmp_path):
    """A caller that joins a name the manifest does not hold is rebuilding
    the phantom-source bug by hand, one `os.path.join` at a time."""
    workspace = create_workspace(str(tmp_path / "w"))
    with pytest.raises(UnknownSource):
        workspace.source_dir("never-committed")


def test_a_commit_is_visible_only_after_the_manifest_names_it(tmp_path):
    """The generation directory can exist and still not be a source.

    That is the whole point of the ordering: a crash between the rename and
    the manifest replacement leaves a complete, invisible, removable
    directory — never a half-visible one.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    staged = _stage(workspace, **{"summary.json": "{}"})

    generation = new_generation_id()
    final = tmp_path / "w" / "sources" / "orphan-1234" / generation
    final.parent.mkdir(parents=True)
    os.rename(staged, final)
    assert (final / "summary.json").exists()

    assert open_workspace(str(tmp_path / "w")).source_ids() == []


def test_a_rerun_leaves_the_previous_generation_intact(tmp_path):
    """Immutable means the old directory is still there and still readable.

    An in-place overwrite is what let an interrupted rerun leave
    `plan.validated.json` from the new attempt beside `statements.csv` from
    the old, with nothing on disk saying so.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    first = _commit(workspace, "s-1", **{"statements.csv": "first"})
    workspace = open_workspace(str(tmp_path / "w"))
    second = _commit(workspace, "s-1", **{"statements.csv": "second"})

    assert first != second
    with open(os.path.join(first, "statements.csv"), encoding="utf-8") as fh:
        assert fh.read() == "first"
    workspace = open_workspace(str(tmp_path / "w"))
    assert workspace.source_dir("s-1") == second


def test_the_mapping_lives_inside_the_generation_it_belongs_to(tmp_path):
    """A shared `mappings/` directory is why a failed first run left one.

    The source directory was rolled back and the mapping was not, so `ftm map`
    could be pointed at a document describing a run that produced nothing.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    committed = _commit(workspace, "s-1", **{"mapping.yml": "x: 1"})
    assert workspace.mapping_path("s-1") == os.path.join(committed, "mapping.yml")


def test_decisions_live_outside_the_generation_because_they_outlive_it(
        tmp_path):
    """The mirror image of the mapping. An analyst decision is made ABOUT a
    generation and has to survive the rerun that replaces it, so it cannot
    live in a directory that rerun makes immutable and superseded."""
    workspace = create_workspace(str(tmp_path / "w"))
    committed = _commit(workspace, "s-1")
    assert not workspace.decisions_path("s-1").startswith(committed)
    assert workspace.decisions_path("s-1").startswith(
        os.path.join(workspace.root, "decisions"))


# ------------------------------------------------------------- staleness


def test_a_source_rerun_makes_the_aggregate_stale_without_rebuilding_it(
        tmp_path):
    """One review action must not re-merge hundreds of megabytes.

    Staleness is DERIVED, not flagged: the aggregate records the generation
    each source was at when it was built, so a source that has moved on since
    is stale whether or not anyone remembered to say so.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    _commit(workspace, "s-1")
    _commit(workspace, "s-2")
    workspace.commit_aggregate(
        _stage(workspace, **{"run.json": '{"accounting_valid": true}'}),
        {"summaries": ["s-1", "s-2"]})
    assert workspace.stale_sources() == []
    assert workspace.publication_grade()

    _commit(workspace, "s-1")
    assert workspace.stale_sources() == ["s-1"]
    assert not workspace.publication_grade()


def test_publication_grade_requires_the_ledger_to_vouch_for_itself(tmp_path):
    """`accounting_valid` must be exactly True. A false flag is an invalid
    ledger; a missing one is a legacy aggregate that predates the gate and
    cannot vouch either way — unvouched, not publication-grade (2026-08-31
    follow-up review P1)."""
    for raw, grade in (('{"accounting_valid": true}', True),
                       ('{"accounting_valid": false}', False),
                       ('{"accounting_valid": null}', False),
                       ("{}", False)):
        workspace = create_workspace(str(tmp_path / f"w-{grade}-{len(raw)}"))
        _commit(workspace, "s-1")
        workspace.commit_aggregate(_stage(workspace, **{"run.json": raw}),
                                   {"summaries": ["s-1"]})
        assert workspace.stale_sources() == []
        assert workspace.publication_grade() is grade, raw


def test_an_incomplete_run_is_never_publication_grade(tmp_path):
    """A cancelled or partly-failed run is a real run over fewer sources.

    It may commit an internally consistent aggregate — the sources that
    finished are still worth having — but a number read out of it is a number
    from a corpus nobody chose.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    _commit(workspace, "s-1")
    workspace.commit_aggregate(_stage(workspace, **{"run.json": "{}"}),
                               {"summaries": ["s-1"]}, incomplete=True,
                               failures=[{"path": "b.xlsx"}])
    assert workspace.stale_sources() == []
    assert not workspace.publication_grade()


# ------------------------------------------------------------ legacy modes


def _legacy_manifest_workspace(tmp_path):
    root = tmp_path / "work-corpus"
    (root / "mappings").mkdir(parents=True)
    for safe_id in ("aaa__", "bbb__Лист1"):
        (root / safe_id).mkdir()
        (root / safe_id / "summary.json").write_text(
            json.dumps({"path": f"/corpus/{safe_id}.csv"}), encoding="utf-8")
        (root / "mappings" / f"{safe_id}.yml").write_text("x: 1", encoding="utf-8")
    (root / "run.json").write_text(
        json.dumps({"summaries": ["aaa__", "bbb__Лист1"]}), encoding="utf-8")
    return root


def test_a_retained_run_json_workspace_reads_through_its_summaries(tmp_path):
    workspace = open_workspace(str(_legacy_manifest_workspace(tmp_path)))
    assert workspace.mode == "legacy-manifest"
    assert workspace.source_ids() == ["aaa__", "bbb__Лист1"]
    assert workspace.mapping_path("aaa__").endswith("mappings/aaa__.yml")


def test_a_retained_workspace_with_no_run_json_is_read_by_scanning(tmp_path):
    """`legacy-scan` is not a fallback to be deleted later.

    The retained `work-corpus` has hundreds of source directories and no
    `run.json` at all, so this is the ONLY mode that can read the corpus the
    published measurement was taken from.
    """
    root = _legacy_manifest_workspace(tmp_path)
    (root / "run.json").unlink()
    (root / ".staging").mkdir()
    (root / ".staging" / "summary.json").write_text("{}", encoding="utf-8")
    (root / "aaa__.rollback-77").mkdir()
    (root / "aaa__.rollback-77" / "summary.json").write_text("{}", encoding="utf-8")

    workspace = open_workspace(str(root))
    assert workspace.mode == "legacy-scan"
    assert workspace.source_ids() == ["aaa__", "bbb__Лист1"]


@pytest.mark.parametrize("drop_run_json", [False, True])
def test_a_legacy_workspace_refuses_a_durable_write(tmp_path, drop_run_json):
    """Retained evidence is not migrated in place, in either mode."""
    root = _legacy_manifest_workspace(tmp_path)
    if drop_run_json:
        (root / "run.json").unlink()
    workspace = open_workspace(str(root))
    assert not workspace.writable
    with pytest.raises(LegacyWorkspace) as excinfo:
        workspace.require_writable()
    assert "ftmap upgrade" in str(excinfo.value)


def test_upgrade_builds_a_new_workspace_and_never_touches_the_old_one(tmp_path):
    old = _legacy_manifest_workspace(tmp_path)
    (old / "aaa__" / "overrides.jsonl").write_text(
        json.dumps({"column": "c0", "prop": "Person:name"}) + "\n",
        encoding="utf-8")
    before = {p.relative_to(old): p.read_bytes()
              for p in old.rglob("*") if p.is_file()}

    new = upgrade(str(old), str(tmp_path / "upgraded"))
    assert new.mode == "generation"
    assert new.source_ids() == ["aaa__", "bbb__Лист1"]
    # The mapping moved inside the generation, the decisions moved out of it.
    assert os.path.exists(new.mapping_path("aaa__"))
    assert os.path.exists(new.decisions_path("aaa__"))
    assert not os.path.exists(os.path.join(new.source_dir("aaa__"),
                                           "overrides.jsonl"))
    # And the retained evidence is byte-for-byte what it was.
    after = {p.relative_to(old): p.read_bytes()
             for p in old.rglob("*") if p.is_file()}
    assert after == before


def test_upgrade_refuses_a_destination_that_already_holds_data(tmp_path):
    old = _legacy_manifest_workspace(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "something.txt").write_text("x", encoding="utf-8")
    with pytest.raises(WorkspaceError):
        upgrade(str(old), str(dest))


def test_a_manifest_from_a_newer_ftmap_is_refused_rather_than_half_read(
        tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    (root / "workspace.json").write_text(
        json.dumps({"format": FORMAT + 1, "sources": {}}), encoding="utf-8")
    with pytest.raises(WorkspaceError):
        open_workspace(str(root))


# ------------------------------------------------------------------- locks


def test_a_second_writer_is_refused_rather_than_queued(tmp_path):
    """A queue would make the second rerun wait and then run against a
    workspace that changed underneath the decision that started it."""
    workspace = create_workspace(str(tmp_path / "w"))
    other = open_workspace(str(tmp_path / "w"))
    with workspace.lock():
        assert workspace.active() == ["workspace.lock"]
        with pytest.raises(GenerationConflict) as excinfo:
            with other.lock():
                pass
        assert str(os.getpid()) in str(excinfo.value)
    assert workspace.active() == []


def test_a_lock_is_released_even_when_the_body_raises(tmp_path):
    workspace = create_workspace(str(tmp_path / "w"))
    with pytest.raises(RuntimeError):
        with workspace.lock():
            raise RuntimeError("stage failed")
    assert workspace.active() == []
    with workspace.lock():
        pass


def test_two_names_are_two_locks(tmp_path):
    """A per-source rerun and a corpus publish are different operations, and
    the workspace lock is what stops THEM interleaving — not two unrelated
    reads of different sources."""
    workspace = create_workspace(str(tmp_path / "w"))
    with workspace.lock("source:s-1"):
        with workspace.lock("source:s-2"):
            assert len(workspace.active()) == 2


# ------------------------------------------------------------ manifest form


def test_the_manifest_is_published_and_carries_no_cell_value(tmp_path):
    """It names sources, digests, generations and counts. Everything in it is
    checked against that claim in `tests/test_boundary.py`; this asserts the
    file mode the claim licenses."""
    workspace = create_workspace(str(tmp_path / "w"))
    _commit(workspace, "s-1")
    path = os.path.join(workspace.root, "workspace.json")
    assert oct(os.stat(path).st_mode)[-3:] == "644"
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["format"] == FORMAT


def test_a_concurrent_commit_is_not_lost_by_a_stale_in_memory_manifest(
        tmp_path):
    """Two `Workspace` objects over one directory, which is what the API and
    a CLI rerun actually are. The one that commits second must not replace
    the first's entry with a manifest that never saw it."""
    first = create_workspace(str(tmp_path / "w"))
    second = open_workspace(str(tmp_path / "w"))
    _commit(first, "s-1")
    _commit(second, "s-2")
    assert open_workspace(str(tmp_path / "w")).source_ids() == ["s-1", "s-2"]


def test_the_same_workspace_object_may_take_a_lock_it_already_holds(tmp_path):
    """A corpus run holds the workspace lock and calls `run_source`, which
    asks for it again. That must not deadlock — while a SECOND object over
    the same directory is exactly the collision the lock is for."""
    workspace = create_workspace(str(tmp_path / "w"))
    other = open_workspace(str(tmp_path / "w"))
    with workspace.lock():
        with workspace.lock():
            with pytest.raises(GenerationConflict):
                with other.lock():
                    pass
        # Still held after the inner block: reentrancy releases once.
        assert workspace.active() == ["workspace.lock"]
    assert workspace.active() == []


def test_a_caller_may_choose_the_generation_id_before_staging(tmp_path):
    """The mapping document names its CSV by absolute path, and that path is
    inside the generation. Compiling against staging and renaming afterwards
    would publish a mapping pointing at a directory that no longer exists."""
    workspace = create_workspace(str(tmp_path / "w"))
    chosen = new_generation_id()
    committed = workspace.commit_source(
        "s-1", _stage(workspace, **{"summary.json": "{}"}),
        {"source_id": "s-1"}, generation=chosen)
    assert committed.endswith(os.path.join("sources", "s-1", chosen))


# --------------------------------------------------------------- retention


def _snapshot(workspace, digest, ext=".csv", body=b"x"):
    path = os.path.join(workspace.root, ".snapshots", digest[:2], digest + ext)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(body)
    return path


def _commit_with_digest(workspace, safe_id, digest):
    summary = json.dumps({"safe_id": safe_id, "source": {"sha256": digest}})
    return workspace.commit_source(
        safe_id, _stage(workspace, **{"summary.json": summary}),
        {"source_id": safe_id, "sha256": digest})


def test_retention_keeps_the_current_and_previous_generation(tmp_path):
    """One previous, because what an operator wants after a bad rerun is the
    generation before it — and not more, because a corpus of 360 sources
    multiplies whatever number is chosen."""
    from ftmap.workspace import Retention

    workspace = create_workspace(str(tmp_path / "w"))
    kept = [_commit(workspace, "s-1") for _ in range(4)]
    workspace.commit_aggregate(_stage(workspace, **{"run.json": "{}"}),
                               {"summaries": ["s-1"]})

    plan = Retention(workspace).reclaimable()
    assert sorted(plan["generations"]) == sorted(kept[:2])
    assert plan["bytes"] > 0

    Retention(workspace).prune()
    workspace = open_workspace(str(tmp_path / "w"))
    assert os.path.isdir(workspace.source_dir("s-1"))
    assert os.path.isdir(kept[2]) and not os.path.isdir(kept[1])


def test_prune_never_removes_what_the_manifest_names(tmp_path):
    """The current generation is kept FIRST and unconditionally. A sort order
    is not an argument for keeping the right file."""
    from ftmap.workspace import Retention

    workspace = create_workspace(str(tmp_path / "w"))
    _commit(workspace, "s-1")
    _commit(workspace, "s-1")
    current = workspace.source_dir("s-1")
    Retention(workspace).prune()
    assert os.path.isdir(current)
    assert os.path.exists(os.path.join(current, "summary.json"))


def test_a_snapshot_a_retained_generation_still_needs_is_not_removed(tmp_path):
    """Reference-counted from the RETAINED generations, not the manifest.

    A superseded generation that prune is about to keep still describes the
    input it was built from, and removing that input leaves a retained
    generation whose provenance cannot be checked.
    """
    from ftmap.workspace import Retention

    workspace = create_workspace(str(tmp_path / "w"))
    live = "a" * 64
    orphan = "b" * 64
    _snapshot(workspace, live)
    _snapshot(workspace, orphan)
    _commit_with_digest(workspace, "s-1", live)

    plan = Retention(workspace).reclaimable()
    assert [os.path.basename(p) for p in plan["snapshots"]] == [orphan + ".csv"]

    Retention(workspace).prune()
    assert os.path.exists(os.path.join(workspace.root, ".snapshots",
                                       live[:2], live + ".csv"))
    assert not os.path.exists(os.path.join(workspace.root, ".snapshots",
                                           orphan[:2], orphan + ".csv"))


def test_prune_refuses_while_another_process_holds_the_workspace(tmp_path):
    """A review server serving a browser request out of a superseded
    generation is the case this exists not to break, and from here it is
    invisible except as a lock."""
    from ftmap.workspace import Retention

    workspace = create_workspace(str(tmp_path / "w"))
    for _ in range(3):
        _commit(workspace, "s-1")
    assert len(Retention(workspace).reclaimable()["generations"]) == 1

    other = open_workspace(str(tmp_path / "w"))
    with other.lock():
        with pytest.raises(GenerationConflict):
            Retention(workspace).prune()
    # And the generation it would have removed is still there.
    assert len(Retention(workspace).reclaimable()["generations"]) == 1


def test_abandoned_staging_is_cleaned_but_only_by_prune(tmp_path):
    """A crashed attempt leaves a staging directory. It is invisible — nothing
    names it and the scanner skips dotted names — so it is a storage question
    rather than a correctness one, and it is answered at the same moment as
    every other storage question."""
    from ftmap.workspace import Retention

    workspace = create_workspace(str(tmp_path / "w"))
    abandoned = _stage(workspace, **{"half.json": "{"})
    assert open_workspace(str(tmp_path / "w")).source_ids() == []
    assert Retention(workspace).reclaimable()["staging"] == [abandoned]
    Retention(workspace).prune()
    assert not os.path.exists(abandoned)


def test_a_legacy_workspace_cannot_be_pruned(tmp_path):
    from ftmap.workspace import Retention

    workspace = open_workspace(str(_legacy_manifest_workspace(tmp_path)))
    with pytest.raises(LegacyWorkspace):
        Retention(workspace).prune()


def test_a_lock_left_by_a_dead_process_does_not_brick_the_workspace(tmp_path):
    """SIGKILL runs no `finally`, so a crashed writer leaves its lock file.

    Refusing every later run on that evidence turns one crash into a workspace
    nobody can write to again without knowing to delete a dotfile — which is a
    worse failure than the one the lock prevents.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    stale = os.path.join(workspace.root, ".locks", "workspace.lock")
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    # A pid that is not running. Chosen high and checked, rather than assumed.
    dead = 999_999
    with pytest.raises(OSError):
        os.kill(dead, 0)
    with open(stale, "w", encoding="utf-8") as fh:
        json.dump({"pid": dead, "at": "2026-08-24T00:00:00+00:00"}, fh)

    with workspace.lock():
        pass
    assert not os.path.exists(stale)


def test_a_lock_held_by_a_live_process_is_still_refused(tmp_path):
    """The liveness check must not turn the lock into a suggestion."""
    workspace = create_workspace(str(tmp_path / "w"))
    other = open_workspace(str(tmp_path / "w"))
    with workspace.lock():
        with pytest.raises(GenerationConflict) as excinfo:
            with other.lock():
                pass
    assert "wait for it to finish" in str(excinfo.value)


def test_a_lock_holder_whose_pid_is_unreadable_is_treated_as_alive(tmp_path):
    """Refusing a write is recoverable; stealing a lock from a live writer is
    not, so a pid the liveness check cannot parse counts as held."""
    workspace = create_workspace(str(tmp_path / "w"))
    stale = os.path.join(workspace.root, ".locks", "workspace.lock")
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    with open(stale, "w", encoding="utf-8") as fh:
        json.dump({"pid": "not-a-number", "at": "2026-08-25"}, fh)
    with pytest.raises(GenerationConflict):
        with workspace.lock():
            pass


def test_a_lock_file_naming_no_pid_at_all_is_recovered(tmp_path):
    """The opposite case, and it used to brick the workspace forever.

    `_take` publishes the holder record ATOMICALLY — written to a temp file
    and linked into place — so a lock this version wrote always names a pid.
    A file without one is a zero-byte remnant from the version that created
    the lock before writing into it, and a process killed in that window left
    every later write refused with no way back but deleting a dotfile.

    So absence of a pid means stale, and the difference from the test above
    is deliberate: an unreadable pid is evidence of a writer, and no pid at
    all is evidence of a crash between two syscalls that are now one.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    stale = os.path.join(workspace.root, ".locks", "workspace.lock")
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    open(stale, "w").close()                      # zero bytes, as the crash left it

    with workspace.lock():
        pass
    assert workspace.active() == []


def test_a_released_lock_is_only_removed_by_the_holder_that_still_owns_it(
        tmp_path):
    """The release used to unlink unconditionally.

    A writer that had lost its lock to a stale-takeover still removed the
    file on the way out, stripping the lock the new holder was inside. The
    record carries a token, and the release checks it.
    """
    workspace = create_workspace(str(tmp_path / "w"))
    path = os.path.join(workspace.root, ".locks", "workspace.lock")

    with workspace.lock():
        # Someone else takes over and records themselves as the holder.
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "token": "someone-else"}, fh)

    assert os.path.exists(path), "the other holder's lock was stripped"
    os.unlink(path)


def test_a_derived_index_is_pruned_with_the_generation_it_describes(tmp_path):
    """It is keyed on a source generation id, so one whose generation is going
    describes a directory that is about to stop existing — and it is a cache,
    so removing it costs a rebuild and nothing else."""
    from ftmap.workspace import DERIVED, Retention

    workspace = create_workspace(str(tmp_path / "w"))
    generations = [_commit(workspace, "s-1") for _ in range(3)]

    for committed in generations:
        directory = os.path.join(workspace.root, DERIVED,
                                 os.path.basename(committed), "s-1")
        os.makedirs(directory)
        with open(os.path.join(directory, "index.sqlite"), "wb") as fh:
            fh.write(b"x" * 1024)

    plan = Retention(workspace).reclaimable()
    assert [os.path.basename(p) for p in plan["derived"]] == [
        os.path.basename(generations[0])]

    Retention(workspace).prune()
    remaining = sorted(os.listdir(os.path.join(workspace.root, DERIVED)))
    assert remaining == sorted(os.path.basename(g) for g in generations[1:])
