# tests/test_fsutil.py
"""The durable-write primitive and the input snapshot store.

Both are about a claim that cannot be checked after the fact: that the bytes a
run reports on are the bytes it read, and that a workspace interrupted at any
moment holds whole artefacts rather than halves of two. The tests are written
against the SEQUENCE of syscalls where the outcome is otherwise invisible —
a write that skipped the parent-directory sync looks identical to one that did
not, right up until the machine loses power.
"""

import errno
import hashlib
import os
import pathlib
import sys

import pytest

from ftmap.fsutil import (PRIVATE, PUBLISHED, SnapshotMismatch, SnapshotStore,
                          durable_write)


def test_a_durable_write_replaces_the_file_whole(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text("old", encoding="utf-8")
    with durable_write(str(target), text=True) as fh:
        fh.write("new")
    assert target.read_text(encoding="utf-8") == "new"


def test_a_failed_write_leaves_the_previous_file_and_no_debris(tmp_path):
    """The destination name is never opened for writing, so there is no
    half-file to recover from — only a temp file to remove."""
    target = tmp_path / "manifest.json"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with durable_write(str(target), text=True) as fh:
            fh.write("half")
            raise RuntimeError("stage failed")
    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_a_keyboard_interrupt_is_cleaned_up_like_any_other_failure(tmp_path):
    """`except Exception` would not have caught this, and Ctrl-C mid-run is
    the case the whole module exists for."""
    target = tmp_path / "manifest.json"
    with pytest.raises(KeyboardInterrupt):
        with durable_write(str(target), text=True) as fh:
            fh.write("half")
            raise KeyboardInterrupt
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_the_file_is_created_at_its_final_mode_not_chmod_ed_after(tmp_path):
    """The window between `open` and `chmod` is a window in which an artefact
    carrying real cell values is world readable."""
    modes: list[int] = []
    private, published = tmp_path / "p.json", tmp_path / "q.json"
    with durable_write(str(private), text=True) as fh:
        modes.append(os.fstat(fh.fileno()).st_mode & 0o777)
        fh.write("{}")
    with durable_write(str(published), mode=PUBLISHED, text=True) as fh:
        modes.append(os.fstat(fh.fileno()).st_mode & 0o777)
        fh.write("{}")
    assert modes == [PRIVATE, PUBLISHED]
    assert oct(private.stat().st_mode)[-3:] == "600"
    assert oct(published.stat().st_mode)[-3:] == "644"


def test_the_temp_file_is_made_in_the_destination_directory(tmp_path):
    """`os.replace` is atomic only WITHIN a filesystem, and the system temp
    directory is routinely a different one — on this machine `/tmp` and a
    workspace under `$HOME` can be two volumes. A cross-device staging file
    turns the atomic rename into a copy with a window in it."""
    target = tmp_path / "deep" / "manifest.json"
    seen: list[str] = []
    with durable_write(str(target), text=True) as fh:
        seen.extend(p.name for p in (tmp_path / "deep").iterdir())
        fh.write("{}")
    assert any(name.startswith(".tmp-") for name in seen), seen


def test_the_syscall_order_is_data_then_replace_then_parent_directory(
        tmp_path, monkeypatch):
    """The step that is easiest to omit and hardest to notice.

    Without the parent-directory sync the file's CONTENT is durable and the
    directory entry naming it is not, so a crash can leave the old name
    pointing at the old inode with the new data stranded. Nothing about the
    resulting workspace looks wrong until it is read after a power cut, which
    is why this is asserted as a sequence rather than as an outcome.
    """
    import ftmap.fsutil as fsutil

    events: list[str] = []
    real_sync_file, real_sync_dir = fsutil.sync_file, fsutil.sync_dir
    real_replace = os.replace

    monkeypatch.setattr(fsutil, "sync_file",
                        lambda fd: (events.append("sync_file"),
                                    real_sync_file(fd))[1])
    monkeypatch.setattr(fsutil, "sync_dir",
                        lambda p: (events.append("sync_dir"),
                                   real_sync_dir(p))[1])
    monkeypatch.setattr(fsutil.os, "replace",
                        lambda a, b: (events.append("replace"),
                                      real_replace(a, b))[1])

    with fsutil.durable_write(str(tmp_path / "m.json"), text=True) as fh:
        fh.write("{}")
    assert events == ["sync_file", "replace", "sync_dir"]


@pytest.mark.skipif(sys.platform != "darwin", reason="F_FULLFSYNC is Darwin's")
def test_on_darwin_a_data_file_gets_f_fullfsync(tmp_path, monkeypatch):
    """macOS's `fsync(2)` returns once the data reaches the DRIVE'S CACHE —
    its own manual page says so and points at `F_FULLFSYNC` for a flush to
    permanent storage. A durability claim resting on `fsync` alone here is a
    claim about a buffer."""
    import ftmap.fsutil as fsutil

    calls: list[int] = []
    real = fsutil.fcntl.fcntl
    monkeypatch.setattr(fsutil.fcntl, "fcntl",
                        lambda fd, op, *a: (calls.append(op),
                                            real(fd, op, *a))[1])
    with fsutil.durable_write(str(tmp_path / "m.json"), text=True) as fh:
        fh.write("{}")
    assert fsutil._F_FULLFSYNC in calls


def test_a_filesystem_that_refuses_full_fsync_falls_back_rather_than_failing(
        tmp_path, monkeypatch):
    """The documented fallback, which is also every non-Darwin platform's
    only path. Exercised here on whatever this is, so the branch is covered
    on both."""
    import ftmap.fsutil as fsutil

    fell_back: list[int] = []
    monkeypatch.setattr(fsutil, "_DARWIN", True)
    monkeypatch.setattr(fsutil.fcntl, "fcntl",
                        lambda fd, op, *a: (_ for _ in ()).throw(
                            OSError(errno.ENOTSUP, "nope")))
    monkeypatch.setattr(fsutil.os, "fsync",
                        lambda fd: fell_back.append(fd))
    with fsutil.durable_write(str(tmp_path / "m.json"), text=True) as fh:
        fh.write("{}")
    assert fell_back, "no fallback to os.fsync"


def test_an_unexpected_fcntl_error_is_not_swallowed(tmp_path, monkeypatch):
    """ENOTSUP means "this filesystem cannot"; EIO means the write failed.
    Treating them alike would turn a real durability failure into silence."""
    import ftmap.fsutil as fsutil

    monkeypatch.setattr(fsutil, "_DARWIN", True)
    monkeypatch.setattr(fsutil.fcntl, "fcntl",
                        lambda fd, op, *a: (_ for _ in ()).throw(
                            OSError(errno.EIO, "disk")))
    with pytest.raises(OSError):
        with fsutil.durable_write(str(tmp_path / "m.json"), text=True) as fh:
            fh.write("{}")


# ---------------------------------------------------------------- snapshots


def _input(tmp_path, name, content=b"col\n1\n"):
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def test_a_snapshot_is_hashed_and_copied_in_one_pass(tmp_path):
    store = SnapshotStore(str(tmp_path / "work"))
    source = _input(tmp_path, "a.csv", "ПІБ\nКоваленко Іван\n".encode())
    digest, path = store.capture(source, ".csv")

    raw = pathlib.Path(source).read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert pathlib.Path(path).read_bytes() == raw
    assert oct(os.stat(path).st_mode)[-3:] == "600"
    # The extension is kept because `io.tabular.read_source` dispatches on it;
    # a digest alone would make every input a CSV.
    assert path.endswith(".csv")


def test_one_snapshot_serves_every_sheet_rerun_and_generation(tmp_path):
    """Content-addressed, so the eleven duplicate-content pairs measured in
    `raw_835` are eleven files rather than twenty-two — and a rerun of the
    same source rewrites nothing that another reader is already open on."""
    store = SnapshotStore(str(tmp_path / "work"))
    source = _input(tmp_path, "a.csv")
    copy = _input(tmp_path, "b_copy.csv")

    first = store.capture(source, ".csv")
    second = store.capture(source, ".csv")
    third = store.capture(copy, ".csv")
    assert first == second == third
    stored = [p for p in (tmp_path / "work" / ".snapshots").rglob("*")
              if p.is_file()]
    assert len(stored) == 1


def test_mutating_the_original_input_does_not_reach_the_snapshot(tmp_path):
    """The failure this whole class is for: a digest recorded once at
    inventory, while every later stage re-opened the ORIGINAL path."""
    store = SnapshotStore(str(tmp_path / "work"))
    source = _input(tmp_path, "a.csv", b"before\n")
    digest, path = store.capture(source, ".csv")

    pathlib.Path(source).write_bytes(b"after-the-fact\n")
    assert pathlib.Path(path).read_bytes() == b"before\n"
    assert store.verify(digest, ".csv") == path


def test_a_corrupted_snapshot_fails_closed_rather_than_being_read(tmp_path):
    """Immutable by contract, not by permission. The owner can still write to
    it, a filesystem can still corrupt it, and a backup restore can still put
    an older one back — so the bytes are re-hashed before they are trusted."""
    store = SnapshotStore(str(tmp_path / "work"))
    digest, path = store.capture(_input(tmp_path, "a.csv"), ".csv")
    pathlib.Path(path).write_bytes(b"tampered\n")

    with pytest.raises(SnapshotMismatch) as excinfo:
        store.verify(digest, ".csv")
    assert digest[:12] in str(excinfo.value)


def test_a_failed_capture_leaves_no_partial_snapshot(tmp_path):
    store = SnapshotStore(str(tmp_path / "work"))
    with pytest.raises(FileNotFoundError):
        store.capture(str(tmp_path / "absent.csv"), ".csv")
    stored = [p for p in (tmp_path / "work" / ".snapshots").rglob("*")
              if p.is_file()]
    assert stored == []
