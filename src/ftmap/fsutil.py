"""Writes that survive a power cut, and one immutable copy of every input."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import IO, Any, Iterator

PRIVATE = 0o600
PUBLISHED = 0o644

_DARWIN = sys.platform == "darwin"
_F_FULLFSYNC = 51


def sync_file(fd: int) -> None:
    """Flush one open data file all the way to stable storage."""
    if _DARWIN:
        try:
            fcntl.fcntl(fd, _F_FULLFSYNC)
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTSUP, errno.EINVAL, errno.ENOTTY):
                raise
    os.fsync(fd)


def sync_dir(path: str) -> None:
    """Make a directory's own entries durable — the step that gets forgotten.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def durable_write(path: str, mode: int = PRIVATE,
                  text: bool = False) -> Iterator[IO]:
    """Write `path` so a reader sees the whole old file or the whole new one.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-",
                              suffix=os.path.basename(path))
    try:
        os.fchmod(fd, mode)
        handle = os.fdopen(fd, "w" if text else "wb",
                           encoding="utf-8" if text else None,
                           newline="" if text else None)
        try:
            yield handle
            handle.flush()
            sync_file(handle.fileno())
        finally:
            handle.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    sync_dir(directory)


def sha256_stream(fh: IO[bytes], sink: IO[bytes] | None = None) -> str:
    """Digest an open binary file in 1 MiB chunks, optionally copying as it
    goes.
    """
    digest = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        digest.update(chunk)
        if sink is not None:
            sink.write(chunk)
    return digest.hexdigest()


def sha256_file(path: str) -> str:
    """`sha256_stream` over a path, for callers that only want the digest."""
    with open(path, "rb") as fh:
        return sha256_stream(fh)


def open_private(path: str, mode: int = PRIVATE, *,
                 append: bool = False, newline: str = "") -> IO[str]:
    """Open a text artefact for writing AT `mode`, never chmod-ed afterwards.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, mode)
    try:
        os.fchmod(fd, mode)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "a" if append else "w",
                     encoding="utf-8", newline=newline)


def dump_json(obj: Any, fh: IO[str], *, newline: bool = True) -> None:
    """Write `obj` as this project's canonical JSON."""
    json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
    if newline:
        fh.write("\n")


def json_line(obj: Any) -> str:
    """One JSONL record, terminator included."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def now_iso() -> str:
    """The one timestamp spelling every artefact records."""
    return datetime.now(timezone.utc).isoformat()


class SnapshotMismatch(RuntimeError):
    """Bytes did not hash to the digest they were supposed to. Always fatal."""


class SnapshotStore:
    """One immutable, content-addressed copy of every physical input."""

    DIRNAME = ".snapshots"

    def __init__(self, root: str) -> None:
        self.root = os.path.join(os.path.abspath(root), self.DIRNAME)

    def path_for(self, digest: str, ext: str) -> str:
        return os.path.join(self.root, digest[:2], digest + ext)

    def capture(self, source: str, ext: str) -> tuple[str, str]:
        """Hash `source` and snapshot it in one pass. Returns (digest, path).
        """
        os.makedirs(self.root, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
        try:
            os.fchmod(fd, PRIVATE)
            with os.fdopen(fd, "wb") as out, open(source, "rb") as fh:
                hexdigest = sha256_stream(fh, sink=out)
                out.flush()
                sync_file(out.fileno())
            final = self.path_for(hexdigest, ext)
            if os.path.exists(final):
                os.unlink(tmp)
            else:
                os.makedirs(os.path.dirname(final), exist_ok=True)
                os.replace(tmp, final)
                sync_dir(os.path.dirname(final))
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        return hexdigest, final

    def verify(self, digest: str, ext: str) -> str:
        """Re-hash a snapshot and fail closed if it has moved."""
        path = self.path_for(digest, ext)
        got = sha256_file(path)
        if got != digest:
            raise SnapshotMismatch(
                f"snapshot {path} hashes to {got[:12]}, not the "
                f"{digest[:12]} recorded for it; the input copy this run reads "
                "is not the one its provenance describes")
        return path
