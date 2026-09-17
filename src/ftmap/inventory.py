"""What is there to process."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import cached_property

from ftmap.fsutil import SnapshotStore, sha256_file
from ftmap.io.tabular import CSV_EXT, XL_EXT, read_source, sheet_names

SUPPORTED = CSV_EXT | XL_EXT

_UNSAFE_PATH = re.compile(r"[/\\:#?$\x00-\x1f\x7f]")
_MAX_PREFIX_BYTES = 120


class SourcePathCollision(RuntimeError):
    """Two sources resolved to one directory. Never expected; always fatal."""


def _path_suffix(source_id: str) -> str:
    """The part of a source's directory name that carries its identity."""
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8]


def _path_prefix(source_id: str) -> str:
    """`source_id`, made safe to be a directory name and still readable."""
    text = _UNSAFE_PATH.sub("_", unicodedata.normalize("NFC", source_id))
    text = "_".join(text.split())
    encoded = text.encode("utf-8")[:_MAX_PREFIX_BYTES]
    return encoded.decode("utf-8", "ignore").strip("_")


@dataclass(frozen=True)
class SourceRef:
    path: str
    sheet_index: int
    sheet: str
    sha256: str
    size: int
    ext: str
    source_id: str
    duplicate_of: str | None = None
    snapshot: str | None = None

    @property
    def read_path(self) -> str:
        """Where the bytes come from — the snapshot when there is one."""
        return self.snapshot or self.path

    def to_dict(self) -> dict:
        return asdict(self)

    @cached_property
    def safe_id(self) -> str:
        """The source's directory name: a readable prefix, then its identity.
        """
        prefix = _path_prefix(self.source_id)
        return f"{prefix}-{_path_suffix(self.source_id)}" if prefix else \
            f"{self.sha256[:12]}-{_path_suffix(self.source_id)}"


def _sheets_of(path: str) -> list[tuple[int, str]]:
    """Every sheet's index and name, without parsing a single value."""
    for read in (lambda: list(enumerate(sheet_names(path))),
                 lambda: [(i, g.sheet) for i, g in enumerate(read_source(path))]):
        try:
            found = read()
        except Exception:
            continue
        if found:
            return found
    return [(0, "")]


def inventory(root: str, snapshots: SnapshotStore | None = None) -> list[SourceRef]:
    """Every supported source under `root`."""
    refs: list[SourceRef] = []
    paths: list[str] = []
    if os.path.isfile(root):
        if os.path.splitext(root)[1].lower() in SUPPORTED:
            paths = [root]
    elif not os.path.isdir(root):
        raise FileNotFoundError(f"corpus root does not exist: {root}")
    else:
        def _raise(error: OSError) -> None:
            raise error

        for dirpath, _dirnames, filenames in os.walk(root, onerror=_raise):
            for name in sorted(filenames):
                if name.startswith("~$") or name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() in SUPPORTED:
                    paths.append(os.path.join(dirpath, name))
    canonical: dict[str, str] = {}
    for path in sorted(paths):
        ext = os.path.splitext(path)[1].lower()
        size = os.path.getsize(path)
        if snapshots is not None:
            digest, snapshot = snapshots.capture(path, ext)
        else:
            digest, snapshot = sha256_file(path), None
        read_from = snapshot or path
        if ext in CSV_EXT:
            sheets = [(0, "")]
        else:
            sheets = _sheets_of(read_from)
        for idx, sheet in sheets:
            source_id = f"{digest[:12]}/{sheet}"
            kept_path = canonical.setdefault(source_id, path)
            refs.append(SourceRef(
                path=path, sheet_index=idx, sheet=sheet, sha256=digest,
                size=size, ext=ext, source_id=source_id,
                duplicate_of=None if kept_path == path else kept_path,
                snapshot=snapshot,
            ))
    seen: dict[str, SourceRef] = {}
    for ref in refs:
        if ref.duplicate_of is not None:
            continue
        clash = seen.setdefault(ref.safe_id, ref)
        if clash.source_id != ref.source_id:
            raise SourcePathCollision(
                f"two sources resolve to the directory {ref.safe_id!r}: "
                f"{clash.source_id!r} ({clash.path}, sheet {clash.sheet!r}) "
                f"and {ref.source_id!r} ({ref.path}, sheet {ref.sheet!r})")
    return refs
