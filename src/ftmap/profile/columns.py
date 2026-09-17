"""Per-column description."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from ftmap.config import Config
from ftmap.io.frame import Frame
from ftmap.normalize.canonical import is_sentinel
from ftmap.profile.detectors import detect
from ftmap.profile.shapes import shape, top_shapes


@dataclass(frozen=True)
class ColumnProfile:
    id: str
    index: int
    header: str | None
    label: str | None
    count: int
    filled: int
    fill_rate: float
    distinct: int
    distinct_ratio: float
    min_len: int
    max_len: int
    shapes: list[tuple[str, float]]
    detectors: dict[str, float]
    samples: list[str]
    sentinel_share: float = 0.0

    modal_share: float = 0.0
    distinct_ex_modal: float = 0.0

    group: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shapes"] = [[s, r] for s, r in self.shapes]
        return d


def _modal_share(counts: Counter[str], total: int) -> float:
    """How much of the column its single commonest value occupies."""
    if not total:
        return 0.0
    return counts.most_common(1)[0][1] / total


def _distinct_ex_modal(counts: Counter[str], total: int) -> float:
    """`distinct_ratio` computed over everything except the commonest value."""
    if not total:
        return 0.0
    rest = total - counts.most_common(1)[0][1]
    return (len(counts) - 1) / rest if rest else 0.0


def _samples(counts: Counter[str], total: int, n: int) -> list[str]:
    """The rows ordered by value length and cut into `n` equal parts; from each
    part, the value that fills most of it.
    """
    shapes = {value: shape(value) for value in counts}
    by_shape: Counter[str] = Counter()
    for value, held in counts.items():
        by_shape[shapes[value]] += held

    parts: list[Counter[str]] = [Counter() for _ in range(n)]
    edges = [(i * total // n, (i + 1) * total // n) for i in range(n)]
    at = 0
    for value, held in sorted(counts.items(), key=lambda kv: (len(kv[0]), kv[0])):
        for part, (lo, hi) in zip(parts, edges):
            inside = min(at + held, hi) - max(at, lo)
            if inside > 0:
                part[value] = inside
        at += held

    chosen: dict[str, None] = {}
    for part in parts:
        pool = [v for v in part if v not in chosen]
        if pool:
            chosen[min(pool, key=lambda v: (-part[v], -by_shape[shapes[v]], v))] = None

    if len(chosen) < n:
        for value in sorted(counts, key=lambda v: (-counts[v], -by_shape[shapes[v]], v)):
            if value not in chosen:
                chosen[value] = None
            if len(chosen) >= n:
                break
    return list(chosen)


def profile_frame(frame: Frame, cfg: Config) -> list[ColumnProfile]:
    out: list[ColumnProfile] = []
    rows = frame.rows
    for col in frame.columns:
        idx = col.index
        raw = [row[idx] if idx < len(row) else None for row in rows]
        vals: list[str] = []
        for value in raw:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                vals.append(text)
        counts = Counter(vals)
        count = len(raw)
        filled = len(vals)
        distinct = len(counts)
        out.append(
            ColumnProfile(
                id=col.id,
                index=col.index,
                header=col.header,
                label=col.label,
                count=count,
                filled=filled,
                fill_rate=(filled / count) if count else 0.0,
                distinct=distinct,
                distinct_ratio=(distinct / filled) if filled else 0.0,
                min_len=min((len(v) for v in counts), default=0),
                max_len=max((len(v) for v in counts), default=0),
                shapes=top_shapes(raw, 5),
                detectors=detect(raw),
                samples=_samples(counts, filled, cfg.sample_size),
                sentinel_share=(sum(held for value, held in counts.items()
                                    if is_sentinel(value)) / filled)
                if filled else 0.0,
                modal_share=_modal_share(counts, filled),
                distinct_ex_modal=_distinct_ex_modal(counts, filled),
                group=col.group,
            )
        )
    return out
