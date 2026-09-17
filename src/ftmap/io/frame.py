"""The Frame: what every later stage reads instead of the file."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Column:
    id: str
    index: int
    header: str | None
    label: str | None
    group: str | None = None


@dataclass(frozen=True)
class FurnitureRow:
    index: int
    kind: str
    reason: str


@dataclass(frozen=True)
class Frame:
    source_id: str
    path: str
    sheet: str
    sha256: str
    columns: list[Column]
    rows: list[list[str | None]]
    row_index: list[int]
    header_row: int | None
    label_row: int | None
    group_row: int | None = None
    furniture: list[FurnitureRow] = field(default_factory=list)
    filled_cells: int = 0
    hidden: bool = False
    repairs: list[str] = field(default_factory=list)

    def column(self, column_id: str) -> Column:
        """The Column with this id, in constant time."""
        by_id = self.__dict__.get("_by_id")
        if by_id is None:
            by_id: dict[str, Column] = {}
            for column in self.columns:
                by_id.setdefault(column.id, column)
            object.__setattr__(self, "_by_id", by_id)
        return by_id[column_id]


def column_values(frame: Frame, column_id: str) -> list[str | None]:
    idx = frame.column(column_id).index
    return [row[idx] if idx < len(row) else None for row in frame.rows]
