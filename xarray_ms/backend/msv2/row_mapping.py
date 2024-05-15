from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any, ClassVar, Tuple

import numpy as np
from arcae.lib.arrow_tables import Table
from cacheout import LRUCache

from xarray_ms.query import select_clause, where_clause
from xarray_ms.utils import baseline_id

if TYPE_CHECKING:
  import numpy.typing as npt

  from xarray_ms.backend.msv2.table_factory import TableFactory

PartitionT = Tuple[Tuple[str, Number], ...]


class RowMapFactory:
  """Hashable Factory for creating a MSv2 Partition Row Map"""

  _MAP_CACHE: ClassVar[LRUCache] = LRUCache(maxsize=100, ttl=1 * 60)
  _MAP_COLUMNS: ClassVar[list[str]] = ["TIME", "ANTENNA1", "ANTENNA2"]
  _table_factory: TableFactory
  _partition: PartitionT
  _na: int
  _auto_corrs: bool

  def __init__(
    self,
    table_factory: TableFactory,
    partition: PartitionT,
    na: int,
    auto_corrs: bool = True,
  ):
    self._table_factory = table_factory
    self._partition = partition
    self._na = na
    self._auto_corrs = auto_corrs

  def __reduce__(self) -> Tuple[type[RowMapFactory], Tuple[Any, ...]]:
    return (
      RowMapFactory,
      (self._table_factory, self._partition, self._na, self._auto_corrs),
    )

  def __hash__(self) -> int:
    return hash((self._table_factory, self._partition, self._na, self._auto_corrs))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, RowMapFactory):
      return NotImplemented
    return (
      self._table_factory == other._table_factory
      and self._partition == other._partition
      and self._na == other._na
      and self._auto_corrs == other._auto_corrs
    )

  def create_map(self):
    select_cols = RowMapFactory._MAP_COLUMNS.copy()
    select_cols.append("ROWID() as ROWS")
    select = select_clause(select_cols)

    names, values = zip(*self._partition)
    where = where_clause(names, values)
    query = f"{select} FROM $1 {where}"

    with Table.from_taql(query, self._table_factory()) as Q:
      result = Q.to_arrow()

    time = result["TIME"].to_numpy()
    ant1 = result["ANTENNA1"].to_numpy()
    ant2 = result["ANTENNA2"].to_numpy()
    rows = result["ROWS"].to_numpy()

    if not np.all(ant1 < self._na):
      raise ValueError(f"Not all ANTENNA1 values are less than {self._na}")

    if not np.all(ant2 < self._na):
      raise ValueError(f"Not all ANTENNA2 values are less than {self._na}")

    na = self._na
    utime, time_id = np.unique(time, return_inverse=True)
    bl_id = baseline_id(ant1, ant2, na, self._auto_corrs)
    nbl = na * (na + (1 if self._auto_corrs else -1)) // 2
    row_map = np.full(utime.size * nbl, -1, dtype=np.int64)
    row_map[time_id * nbl + bl_id] = rows
    return row_map.reshape((utime.size, nbl))

  def __call__(self, *args, **kw) -> npt.NDArray[np.integer]:
    assert not args and not kw
    return self._MAP_CACHE.get(self, lambda self: self.create_map())
