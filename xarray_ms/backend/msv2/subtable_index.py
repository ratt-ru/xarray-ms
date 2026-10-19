from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from typing import Any, Dict, Mapping, Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from arcae.lib.arrow_tables import Table
from rarg_python_patterns.multiton import Multiton

from xarray_ms.backend.msv2.partition import PartitionKeyT, TablePartitioner
from xarray_ms.backend.msv2.subtable_grid import effective_interval

SUBTABLE_SORT_COLUMNS = ["ANTENNA_ID", "TIME"]
SUBTABLE_OTHER_COLUMNS = ["INTERVAL", "row"]
#: Default number of table instances opened for indexed subtables,
#: so that data columns can be read concurrently
DEFAULT_SUBTABLE_NINSTANCES = 4

SubtableTableFactory: TypeAlias = Multiton[Table]
"""Multiton producing an arcae subtable :class:`~arcae.lib.arrow_tables.Table`"""


class SubtableIndex:
  """Indexes an MSv2 subtable keyed on ``(ANTENNA_ID, TIME)``,
  such as POINTING, SYSCAL or PHASE_CAL.

  Some subtables key their rows on another identifier.
  For example, WEATHER may identify weather stations
  by ``NS_WX_STATION_ID`` rather than ``ANTENNA_ID``.
  The first of ``key_columns`` present in the subtable
  is used as the key and is presented as ``ANTENNA_ID``.

  Only the indexing columns of the subtable are read. These are
  partitioned on ``partition_columns`` (e.g. ``SPECTRAL_WINDOW_ID``)
  and sorted on ``(ANTENNA_ID, TIME)``, in the same manner as the MAIN table.
  The original subtable row numbers are retained so that
  data columns can subsequently be read for selected rows only.
  """

  _table_factory: SubtableTableFactory
  _table_desc: Dict[str, Any]
  _key_column: str
  _groups: Dict[PartitionKeyT, Dict[str, npt.NDArray]]

  def __init__(
    self,
    table_factory: SubtableTableFactory,
    partition_columns: Sequence[str] = (),
    key_columns: Sequence[str] = ("ANTENNA_ID",),
  ):
    self._table_factory = table_factory
    self._table_desc = {}
    self._key_column = key_columns[-1]
    self._groups = {}

    try:
      table = table_factory.instance
    except pa.lib.ArrowInvalid as e:
      # The subtable does not exist
      if "subtable" in str(e) and "is invalid" in str(e):
        return
      raise

    self._table_desc = table.tabledesc()

    if table.nrow() == 0:
      return

    columns = set(table.columns())
    self._key_column = next((c for c in key_columns if c in columns), key_columns[-1])
    partition_columns = list(partition_columns)
    index_columns = partition_columns + [self._key_column, "TIME", "INTERVAL"]

    if missing := set(index_columns) - columns:
      raise ValueError(f"{table.name()} is missing indexing columns {sorted(missing)}")

    index = table.to_arrow(columns=index_columns)

    if self._key_column != "ANTENNA_ID":
      index = index.rename_columns(
        ["ANTENNA_ID" if c == self._key_column else c for c in index.column_names]
      )

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as pool:
      self._groups = TablePartitioner(
        partition_columns, SUBTABLE_SORT_COLUMNS, SUBTABLE_OTHER_COLUMNS
      ).partition(index, pool)

    for group in self._groups.values():
      group["VALID_INTERVAL"] = effective_interval(
        group["TIME"], group["INTERVAL"], group["ANTENNA_ID"]
      )

  @property
  def name(self) -> str:
    return self._table_factory.instance.name()

  @property
  def table_desc(self) -> Dict[str, Any]:
    """The subtable descriptor. Empty if the subtable does not exist"""
    return self._table_desc

  @property
  def groups(self) -> Mapping[PartitionKeyT, Dict[str, npt.NDArray]]:
    """A :code:`{partition_key: {column: values}}` mapping
    of the indexing columns, sorted on ``(ANTENNA_ID, TIME)``.
    The original subtable row numbers are in the ``row`` column and
    the interval over which each row is valid in the ``VALID_INTERVAL`` column.
    Empty if the subtable does not exist or contains no rows"""
    return self._groups

  @property
  def key_column(self) -> str:
    """The subtable column presented as ``ANTENNA_ID`` in :attr:`groups`"""
    return self._key_column

  def has_column(self, column: str) -> bool:
    return column in self._table_desc

  @property
  def table_factory(self) -> SubtableTableFactory:
    """Factory producing the subtable, for reading data columns"""
    return self._table_factory

  def column_row_type(
    self, column: str, row: int
  ) -> Tuple[Tuple[int, ...], np.dtype] | None:
    """Returns the shape and dtype of ``column`` in subtable ``row``,
    or None if the column is absent or its cell is undefined"""
    if not self.has_column(column):
      return None

    try:
      data = self._table_factory.instance.getcol(column, (np.array([row]),))
    except (pa.lib.ArrowInvalid, TypeError):
      return None

    return data.shape[1:], data.dtype


SubtableIndexFactory: TypeAlias = Multiton[SubtableIndex]
"""Multiton producing a :class:`SubtableIndex`"""
