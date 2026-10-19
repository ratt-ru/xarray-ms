from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from typing import Any, Dict, Mapping, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from arcae.lib.arrow_tables import Table
from rarg_python_patterns.multiton import Multiton

from xarray_ms.backend.msv2.partition import PartitionKeyT, TablePartitioner
from xarray_ms.backend.msv2.subtable_grid import effective_interval

SUBTABLE_SORT_COLUMNS = ["ANTENNA_ID", "TIME"]
SUBTABLE_OTHER_COLUMNS = ["INTERVAL", "row"]


class SubtableIndex:
  """Indexes an MSv2 subtable keyed on ``(ANTENNA_ID, TIME)``,
  such as POINTING, SYSCAL or PHASE_CAL.

  Only the indexing columns of the subtable are read. These are
  partitioned on ``partition_columns`` (e.g. ``SPECTRAL_WINDOW_ID``)
  and sorted on ``(ANTENNA_ID, TIME)``, in the same manner as the MAIN table.
  The original subtable row numbers are retained so that
  data columns can subsequently be read for selected rows only.
  """

  _name: str
  _driver_kwargs: Dict[str, Any]
  _table_desc: Dict[str, Any]
  _groups: Dict[PartitionKeyT, Dict[str, npt.NDArray]]

  def __init__(
    self,
    name: str,
    partition_columns: Sequence[str] = (),
    **driver_kwargs: Any,
  ):
    driver_kwargs.setdefault("ninstances", 1)
    self._name = name
    self._driver_kwargs = driver_kwargs
    self._table_desc = {}
    self._groups = {}

    try:
      table = self._open()
    except pa.lib.ArrowInvalid as e:
      # The subtable does not exist
      if "subtable" in str(e) and "is invalid" in str(e):
        return
      raise

    with table:
      self._table_desc = table.tabledesc()

      if table.nrow() == 0:
        return

      columns = set(table.columns())
      partition_columns = list(partition_columns)
      index_columns = partition_columns + SUBTABLE_SORT_COLUMNS + ["INTERVAL"]

      if missing := set(index_columns) - columns:
        raise ValueError(f"{name} is missing indexing columns {sorted(missing)}")

      index = table.to_arrow(columns=index_columns)

    with cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) as pool:
      self._groups = TablePartitioner(
        partition_columns, SUBTABLE_SORT_COLUMNS, SUBTABLE_OTHER_COLUMNS
      ).partition(index, pool)

    for group in self._groups.values():
      group["VALID_INTERVAL"] = effective_interval(
        group["TIME"], group["INTERVAL"], group["ANTENNA_ID"]
      )

  def _open(self) -> Table:
    return Table.from_filename(
      self._name, readonly=True, lockoptions="nolock", **self._driver_kwargs
    )

  @property
  def name(self) -> str:
    return self._name

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

  def has_column(self, column: str) -> bool:
    return column in self._table_desc

  def getcols(
    self, columns: Sequence[str], rows: npt.NDArray[np.integer]
  ) -> Dict[str, npt.NDArray | None]:
    """Read ``columns`` for the given subtable ``rows``.

    Rows are read in ascending order and returned in the order
    of ``rows``. Columns which cannot be read, generally because their
    cells are undefined, are returned as ``None``."""
    order = np.argsort(rows, kind="stable")
    index = (np.ascontiguousarray(rows[order]),)
    result: Dict[str, npt.NDArray | None] = {}

    with self._open() as table:
      for column in columns:
        try:
          data = table.getcol(column, index)
        except (pa.lib.ArrowInvalid, TypeError):
          result[column] = None
        else:
          unordered = np.empty_like(data)
          unordered[order] = data
          result[column] = unordered

    return result


SubtableIndexFactory: TypeAlias = Multiton[SubtableIndex]
"""Multiton producing a :class:`SubtableIndex`"""
