"""Common functionality for factories producing datasets from
MSv2 subtables keyed on ``(ANTENNA_ID, TIME)``"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
import numpy.typing as npt
from rarg_python_patterns.multiton import Multiton
from xarray import Variable
from xarray.core.indexing import LazilyIndexedArray

from xarray_ms.backend.msv2.array import SubtableMSv2Array
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory
from xarray_ms.backend.msv2.subtable_grid import (
  GriddedSubtable,
  GriddedSubtableFactory,
  grid_subtable,
)

if TYPE_CHECKING:
  from xarray_ms.backend.msv2.partition import PartitionKeyT
  from xarray_ms.backend.msv2.structure import PartitionData
  from xarray_ms.backend.msv2.subtable_index import (
    SubtableIndex,
    SubtableIndexFactory,
  )


def partition_span(partition: PartitionData) -> Tuple[float, float]:
  """The time span covered by the timesteps of a partition"""
  half_interval = partition.interval.max() / 2 if partition.interval.size else 0.0
  return (partition.time.min() - half_interval, partition.time.max() + half_interval)


def as_float64(data: npt.NDArray) -> npt.NDArray:
  return data.astype(np.float64)


class LazySubtableColumns:
  """Creates lazily read variables from the columns of a gridded subtable"""

  __slots__ = (
    "_index",
    "_grid_factory",
    "_grid",
    "_first_row",
    "_antenna_first",
    "_coder_factory",
  )

  _index: SubtableIndex
  _grid_factory: GriddedSubtableFactory
  _grid: GriddedSubtable
  _first_row: int
  _antenna_first: bool
  _coder_factory: MSv2CoderFactory

  def __init__(
    self,
    index: SubtableIndex,
    grid_factory: GriddedSubtableFactory,
    grid: GriddedSubtable,
    antenna_first: bool,
  ):
    self._index = index
    self._grid_factory = grid_factory
    self._grid = grid
    # Column shapes are established from the first row present in the grid
    self._first_row = int(grid.rows[grid.rows >= 0][0])
    self._antenna_first = antenna_first
    self._coder_factory = MSv2CoderFactory.from_table_desc(index.table_desc)

  @staticmethod
  def from_partition(
    index_factory: SubtableIndexFactory,
    group: PartitionKeyT,
    antenna_ids: npt.NDArray[np.integer],
    partition: PartitionData,
    interpolate: bool,
    antenna_first: bool,
  ) -> LazySubtableColumns | None:
    """Grids the subtable rows relevant to ``partition``.

    Args:
      index_factory: Factory producing the subtable index.
      group: Partition key of the subtable index group.
      antenna_ids: Antenna (or other key) ids defining the antenna axis.
      partition: The correlated data partition.
      interpolate: If True, grid on the partition timesteps.
        Otherwise, grid on the subtable timesteps.
      antenna_first: Present ``(antenna, time)``, rather than ``(time, antenna)``,
        as the first dimensions.

    Returns:
      The gridded columns, or None if no subtable rows are relevant.
    """
    # Partitions sharing antennas and time span share the grid.
    # The grid is not released when the store is closed, but expires.
    # Lazy reads after expiry recreate the grid, and the subtable index
    # if necessary. This is rare, as the arrays are usually read soon after
    # opening, and is preferred to holding grids for the lifetime of the store
    grid_factory = Multiton(
      grid_subtable,
      index_factory,
      group,
      antenna_ids,
      partition_span(partition),
      partition.time if interpolate else None,
    )

    if (grid := grid_factory.instance) is None:
      return None

    return LazySubtableColumns(
      index_factory.instance, grid_factory, grid, antenna_first
    )

  @property
  def grid(self) -> GriddedSubtable:
    return self._grid

  @property
  def coder_factory(self) -> MSv2CoderFactory:
    return self._coder_factory

  @property
  def grid_shape(self) -> Tuple[int, int]:
    """Shape of the grid in presentation order"""
    ntime, nant = self._grid.rows.shape
    return (nant, ntime) if self._antenna_first else (ntime, nant)

  def column_type(self, column: str) -> Tuple[Tuple[int, ...], np.dtype] | None:
    """Row shape and dtype of ``column``,
    or None if absent or its cells are undefined"""
    return self._index.column_row_type(column, self._first_row)

  def time_variable(self, dim: str, time_type: str) -> Variable:
    """Decoded time coordinate of the grid, with the given ``type`` attribute"""
    time = self._coder_factory.create("TIME").decode(Variable(dim, self._grid.time))
    time.attrs["type"] = time_type
    return time

  def array(
    self,
    column: str,
    row_shape: Tuple[int, ...],
    dtype: npt.DTypeLike,
    default: Any,
    **kwargs: Any,
  ) -> LazilyIndexedArray:
    """Lazily read array of ``column``, whose presented
    trailing dimensions have shape ``row_shape``"""
    return LazilyIndexedArray(
      SubtableMSv2Array(
        self._index.table_factory,
        self._grid_factory,
        column,
        self.grid_shape + row_shape,
        dtype,
        default,
        antenna_first=self._antenna_first,
        **kwargs,
      )
    )

  def float_variable(
    self,
    column: str,
    dims: Tuple[str, ...],
    row_shape: Tuple[int, ...],
    read_dtype: npt.DTypeLike,
    **kwargs: Any,
  ) -> Variable:
    """Decoded, lazily read float64 variable of ``column``.
    Missing values are ``nan``"""
    read_dtype = np.dtype(read_dtype)
    if read_dtype != np.float64:
      kwargs.setdefault("transform", as_float64)
    data = self.array(
      column, row_shape, np.float64, np.nan, read_dtype=read_dtype, **kwargs
    )
    return self._coder_factory.create(column).decode(Variable(dims, data))
