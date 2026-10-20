"""Gridding of ``(ANTENNA_ID, TIME)`` keyed MSv2 subtables
onto ``(time, antenna)`` grids.

Both grids produced here are analogous to
:attr:`~xarray_ms.backend.msv2.structure.PartitionData.row_map`:
they contain indices into a set of selected subtable rows,
with ``-1`` indicating that no row exists for a grid cell.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from rarg_python_patterns.multiton import Multiton

from xarray_ms.errors import DuplicateSubtableRowWarning, IrregularSubtableGridWarning

if TYPE_CHECKING:
  from xarray_ms.backend.msv2.partition import PartitionKeyT
  from xarray_ms.backend.msv2.subtable_index import SubtableIndexFactory

# Intervals smaller than this (seconds) are considered unusable
# for establishing the validity of a subtable row
MINIMUM_INTERVAL = 1e-6


@dataclass(frozen=True, slots=True)
class SubtableGrid:
  """A ``(time, antenna)`` grid of indices into selected subtable rows"""

  time: npt.NDArray[np.float64]
  """Time coordinate of the grid"""

  index: npt.NDArray[np.int64]
  """``(time, antenna)`` grid of indices into the selected rows.
  ``-1`` indicates a missing row"""


def effective_interval(
  time: npt.NDArray[np.float64],
  interval: npt.NDArray[np.float64],
  antenna_id: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
  """Returns an interval over which each subtable row is valid.

  Some subtables record zero or near-zero intervals.
  The median sampling cadence of the subtable is substituted
  for such intervals, or an infinite interval if the cadence
  cannot be determined.

  Args:
    time: Row times, sorted per antenna.
    interval: Row intervals.
    antenna_id: Row antenna ids, sorted.

  Returns:
    Row intervals.
  """
  if np.all(valid := interval > MINIMUM_INTERVAL):
    return interval

  same_antenna = antenna_id[1:] == antenna_id[:-1]
  diffs = np.diff(time)[same_antenna]
  diffs = diffs[diffs > 0]
  cadence = np.median(diffs) if diffs.size > 0 else np.inf
  return np.where(valid, interval, cadence)


def select_rows(
  time: npt.NDArray[np.float64],
  valid_interval: npt.NDArray[np.float64],
  antenna_id: npt.NDArray[np.integer],
  antenna_ids: npt.NDArray[np.integer],
  span: Tuple[float, float],
) -> npt.NDArray[np.intp]:
  """Select subtable rows associated with ``antenna_ids``, whose
  validity interval ``[time - valid_interval / 2, time + valid_interval / 2]``
  overlaps ``span``.

  Args:
    time: Row times, sorted per antenna.
    valid_interval: Row validity intervals, see :func:`effective_interval`.
    antenna_id: Row antenna ids, sorted.
    antenna_ids: Antenna ids to select.
    span: ``(start, end)`` time span to select.

  Returns:
    Sorted indices of the selected rows.
  """
  start, end = span
  half = valid_interval / 2
  max_half = half.max(initial=0.0)
  ant_starts = np.searchsorted(antenna_id, antenna_ids, side="left")
  ant_ends = np.searchsorted(antenna_id, antenna_ids, side="right")
  selected = []

  # Bound the candidate rows of each antenna by binary search,
  # then test the validity interval of each candidate exactly
  for ant_start, ant_end in zip(ant_starts, ant_ends):
    ant_time = time[ant_start:ant_end]
    lo = np.searchsorted(ant_time, start - max_half, side="left")
    hi = np.searchsorted(ant_time, end + max_half, side="right")
    candidates = np.arange(ant_start + lo, ant_start + hi)
    keep = np.logical_and(
      time[candidates] + half[candidates] >= start,
      time[candidates] - half[candidates] <= end,
    )
    selected.append(candidates[keep])

  return np.sort(np.concatenate(selected)) if selected else np.empty(0, np.intp)


def antenna_index(
  antenna_id: npt.NDArray[np.integer], antenna_ids: npt.NDArray[np.integer]
) -> npt.NDArray[np.intp]:
  """Maps each of ``antenna_id`` onto its position in ``antenna_ids``"""
  sorter = np.argsort(antenna_ids)
  return sorter[np.searchsorted(antenna_ids, antenna_id, sorter=sorter)]


def has_duplicate_keys(
  time: npt.NDArray[np.float64], antenna_id: npt.NDArray[np.integer]
) -> bool:
  """Returns True if any ``(TIME, ANTENNA_ID)`` tuple occurs more than once.

  Args:
    time: Row times, sorted per antenna.
    antenna_id: Row antenna ids, sorted.
  """
  # Sorting places duplicates next to each other
  return bool(
    np.any(np.logical_and(antenna_id[1:] == antenna_id[:-1], time[1:] == time[:-1]))
  )


def native_grid(
  time: npt.NDArray[np.float64],
  antenna_id: npt.NDArray[np.integer],
  antenna_ids: npt.NDArray[np.integer],
  name: str,
) -> SubtableGrid:
  """Grids selected subtable rows on the unique subtable times.
  Each ``(TIME, ANTENNA_ID)`` tuple must be unique, see :func:`has_duplicate_keys`.

  Args:
    time: Selected row times.
    antenna_id: Selected row antenna ids.
    antenna_ids: Antenna ids defining the antenna axis of the grid.
    name: Subtable name, used in error and warning messages.

  Returns:
    A grid over the unique subtable times.
  """
  utime, time_index = np.unique(time, return_inverse=True)
  nant = len(antenna_ids)
  flat_index = time_index * nant + antenna_index(antenna_id, antenna_ids)
  index = np.full(utime.size * nant, -1, dtype=np.int64)
  index[flat_index] = np.arange(flat_index.size)

  if (nmissing := index.size - flat_index.size) > 0:
    warnings.warn(
      f"{nmissing} / {index.size} cells of the (time, antenna) grid "
      f"of {name} are missing and will be filled with default values. "
      f"This occurs when antennas record values at differing times.",
      IrregularSubtableGridWarning,
    )

  return SubtableGrid(utime, index.reshape(utime.size, nant))


def _to_nanoseconds(
  seconds: npt.NDArray[np.float64], origin: float
) -> npt.NDArray[np.int64]:
  """Converts times in seconds to integer nanoseconds relative to ``origin``,
  avoiding int64 overflow for large epochs"""
  return np.round((seconds - origin) * 1e9).astype(np.int64)


def asof_grid(
  time: npt.NDArray[np.float64],
  valid_interval: npt.NDArray[np.float64],
  antenna_id: npt.NDArray[np.integer],
  antenna_ids: npt.NDArray[np.integer],
  grid_time: npt.NDArray[np.float64],
) -> SubtableGrid:
  """Grids selected subtable rows on ``grid_time``.

  Each grid cell is associated with the subtable row of the same antenna
  that most recently became valid at, or before, the grid time,
  provided that the row's validity interval covers the grid time.

  Args:
    time: Selected row times, sorted per antenna.
    valid_interval: Selected row validity intervals,
      see :func:`effective_interval`.
    antenna_id: Selected row antenna ids, sorted.
    antenna_ids: Antenna ids defining the antenna axis of the grid.
    grid_time: Sorted times defining the time axis of the grid.

  Returns:
    A grid over ``grid_time``.
  """
  start = time - valid_interval / 2
  # Rows with infinite intervals are valid from the earliest time
  earliest = min(start[np.isfinite(start)].min(initial=time.min()), grid_time.min())
  start = np.where(np.isfinite(start), start, earliest)
  end = time + valid_interval / 2

  ntime, nant = grid_time.size, antenna_ids.size
  left = pa.table(
    {
      "time": pa.array(_to_nanoseconds(np.repeat(grid_time, nant), earliest)),
      "antenna": pa.array(np.tile(antenna_ids, ntime).astype(np.int32)),
      "cell": pa.array(np.arange(ntime * nant, dtype=np.int64)),
    }
  )

  order = np.argsort(start, kind="stable")
  right = pa.table(
    {
      "time": pa.array(_to_nanoseconds(start[order], earliest)),
      "antenna": pa.array(antenna_id[order].astype(np.int32)),
      "row": pa.array(order.astype(np.int64)),
    }
  )

  # Match the most recently valid row, however long ago that was
  tolerance = -int(_to_nanoseconds(grid_time.max(), earliest) + 1)
  joined = left.join_asof(right, on="time", by="antenna", tolerance=tolerance)

  cell = joined["cell"].to_numpy()
  row = joined["row"].fill_null(-1).to_numpy()
  index = np.full(ntime * nant, -1, dtype=np.int64)
  index[cell] = row

  # Discard matches whose validity ends before the grid time
  matched = index >= 0
  flat_time = np.repeat(grid_time, nant)
  stale = np.zeros_like(matched)
  stale[matched] = end[index[matched]] < flat_time[matched]
  index[stale] = -1

  return SubtableGrid(grid_time, index.reshape(ntime, nant))


@dataclass(frozen=True, slots=True)
class GriddedSubtable:
  """A ``(time, antenna)`` grid of subtable row numbers"""

  time: npt.NDArray[np.float64]
  """Time coordinate of the grid"""

  rows: npt.NDArray[np.int64]
  """Read-only ``(time, antenna)`` grid of subtable row numbers.
  ``-1`` indicates a missing row"""


GriddedSubtableFactory: TypeAlias = Multiton[GriddedSubtable | None]
"""Multiton producing a :class:`GriddedSubtable`"""


def grid_subtable(
  index_factory: SubtableIndexFactory,
  group: PartitionKeyT,
  antenna_ids: npt.NDArray[np.integer],
  span: Tuple[float, float],
  grid_time: npt.NDArray[np.float64] | None = None,
) -> GriddedSubtable | None:
  """Select subtable rows and grid them onto ``(time, antenna)``.

  This is intended to be wrapped in a
  :class:`~rarg_python_patterns.multiton.Multiton`, so that partitions
  sharing the same antennas, time span and grid times
  (e.g. partitions differing only in spectral window)
  share the grid, rather than recomputing it.

  Args:
    index_factory: Factory producing the subtable index.
    group: Partition key of the subtable index group.
    antenna_ids: Antenna ids defining the antenna axis of the grid.
    span: ``(start, end)`` time span of rows to select.
    grid_time: If supplied, rows are gridded on these times via :func:`asof_grid`.
      Otherwise, rows are gridded on their own times via :func:`native_grid`.

  Returns:
    The grid, or None if no subtable rows were selected, or if the
    selected rows do not uniquely identify each ``(time, antenna)`` cell.
  """
  index = index_factory.instance

  try:
    group_index = index.groups[group]
  except KeyError:
    return None

  selected = select_rows(
    group_index["TIME"],
    group_index["VALID_INTERVAL"],
    group_index["ANTENNA_ID"],
    antenna_ids,
    span,
  )

  if selected.size == 0:
    return None

  time = group_index["TIME"][selected]
  antenna_id = group_index["ANTENNA_ID"][selected]

  # The subtable is optional: rather than failing to open the
  # Measurement Set, omit it if it cannot be gridded unambiguously
  if has_duplicate_keys(time, antenna_id):
    warnings.warn(
      f"Multiple rows of {index.name} share a (TIME, ANTENNA_ID) tuple, "
      f"so it is not possible to establish a unique (time, antenna) grid. "
      f"{index.name} will be omitted for antennas {antenna_ids.tolist()} "
      f"over the time span {span}.",
      DuplicateSubtableRowWarning,
    )
    return None

  if grid_time is not None:
    valid_interval = group_index["VALID_INTERVAL"][selected]
    grid = asof_grid(time, valid_interval, antenna_id, antenna_ids, grid_time)
  else:
    grid = native_grid(time, antenna_id, antenna_ids, index.name)

  if not np.any(valid := grid.index >= 0):
    return None

  # Convert indices into selected rows into subtable row numbers
  rows = np.full_like(grid.index, -1)
  rows[valid] = group_index["row"][selected][grid.index[valid]]
  # The grid may be shared between partitions
  rows.flags.writeable = False
  return GriddedSubtable(grid.time, rows)
