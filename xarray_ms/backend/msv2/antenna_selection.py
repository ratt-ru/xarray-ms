from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from xarray_ms.backend.msv2.structure import PartitionData
from xarray_ms.errors import DuplicateAntennaNameWarning, HeterogeneousFeedWarning

# FEED columns describing per-feed quantities that antenna_xds
# represents once per antenna
PER_FEED_COLUMNS = ["RECEPTOR_ANGLE", "POLARIZATION_TYPE", "FOCUS_LENGTH"]


def unique_antenna_names(names: np.ndarray) -> np.ndarray:
  """Return antenna names with duplicates made unique by appending -N suffixes.

  All occurrences of a duplicated name are renamed: the first gets ``-1``,
  the second ``-2``, and so on. Names that are already unique are left
  unchanged. A :class:`~xarray_ms.errors.DuplicateAntennaNameWarning` is
  emitted when any renaming is performed.
  """
  unique, counts = np.unique(names, return_counts=True)
  if not (duplicates := set(unique[counts > 1].tolist())):
    return names

  warnings.warn(
    f"Duplicate antenna names detected in the ANTENNA table: "
    f"{sorted(duplicates)}. "
    f"A numeric suffix will be appended to make names unique.",
    DuplicateAntennaNameWarning,
    stacklevel=2,
  )
  # Build as a Python list to avoid numpy fixed-width string truncation,
  # then convert back to an array with a wide enough dtype.
  result = names.tolist()
  counters: Dict[str, int] = {}
  for i, name in enumerate(result):
    if name in duplicates:
      counters[name] = counters.get(name, 0) + 1
      result[i] = f"{name}-{counters[name]}"
  return np.array(result, dtype=str)


@dataclass(frozen=True, slots=True)
class PartitionAntennaSelection:
  """Antenna and feed rows selected for a partition."""

  feed_row_indices: npt.NDArray[np.int32]
  """The first FEED row selected for each antenna."""

  antenna_ids: npt.NDArray[np.int32]
  """Unique antenna ids, in order of first occurrence in the FEED table."""

  antenna_names: npt.NDArray[np.str_]
  """Canonical antenna names, aligned with :attr:`antenna_ids`."""


def differing_feed_columns(
  feed_table: pa.Table,
  feed_row_indices: npt.NDArray[np.integer],
  first_row_indices: npt.NDArray[np.integer],
) -> List[str]:
  """Return the per-feed FEED columns whose values differ between
  each selected row and the first selected row of the same antenna."""
  import pyarrow.compute as pac

  differing = []

  for column in PER_FEED_COLUMNS:
    if column not in feed_table.column_names:
      continue

    values = feed_table[column]
    rows = values.take(feed_row_indices)
    first_rows = values.take(first_row_indices)

    if pa.types.is_list(values.type) or pa.types.is_fixed_size_list(values.type):
      rows = pac.list_flatten(rows)
      first_rows = pac.list_flatten(first_rows)

    rows = rows.to_numpy(zero_copy_only=False)
    first_rows = first_rows.to_numpy(zero_copy_only=False)
    equal_nan = np.issubdtype(rows.dtype, np.inexact)

    if not np.array_equal(rows, first_rows, equal_nan=equal_nan):
      differing.append(column)

  return differing


def select_partition_antennas(
  antenna_table: pa.Table,
  feed_table: pa.Table,
  partition: PartitionData,
) -> PartitionAntennaSelection:
  """Select antenna and feed rows for a partition."""
  feed_ids = feed_table["FEED_ID"].to_numpy()
  spw_ids = feed_table["SPECTRAL_WINDOW_ID"].to_numpy()
  antenna_ids = feed_table["ANTENNA_ID"].to_numpy()

  # Select feeds with global spws (-1) or that match the partition spw.
  mask = np.logical_or.reduce(
    (
      spw_ids == -1,
      spw_ids == partition.spw_id,
    )
  )

  np.logical_and.reduce(
    (
      mask,
      np.isin(feed_ids, partition.feed_ids),
      np.isin(antenna_ids, partition.antenna_ids),
    ),
    out=mask,
  )

  # Prefer rows matching the partition spw over global spw (-1) rows
  # for antennas that have both.
  spw_antennas = antenna_ids[mask & (spw_ids == partition.spw_id)]
  mask &= ~((spw_ids == -1) & np.isin(antenna_ids, spw_antennas))

  # FEED may contain multiple rows per antenna (e.g. one per feed and time).
  # Select the first row of each antenna, in FEED-table order.
  feed_row_indices = np.where(mask)[0]
  _, first, inverse = np.unique(
    antenna_ids[feed_row_indices], return_index=True, return_inverse=True
  )
  first_row_indices = feed_row_indices[first[inverse]]

  if differing := differing_feed_columns(
    feed_table, feed_row_indices, first_row_indices
  ):
    warnings.warn(
      f"FEED columns {differing} differ between the FEED rows of an antenna "
      f"for SPECTRAL_WINDOW_ID={partition.spw_id} and feeds={partition.feed_ids}. "
      f"antenna_xds uses the first FEED row of each antenna.",
      HeterogeneousFeedWarning,
      stacklevel=2,
    )

  feed_row_indices = feed_row_indices[np.sort(first)].astype(np.int32, copy=False)
  selected_antenna_ids = antenna_ids[feed_row_indices].astype(np.int32, copy=False)
  # Deduplicate against the full ANTENNA table so suffix assignments are
  # consistent with those produced in the correlated dataset factory.
  antenna_names = antenna_table["NAME"].to_numpy().astype(str)
  antenna_names = unique_antenna_names(antenna_names)[selected_antenna_ids]

  return PartitionAntennaSelection(
    feed_row_indices=feed_row_indices,
    antenna_ids=selected_antenna_ids,
    antenna_names=antenna_names,
  )
