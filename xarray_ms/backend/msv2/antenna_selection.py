from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from xarray_ms.backend.msv2.structure import PartitionData
from xarray_ms.errors import DuplicateAntennaNameWarning


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
  """Row indices selected from the FEED table."""

  antenna_ids: npt.NDArray[np.int32]
  """Antenna ids in FEED-table order."""

  antenna_names: npt.NDArray[np.str_]
  """Canonical antenna names in FEED-table order."""


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

  feed_row_indices = np.where(mask)[0].astype(np.int32)
  selected_antenna_ids = antenna_ids[feed_row_indices].astype(np.int32, copy=False)
  antenna_names = unique_antenna_names(
    antenna_table["NAME"].to_numpy().astype(str)
  )[selected_antenna_ids]

  return PartitionAntennaSelection(
    feed_row_indices=feed_row_indices,
    antenna_ids=selected_antenna_ids,
    antenna_names=antenna_names,
  )
