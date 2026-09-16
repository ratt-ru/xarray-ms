import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from xarray_ms.errors import DuplicateAntennaNameWarning


def unique_antenna_names(names: np.ndarray) -> np.ndarray:
  """Return antenna names with duplicates made unique by appending -N suffixes.

  All occurrences of a duplicated name are renamed: the first gets ``-1``,
  the second ``-2``, and so on.  Names that are already unique are left
  unchanged.  A :class:`~xarray_ms.errors.DuplicateAntennaNameWarning` is
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


def extract_table_desc(table: pa.Table) -> Dict[str, Any]:
  """Extract the CASA table descriptor stored in an Arrow table constructed by arcae"""
  try:
    arcae_metadata = table.schema.metadata[b"__arcae_metadata__"]
  except KeyError:
    raise KeyError("__arcae_metadata__ was not present in the table metadata")

  try:
    return json.loads(arcae_metadata)["__casa_descriptor__"]
  except KeyError:
    raise KeyError(
      f"arcae metadata {arcae_metadata} does not contain a __casa_descriptor__"
    )


@dataclass
class AntennaSelection:
  """Antenna and feed ids for a given partition.

  Used both in the creation of antenna_xds and phased_array_xds.
  """

  feed_rows: npt.NDArray[np.int32]
  """Row indices that were selected in the FEED table."""

  antenna_ids: npt.NDArray[np.int32]
  """Antenna ids for the given partition, in order of appearance in the FEED table."""

  unique_antenna_names: npt.NDArray[np.str_]
  """Antenna names (made unique if necessary) for the given partition.

  Given in order of appearance in the FEED table.
  """


def select_partition_antennas(
  antenna_table: pa.Table,
  feed_table: pa.Table,
  partition_spw_id: int,
  partition_feed_ids: npt.NDArray[np.int32],
  partition_antenna_ids: npt.NDArray[np.int32],
) -> AntennaSelection:
  """Select the antenna and feed ids for a given partition.

  Parameters
  ----------
  antenna_table : pa.Table
      The ANTENNA subtable.
  feed_table : pa.Table
      The FEED subtable.
  partition_spw_id : int
      The spectral window id for the partition.
  partition_feed_ids : np.ndarray
      The feed ids for the partition.
  partition_antenna_ids : np.ndarray
      The antenna ids for the partition.

  Returns
  -------
  AntennaSelection
      A dataclass containing the selected feed rows, antenna ids, and
      unique (deduplicated) antenna names for the partition.
  """
  feed_ids = feed_table["FEED_ID"].to_numpy()
  spw_ids = feed_table["SPECTRAL_WINDOW_ID"].to_numpy()
  antenna_ids = feed_table["ANTENNA_ID"].to_numpy()

  # Select feeds with global spws (-1) or that match the partition spw
  mask = np.logical_or.reduce(
    (
      spw_ids == -1,
      spw_ids == partition_spw_id,
    )
  )

  np.logical_and.reduce(
    (
      mask,
      np.isin(feed_ids, partition_feed_ids),
      np.isin(antenna_ids, partition_antenna_ids),
    ),
    out=mask,
  )

  selected_feed_rows = np.where(mask)[0].astype(np.int32)
  selected_antenna_ids = antenna_ids[selected_feed_rows]

  # Deduplicate against the full ANTENNA table so suffix assignments are
  # consistent with those produced in the correlated dataset factory.
  antenna_names = antenna_table["NAME"].to_numpy().astype(str)
  selected_antenna_names = unique_antenna_names(antenna_names)[selected_antenna_ids]

  return AntennaSelection(
    feed_rows=selected_feed_rows,
    antenna_ids=selected_antenna_ids,
    unique_antenna_names=selected_antenna_names,
  )
