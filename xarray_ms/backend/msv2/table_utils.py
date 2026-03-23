import json
import warnings
from typing import Any, Dict

import numpy as np
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
