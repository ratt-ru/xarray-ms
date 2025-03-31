from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from xarray_ms.errors import ImputedMetadataWarning

if TYPE_CHECKING:
  import pyarrow as pa


def _maybe_return_table_or_max_id(
  table: pa.Table, table_name: str, ids: npt.NDArray[np.int32], id_column_name: str
) -> pa.Table | int:
  """Returns the existing table if a row entry exists,
  else returns the maximum id"""
  max_id = np.max(ids)

  if max_id < len(table):
    return table

  warnings.warn(
    f"No row exists in the {table_name} table of length {len(table)} "
    f"for {id_column_name}={max_id}. "
    f"Artificial metadata will be substituted.",
    ImputedMetadataWarning,
  )

  return max_id


def maybe_impute_field_table(
  field: pa.Table, field_id: npt.NDArray[np.int32]
) -> pa.Table:
  """Generates a FIELD subtable if there are no row ids
  associated with the given FIELD_ID values"""

  import pyarrow as pa

  result = _maybe_return_table_or_max_id(field, "FIELD", field_id, "FIELD_ID")
  if isinstance(result, pa.Table):
    return result

  return pa.Table.from_pydict(
    {
      "NAME": np.array([f"UNKNOWN-{i}" for i in range(result + 1)], dtype=object),
      "SOURCE_ID": np.zeros(result + 1, np.int32),
    }
  )


def maybe_impute_state_table(
  state: pa.Table, state_id: npt.NDArray[np.int32]
) -> pa.Table:
  """Generates a STATE subtable if there are no row ids
  associated with the given STATE_ID values"""
  import pyarrow as pa

  result = _maybe_return_table_or_max_id(state, "STATE", state_id, "STATE_ID")
  if isinstance(result, pa.Table):
    return result

  return pa.Table.from_pydict(
    {
      "OBS_MODE": np.array(["UNSPECIFIED"] * (result + 1), dtype=object),
      "SUB_SCAN": np.zeros(result + 1, np.int32),
    }
  )


def maybe_impute_observation_table(
  observation: pa.Table, observation_id: npt.NDArray[np.int32]
) -> pa.Table:
  """Generates an OBSERVATION table if there are no row ids
  associated with the given OBSERVATION_ID values"""
  import pyarrow as pa

  result = _maybe_return_table_or_max_id(
    observation, "OBSERVATION", observation_id, "OBSERVATION_ID"
  )
  if isinstance(result, pa.Table):
    return result

  unknown = np.array(["unknown"] * (result + 1), dtype=object)

  return pa.Table.from_pydict(
    {
      "OBSERVER": unknown,
      "PROJECT": unknown,
      "TELESCOPE_NAME": unknown,
    }
  )
