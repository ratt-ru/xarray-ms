from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from xarray import Variable

from xarray_ms.backend.msv2.encoders import UTCCoder
from xarray_ms.casa_types import ColumnDesc, DirectionMeasures
from xarray_ms.errors import ImputedMetadataWarning

if TYPE_CHECKING:
  import pyarrow as pa


def _maybe_return_table_or_max_id(
  table: pa.Table, table_name: str, ids: npt.NDArray[np.int32], id_column_name: str
) -> pa.Table | int:
  """Returns the existing table if a row entry exists,
  else returns the maximum id"""

  # NOTE(sjperkins)
  # Negative ids are invalid foreign keys
  # AFAICT this implies there's no entry in the associated subtable
  # Set a ceiling of 0 for imputation purposes
  ids = np.concatenate([ids, [0]])
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

  npoly = 0
  num_poly = np.full(result + 1, npoly, np.int32)
  direction = pa.array([[0.0, 0.0]], pa.list_(pa.float64(), 2))
  row_ref_codes = pa.array([DirectionMeasures.J2000.value] * (result + 1), pa.int32())

  table = pa.Table.from_pydict(
    {
      "NAME": np.array([f"UNKNOWN-{i}" for i in range(result + 1)], dtype=object),
      "NUM_POLY": num_poly,
      "DELAY_DIR": direction,
      "DelayDir_Ref": row_ref_codes,
      "PHASE_DIR": direction,
      "PhaseDir_Ref": row_ref_codes,
      "REFERENCE_DIR": direction,
      "RefDir_Ref": row_ref_codes,
      "SOURCE_ID": np.arange(result + 1, dtype=np.int32),
      # TODO: Both TIME and INTERVAL could be improved
      "TIME": np.zeros(result + 1, np.float64),
    }
  )

  ref_types, ref_codes = map(list, zip(*((m.name, m.value) for m in DirectionMeasures)))

  # Create a minimal table descriptor for the three direction
  # columns containing synthesised measures data
  table_desc = {
    column: {
      "option": 0,
      "valueType": "DOUBLE",
      "keywords": {
        "QuantumUnits": ["rad", "rad"],
        "MEASINFO": {
          "type": "direction",
          "VarRefCol": ref_column,
          "TabRefTypes": ref_types,
          "TabRefCodes": ref_codes,
        },
      },
    }
    for column, ref_column in [
      ("DELAY_DIR", "DelayDir_Ref"),
      ("PHASE_DIR", "PhaseDir_Ref"),
      ("REFERENCE_DIR", "RefDir_Ref"),
    ]
  }

  # Add the table descriptor to the schema metadata
  updated_schema = table.schema.with_metadata(
    {"__arcae_metadata__": json.dumps({"__casa_descriptor__": table_desc})}
  )

  for column in table_desc.keys():
    i = updated_schema.get_field_index(column)
    field_metadata = {
      "__arcae_metadata__": json.dumps({"__casa_descriptor__": table_desc[column]})
    }
    field = updated_schema.field(column).with_metadata(field_metadata)
    updated_schema = updated_schema.set(i, field)

  return table.cast(target_schema=updated_schema)


def maybe_impute_source_table(
  source: pa.Table, source_id: npt.NDArray[np.int32]
) -> pa.Table:
  """Generates a SOURCE subtable if there are no row ids
  associated with the given SOURCE_ID values"""

  import pyarrow as pa

  result = _maybe_return_table_or_max_id(source, "SOURCE", source_id, "SOURCE_ID")
  if isinstance(result, pa.Table):
    return result

  np_direction = np.zeros((result + 1, 2), dtype=np.float64)
  direction = pa.FixedSizeListArray.from_arrays(np_direction.ravel(), 2)

  return pa.Table.from_pydict(
    {
      "SOURCE_ID": np.arange(result + 1, dtype=np.int32),
      "NAME": np.array([f"UNKNOWN-{i}" for i in range(result + 1)], dtype=object),
      "NUM_LINES": np.ones(result + 1, np.int32),
      # TODO: Both TIME and INTERVAL could be improved
      "TIME": np.zeros(result + 1, np.float64),
      "INTERVAL": np.zeros(result + 1, np.float64),
      "DIRECTION": direction,
      "SPECTRAL_WINDOW_ID": np.full(result + 1, -1, np.int32),
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

  # Create a minimal table descriptor
  table_desc = {
    "RELEASE_DATE": {
      "option": 0,
      "valueType": "DOUBLE",
      "keywords": {
        "QuantumUnits": ["s"],
        "MEASINFO": {"type": "epoch", "Ref": "UTC"},
      },
    }
  }

  release_date_coldesc = ColumnDesc.from_descriptor("RELEASE_DATE", table_desc)
  dt = datetime(1978, 10, 9, 8, 0, 0, tzinfo=timezone.utc).timestamp()
  release_date_var = Variable("time", [dt] * (result + 1))
  release_date_var = UTCCoder(release_date_coldesc).encode(release_date_var)
  unknown = np.array(["unknown"] * (result + 1), dtype=object)

  table = pa.Table.from_pydict(
    {
      "OBSERVER": unknown,
      "PROJECT": unknown,
      "TELESCOPE_NAME": unknown,
      "RELEASE_DATE": release_date_var.values,
    }
  )

  # Add the table descriptor to the schema metadata
  updated_schema = table.schema.with_metadata(
    {"__arcae_metadata__": json.dumps({"__casa_descriptor__": table_desc})}
  )

  for column in table_desc.keys():
    i = updated_schema.get_field_index(column)
    field_metadata = {
      "__arcae_metadata__": json.dumps({"__casa_descriptor__": table_desc[column]})
    }
    field = updated_schema.field(column).with_metadata(field_metadata)
    updated_schema = updated_schema.set(i, field)

  return table.cast(target_schema=updated_schema)


def maybe_impute_processor_table(
  processor: pa.Table, processor_id: npt.NDArray[np.int32]
) -> pa.Table:
  """Generates a PROCESSOR table if there are no row ids
  associated with the given PROCESSOR_ID values"""
  import pyarrow as pa

  result = _maybe_return_table_or_max_id(
    processor, "PROCESSOR", processor_id, "PROCESSOR_ID"
  )
  if isinstance(result, pa.Table):
    return result

  unknown = np.array(["unknown"] * (result + 1), dtype=object)

  return pa.Table.from_pydict(
    {
      "TYPE": unknown,
      "SUB_TYPE": unknown,
    }
  )
