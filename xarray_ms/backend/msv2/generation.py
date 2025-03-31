from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from xarray_ms.errors import GeneratedMetadataWarning
from xarray_ms.multiton import Multiton

if TYPE_CHECKING:
  import pyarrow as pa


def maybe_generate_field_table(
  field: pa.Table, field_id: npt.NDArray[np.int32]
) -> pa.Table:
  """Generates a good enough FIELD subtable if there are no row ids
  associated with the given FIELD_ID values"""
  max_field_id = np.max(field_id)

  if max_field_id < len(field):
    return field

  warnings.warn(
    f"No row exists in the FIELD table of length {len(field)} "
    f"for FIELD_ID={max_field_id}. "
    f"An artificial FIELD table will be generated.",
    GeneratedMetadataWarning,
  )

  import pyarrow as pa

  return pa.Table.from_pydict(
    {
      "NAME": np.array([f"Unknown-{i}" for i in range(max_field_id + 1)], dtype=object),
      "SOURCE_ID": np.zeros(max_field_id + 1, np.int32),
    }
  )


def maybe_generate_state_table(
  state: pa.Table, state_id: npt.NDArray[npt.int32]
) -> pa.Table:
  """Generates a STATE subtable if there are no row ids
  associated with the given STATE_ID values"""
  max_state_id = np.max(state_id)

  if max_state_id < len(state):
    return state

  warnings.warn(
    f"No row exists in the STATE table of length {len(state)} "
    f"for STATE_ID={max_state_id}. "
    f"An artifical STATE table will be generated.",
    GeneratedMetadataWarning,
  )

  import pyarrow as pa

  return pa.Table.from_pydict(
    {
      "OBS_MODE": np.array(["UNSPECIFIED"] * (max_state_id + 1), dtype=object),
      "SUB_SCAN": np.zeros(max_state_id + 1, np.int32),
    }
  )


def maybe_generate_observation_row(
  observation: Multiton, observation_id: int
) -> pa.Table:
  obs = observation.instance

  if observation_id < len(obs):
    return obs.take([observation_id])

  warnings.warn(
    f"No row exists in the OBSERVATION table of length {len(obs)} "
    f"for OBSERVATION_ID={observation_id}. "
    f"Artificial metadata will be generated.",
    GeneratedMetadataWarning,
  )

  unknown = np.array(["unknown"], dtype=object)

  import pyarrow as pa

  return pa.Table.from_pydict(
    {
      "OBSERVER": unknown,
      "PROJECT": unknown,
      "TELESCOPE_NAME": unknown,
    }
  )
