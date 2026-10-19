import numpy as np
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.antenna_selection import PartitionAntennaSelection
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.factories.subtable import LazySubtableColumns
from xarray_ms.backend.msv2.subtable_index import SubtableIndexFactory

# Direction labels for horizontal and equatorial pointing frames
LOCAL_SKY_DIR_LABELS = {"altaz": ["az", "alt"]}
DEFAULT_LOCAL_SKY_DIR_LABELS = ["ra", "dec"]


def _as_bool(data: np.ndarray) -> np.ndarray:
  return data.astype(bool)


class PointingFactory(DatasetFactory):
  """Factory class for generating the pointing_xds dataset
  for a partition of the Measurement Set"""

  def get_dataset(
    self,
    selection: PartitionAntennaSelection,
    pointing_factory: SubtableIndexFactory,
    interpolate: bool = False,
  ) -> Dataset | None:
    """Generate the pointing_xds dataset for a partition.

    POINTING data is read lazily.

    Parameters
    ----------
    selection : PartitionAntennaSelection
        The partition-specific antenna selection shared with antenna_xds.
    pointing_factory : SubtableIndexFactory
        Factory producing the index of the POINTING subtable.
    interpolate : bool
        If True, POINTING rows are associated with the
        correlated data timesteps, producing a ``time`` dimension.
        Otherwise, the POINTING timesteps are retained,
        producing a ``time_pointing`` dimension.

    Returns
    -------
    Dataset | None
        The pointing_xds dataset, or None if the POINTING subtable
        is absent, empty or has no rows relevant to this partition.
    """
    pointing = pointing_factory.instance

    if len(pointing.groups) == 0:
      return None

    (group,) = pointing.groups.keys()
    partition = self._structure_factory.instance[self._partition_key]
    columns = LazySubtableColumns.from_partition(
      pointing_factory,
      group,
      selection.antenna_ids,
      partition,
      interpolate,
      antenna_first=False,
    )

    if columns is None or (direction := columns.column_type("DIRECTION")) is None:
      return None

    direction_shape, direction_dtype = direction

    if direction_shape[0] != 1:
      raise NotImplementedError(
        f"POINTING NUM_POLY {direction_shape[0] - 1} > 0 is not currently supported"
      )

    time_dim = "time" if interpolate else "time_pointing"
    dims = (time_dim, "antenna_name", "local_sky_dir_label")
    data_vars = {
      "POINTING_BEAM": columns.float_variable(
        "DIRECTION", dims, direction_shape[1:], direction_dtype, leading_axes=1
      )
    }

    if (encoder := columns.column_type("ENCODER")) is not None:
      encoder_shape, encoder_dtype = encoder
      data_vars["POINTING_DISH_MEASURED"] = columns.float_variable(
        "ENCODER", dims, encoder_shape, encoder_dtype
      )

    if columns.column_type("OVER_THE_TOP") is not None:
      # Boolean columns are read as bytes
      data_vars["POINTING_OVER_THE_TOP"] = Variable(
        dims[:2],
        columns.array(
          "OVER_THE_TOP", (), bool, 0, read_dtype=np.uint8, transform=_as_bool
        ),
      )

    # The time coordinate type is named after its dimension
    time_var = columns.time_variable(time_dim, time_dim)
    frame = data_vars["POINTING_BEAM"].attrs["frame"]
    local_sky_dir_labels = LOCAL_SKY_DIR_LABELS.get(frame, DEFAULT_LOCAL_SKY_DIR_LABELS)

    return Dataset(
      data_vars=data_vars,
      coords={
        time_dim: time_var,
        "antenna_name": Variable("antenna_name", selection.antenna_names),
        "local_sky_dir_label": Variable("local_sky_dir_label", local_sky_dir_labels),
      },
      attrs={"type": "pointing"},
    )
