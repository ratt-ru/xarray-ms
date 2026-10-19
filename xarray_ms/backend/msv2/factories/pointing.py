import numpy as np
from rarg_python_patterns.multiton import Multiton
from xarray import Dataset, Variable
from xarray.core.indexing import LazilyIndexedArray

from xarray_ms.backend.msv2.antenna_selection import PartitionAntennaSelection
from xarray_ms.backend.msv2.array import SubtableMSv2Array
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory
from xarray_ms.backend.msv2.subtable_grid import grid_subtable
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
    half_interval = partition.interval.max() / 2 if partition.interval.size else 0.0
    span = (partition.time.min() - half_interval, partition.time.max() + half_interval)

    # Partitions sharing antennas and time span share the grid
    grid_factory = Multiton(
      grid_subtable,
      pointing_factory,
      group,
      selection.antenna_ids,
      span,
      partition.time if interpolate else None,
    )

    if (grid := grid_factory.instance) is None:
      return None

    # Establish column shapes from the first row present in the grid
    first_row = int(grid.rows[grid.rows >= 0][0])

    if (direction := pointing.column_row_type("DIRECTION", first_row)) is None:
      return None

    direction_shape, direction_dtype = direction

    if direction_shape[0] != 1:
      raise NotImplementedError(
        f"POINTING NUM_POLY {direction_shape[0] - 1} > 0 is not currently supported"
      )

    time_dim = "time" if interpolate else "time_pointing"
    dims = (time_dim, "antenna_name", "local_sky_dir_label")
    table_factory = pointing.table_factory

    def lazy(column, shape, dtype, default, **kw) -> LazilyIndexedArray:
      return LazilyIndexedArray(
        SubtableMSv2Array(
          table_factory, grid_factory, column, shape, dtype, default, **kw
        )
      )

    coder_factory = MSv2CoderFactory.from_table_desc(pointing.table_desc)
    beam_shape = grid.rows.shape + direction_shape[1:]
    data_vars = {
      "POINTING_BEAM": coder_factory.create("DIRECTION").decode(
        Variable(
          dims, lazy("DIRECTION", beam_shape, direction_dtype, np.nan, leading_axes=1)
        )
      )
    }

    if (encoder := pointing.column_row_type("ENCODER", first_row)) is not None:
      encoder_shape, encoder_dtype = encoder
      data_vars["POINTING_DISH_MEASURED"] = coder_factory.create("ENCODER").decode(
        Variable(
          dims,
          lazy("ENCODER", grid.rows.shape + encoder_shape, encoder_dtype, np.nan),
        )
      )

    if pointing.column_row_type("OVER_THE_TOP", first_row) is not None:
      # Boolean columns are read as bytes
      data_vars["POINTING_OVER_THE_TOP"] = Variable(
        dims[:2],
        lazy(
          "OVER_THE_TOP",
          grid.rows.shape,
          bool,
          0,
          read_dtype=np.uint8,
          transform=_as_bool,
        ),
      )

    time_var = coder_factory.create("TIME").decode(Variable(time_dim, grid.time))
    # The time coordinate type is named after its dimension
    time_var.attrs["type"] = time_dim

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
