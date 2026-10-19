import numpy as np
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.antenna_selection import PartitionAntennaSelection
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory
from xarray_ms.backend.msv2.subtable_grid import (
  asof_grid,
  gather,
  native_grid,
  select_rows,
)
from xarray_ms.backend.msv2.subtable_index import SubtableIndex

# Direction labels for horizontal and equatorial pointing frames
LOCAL_SKY_DIR_LABELS = {"altaz": ["az", "alt"]}
DEFAULT_LOCAL_SKY_DIR_LABELS = ["ra", "dec"]


class PointingFactory(DatasetFactory):
  """Factory class for generating the pointing_xds dataset
  for a partition of the Measurement Set"""

  def get_dataset(
    self,
    selection: PartitionAntennaSelection,
    pointing: SubtableIndex,
    interpolate: bool = False,
  ) -> Dataset | None:
    """Generate the pointing_xds dataset for a partition.

    Parameters
    ----------
    selection : PartitionAntennaSelection
        The partition-specific antenna selection shared with antenna_xds.
        Antennas repeated over multiple feeds are only included once.
    pointing : SubtableIndex
        Index of the POINTING subtable.
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
    if len(pointing.groups) == 0:
      return None

    (index,) = pointing.groups.values()
    partition = self._structure_factory.instance[self._partition_key]
    half_interval = partition.interval.max() / 2 if partition.interval.size else 0.0
    span = (partition.time.min() - half_interval, partition.time.max() + half_interval)

    # POINTING has no feed axis, but the selection contains an antenna
    # per selected feed. Use each antenna once, in order of first occurrence
    _, first = np.unique(selection.antenna_ids, return_index=True)
    first = np.sort(first)
    antenna_ids = selection.antenna_ids[first]
    antenna_names = selection.antenna_names[first]

    selected = select_rows(
      index["TIME"], index["VALID_INTERVAL"], index["ANTENNA_ID"], antenna_ids, span
    )

    if selected.size == 0:
      return None

    time = index["TIME"][selected]
    antenna_id = index["ANTENNA_ID"][selected]

    if interpolate:
      grid = asof_grid(
        time,
        index["VALID_INTERVAL"][selected],
        antenna_id,
        antenna_ids,
        partition.time,
      )
      time_dim = "time"
      # Read only the rows referenced by the grid
      rows, grid_index = np.unique(grid.index, return_inverse=True)
      grid_index = grid_index.reshape(grid.index.shape) - int(rows[0] < 0)
      rows = rows[rows >= 0]
    else:
      # The native grid references every selected row
      grid = native_grid(time, antenna_id, antenna_ids, "POINTING")
      time_dim = "time_pointing"
      rows, grid_index = np.arange(time.size), grid.index

    if rows.size == 0:
      return None

    columns = ["DIRECTION"] + [
      c for c in ("ENCODER", "OVER_THE_TOP") if pointing.has_column(c)
    ]
    data = pointing.getcols(columns, index["row"][selected][rows])

    if (direction := data["DIRECTION"]) is None:
      return None

    if direction.shape[1] != 1:
      raise NotImplementedError(
        f"POINTING NUM_POLY {direction.shape[1] - 1} > 0 is not currently supported"
      )

    coder_factory = MSv2CoderFactory.from_table_desc(pointing.table_desc)
    dims = (time_dim, "antenna_name", "local_sky_dir_label")
    data_vars = {
      "POINTING_BEAM": coder_factory.create("DIRECTION").decode(
        Variable(dims, gather(direction[:, 0, :], grid_index, np.nan))
      )
    }

    if (encoder := data.get("ENCODER")) is not None:
      data_vars["POINTING_DISH_MEASURED"] = coder_factory.create("ENCODER").decode(
        Variable(dims, gather(encoder, grid_index, np.nan))
      )

    if (over_the_top := data.get("OVER_THE_TOP")) is not None:
      data_vars["POINTING_OVER_THE_TOP"] = Variable(
        dims[:2], gather(over_the_top.astype(bool), grid_index, False)
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
        "antenna_name": Variable("antenna_name", antenna_names),
        "local_sky_dir_label": Variable("local_sky_dir_label", local_sky_dir_labels),
      },
      attrs={"type": "pointing"},
    )
