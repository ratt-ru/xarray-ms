import numpy as np
from rarg_python_patterns.multiton import Multiton
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.antenna_selection import PartitionAntennaSelection
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory
from xarray_ms.backend.msv2.subtable_grid import grid_subtable
from xarray_ms.backend.msv2.subtable_index import SubtableIndexFactory

POINTING_COLUMNS = ("DIRECTION", "ENCODER", "OVER_THE_TOP")

# Direction labels for horizontal and equatorial pointing frames
LOCAL_SKY_DIR_LABELS = {"altaz": ["az", "alt"]}
DEFAULT_LOCAL_SKY_DIR_LABELS = ["ra", "dec"]


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

    Parameters
    ----------
    selection : PartitionAntennaSelection
        The partition-specific antenna selection shared with antenna_xds.
        Antennas repeated over multiple feeds are only included once.
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

    # POINTING has no feed axis, but the selection contains an antenna
    # per selected feed. Use each antenna once, in order of first occurrence
    _, first = np.unique(selection.antenna_ids, return_index=True)
    first = np.sort(first)
    antenna_ids = selection.antenna_ids[first]
    antenna_names = selection.antenna_names[first]

    # Partitions sharing antennas and time span share the gridded columns
    gridded = Multiton(
      grid_subtable,
      pointing_factory,
      group,
      antenna_ids,
      span,
      POINTING_COLUMNS,
      partition.time if interpolate else None,
    ).instance

    if gridded is None or (direction := gridded.data["DIRECTION"]) is None:
      return None

    if direction.shape[2] != 1:
      raise NotImplementedError(
        f"POINTING NUM_POLY {direction.shape[2] - 1} > 0 is not currently supported"
      )

    time_dim = "time" if interpolate else "time_pointing"
    coder_factory = MSv2CoderFactory.from_table_desc(pointing.table_desc)
    dims = (time_dim, "antenna_name", "local_sky_dir_label")
    data_vars = {
      "POINTING_BEAM": coder_factory.create("DIRECTION").decode(
        Variable(dims, direction[:, :, 0, :])
      )
    }

    if (encoder := gridded.data["ENCODER"]) is not None:
      data_vars["POINTING_DISH_MEASURED"] = coder_factory.create("ENCODER").decode(
        Variable(dims, encoder)
      )

    if (over_the_top := gridded.data["OVER_THE_TOP"]) is not None:
      data_vars["POINTING_OVER_THE_TOP"] = Variable(dims[:2], over_the_top.astype(bool))

    time_var = coder_factory.create("TIME").decode(Variable(time_dim, gridded.time))
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
