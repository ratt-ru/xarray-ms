from typing import Mapping

import numpy as np
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.errors import InvalidMeasurementSet

RELOCATABLE_ARRAY = {"ALMA", "VLA", "NOEMA", "EVLA"}


class AntennaDatasetFactory:
  _partition_key: PartitionKeyT
  _structure_factory: MSv2StructureFactory

  def __init__(
    self, partition_key: PartitionKeyT, structure_factory: MSv2StructureFactory
  ):
    self._partition_key = partition_key
    self._structure_factory = structure_factory

  def get_dataset(self) -> Mapping[str, Variable]:
    structure = self._structure_factory()
    partition = structure[self._partition_key]
    ants = structure._ant
    feeds = structure._feed

    import pyarrow.compute as pac

    feed_id = feeds["FEED_ID"].to_numpy()
    spw_id = feeds["SPECTRAL_WINDOW_ID"].to_numpy()
    feed_ant_id = feeds["ANTENNA_ID"].to_numpy()
    # Select feeds with global spws (-1) or that match the partition spw
    feed_mask = np.logical_or(spw_id == -1, spw_id == partition.spw_id)
    # Select feed_id's matching the partition feeds
    np.logical_and(feed_mask, feed_id == partition.feed_id, out=feed_mask)

    # NOTE: This outer product could potentially be large
    # and could be ameliorated by selecting out feed_ant_id[None, feed_mask]
    ant_id = np.arange(len(ants))
    ant_feed_map = ant_id[:, None] == feed_ant_id[None, :]
    np.logical_and(ant_feed_map, feed_mask[None, :], out=ant_feed_map)
    ant_mask = np.any(ant_feed_map, axis=1)

    filtered_ants = ants.filter(ant_mask)

    if len(filtered_ants) == 0:
      raise InvalidMeasurementSet(
        f"No antennas were found in FEED matching "
        f"feed_id = {partition.feed_id} and spw_id = {partition.spw_id}"
      )

    antenna_names = filtered_ants["NAME"].to_numpy()
    telescope_names = np.asarray([partition.telescope_name] * len(antenna_names))
    position = pac.list_flatten(filtered_ants["POSITION"]).to_numpy().reshape(-1, 3)
    diameter = filtered_ants["DISH_DIAMETER"].to_numpy()
    station = filtered_ants["STATION"].to_numpy()
    mount = filtered_ants["MOUNT"].to_numpy()

    filtered_feeds = feeds.take(np.where(ant_feed_map)[-1])
    nreceptors = filtered_feeds["NUM_RECEPTORS"].unique().to_numpy()

    if len(nreceptors) != 1 or nreceptors.item() != 2:
      raise NotImplementedError(
        f"Representation of Measurement Sets with "
        f"FEED::NUM_RECEPTORS != 2: {nreceptors.tolist()}"
      )

    receptor_angle = (
      pac.list_flatten(filtered_feeds["RECEPTOR_ANGLE"]).to_numpy().reshape(-1, 2)
    )
    pol_type = (
      pac.list_flatten(filtered_feeds["POLARIZATION_TYPE"]).to_numpy().reshape(-1, 2)
    )
    receptor_labels = [f"pol_{i}" for i in range(nreceptors.item())]

    metre_attrs = {"units": ["m"], "type": "quantity"}
    rad_attrs = {"units": ["rad"], "type": "quantity"}

    data_vars = {
      "ANTENNA_POSITION": Variable(
        ("antenna_name", "cartesian_pos_label"),
        position,
        {
          "coordinate_system": "geocentric",
          "origin_object_name": "earth",
        },
      ),
      "ANTENNA_DISH_DIAMETER": Variable("antenna_name", diameter, metre_attrs),
      "ANTENNA_EFFECTIVE_DISH_DIAMETER": Variable(
        "antenna_name", diameter, metre_attrs
      ),
      "ANTENNA_RECEPTOR_ANGLE": Variable(
        ("antenna_name", "receptor_label"), receptor_angle, rad_attrs
      ),
    }

    if "FOCUS_LENGTH" in filtered_feeds:
      focus_length = filtered_feeds["FOCUS_LENGTH"].to_numpy()
      data_vars["ANTENNA_FOCUS_LENGTH"] = Variable(
        "antenna_name", focus_length, metre_attrs
      )

    return Dataset(
      data_vars=data_vars,
      coords={
        "antenna_name": Variable("antenna_name", antenna_names),
        "mount": Variable("antenna_name", mount),
        "telescope_name": Variable("telescope_name", telescope_names),
        "station": Variable("antenna_name", station),
        "cartesian_pos_label": Variable("cartesian_pos_label", ["x", "y", "z"]),
        "polarization_type": Variable(("antenna_name", "receptor_label"), pol_type),
        "receptor_label": Variable("receptor_label", receptor_labels),
      },
      attrs={
        "type": "antenna",
        "overall_telescope_name": partition.telescope_name,
        "relocatable_antennas": partition.telescope_name in RELOCATABLE_ARRAY,
      },
    )
