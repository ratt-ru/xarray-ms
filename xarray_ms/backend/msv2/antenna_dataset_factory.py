from typing import Mapping

from xarray import Dataset, Variable

from xarray_ms.backend.msv2.structure import MSv2StructureFactory


class AntennaDatasetFactory:
  _structure_factory: MSv2StructureFactory

  def __init__(self, structure_factory: MSv2StructureFactory):
    self._structure_factory = structure_factory

  def get_dataset(self) -> Mapping[str, Variable]:
    ants = self._structure_factory()._ant

    import pyarrow.compute as pac

    ant_pos = pac.list_flatten(ants["POSITION"]).to_numpy().reshape(-1, 3)

    return Dataset(
      data_vars={
        "ANTENNA_POSITION": Variable(
          ("antenna_name", "cartesian_pos_label/ellipsoid_pos_label"), ant_pos
        )
      },
      coords={
        "antenna_name": Variable("antenna_name", ants["NAME"].to_numpy()),
        "mount": Variable(
          "antenna_name",
          ants["MOUNT"].to_numpy(),
        ),
        "station": Variable(
          "antenna_name",
          ants["STATION"].to_numpy(),
        ),
      },
    )
