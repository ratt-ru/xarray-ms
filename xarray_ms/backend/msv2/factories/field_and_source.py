from typing import Dict, Mapping

from xarray import Dataset, Variable

from xarray_ms.backend.msv2.imputation import (
  maybe_impute_field_table,
  maybe_impute_source_table,
)
from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.multiton import Multiton


class FieldAndSourceDatasetFactory:
  _partition_key: PartitionKeyT
  _structure_factory: MSv2StructureFactory
  _subtable_factories: Dict[str, Multiton]

  def __init__(
    self,
    partition_key: PartitionKeyT,
    structure_factory: MSv2StructureFactory,
    subtable_factories: Dict[str, Multiton],
  ):
    self._partition_key = partition_key
    self._structure_factory = structure_factory
    self._subtable_factories = subtable_factories

  def get_dataset(self) -> Mapping[str, Variable]:
    import pyarrow.compute as pac

    partition = self._structure_factory.instance[self._partition_key]
    field = self._subtable_factories["FIELD"].instance
    source = self._subtable_factories["SOURCE"].instance

    field = maybe_impute_field_table(field, partition.field_ids)
    source = maybe_impute_source_table(source, partition.source_ids)

    num_poly = field["NUM_POLY"].to_numpy()
    phase_centre = (
      pac.list_flatten(field["PHASE_DIR"]).to_numpy().reshape((2, num_poly))
    )

    return Dataset(
      {
        "FIELD_PHASE_CENTER_DIRECTION": Variable(
          ("field_name", "sky_dir_label"), phase_centre
        ),
      }
    )
