from typing import Dict, Mapping

import numpy as np
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

    num_poly = np.unique(field["NUM_POLY"].to_numpy())
    if not num_poly == [0]:
      raise NotImplementedError(
        f"FIELD subtable NUM_POLY {num_poly} != 0 "
        f"are not currently supported for FIELD_IDs "
        f"{partition.field_ids}"
      )

    field_columns = set(field.column_names)
    data_vars = {}
    if "PHASE_DIR" in field_columns:
      phase_centre = pac.list_flatten(field["PHASE_DIR"], recursive=True)
      phase_centre = phase_centre.to_numpy().reshape(len(field), 2)
      data_vars["FIELD_PHASE_CENTER_DIRECTION"] = Variable(
        ("field_name", "sky_dir_label"), phase_centre
      )

    field_names = field["NAME"].to_numpy()

    return Dataset(
      data_vars=data_vars,
      coords={
        "field_name": Variable("field_name", field_names),
        "sky_dir_label": Variable("sky_dir_label", ["ra", "dec"]),
      },
      attrs={"type": "field_and_source"},
    )
