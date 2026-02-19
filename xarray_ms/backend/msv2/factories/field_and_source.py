import numpy as np
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.imputation import (
  maybe_impute_field_table,
  maybe_impute_source_table,
)
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory


class FieldAndSourceFactory(DatasetFactory):
  """Factory class for generating the field_and_source_xds dataset
  for a partition of the Measurement Set"""

  def get_dataset(self) -> Dataset:
    import pyarrow as pa
    import pyarrow.compute as pac

    partition = self._structure_factory.instance[self._partition_key]
    field = self._subtable_factories["FIELD"].instance
    source = self._subtable_factories["SOURCE"].instance
    ufield_ids = np.unique(partition.field_ids)

    field = maybe_impute_field_table(field, ufield_ids)
    field = field.take(ufield_ids)
    field_coder = MSv2CoderFactory.from_arrow_table(field)
    source_ids = field["SOURCE_ID"].to_numpy()
    source = maybe_impute_source_table(source, source_ids)

    if (num_poly := np.unique(field["NUM_POLY"].to_numpy())) != [0]:
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
      field_phase_centre_dir = Variable(("field_name", "sky_dir_label"), phase_centre)
      data_vars["FIELD_PHASE_CENTER_DIRECTION"] = field_coder.create(
        "PHASE_DIR"
      ).decode(field_phase_centre_dir)

    field_names = field["NAME"].to_numpy().astype(str)
    # Filter out negative source ids
    pa_source_ids = pa.array(source_ids)
    pa_source_ids = pac.replace_with_mask(
      pa_source_ids,
      pac.equal(pa_source_ids, -1),
      pa.array([None] * len(source_ids), pa_source_ids.type),
    )
    pa_source_names = pac.take(source["NAME"], pa_source_ids)
    source_names = pac.fill_null(pa_source_names, "UNKNOWN").to_numpy().astype(str)

    return Dataset(
      data_vars=data_vars,
      coords={
        "field_name": Variable("field_name", field_names),
        "sky_dir_label": Variable("sky_dir_label", ["ra", "dec"]),
        "source_name": Variable("field_name", source_names),
      },
      attrs={"type": "field_and_source"},
    )
