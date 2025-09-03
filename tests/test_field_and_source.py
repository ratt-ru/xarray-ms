import numpy as np
import xarray


def test_field_and_source(simmed_ms):
  """A basic field and source test case"""
  dt = xarray.open_datatree(simmed_ms)
  assert len(dt.children) == 1

  assert list(dt.children) == ["test_partition_000"]
  dt = dt.load()

  p0 = dt["test_partition_000"]

  for _, data_group in p0.attrs.get("data_group", {}).items():
    field_source = data_group["field_and_source"]
    assert field_source.field_name == ["FIELD-0"]
    assert field_source.source_name == ["SOURCE-0"]
    np.testing.assert_array_equal(
      field_source.FIELD_PHASE_CENTER_DIRECTION.values, [[0, 0]]
    )
