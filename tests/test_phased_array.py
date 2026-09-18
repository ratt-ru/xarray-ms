import numpy as np
import numpy.testing as npt
import pytest
import xarray

NANTENNA = 4
MAX_ELEMENTS = NANTENNA


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "phased_array.ms",
      "nantenna": NANTENNA,
      "phased_array": True,
    }
  ],
  indirect=True,
)
def test_phased_array_dataset(simmed_ms):
  datatree = xarray.open_datatree(simmed_ms).load()
  partition = datatree["phased_array_partition_000"]
  phased_array = partition["phased_array_xds"].ds
  antenna = partition["antenna_xds"].ds
  nreceptors = len(antenna.receptor_label)

  assert {
    dimension: phased_array.sizes[dimension]
    for dimension in (
      "antenna_name",
      "cartesian_pos_label_local",
      "cartesian_pos_label",
      "receptor_label",
      "element_id",
    )
  } == {
    "antenna_name": NANTENNA,
    "cartesian_pos_label_local": 3,
    "cartesian_pos_label": 3,
    "receptor_label": 2,
    "element_id": MAX_ELEMENTS,
  }
  npt.assert_array_equal(phased_array.antenna_name, antenna.antenna_name)
  npt.assert_array_equal(phased_array.receptor_label, antenna.receptor_label)
  npt.assert_array_equal(phased_array.polarization_type, antenna.polarization_type)
  npt.assert_array_equal(phased_array.element_id, np.arange(MAX_ELEMENTS))

  expected_coordinate_axes = np.stack(
    [(antenna_id + 1) * np.eye(3) for antenna_id in range(NANTENNA)]
  )
  expected_element_offsets = np.full(
    (NANTENNA, 3, MAX_ELEMENTS), np.nan, dtype=np.float64
  )
  expected_element_flags = np.full((NANTENNA, 2, MAX_ELEMENTS), True, dtype=bool)
  for antenna_id in range(NANTENNA):
    nelements = antenna_id + 1
    expected_element_offsets[antenna_id, :, :nelements] = np.tile(
      np.arange(nelements, dtype=np.float64), (3, 1)
    )
    expected_element_flags[antenna_id, :, :nelements] = np.tile(
      np.full(nelements, False, dtype=bool), (nreceptors, 1)
    )

  npt.assert_array_equal(
    phased_array["PHASED_ARRAY_ELEMENT_COUNT"],
    np.arange(MAX_ELEMENTS) + 1,
  )

  npt.assert_allclose(
    phased_array["PHASED_ARRAY_COORDINATE_AXES"],
    expected_coordinate_axes,
  )
  npt.assert_allclose(
    phased_array["PHASED_ARRAY_ELEMENT_OFFSET"],
    expected_element_offsets,
    equal_nan=True,
  )
  npt.assert_array_equal(
    phased_array["PHASED_ARRAY_ELEMENT_FLAG"],
    expected_element_flags,
  )

  assert phased_array.attrs == {"type": "phased_array"}
  assert phased_array["PHASED_ARRAY_COORDINATE_AXES"].attrs == {
    "units": "dimensionless",
    "type": "rotation_matrix",
  }
  assert phased_array["PHASED_ARRAY_ELEMENT_OFFSET"].attrs == {
    "units": "m",
    "type": "location",
    "coordinate_system": "topocentric",
    "origin": "ANTENNA_POSITION",
  }


def test_phased_array_dataset_is_absent_when_ms_table_absent(simmed_ms):
  datatree = xarray.open_datatree(simmed_ms).load()
  partition = datatree["test_partition_000"]

  assert "phased_array_xds" not in partition
