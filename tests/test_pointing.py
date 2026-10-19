import numpy as np
import numpy.testing as npt
import pytest
import xarray

from xarray_ms.backend.msv2.measures_encoders import EpochCoder
from xarray_ms.errors import IrregularSubtableGridWarning
from xarray_ms.testing.simulator import FIRST_FEB_2023_MJDS

NTIME = 5
NANTENNA = 3
DUMP_RATE = 8.0


@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "pointing.ms", "ntime": NTIME, "nantenna": NANTENNA, "pointing": True}],
  indirect=True,
)
def test_pointing_native(simmed_ms):
  with pytest.warns(IrregularSubtableGridWarning):
    dt = xarray.open_datatree(simmed_ms)

  partition = dt["pointing_partition_000"]
  antenna = partition["antenna_xds"].ds
  pointing = partition["pointing_xds"].to_dataset(inherit=False)

  # POINTING rows overlapping the correlated data time span are selected:
  # POINTING timesteps j in [1, 2 * NTIME + 1] at half the dump rate
  ntime = 2 * NTIME + 1
  assert dict(pointing.sizes) == {
    "time_pointing": ntime,
    "antenna_name": NANTENNA,
    "local_sky_dir_label": 2,
  }
  assert pointing.attrs["type"] == "pointing"
  npt.assert_array_equal(pointing.antenna_name, antenna.antenna_name)
  npt.assert_array_equal(pointing.local_sky_dir_label, ["az", "alt"])

  j = np.arange(1, ntime + 1)
  expected_time = FIRST_FEB_2023_MJDS + (j / 2 - 1) * DUMP_RATE
  npt.assert_allclose(pointing.time_pointing, EpochCoder.decode_array(expected_time))
  assert pointing.time_pointing.attrs == {
    "type": "time_pointing",
    "units": "s",
    "format": "unix",
    "scale": "utc",
  }

  beam = pointing.POINTING_BEAM
  assert beam.attrs == {"type": "sky_coord", "units": "rad", "frame": "altaz"}
  expected = np.empty((ntime, NANTENNA, 2))
  expected[:, :, 0] = j[:, None] * 0.01
  expected[:, :, 1] = np.arange(NANTENNA)[None, :] * 0.1
  # The last antenna is only present at every second POINTING timestep
  expected[j % 2 == 1, -1, :] = np.nan
  npt.assert_allclose(beam.values, expected)

  npt.assert_allclose(pointing.POINTING_DISH_MEASURED.values, expected + 0.001)
  assert pointing.POINTING_DISH_MEASURED.attrs["frame"] == "altaz"
  assert pointing.POINTING_OVER_THE_TOP.dtype == bool


@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "pointing.ms", "ntime": NTIME, "nantenna": NANTENNA, "pointing": True}],
  indirect=True,
)
@pytest.mark.parametrize("interpolate", ["pointing", ["pointing"]])
def test_pointing_interpolated(simmed_ms, interpolate):
  dt = xarray.open_datatree(simmed_ms, interpolate=interpolate)
  partition = dt["pointing_partition_000"]
  pointing = partition["pointing_xds"].to_dataset(inherit=False)

  assert dict(pointing.sizes) == {
    "time": NTIME,
    "antenna_name": NANTENNA,
    "local_sky_dir_label": 2,
  }
  # The time coordinate is shared with, and inherited from, the correlated dataset
  assert "time" not in pointing.coords
  npt.assert_array_equal(partition["pointing_xds"].time, partition.time)

  # Each correlated timestep k takes the POINTING row
  # most recently valid at that time: POINTING timestep j = 2 * k + 2
  j = 2 * np.arange(NTIME) + 2
  expected = np.empty((NTIME, NANTENNA, 2))
  expected[:, :, 0] = j[:, None] * 0.01
  expected[:, :, 1] = np.arange(NANTENNA)[None, :] * 0.1
  npt.assert_allclose(pointing.POINTING_BEAM.values, expected)


@pytest.mark.parametrize("simmed_ms", [{"name": "no_pointing.ms"}], indirect=True)
def test_pointing_absent(simmed_ms):
  dt = xarray.open_datatree(simmed_ms)
  assert not any(path.endswith("pointing_xds") for path in dt.groups)


@pytest.mark.parametrize("simmed_ms", [{"name": "no_pointing.ms"}], indirect=True)
def test_invalid_interpolate(simmed_ms):
  with pytest.raises(ValueError, match="Invalid interpolate values"):
    xarray.open_datatree(simmed_ms, interpolate=["pointing", "foo"])
