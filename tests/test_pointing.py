import numpy as np
import numpy.testing as npt
import pytest
import xarray
from arcae.lib.arrow_tables import Table, ms_descriptor

from xarray_ms.backend.msv2.array import SubtableMSv2Array
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


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "pointing_spws.ms",
      "ntime": NTIME,
      "nantenna": NANTENNA,
      "pointing": True,
      "data_description": [(8, ["XX", "YY"]), (4, ["XX", "YY"])],
    }
  ],
  indirect=True,
)
def test_pointing_shared_between_partitions(simmed_ms):
  """Partitions differing only in spectral window share
  the lazily read POINTING row grid"""
  with pytest.warns(IrregularSubtableGridWarning):
    dt = xarray.open_datatree(simmed_ms)

  def backend_array(variable):
    data = variable._data
    while not isinstance(data, SubtableMSv2Array):
      data = data.array
    return data

  arrays = [
    backend_array(dt[path].POINTING_BEAM.variable)
    for path in sorted(dt.groups)
    if path.endswith("/pointing_xds")
  ]
  assert len(arrays) == 2
  grids = [a._grid_factory.instance for a in arrays]
  assert grids[0] is grids[1]
  assert not grids[0].rows.flags.writeable


@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "pointing_validity.ms", "ntime": NTIME, "nantenna": NANTENNA}],
  indirect=True,
)
@pytest.mark.parametrize("interpolate", [None, "pointing"])
def test_pointing_single_validity_row(simmed_ms, interpolate):
  """A single POINTING row per antenna, whose time lies outside the
  correlated data, but whose interval covers it, is selected and
  associated with every correlated timestep"""
  time = FIRST_FEB_2023_MJDS + 10 * NTIME * DUMP_RATE
  direction = np.stack([np.full(NANTENNA, 0.5), np.arange(NANTENNA) * 0.1], axis=-1)

  with Table.ms_from_descriptor(
    simmed_ms, "POINTING", table_desc=ms_descriptor("POINTING", complete=True)
  ) as T:
    T.addrows(NANTENNA)
    T.putcol("ANTENNA_ID", np.arange(NANTENNA, dtype=np.int32))
    T.putcol("TIME", np.full(NANTENNA, time))
    T.putcol("INTERVAL", np.full(NANTENNA, 100 * NTIME * DUMP_RATE))
    T.putcol("NUM_POLY", np.zeros(NANTENNA, np.int32))
    T.putcol("DIRECTION", direction[:, None, :])

  dt = xarray.open_datatree(simmed_ms, interpolate=interpolate)
  pointing = dt["pointing_validity_partition_000/pointing_xds"]
  beam = pointing.POINTING_BEAM.values

  if interpolate:
    assert beam.shape == (NTIME, NANTENNA, 2)
  else:
    assert beam.shape == (1, NANTENNA, 2)

  npt.assert_allclose(beam, np.broadcast_to(direction, beam.shape))
  assert "POINTING_DISH_MEASURED" not in pointing
