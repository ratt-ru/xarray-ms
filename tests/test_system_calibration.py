import numpy as np
import numpy.testing as npt
import pytest
import xarray

from xarray_ms.backend.msv2.measures_encoders import EpochCoder
from xarray_ms.testing.simulator import FIRST_FEB_2023_MJDS, MSStructureSimulator

NTIME = 5
NANTENNA = 3
DUMP_RATE = 8.0
NRECEPTOR = 2
NCHAN = MSStructureSimulator.SYSCAL_NCHAN

SIM_PARAMS = {
  "name": "syscal.ms",
  "ntime": NTIME,
  "nantenna": NANTENNA,
  "syscal": True,
  "data_description": [(8, ["XX", "YY"]), (4, ["XX", "YY"])],
}


def expected_base(spw, samples):
  return spw * 100.0 + np.arange(NANTENNA)[:, None] * 10.0 + samples[None, :]


@pytest.mark.parametrize("simmed_ms", [{**SIM_PARAMS}], indirect=True)
def test_system_calibration_native(simmed_ms):
  dt = xarray.open_datatree(simmed_ms)

  for spw in range(2):
    partition = dt[f"syscal_partition_{spw:03}"]
    antenna = partition["antenna_xds"].ds
    syscal = partition["system_calibration_xds"].to_dataset(inherit=False)

    # SYSCAL samples every second timestep, overlapping
    # the correlated data time span are selected
    samples = np.arange(3)
    assert dict(syscal.sizes) == {
      "antenna_name": NANTENNA,
      "time_system_cal": samples.size,
      "receptor_label": NRECEPTOR,
      "frequency_system_cal": NCHAN,
    }
    assert syscal.attrs["type"] == "system_calibration"
    npt.assert_array_equal(syscal.antenna_name, antenna.antenna_name)
    npt.assert_array_equal(syscal.station_name, antenna.station_name)
    npt.assert_array_equal(syscal.polarization_type, antenna.polarization_type)
    npt.assert_array_equal(syscal.frequency_system_cal, np.arange(NCHAN))
    npt.assert_allclose(
      syscal.time_system_cal,
      EpochCoder.decode_array(FIRST_FEB_2023_MJDS + samples * 2 * DUMP_RATE),
    )
    assert syscal.time_system_cal.attrs["type"] == "time"

    # TANT_SPECTRUM is undefined, other columns are not written
    assert set(syscal.data_vars) == {"TSYS", "TRX", "PHASE_DIFFERENCE"}

    base = expected_base(spw, samples)
    receptor = np.arange(NRECEPTOR) * 0.1
    chan = np.arange(NCHAN) * 0.01

    # (chan, receptor) spectra are presented as (receptor, chan)
    tsys = syscal.TSYS
    assert tsys.dims == (
      "antenna_name",
      "time_system_cal",
      "receptor_label",
      "frequency_system_cal",
    )
    assert tsys.dtype == np.float64
    assert tsys.attrs == {"type": "quantity", "units": "K"}
    expected = base[:, :, None, None] + receptor[:, None] + chan[None, :]
    npt.assert_allclose(tsys.values, expected, rtol=1e-6)

    npt.assert_allclose(
      syscal.TRX.values, base[:, :, None] + receptor[None, None, :], rtol=1e-6
    )
    npt.assert_allclose(syscal.PHASE_DIFFERENCE.values, base, rtol=1e-6)
    assert syscal.PHASE_DIFFERENCE.attrs["units"] == "rad"


@pytest.mark.parametrize("simmed_ms", [{**SIM_PARAMS}], indirect=True)
def test_system_calibration_interpolated(simmed_ms):
  dt = xarray.open_datatree(simmed_ms, interpolate="sys_cal")
  partition = dt["syscal_partition_001"]
  syscal = partition["system_calibration_xds"].to_dataset(inherit=False)

  assert syscal.TRX.dims == ("antenna_name", "time", "receptor_label")
  assert syscal.sizes["time"] == NTIME
  # SYSCAL samples are valid from (sample - 0.5) * 2 * DUMP_RATE.
  # Each timestep takes the most recently valid sample
  samples = np.array([0, 1, 1, 2, 2])
  npt.assert_allclose(syscal.PHASE_DIFFERENCE.values, expected_base(1, samples))


@pytest.mark.parametrize("simmed_ms", [{"name": "no_syscal.ms"}], indirect=True)
def test_system_calibration_absent(simmed_ms):
  dt = xarray.open_datatree(simmed_ms)
  assert not any(path.endswith("system_calibration_xds") for path in dt.groups)


@pytest.mark.parametrize("simmed_ms", [{**SIM_PARAMS}], indirect=True)
def test_system_calibration_outer_indexing(simmed_ms):
  """Array selections along both grid dimensions are outer selections"""
  dt = xarray.open_datatree(simmed_ms)
  syscal = dt["syscal_partition_000/system_calibration_xds"].to_dataset(inherit=False)
  selection = {"antenna_name": [0, 2], "time_system_cal": [0, 1, 2]}
  expected = syscal.TSYS.load().isel(selection)
  actual = xarray.open_datatree(simmed_ms)[
    "syscal_partition_000/system_calibration_xds"
  ].TSYS.isel(selection)
  assert actual.shape == (2, 3, NRECEPTOR, NCHAN)
  npt.assert_array_equal(actual.values, expected.values)
