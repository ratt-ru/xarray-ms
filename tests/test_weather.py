import numpy as np
import numpy.testing as npt
import pytest
import xarray

from xarray_ms.backend.msv2.measures_encoders import EpochCoder
from xarray_ms.testing.simulator import FIRST_FEB_2023_MJDS

NTIME = 5
NANTENNA = 3
DUMP_RATE = 8.0
DATA_DESCRIPTION = [(8, ["XX", "YY"])]


def sim_params(name, weather):
  return {
    "name": name,
    "ntime": NTIME,
    "nantenna": NANTENNA,
    "weather": weather,
    "data_description": DATA_DESCRIPTION,
  }


@pytest.mark.parametrize(
  "simmed_ms", [sim_params("weather_antenna.ms", "antenna")], indirect=True
)
def test_weather_antenna_stations(simmed_ms):
  """Weather stations identified by ANTENNA_ID are
  the partition antennas with WEATHER rows"""
  dt = xarray.open_datatree(simmed_ms)
  partition = dt["weather_antenna_partition_000"]
  antenna = partition["antenna_xds"].ds
  weather = partition["weather_xds"].to_dataset(inherit=False)

  # The last antenna has no WEATHER rows. Zero intervals are extended
  # to the 24s sampling cadence, so the sample at 48s, valid from 36s,
  # overlaps the end of the correlated data
  stations = NANTENNA - 1
  samples = np.arange(3)
  assert weather.attrs["type"] == "weather"
  assert weather.sizes["station_name"] == stations
  assert weather.sizes["time_weather"] == samples.size
  npt.assert_array_equal(weather.station_name, antenna.station_name[:stations])
  npt.assert_array_equal(
    weather.STATION_POSITION, antenna.ANTENNA_POSITION.values[:stations]
  )
  assert weather.STATION_POSITION.attrs == antenna.ANTENNA_POSITION.attrs
  npt.assert_allclose(
    weather.time_weather,
    EpochCoder.decode_array(FIRST_FEB_2023_MJDS + samples * 3 * DUMP_RATE),
  )
  assert weather.time_weather.attrs["type"] == "time_weather"

  # hPa are converted to Pa
  pressure = weather.PRESSURE
  assert pressure.dims == ("station_name", "time_weather")
  assert pressure.dtype == np.float64
  assert pressure.attrs == {"type": "quantity", "units": "Pa"}
  key = np.arange(stations)[:, None]
  npt.assert_allclose(pressure.values, (1000.0 + key + samples) * 100.0)
  npt.assert_allclose(weather.TEMPERATURE.values, 280.0 + key + samples)
  assert weather.TEMPERATURE.attrs["units"] == "K"


@pytest.mark.parametrize(
  "simmed_ms", [sim_params("weather_station.ms", "station")], indirect=True
)
@pytest.mark.parametrize("interpolate", [None, "weather"])
def test_weather_ns_wx_stations(simmed_ms, interpolate):
  """Weather stations identified by NS_WX_STATION_ID"""
  dt = xarray.open_datatree(simmed_ms, interpolate=interpolate)
  partition = dt["weather_station_partition_000"]
  weather = partition["weather_xds"].to_dataset(inherit=False)

  station_ids = np.array([5, 7])
  npt.assert_array_equal(weather.station_name, ["Station_5", "Station_7"])
  npt.assert_array_equal(
    weather.STATION_POSITION.values,
    np.stack([station_ids * 1.0, station_ids * 2.0, station_ids * 3.0], axis=-1),
  )
  assert weather.STATION_POSITION.attrs["frame"] == "ITRS"

  if interpolate:
    # Zero intervals are extended to the 24s sampling cadence, so samples are
    # valid from 12s before their time. Each timestep takes the most recent sample
    samples = np.array([0, 0, 1, 1, 1])
    assert weather.PRESSURE.dims == ("station_name", "time")
  else:
    samples = np.arange(3)
    assert weather.PRESSURE.dims == ("station_name", "time_weather")

  npt.assert_allclose(
    weather.PRESSURE.values, (1000.0 + station_ids[:, None] + samples) * 100.0
  )


@pytest.mark.parametrize("simmed_ms", [{"name": "no_weather.ms"}], indirect=True)
def test_weather_absent(simmed_ms):
  dt = xarray.open_datatree(simmed_ms)
  assert not any(path.endswith("weather_xds") for path in dt.groups)
