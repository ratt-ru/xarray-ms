import pytest
import xarray


@pytest.mark.skip
@pytest.mark.filterwarnings("ignore:.*?matched multiple partitions")
@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
    }
  ],
  indirect=True,
)
def test_accessors(simmed_ms):
  dt = xarray.open_datatree(simmed_ms)
  dt["/backend/partition_000"].attrs["links"] = {
    "antenna": "/backend/partition_000/ANTENNA",
    "weather": "/backend/partition_000/weather_xds",
  }
  # print(dt)
  print(dt["/backend/partition_000"].links.antenna)
