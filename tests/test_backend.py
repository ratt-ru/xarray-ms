import pytest

from xarray_ms.backend.msv2.entrypoint import MSv2PartitionEntryPoint


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "proxy.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
    }
  ],
  indirect=True,
)
def test_msv2_backend(simmed_ms):
  entrypoint = MSv2PartitionEntryPoint()
  assert entrypoint.guess_can_open(simmed_ms) is True

  # ds = entrypoint.open_dataset(simmed_ms, partition=None)

  import xarray

  ds = xarray.open_dataset(simmed_ms)
  ds.compute()
  # breakpoint()
  None
