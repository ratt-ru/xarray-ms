from contextlib import ExitStack

import pytest
import xarray

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
  distributed = pytest.importorskip("dask.distributed")
  da = pytest.importorskip("dask.array")
  np = pytest.importorskip("numpy")

  Client = distributed.Client
  LocalCluster = distributed.LocalCluster

  entrypoint = MSv2PartitionEntryPoint()
  assert entrypoint.guess_can_open(simmed_ms) is True

  with ExitStack() as stack:
    mem_ds = stack.enter_context(xarray.open_dataset(simmed_ms))
    mem_ds.load()
    assert isinstance(mem_ds.VISIBILITY.data, np.ndarray)

  chunks = {"time": 2, "frequency": 2}

  with ExitStack() as stack:
    ds = stack.enter_context(xarray.open_dataset(simmed_ms, chunks=chunks))
    assert isinstance(ds.VISIBILITY.data, da.Array)

  with ExitStack() as stack:
    cluster = stack.enter_context(LocalCluster(processes=True, n_workers=4))
    stack.enter_context(Client(cluster))
    ds = stack.enter_context(xarray.open_dataset(simmed_ms, chunks=chunks))
    assert isinstance(ds.VISIBILITY.data, da.Array)
    assert ds.equals(mem_ds)
    assert ds.identical(mem_ds)
    None

  del mem_ds
  del ds
  import gc

  gc.collect()
