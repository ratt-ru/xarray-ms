from contextlib import ExitStack

import numpy as np
import pytest
import xarray
import xarray.testing as xt
from xarray.backends.api import open_datatree

from xarray_ms.backend.msv2.entrypoint import MSv2PartitionEntryPoint


def test_entrypoint(simmed_ms):
  # The entrypoint thinks it can open the MS
  entrypoint = MSv2PartitionEntryPoint()
  assert entrypoint.guess_can_open(simmed_ms) is True


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
def test_open_dataset(simmed_ms):
  distributed = pytest.importorskip("dask.distributed")
  Client = distributed.Client
  LocalCluster = distributed.LocalCluster

  # Works with xarray default load mechanism
  with ExitStack() as stack:
    mem_ds = stack.enter_context(xarray.open_dataset(simmed_ms))
    mem_ds.load()
    del mem_ds.attrs["creation_date"]
    assert isinstance(mem_ds.VISIBILITY.data, np.ndarray)

  chunks = {"time": 2, "frequency": 2}

  # Works with default dask scheduler
  with ExitStack() as stack:
    ds = stack.enter_context(xarray.open_dataset(simmed_ms, chunks=chunks))
    del ds.attrs["creation_date"]
    xt.assert_equal(ds, mem_ds)
    # assert ds.identical(mem_ds)

  # Works with a LocalCluster
  with ExitStack() as stack:
    cluster = stack.enter_context(LocalCluster(processes=True, n_workers=4))
    stack.enter_context(Client(cluster))
    ds = stack.enter_context(xarray.open_dataset(simmed_ms, chunks=chunks))
    del ds.attrs["creation_date"]
    xt.assert_equal(ds, mem_ds)
    xt.assert_identical(ds, mem_ds)


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
def test_open_datatree(simmed_ms):
  distributed = pytest.importorskip("dask.distributed")
  Client = distributed.Client
  LocalCluster = distributed.LocalCluster

  # Works with xarray default load mechanism
  with ExitStack() as stack:
    mem_dt = open_datatree(simmed_ms)
    mem_dt.load()
    for ds in mem_dt.values():
      del ds.attrs["creation_date"]
      assert isinstance(ds.VISIBILITY.data, np.ndarray)

  chunks = {"time": 2, "frequency": 2}

  # Works with default dask scheduler
  with ExitStack() as stack:
    dt = open_datatree(simmed_ms, chunks=chunks)
    for ds in dt.values():
      del ds.attrs["creation_date"]
    xt.assert_equal(dt, mem_dt)

  # Works with a LocalCluster
  with ExitStack() as stack:
    cluster = stack.enter_context(LocalCluster(processes=True, n_workers=4))
    stack.enter_context(Client(cluster))
    dt = open_datatree(simmed_ms, chunks=chunks)
    for ds in dt.values():
      del ds.attrs["creation_date"]
    xt.assert_equal(dt, mem_dt)
    xt.assert_identical(dt, mem_dt)

  with ExitStack() as stack:
    dt = open_datatree(
      simmed_ms,
      chunks={"D=0": {"time": 2, "baseline": 2}, "D=1": {"time": 3, "frequency": 2}},
    )
