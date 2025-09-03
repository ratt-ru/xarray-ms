from contextlib import ExitStack

import arcae
import numpy as np
import pytest
import xarray
from xarray import Dataset, DataTree

from xarray_ms.backend.msv2.writes import dataset_to_msv2, datatree_to_msv2
from xarray_ms.errors import MismatchedWriteRegion
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


@pytest.mark.parametrize("simmed_ms", [{"name": "test_store.ms"}], indirect=True)
def test_store(monkeypatch, simmed_ms):
  monkeypatch.setattr(Dataset, "to_msv2", dataset_to_msv2, raising=False)
  monkeypatch.setattr(DataTree, "to_msv2", datatree_to_msv2, raising=False)

  read = written = False

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    # Overwrite UVW coordinates with zeroes
    # Add a CORRECTED column
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert not np.all(node.UVW == 0)
        node.UVW[:] = 0
        assert len(node.encoding) > 0
        ds = node.ds.assign(CORRECTED=xarray.full_like(node.VISIBILITY, 2 + 3j))
        xdt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    xdt.to_msv2(["UVW", "CORRECTED"])
    written = written or True

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert np.all(node.UVW == 0)
        # Non-standard columns aren't yet exposed
        # assert np.all(node.CORRECTED == 1 + 2j)
        read = read or True

  assert read
  assert written

  # But we can check that CORRECTED has been written correctly
  with arcae.table(simmed_ms) as T:
    np.testing.assert_array_equal(T.getcol("CORRECTED"), 2 + 3j)


@pytest.mark.parametrize("simmed_ms", [{"name": "test_store_region.ms"}], indirect=True)
def test_store_region(monkeypatch, simmed_ms):
  monkeypatch.setattr(Dataset, "to_msv2", dataset_to_msv2, raising=False)
  monkeypatch.setattr(DataTree, "to_msv2", datatree_to_msv2, raising=False)

  region = {"time": slice(0, 2), "frequency": slice(2, 4)}

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    # Add a CORRECTED column
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        ds = node.ds.assign(CORRECTED=xarray.zeros_like(node.VISIBILITY))
        xdt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    # Create the new MS columns
    xdt.to_msv2(["CORRECTED"], compute=False)

    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        sizes = node.sizes
        ds = ds.isel(**region)
        ds = ds.assign(CORRECTED=xarray.full_like(ds.CORRECTED, 1 + 2j))
        # Now write it out
        ds.to_msv2(["CORRECTED"], compute=False, region=region)

  # We can check that CORRECTED has been written correctly
  with arcae.table(simmed_ms) as T:
    corrected = T.getcol("CORRECTED")
    nt, nbl, nf, npol = (
      sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    corrected = corrected.reshape((nt, nbl, nf, npol))
    ts, fs = (region[d] for d in ("time", "frequency"))
    mask = np.full(corrected.shape, False, np.bool_)
    mask[ts, :, fs, :] = True
    np.testing.assert_array_equal(corrected[mask], 1 + 2j)
    np.testing.assert_array_equal(corrected[~mask], 0 + 0j)


@pytest.mark.parametrize("chunks", [{"time": 2, "frequency": 2}])
@pytest.mark.parametrize("simmed_ms", [{"name": "distributed-write.ms"}], indirect=True)
@pytest.mark.parametrize("nworkers", [4])
@pytest.mark.parametrize("processes", [True, False])
def test_distributed_write(simmed_ms, monkeypatch, processes, nworkers, chunks):
  monkeypatch.setattr(Dataset, "to_msv2", dataset_to_msv2, raising=False)
  monkeypatch.setattr(DataTree, "to_msv2", datatree_to_msv2, raising=False)
  da = pytest.importorskip("dask.array")
  distributed = pytest.importorskip("dask.distributed")
  Client = distributed.Client
  LocalCluster = distributed.LocalCluster

  with ExitStack() as stack:
    cluster = stack.enter_context(LocalCluster(processes=processes, n_workers=nworkers))
    stack.enter_context(Client(cluster))
    dt = stack.enter_context(
      xarray.open_datatree(simmed_ms, chunks=chunks, auto_corrs=True)
    )
    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        vis = node.VISIBILITY
        sizes = node.sizes
        corrected = da.arange(np.prod(vis.shape), dtype=np.int32)
        corrected = corrected.reshape(vis.shape).rechunk(vis.data.chunks)
        ds = node.ds.assign(CORRECTED=(vis.dims, corrected))
        dt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    # Create the new MS columns
    dt.to_msv2(["CORRECTED"], compute=False)

    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        print("Writing")
        node.ds.to_msv2(["CORRECTED"], compute=True)

  with arcae.table(simmed_ms) as T:
    corrected = T.getcol("CORRECTED")
    shape = tuple(
      sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    expected = np.arange(np.prod(vis.shape), dtype=np.int32)
    expected = expected.reshape((-1,) + shape[2:])
    np.testing.assert_array_equal(corrected, expected)


@pytest.mark.parametrize("simmed_ms", [{"name": "indexed-write.ms"}], indirect=True)
def test_indexed_write(monkeypatch, simmed_ms):
  """Check that we throw if we select a variable out with an integer index
  and then try write that sub-selection out"""
  monkeypatch.setattr(Dataset, "to_msv2", dataset_to_msv2, raising=False)
  monkeypatch.setattr(DataTree, "to_msv2", datatree_to_msv2, raising=False)
  dt = xarray.open_datatree(simmed_ms)
  assert len(dt.children) == 1

  for node in dt.subtree:
    if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
      ds = node.ds.assign(CORRECTED=xarray.full_like(node.VISIBILITY, 1 + 2j))
      dt[node.path] = DataTree(ds)

  dt.to_msv2(["CORRECTED"], compute=False)

  for node in dt.subtree:
    if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
      ds = node.ds.isel(time=slice(0, 2), baseline_id=slice(0, 2), frequency=1)
      with pytest.raises(MismatchedWriteRegion):
        ds.to_msv2(["CORRECTED"], compute=True)
