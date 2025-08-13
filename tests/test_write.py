import arcae
import numpy as np
import xarray
from xarray import Dataset, DataTree

from xarray_ms.backend.msv2.writes import dataset_to_msv2, datatree_to_msv2
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


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


def test_store_region(monkeypatch, simmed_ms):
  monkeypatch.setattr(Dataset, "to_msv2", dataset_to_msv2, raising=False)
  monkeypatch.setattr(DataTree, "to_msv2", datatree_to_msv2, raising=False)

  region = {"time": slice(0, 2), "frequency": slice(2, 4)}

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    # Overwrite UVW coordinates with zeroes
    # Add a CORRECTED column
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        ds = node.ds.assign(CORRECTED=xarray.full_like(node.VISIBILITY, 1 + 2j))
        xdt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    xdt.to_msv2(["UVW", "CORRECTED"], compute=False)

    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        sizes = node.sizes
        # Slice out the region
        ds = node.ds.isel(**region)
        # Now write it out
        ds.to_msv2(["UVW", "CORRECTED"], compute=False, region=region)

  # But we can check that CORRECTED has been written correctly
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
