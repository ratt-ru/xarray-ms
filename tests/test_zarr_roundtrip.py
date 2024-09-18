import xarray.testing as xt
from xarray.backends.api import open_dataset, open_datatree


def test_dataset_roundtrip(simmed_ms, tmp_path):
  ds = open_dataset(simmed_ms)
  zarr_path = tmp_path / "test_dataset.zarr"
  ds.to_zarr(zarr_path, compute=True, consolidated=True)
  ds2 = open_dataset(zarr_path)
  xt.assert_identical(ds, ds2)


def test_datatree_roundtrip(simmed_ms, tmp_path):
  dt = open_datatree(simmed_ms)
  zarr_path = tmp_path / "test_datatree.zarr"
  dt.to_zarr(zarr_path, compute=True, consolidated=True)
  dt2 = open_datatree(zarr_path)
  xt.assert_identical(dt, dt2)
