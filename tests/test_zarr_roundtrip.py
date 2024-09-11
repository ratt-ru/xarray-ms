import xarray.testing as xt
from xarray.backends.api import open_dataset, open_datatree

from xarray_ms import xds_from_zarr, xds_to_zarr, xdt_from_zarr, xdt_to_zarr


def test_dataset_roundtrip(simmed_ms, tmp_path):
  ds = open_dataset(simmed_ms)
  zarr_path = tmp_path / "test_dataset.zarr"
  xds_to_zarr(ds, zarr_path, compute=True, consolidated=True)
  ds2 = xds_from_zarr(zarr_path)
  xt.assert_identical(ds, ds2)


def test_datatree_roundtrip(simmed_ms, tmp_path):
  dt = open_datatree(simmed_ms)
  zarr_path = tmp_path / "test_datatree.zarr"
  xdt_to_zarr(dt, zarr_path, compute=True, consolidated=True)
  dt2 = xdt_from_zarr(zarr_path)
  xt.assert_identical(dt, dt2)
