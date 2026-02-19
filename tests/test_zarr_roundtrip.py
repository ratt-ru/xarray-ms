import xarray
import xarray.testing as xt


def test_dataset_roundtrip(simmed_ms, tmp_path):
  ds = xarray.open_dataset(simmed_ms)
  zarr_path = tmp_path / "test_dataset.zarr"
  ds.to_zarr(zarr_path, compute=True, consolidated=True)
  ds2 = xarray.open_dataset(zarr_path)
  xt.assert_identical(ds, ds2)


def test_datatree_roundtrip(simmed_ms, tmp_path):
  dt = xarray.open_datatree(simmed_ms)
  zarr_path = tmp_path / "test_datatree.zarr"
  dt.to_zarr(zarr_path, compute=True, consolidated=True)
  # TODO Remove forcing of engine once
  # https://github.com/pydata/xarray/issues/10808
  # is resolved
  dt2 = xarray.open_datatree(zarr_path, engine="zarr")
  xt.assert_identical(dt, dt2)
