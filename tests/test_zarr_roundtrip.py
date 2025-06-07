from contextlib import nullcontext

import pytest
import xarray.testing as xt
from xarray.backends.api import open_dataset, open_datatree

ZARR_V3_WARNINGS = (FutureWarning, UserWarning)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_dataset_roundtrip(simmed_ms, tmp_path, zarr_format):
  ds = open_dataset(simmed_ms)
  zarr_path = tmp_path / "test_dataset.zarr"
  with pytest.warns(ZARR_V3_WARNINGS) if zarr_format == 3 else nullcontext():
    ds.to_zarr(zarr_path, compute=True, consolidated=True, zarr_format=zarr_format)
    ds2 = open_dataset(zarr_path)
  xt.assert_identical(ds, ds2)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_datatree_roundtrip(simmed_ms, tmp_path, zarr_format):
  dt = open_datatree(simmed_ms)
  zarr_path = tmp_path / "test_datatree.zarr"
  with pytest.warns(ZARR_V3_WARNINGS) if zarr_format == 3 else nullcontext():
    dt.to_zarr(zarr_path, compute=True, consolidated=True, zarr_format=zarr_format)
    dt2 = open_datatree(zarr_path)
  xt.assert_identical(dt, dt2)
