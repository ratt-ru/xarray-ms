import json

import pytest
import xarray
import xarray.testing as xt

# zarr-python stores Zarr v3 consolidated metadata inline in the root
# zarr.json, but this is not yet part of the Zarr v3 specification
# https://github.com/zarr-developers/zarr-specs/pull/309
# Fixed-length string dtypes also lack a Zarr v3 specification.
# The warnings are matched on their messages, which vary across
# zarr versions, as their categories are not available in all of them
pytestmark = [
  pytest.mark.filterwarnings(
    "ignore:Consolidated metadata is currently not part:UserWarning"
  ),
  # zarr >= 3.1
  pytest.mark.filterwarnings(
    "ignore:The data type .* does not have a Zarr V3 specification:FutureWarning"
  ),
  # zarr < 3.1
  pytest.mark.filterwarnings(
    "ignore:The (codec|dtype) .* is currently not part in the Zarr format 3"
    " specification:UserWarning"
  ),
]


def assert_consolidated_v3(zarr_path):
  with open(zarr_path / "zarr.json") as f:
    metadata = json.load(f)

  assert metadata["zarr_format"] == 3
  consolidated = metadata["consolidated_metadata"]
  assert consolidated["kind"] == "inline"
  assert len(consolidated["metadata"]) > 0


def test_dataset_roundtrip(simmed_ms, tmp_path):
  ds = xarray.open_dataset(simmed_ms)
  zarr_path = tmp_path / "test_dataset.zarr"
  ds.to_zarr(zarr_path, compute=True, consolidated=True, zarr_format=3)
  assert_consolidated_v3(zarr_path)
  ds2 = xarray.open_dataset(zarr_path, consolidated=True)
  xt.assert_identical(ds, ds2)


def test_datatree_roundtrip(simmed_ms, tmp_path):
  dt = xarray.open_datatree(simmed_ms)
  zarr_path = tmp_path / "test_datatree.zarr"
  dt.to_zarr(zarr_path, compute=True, consolidated=True, zarr_format=3)
  assert_consolidated_v3(zarr_path)
  # TODO Remove forcing of engine once
  # https://github.com/pydata/xarray/issues/10808
  # is resolved
  dt2 = xarray.open_datatree(zarr_path, engine="zarr", consolidated=True)
  xt.assert_identical(dt, dt2)
