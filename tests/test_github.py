import numpy as np
import pytest
import xarray
from arcae.lib.arrow_tables import Table
from numpy.testing import assert_array_equal


def test_gh157(simmed_ms):
  """Tests that #157 is fixed

  ``file://`` URIs are resolved to local filesystem paths.

  https://github.com/ratt-ru/xarray-ms/issues/157
  """
  reference = xarray.open_datatree(simmed_ms)

  for uri in (f"file://{simmed_ms}", f"file://localhost{simmed_ms}"):
    dt = xarray.open_datatree(uri)
    assert dt.children.keys() == reference.children.keys()

    ds = xarray.open_dataset(uri)
    assert ds.sizes == reference[next(iter(reference.children))].ds.sizes


def test_gh116(simmed_ms):
  """Tests that #116 is fixed

  https://github.com/ratt-ru/xarray-ms/issues/116
  https://github.com/ratt-ru/xarray-ms/pull/117
  """
  dt = xarray.open_datatree(
    simmed_ms, partition_schema=["SCAN_NUMBER", "FIELD_ID"], auto_corrs=True
  )

  for k in dt.children:
    # Get a dataset from the datatree.
    ds = dt[k].ds

    # Select out the XY correlation.
    ds = ds.sel(polarization="XY")

    # Take the mean along the time axis for a given scan.
    # This raises an attribute error at present. The error goes away
    # if the prior selection operation is done after taking the mean.
    ds = ds.mean("time")


def _store_02_as_20(chunk_desc, data_dict):
  a1 = data_dict["ANTENNA1"][1].copy()
  a2 = data_dict["ANTENNA2"][1].copy()
  mask = (a1 == 0) & (a2 == 2)
  a1[mask], a2[mask] = 2, 0
  data_dict["ANTENNA1"] = (data_dict["ANTENNA1"][0], a1)
  data_dict["ANTENNA2"] = (data_dict["ANTENNA2"][0], a2)
  return data_dict


@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "gh199.ms", "transform_data": _store_02_as_20}],
  indirect=True,
)
def test_gh199(simmed_ms):
  """Tests that #199 is fixed

  Baselines stored as ``ANTENNA1 > ANTENNA2`` are labelled
  in their stored orientation, matching their data.

  https://github.com/ratt-ru/xarray-ms/issues/199
  """
  with Table.from_filename(simmed_ms) as T:
    a1, a2 = T.getcol("ANTENNA1"), T.getcol("ANTENNA2")
    time = T.getcol("TIME")
    (rows,) = np.where((a1 == 2) & (a2 == 0) & (time == time.min()))
    (r,) = rows
    data = T.getcol("DATA", index=(slice(r, r + 1),))[0]
    uvw = T.getcol("UVW", index=(slice(r, r + 1),))[0]

  dt = xarray.open_datatree(simmed_ms)
  ds = dt[next(iter(dt.children))].ds
  n1 = ds.baseline_antenna1_name.values
  n2 = ds.baseline_antenna2_name.values
  (k,) = np.where(
    ((n1 == "ANTENNA-0") & (n2 == "ANTENNA-2"))
    | ((n1 == "ANTENNA-2") & (n2 == "ANTENNA-0"))
  )[0]
  assert (n1[k], n2[k]) == ("ANTENNA-2", "ANTENNA-0")

  cell = ds.isel(baseline_id=k, time=0)
  assert_array_equal(cell.VISIBILITY.values, data)
  assert_array_equal(cell.UVW.values, uvw)
