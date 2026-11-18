import xarray


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


def test_gh199(tmp_path):
  """Tests that #199 is fixed

  A baseline stored as ``ANTENNA1 > ANTENNA2`` keeps its upper-triangular
  labels, while its VISIBILITY is conjugated and its UVW negated.

  https://github.com/ratt-ru/xarray-ms/issues/199
  """
  import numpy as np

  from xarray_ms.testing.simulator import DEFAULT_SIM_PARAMS, MSStructureSimulator

  def store_reversed(chunk_desc, data_dict):
    (d1, a1), (d2, a2) = data_dict["ANTENNA1"], data_dict["ANTENNA2"]
    mask = (a1 == 0) & (a2 == 2)
    return {
      **data_dict,
      "ANTENNA1": (d1, np.where(mask, a2, a1)),
      "ANTENNA2": (d2, np.where(mask, a1, a2)),
    }

  def open_ms(name, **kw):
    ms = str(tmp_path / name)
    MSStructureSimulator(**{**DEFAULT_SIM_PARAMS, **kw}).simulate_ms(ms)
    ds = xarray.open_dataset(ms, engine="xarray-ms:msv2")
    n1, n2 = ds.baseline_antenna1_name.values, ds.baseline_antenna2_name.values
    (k,) = np.flatnonzero((n1 == "ANTENNA-0") & (n2 == "ANTENNA-2"))
    return ds.isel(baseline_id=k, time=0).load()

  stored = open_ms("stored.ms")
  cell = open_ms("reversed.ms", transform_data=store_reversed)

  # V_02 = V_20^H: cross-hands swapped and conjugated
  expected = np.conj(stored.VISIBILITY.values[..., [0, 2, 1, 3]])
  np.testing.assert_array_equal(cell.VISIBILITY.values, expected)
  np.testing.assert_array_equal(cell.UVW.values, -stored.UVW.values)
