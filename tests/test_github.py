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
