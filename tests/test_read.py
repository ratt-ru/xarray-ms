from contextlib import nullcontext
from functools import reduce
from operator import mul

import arcae
import numpy as np
import pytest
import xarray
from numpy.testing import assert_array_equal

from xarray_ms.errors import (
  ColumnShapeImputationWarning,
  IrregularBaselineGridWarning,
  IrregularChannelGridWarning,
  IrregularTimeGridWarning,
)


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "auto_corrs": True,
    }
  ],
  indirect=True,
)
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_regular_read(simmed_ms, auto_corrs):
  """Test for ramp function values produced by simulator"""
  xdt = xarray.open_datatree(simmed_ms, auto_corrs=auto_corrs)

  def _selection(n):
    a1 = n.baseline_antenna1_name.values
    a2 = n.baseline_antenna2_name.values
    dtype = [("a1", a1.dtype), ("a2", a2.dtype)]
    ds_baselines = np.rec.fromarrays([a1, a2], dtype)
    ant_names = node["antenna_xds"].antenna_name.values
    # Baselines containing auto_corrs
    ant1i, ant2i = np.triu_indices(len(ant_names), 0)
    all_baselines = np.rec.fromarrays([ant_names[ant1i], ant_names[ant2i]], dtype)
    bl_index = np.isin(all_baselines, ds_baselines)
    shape = (n.time.size, ant1i.size, n.frequency.size, n.polarization.size)
    return bl_index, shape

  for p in ["000", "001"]:
    node = xdt[f"backend_partition_{p}"]
    all_bl, vis_shape = _selection(node)

    # Produce expected full resolution values
    nelements = reduce(mul, vis_shape, 1)
    expected = np.arange(nelements, dtype=np.float64)
    expected = (expected + expected * 1j).reshape(vis_shape)
    assert_array_equal(node.VISIBILITY, expected[:, all_bl])

    uvw_shape = vis_shape[:2] + (3,)
    nelements = reduce(mul, uvw_shape, 1)
    expected = np.arange(nelements, dtype=np.float64).reshape(uvw_shape)
    assert_array_equal(node.UVW, expected[:, all_bl])


# Baseline subset present
ANT1_SUBSET = [0, 1, 2]
ANT2_SUBSET = [2, 2, 3]


def _excise_some_baselines(chunk_desc, data_dict):
  _, ant1 = data_dict["ANTENNA1"]
  _, ant2 = data_dict["ANTENNA2"]
  dtype = [("a1", ant1.dtype), ("a2", ant2.dtype)]
  baselines = np.rec.fromarrays([ant1, ant2], dtype)
  desired = np.rec.fromarrays([ANT1_SUBSET, ANT2_SUBSET], dtype)
  index = np.isin(baselines, desired)
  return {k: (d, v[index]) for k, (d, v) in data_dict.items()}


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": 4,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_data": _excise_some_baselines,
      "auto_corrs": True,
    }
  ],
  indirect=True,
)
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_irregular_read(simmed_ms, auto_corrs):
  """Test that excluding baselines works"""
  with pytest.warns(IrregularBaselineGridWarning, match="rows missing from the full"):
    xdt = xarray.open_datatree(simmed_ms, auto_corrs=auto_corrs)

  def _selection(n):
    a1 = n.baseline_antenna1_name.values
    a2 = n.baseline_antenna2_name.values
    dtype = [("a1", a1.dtype), ("a2", a2.dtype)]
    ds_baselines = np.rec.fromarrays([a1, a2], dtype)
    ant_names = node["antenna_xds"].antenna_name.values
    # All baselines, including autocorrs
    ant1i, ant2i = np.triu_indices(len(ant_names), 0)
    all_baselines = np.rec.fromarrays([ant_names[ant1i], ant_names[ant2i]], dtype)
    present_baselines = np.rec.fromarrays(
      [
        [f"ANTENNA-{i}" for i in ANT1_SUBSET],
        [f"ANTENNA-{i}" for i in ANT2_SUBSET],
      ],
      dtype,
    )
    # Selects out baselines from the full resolution data that are
    #   1. in the possibly lower resolution dataset
    #   2. present in the the data
    full_bl_index = np.logical_and(
      np.isin(all_baselines, ds_baselines), np.isin(all_baselines, present_baselines)
    )
    # Selects out baselines from the possibly lower resolution dataset
    # that are present in the data
    ds_bl_index = np.isin(ds_baselines, present_baselines)
    shape = (n.time.size, ant1i.size, n.frequency.size, n.polarization.size)
    return full_bl_index, ds_bl_index, shape

  for p in ["000", "001"]:
    node = xdt[f"backend_partition_{p}"]

    all_bl, ds_bl, full_vis_shape = _selection(node)

    # Selected baseline elements are as expected
    nelements = reduce(mul, full_vis_shape, 1)
    expected = np.arange(nelements, dtype=np.float32)
    expected = (expected + expected * 1j).reshape(full_vis_shape)
    assert_array_equal(node.VISIBILITY[:, ds_bl], expected[:, all_bl])
    # Other baseline elements are nan
    assert np.all(np.isnan((node.VISIBILITY[:, ~ds_bl])))

    full_uvw_shape = full_vis_shape[:2] + (3,)
    # Selected baseline elements are as expected
    nelements = reduce(mul, full_uvw_shape, 1)
    expected = np.arange(nelements, dtype=np.float64).reshape(full_uvw_shape)
    assert_array_equal(node.UVW[:, ds_bl], expected[:, all_bl])
    # Other baseline elements are nan
    assert np.all(np.isnan((node.UVW[:, ~ds_bl, ...])))

    flag = node.FLAG.values
    # Selected baseline elements are as expected
    nelements = reduce(mul, full_vis_shape, 1)
    expected = np.where(np.arange(nelements) & 0x1, 0, 1)
    expected = expected.reshape(full_vis_shape)
    assert_array_equal(flag[:, ds_bl], expected[:, all_bl])
    # Other baseline elements are flagged
    assert np.all(flag[:, ~ds_bl, ...] == 1)


def _randomise_trailing_intervals(chunk_desc, data_dict):
  _, interval = data_dict["INTERVAL"]
  nbl = chunk_desc.ANTENNA1.size
  ntime = chunk_desc.TIME.size

  interval.reshape((ntime, nbl), copy=False)[-1, :] = np.random.random(nbl)
  return data_dict


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": 3,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_data": _randomise_trailing_intervals,
    }
  ],
  indirect=True,
)
def test_differing_trailing_intervals(simmed_ms):
  """Test that differing interval values in the trailing timestep are ignored"""
  xdt = xarray.open_datatree(simmed_ms, auto_corrs=True)

  for p in ["000", "001"]:
    node = xdt[f"backend_partition_{p}"]
    assert node.time.integration_time["data"] == 8.0


def _randomise_starting_intervals(chunk_desc, data_dict):
  _, interval = data_dict["INTERVAL"]
  nbl = chunk_desc.ANTENNA1.size
  ntime = chunk_desc.TIME.size

  interval.reshape((ntime, nbl), copy=False)[0, :] = np.random.random(nbl)
  return data_dict


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": 3,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_data": _randomise_starting_intervals,
    }
  ],
  indirect=True,
)
def test_differing_start_intervals(simmed_ms):
  """Test that starting differing interval values result in nan intervals"""
  with pytest.warns(IrregularTimeGridWarning, match="Multiple intervals"):
    xdt = xarray.open_datatree(simmed_ms, auto_corrs=True)

  for p in ["000", "001"]:
    node = xdt[f"backend_partition_{p}"]
    assert np.isnan(node.time.integration_time["data"])
    assert "TIME" in node.data_vars
    assert "INTEGRATION_TIME" in node.data_vars


DUMP_RATE = 8.0


def _jitter_intervals(chunk_desc, data_dict):
  """Jitter the INTERVAL column"""
  interval = data_dict["INTERVAL"][-1]
  assert np.all(interval == DUMP_RATE)
  rng = np.random.default_rng()
  interval[:] += rng.standard_normal(interval.shape, interval.dtype) * 1e-6
  return data_dict


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "dump_rate": DUMP_RATE,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_data": _jitter_intervals,
    }
  ],
  indirect=True,
)
def test_jittered_intervals(simmed_ms):
  """Test that intervals with very small differences results
  in the use of their mean as the integration time"""
  xdt = xarray.open_datatree(simmed_ms, auto_corrs=True)

  for p in ["000", "001"]:
    node = xdt[f"backend_partition_{p}"]
    assert np.isclose(node.time.integration_time["data"], DUMP_RATE)


def _remove_weight_spectrum_add_weight(chunk_desc, data_dict):
  data_dict = data_dict.copy()
  del data_dict["WEIGHT_SPECTRUM"]

  ddid = chunk_desc.DATA_DESC_ID.item()
  _, ncorr = map(len, chunk_desc.data_description[ddid])
  nrow = data_dict["DATA_DESC_ID"][-1].size

  # Generate a ramp function in the WEIGHT column
  shape = (nrow, ncorr)
  weight = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
  data_dict["WEIGHT"] = ("row", weight)

  return data_dict


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "table_desc": {"__remove_columns__": ["WEIGHT_SPECTRUM"]},
      "transform_data": _remove_weight_spectrum_add_weight,
    }
  ],
  indirect=True,
)
def test_low_resolution_read(simmed_ms):
  """Test that a missing WEIGHT_SPECTRUM results in a broadcasted WEIGHT column"""
  dt = xarray.open_datatree(simmed_ms, auto_corrs=True)

  for p in ["000", "001"]:
    node = dt[f"backend_partition_{p}"]
    ntime, nbl, nfreq, npol = (
      node.sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    values = np.arange(np.prod((ntime, nbl, npol)), dtype=np.float64).reshape(
      ntime, nbl, 1, npol
    )
    assert_array_equal(
      np.broadcast_to(values, (ntime, nbl, nfreq, npol)), node.WEIGHT.values
    )


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": 7,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "table_desc": {"__remove_columns__": ["WEIGHT_SPECTRUM"]},
      "transform_data": _remove_weight_spectrum_add_weight,
    }
  ],
  indirect=True,
)
def test_isel_loads_on_subsets(simmed_ms):
  """Test that isels on a broadcasted WEIGHT column"""
  dt = xarray.open_datatree(simmed_ms, auto_corrs=True)
  assert len(dt.children) == 2

  p0_sel = {"time": slice(1, 2), "baseline_id": [0, 2], "frequency": slice(4, 6)}
  # Second partition only has 4 frequencies, tests slicing outside the range
  p1_sel = {"time": [0, 2], "baseline_id": slice(2, 5), "frequency": slice(4, 6)}

  node0 = dt["backend_partition_000"].isel(**p0_sel)
  node0.load()
  assert node0.sizes == {
    "time": 1,
    "baseline_id": 2,
    "frequency": 2,
    "polarization": 4,
    "uvw_label": 3,
  }

  node1 = dt["backend_partition_001"].isel(**p1_sel)
  node1.load()
  assert node1.VISIBILITY.shape == (2, 3, 0, 2)
  assert node1.sizes == {
    "time": 2,
    "baseline_id": 3,
    "frequency": 0,
    "polarization": 2,
    "uvw_label": 3,
  }

  dt.load()
  assert dt["backend_partition_000"].isel(**p0_sel).identical(node0)
  assert dt["backend_partition_001"].isel(**p1_sel).identical(node1)


def test_isel_loads_on_integers(simmed_ms):
  """Tests isel using integer coordinates"""
  dt = xarray.open_datatree(simmed_ms)
  assert len(dt.children) == 1
  node = dt["test_partition_000"].isel(
    time=slice(1, 2), baseline_id=1, frequency=slice(4, 6)
  )
  node.load()


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"])],
    }
  ],
  indirect=True,
)
@pytest.mark.parametrize("end", [True, False])
def test_irregular_chan_width(simmed_ms, end):
  """Warn about irregular channel widths, if the the first
  N-1 CHAN_WIDTH values differ"""
  # Bump the first channel width in each spectral window
  with arcae.table(f"{simmed_ms}::SPECTRAL_WINDOW") as spw:
    for r in range(spw.nrow()):
      chan_width = spw.getcol("CHAN_WIDTH", index=(slice(r, r + 1),)).copy()
      chan_width[0][-1 if end else 0] += 1e5
      spw.putcol("CHAN_WIDTH", chan_width, index=(slice(r, r + 1),))

  with nullcontext() if end else pytest.warns(IrregularChannelGridWarning):
    with xarray.open_datatree(simmed_ms):
      pass


NONSTANDARD_TABLE_DESC = {
  "CORRELATED_DATA": {
    "_c_order": True,
    "comment": "CORRELATED_DATA column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    # 'shape': ...,  # Variably shaped
    "valueType": "COMPLEX",
  },
  "CORRELATED_WEIGHT_SPECTRUM": {
    "_c_order": True,
    "comment": "CORRELATED_WEIGHT_SPECTRUM column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    # 'shape': ...,  # Variably shaped
    "valueType": "FLOAT",
  },
}


def _add_non_standard_columns(chunk_desc, data_dict):
  data_dict["CORRELATED_DATA"] = data_dict["DATA"]
  return data_dict


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "additional_columns.ms",
      "transform_data": _add_non_standard_columns,
      "table_desc": NONSTANDARD_TABLE_DESC,
    }
  ],
  indirect=True,
)
def test_additional_columns(simmed_ms):
  """CORRELATED_DATA is fully populated with data,
  while CORRELATED_WEIGHT_SPECTRUM's rows are missing"""
  with pytest.warns(
    ColumnShapeImputationWarning,
    match="Ignoring secondary column CORRELATED_WEIGHT_SPECTRUM",
  ):
    with xarray.open_datatree(simmed_ms) as dt:
      dt.load()
      node = dt["additional_columns_partition_000"]
      assert node["CORRELATED_DATA"].dims == (
        "time",
        "baseline_id",
        "frequency",
        "polarization",
      )
      assert node["CORRELATED_DATA"].equals(node["VISIBILITY"])
