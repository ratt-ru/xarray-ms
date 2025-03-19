from functools import reduce
from operator import mul

import numpy as np
import pytest
import xarray
from numpy.testing import assert_array_equal

from xarray_ms.errors import IrregularGridWarning


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
  with pytest.warns(IrregularGridWarning, match="rows missing from the full"):
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
    assert node.time.attrs["integration_time"] == 8.0


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
  with pytest.warns(IrregularGridWarning, match="Multiple intervals"):
    xdt = xarray.open_datatree(simmed_ms, auto_corrs=True)

  for p in ["000", "001"]:
    node = xdt[f"backend_partition_{p}"]
    assert np.isnan(node.time.attrs["integration_time"])
