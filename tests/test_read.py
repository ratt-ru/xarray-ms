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
    }
  ],
  indirect=True,
)
def test_regular_read(simmed_ms):
  """Test for ramp function values produced by simulator"""
  xdt = xarray.open_datatree(simmed_ms)

  for p in ["000", "001"]:
    node = xdt[f"backend/partition_{p}"]
    vis = node.VISIBILITY.values
    nelements = reduce(mul, vis.shape, 1)
    expected = np.arange(nelements, dtype=np.float64)
    expected = (expected + expected * 1j).reshape(vis.shape)
    assert_array_equal(vis, expected)

    uvw = node.UVW.values
    nelements = reduce(mul, uvw.shape, 1)
    expected = np.arange(nelements, dtype=np.float64).reshape(uvw.shape)
    assert_array_equal(uvw, expected)


ANT1_SUBSET = [0, 0, 1]
ANT2_SUBSET = [0, 1, 2]


def _select_baseline_rows(antenna1, antenna2, ant1_subset, ant2_subset):
  dtype = [("a1", antenna1.dtype), ("a2", antenna2.dtype)]
  baselines = np.rec.fromarrays([antenna1, antenna2], dtype=dtype)
  desired = np.rec.fromarrays([ant1_subset, ant2_subset], dtype=dtype)
  return np.isin(baselines, desired)


def _excise_some_baselines(data_dict):
  _, ant1 = data_dict["ANTENNA1"]
  _, ant2 = data_dict["ANTENNA2"]
  index = _select_baseline_rows(ant1, ant2, ANT1_SUBSET, ANT2_SUBSET)
  return {k: (d, v[index]) for k, (d, v) in data_dict.items()}


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": 3,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_data": _excise_some_baselines,
    }
  ],
  indirect=True,
)
def test_irregular_read(simmed_ms):
  """Test that excluding baselines works"""
  with pytest.warns(IrregularGridWarning, match="rows missing from the full"):
    xdt = xarray.open_datatree(simmed_ms)

  for p in ["000", "001"]:
    node = xdt[f"backend/partition_{p}"]

    bl_index = _select_baseline_rows(
      node.baseline_antenna1_name.values,
      node.baseline_antenna2_name.values,
      [f"ANTENNA-{i}" for i in ANT1_SUBSET],
      [f"ANTENNA-{i}" for i in ANT2_SUBSET],
    )

    vis = node.VISIBILITY.values
    # Selected baseline elements are as expected
    nelements = reduce(mul, vis.shape, 1)
    expected = np.arange(nelements, dtype=np.float32)
    expected = (expected + expected * 1j).reshape(vis.shape)
    assert_array_equal(vis[:, bl_index], expected[:, bl_index])
    # Other baseline elements are nan
    vis = node.VISIBILITY.values
    assert np.all(np.isnan((vis[:, ~bl_index])))

    uvw = node.UVW.values
    # Selected baseline elements are as expected
    nelements = reduce(mul, uvw.shape, 1)
    expected = np.arange(nelements, dtype=np.float64).reshape(uvw.shape)
    assert_array_equal(uvw[:, bl_index], expected[:, bl_index])
    # Other baseline elements are nan
    assert np.all(np.isnan((uvw[:, ~bl_index, ...])))

    flag = node.FLAG.values
    # Selected baseline elements are as expected
    nelements = reduce(mul, flag.shape, 1)
    expected = np.where(np.arange(nelements) & 0x1, 0, 1)
    expected = expected.reshape(flag.shape)
    assert_array_equal(flag[:, bl_index], expected[:, bl_index])
    # Other baseline elements are flagged
    assert np.all(flag[:, ~bl_index, ...] == 1)


def _randomise_trailing_intervals(data_dict):
  _, ant1 = data_dict["ANTENNA1"]
  _, ant2 = data_dict["ANTENNA2"]
  _, interval = data_dict["INTERVAL"]

  ubl = np.unique(np.stack([ant1, ant2], axis=1), axis=0)
  shape = (-1, ubl.shape[0])
  interval.reshape(shape, copy=False)[-1, :] = np.random.random(ubl.shape[0])
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
  xdt = xarray.open_datatree(simmed_ms)

  for p in ["000", "001"]:
    node = xdt[f"backend/partition_{p}"]
    assert node.time.attrs["integration_time"] == 8.0


def _randomise_starting_intervals(data_dict):
  _, ant1 = data_dict["ANTENNA1"]
  _, ant2 = data_dict["ANTENNA2"]
  _, interval = data_dict["INTERVAL"]

  ubl = np.unique(np.stack([ant1, ant2], axis=1), axis=0)
  nbl, _ = ubl.shape
  interval.reshape((-1, nbl), copy=False)[0, :] = np.random.random(nbl)
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
    xdt = xarray.open_datatree(simmed_ms)

  for p in ["000", "001"]:
    node = xdt[f"backend/partition_{p}"]
    assert np.isnan(node.time.attrs["integration_time"])
