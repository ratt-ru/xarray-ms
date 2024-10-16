from functools import reduce
from operator import mul

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from xarray.backends.api import open_datatree


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
  xdt = open_datatree(simmed_ms)

  for node in xdt.subtree:
    if "data_description_id" in node.attrs:
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


def _select_rows(antenna1, antenna2, ant1_subset, ant2_subset):
  dtype = [("a1", antenna1.dtype), ("a2", antenna2.dtype)]
  baselines = np.rec.fromarrays([antenna1, antenna2], dtype=dtype)
  desired = np.rec.fromarrays([ant1_subset, ant2_subset], dtype=dtype)
  return np.isin(baselines, desired)


def _excise_rows(data_dict):
  _, ant1 = data_dict["ANTENNA1"]
  _, ant2 = data_dict["ANTENNA2"]
  index = _select_rows(ant1, ant2, ANT1_SUBSET, ANT2_SUBSET)
  return {k: (d, v[index]) for k, (d, v) in data_dict.items()}


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": 3,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_data": _excise_rows,
    }
  ],
  indirect=True,
)
def test_irregular_read(simmed_ms):
  """Test that excluding baselines works"""
  xdt = open_datatree(simmed_ms)

  for node in xdt.subtree:
    if "data_description_id" in node.attrs:
      bl_index = _select_rows(
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
