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
def test_basic_read(simmed_ms):
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
