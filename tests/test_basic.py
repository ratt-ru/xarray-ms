import numpy as np
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.skip(reason="https://github.com/numpy/numpy/issues/28190")
@pytest.mark.parametrize("na", [7])
def test_lexical_binary_search(na):
  rng = np.random.default_rng(seed=42)

  time = np.arange(20.0, dtype=np.float64)[:, None]
  ant1, ant2 = (a.astype(np.int32)[None, :] for a in np.triu_indices(na, 1))
  named_arrays = [("time", time), ("antenna1", ant1), ("antenna2", ant2)]
  names, arrays = zip(*named_arrays)
  arrays = tuple(a.ravel() for a in np.broadcast_arrays(*arrays))
  structured_dtype = np.dtype([(n, a.dtype) for n, a in zip(names, arrays)])

  carray = np.zeros(arrays[0].size, structured_dtype)
  for n, a in zip(names, arrays):
    carray[n] = a

  choice = rng.choice(np.arange(carray.size), 10)

  sarray = np.zeros(choice.size, structured_dtype)

  sarray["time"] = carray["time"][choice]
  sarray["antenna1"] = carray["antenna1"][choice]
  sarray["antenna2"] = carray["antenna2"][choice]

  idx = np.searchsorted(carray, sarray)
  assert_array_equal(carray[idx], sarray)
