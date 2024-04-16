import arcae
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from xarray_ms.msv2 import partition


# @pytest.mark.skip
@pytest.mark.parametrize(
  "ms",
  [
    "/home/simon/data/HLTau_B6cont.calavg.tav300s",
    # "/home/simon/data/C147_unflagged.MS"
    # "/home/simon/data/WSRT_polar.MS_p0"
  ],
)
def test_partition(ms):
  pmap = partition(ms)  # noqa


def test_partioning(partitioned_ms):
  with arcae.table(partitioned_ms) as T:
    assert_array_equal(T.getcol("TIME"), np.arange(10, dtype=np.float64))
    assert T.nrow() == 10


@pytest.mark.parametrize("na", [7])
def test_lexical_binary_search(na):
  time = np.arange(20.0, dtype=np.float64)[:, None]
  ant1, ant2 = (a.astype(np.int32)[None, :] for a in np.triu_indices(na, 1))
  named_arrays = [("time", time), ("antenna1", ant1), ("antenna2", ant2)]
  names, arrays = zip(*named_arrays)
  arrays = tuple(a.ravel() for a in np.broadcast_arrays(*arrays))
  structured_dtype = np.dtype([(n, a.dtype) for n, a in zip(names, arrays)])

  carray = np.zeros(arrays[0].size, structured_dtype)
  for n, a in zip(names, arrays):
    carray[n] = a

  sarray = np.zeros(3, structured_dtype)
  sarray["time"] = [1.0, 3.0, 5.0]
  sarray["antenna1"] = [1, 5, 7]
  sarray["antenna2"] = [2, 8, 9]

  idx = np.searchsorted(carray, sarray)

  print(carray[idx] == sarray, carray[idx])
