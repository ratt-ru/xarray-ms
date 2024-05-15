import numpy as np
import pytest
from numpy.testing import assert_array_equal

from xarray_ms.utils import FrozenKey, baseline_id


def test_frozen_key():
  """FrozenKey tests"""
  assert FrozenKey([1, 2, 3]).frozen == ((1, 2, 3), frozenset())
  assert FrozenKey(a=[1, 2, 3]).frozen == (frozenset({("a", (1, 2, 3))}),)
  assert FrozenKey([1, 2], a=[1, 2, 3]).frozen == (
    (1, 2),
    frozenset({("a", (1, 2, 3))}),
  )
  assert FrozenKey(np.ones((2, 2))).frozen == (
    (
      b"\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"
      b"\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?",
      (2, 2),
      "d",
    ),
    frozenset(),
  )

  assert FrozenKey("abc", 2, a=np.ones(2, np.complex64)).frozen == (
    (
      "abc",
      2,
      frozenset(
        {
          (
            "a",
            (b"\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00\x00", (2,), "F"),
          )
        }
      ),
    )
  )


@pytest.mark.parametrize("na", [4, 7])
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_baseline_id(na, auto_corrs):
  ant1, ant2 = np.triu_indices(na, 0 if auto_corrs else 1)
  bl_id = baseline_id(ant1, ant2, na, auto_corrs)
  assert_array_equal(np.arange(ant1.size), bl_id)
