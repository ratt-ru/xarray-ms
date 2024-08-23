import numpy as np

from xarray_ms.utils import FrozenKey


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
