import pickle

import xarray
from arcae.lib.arrow_tables import Table

from xarray_ms.multiton import Multiton


def test_multiton_duplicate_args(simmed_ms):
  """Tests that when opened with duplicate arguments,
  the same Multiton is returned"""

  multiton = Multiton(Table.from_filename, simmed_ms, readonly=False)
  data = multiton.instance.getcol("DATA")
  assert data.shape == (30, 8, 4)
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton}

  multiton2 = Multiton(Table.from_filename, simmed_ms, readonly=False)
  assert multiton2.instance is multiton.instance
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton}

  multiton3 = Multiton(Table.from_filename, simmed_ms)
  assert multiton3.instance is not multiton.instance
  assert multiton3.instance is not multiton2.instance
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton, multiton3}

  # Check the positional args passed as kwargs are normalised
  # as positional args
  multiton4 = Multiton(Table.from_filename, filename=simmed_ms, readonly=False)
  assert multiton4.instance is multiton.instance
  assert multiton4.instance is multiton2.instance
  assert multiton4.instance is not multiton3.instance
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton, multiton3}


def test_multiton_release():
  data = {"closed": False}

  class A:
    def close(self):
      data["closed"] = True

  multiton = Multiton(A)
  multiton.instance
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton}
  multiton.release()
  assert set(multiton._INSTANCE_CACHE.keys()) == set()
  assert data["closed"] is True


def test_multiton_datatree_release(simmed_ms):
  assert len(Multiton._INSTANCE_CACHE) == 0

  with xarray.open_datatree(simmed_ms):
    assert len(Multiton._INSTANCE_CACHE) > 0

  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_pickling(simmed_ms):
  """Test that a pickle roundtrip of the Multiton results in the same object"""

  multiton = Multiton(Table.from_filename, simmed_ms)
  data = multiton.instance.getcol("DATA")
  assert data.shape == (30, 8, 4)
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton}

  multiton2 = pickle.loads(pickle.dumps(multiton))
  assert multiton.instance is multiton2.instance
  assert set(multiton._INSTANCE_CACHE.keys()) == {multiton}
