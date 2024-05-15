import pickle

import pytest
from arcae.lib.arrow_tables import Table

from xarray_ms.backend.msv2.table_factory import TableFactory


@pytest.fixture
def clear_table_cache():
  yield
  TableFactory._TABLE_CACHE.clear()


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
def test_table_factory_duplicate_args(simmed_ms, clear_table_cache):
  """Tests that when opened with duplicate arguments,
  the same TableFactory is returned"""

  factory = TableFactory(Table.from_filename, simmed_ms, readonly=False)
  data = factory().getcol("DATA")
  assert data.shape == (30, 8, 4)
  assert len(factory._TABLE_CACHE) == 1
  assert set(factory._TABLE_CACHE.keys()) == {factory}

  factory2 = TableFactory(Table.from_filename, simmed_ms, readonly=False)
  assert factory2() is factory()
  assert len(factory._TABLE_CACHE) == 1
  assert set(factory._TABLE_CACHE.keys()) == {factory}

  factory3 = TableFactory(Table.from_filename, simmed_ms)
  assert factory3() is not factory()
  assert factory3() is not factory2()
  assert len(factory._TABLE_CACHE) == 2
  assert set(factory._TABLE_CACHE.keys()) == {factory, factory3}

  # Check the positional args passed as kwargs are normalised
  # as positional args
  factory4 = TableFactory(Table.from_filename, filename=simmed_ms, readonly=False)
  assert factory4() is factory()
  assert factory4() is factory2()
  assert factory4() is not factory3()
  assert len(factory._TABLE_CACHE) == 2
  assert set(factory._TABLE_CACHE.keys()) == {factory, factory3}


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
def test_table_factory_pickling(simmed_ms, clear_table_cache):
  """Test that a pickle roundtrip of the table proxy results in the same object"""

  factory = TableFactory(Table.from_filename, simmed_ms)
  data = factory().getcol("DATA")
  assert data.shape == (30, 8, 4)
  assert set(factory._TABLE_CACHE.keys()) == {factory}

  factory2 = pickle.loads(pickle.dumps(factory))
  assert factory() is factory2()
  assert set(factory._TABLE_CACHE.keys()) == {factory}
