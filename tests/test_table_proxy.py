import pickle

import pytest
from arcae.lib.arrow_tables import Table

from xarray_ms.backend.msv2.table_proxy import _TABLE_CACHE, TableProxy


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
def test_table_proxy_duplicate_args(simmed_ms):
  """Tests that when opened with duplicate arguments, the same TableProxy is returned"""

  def do_test(ms):
    proxy = TableProxy(Table.from_filename, ms, readonly=False)
    data = proxy.table.getcol("DATA")
    assert data.shape == (30, 8, 4)
    assert len(_TABLE_CACHE) == 1
    assert next(_TABLE_CACHE.values()) == proxy

    proxy2 = TableProxy(Table.from_filename, ms, readonly=False)
    assert proxy2 is proxy
    assert len(_TABLE_CACHE) == 1
    assert next(_TABLE_CACHE.values()) == proxy

    proxy3 = TableProxy(Table.from_filename, ms)
    assert proxy3 is not proxy
    assert proxy3 is not proxy2
    assert len(_TABLE_CACHE) == 2
    assert set(_TABLE_CACHE.values()) == {proxy, proxy3}

    # Check the positional args passed as kwargs are normalised
    # as positional args
    proxy4 = TableProxy(Table.from_filename, filename=ms, readonly=False)
    assert proxy4 is proxy
    assert proxy4 is proxy2
    assert proxy4 is not proxy3
    assert len(_TABLE_CACHE) == 2
    assert set(_TABLE_CACHE.values()) == {proxy, proxy3}

  do_test(simmed_ms)
  assert len(_TABLE_CACHE) == 0


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
def test_table_proxy_pickling(simmed_ms):
  """Test that a pickle roundtrip of the table proxy results in the same object"""

  def do_test(ms):
    proxy = TableProxy(Table.from_filename, ms)
    data = proxy.table.getcol("DATA")
    assert data.shape == (30, 8, 4)
    assert len(_TABLE_CACHE) == 1
    assert next(_TABLE_CACHE.values()) == proxy

    proxy2 = pickle.loads(pickle.dumps(proxy))
    assert proxy is proxy2
    assert len(_TABLE_CACHE) == 1

  do_test(simmed_ms)
  assert len(_TABLE_CACHE) == 0
