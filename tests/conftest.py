import numpy as np
import pytest
from arcae.lib.arrow_tables import Table, ms_descriptor

from xarray_ms.backend.msv2.structure import MSv2StructureFactory
from xarray_ms.multiton import Multiton
from xarray_ms.testing.simulator import DEFAULT_SIM_PARAMS, MSStructureSimulator


@pytest.fixture(autouse=True)
def clear_caches():
  yield
  Multiton._INSTANCE_CACHE.clear()
  MSv2StructureFactory._STRUCTURE_CACHE.clear()


@pytest.fixture(scope="function", params=[DEFAULT_SIM_PARAMS])
def simmed_ms(request, tmp_path_factory):
  params = request.param.copy()
  ms = tmp_path_factory.mktemp("simulated") / params.pop("name", "test.ms")
  simulator = MSStructureSimulator(**{**DEFAULT_SIM_PARAMS, **params})
  simulator.simulate_ms(str(ms))
  return str(ms)


@pytest.fixture
def partitioned_ms(tmp_path):
  name = str(tmp_path / "partitioned.ms")

  table_desc = ms_descriptor("MAIN")

  with Table.ms_from_descriptor(name, "MAIN", table_desc) as T:
    T.addrows(10)
    T.putcol("TIME", np.arange(10, dtype=np.float64))

  return name
