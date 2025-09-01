import numpy as np
import pytest
from arcae.lib.arrow_tables import Table, ms_descriptor

from xarray_ms.backend.msv2.structure import MSv2StructureFactory
from xarray_ms.multiton import Multiton
from xarray_ms.testing.simulator import DEFAULT_SIM_PARAMS, MSStructureSimulator

MSV4_TEST_CORPUS = "msv4_test_corpus"
MSV4_TEST_CORPUS_CMDLINE = f"--{MSV4_TEST_CORPUS}"


def pytest_addoption(parser):
  parser.addoption(
    MSV4_TEST_CORPUS_CMDLINE,
    action="store_true",
    help="Run suite of tests on the MSv4 Test Corpus",
  )


def pytest_configure(config):
  config.addinivalue_line(
    "markers", f"{MSV4_TEST_CORPUS}: mark tests as part of the MSv4 Test Corpus suite"
  )


def pytest_collection_modifyitems(config, items):
  if config.getoption(MSV4_TEST_CORPUS_CMDLINE):
    return
  skip_msv4_corpus = pytest.mark.skip(
    reason=f"need {MSV4_TEST_CORPUS_CMDLINE} option to run"
  )
  for item in items:
    if MSV4_TEST_CORPUS in item.keywords:
      item.add_marker(skip_msv4_corpus)


@pytest.fixture(autouse=True)
def clear_caches():
  yield
  Multiton._INSTANCE_CACHE.clear()
  MSv2StructureFactory._STRUCTURE_CACHE.clear()


@pytest.fixture(scope="session", params=[DEFAULT_SIM_PARAMS])
def simmed_ms(request, tmp_path_factory):
  ms = tmp_path_factory.mktemp("simulated") / request.param.pop("name", "test.ms")
  simulator = MSStructureSimulator(**{**DEFAULT_SIM_PARAMS, **request.param})
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
