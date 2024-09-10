import pickle

import numpy as np
import pytest
from arcae.lib.arrow_tables import Table
from numpy.testing import assert_array_equal

from xarray_ms.backend.msv2.structure import MSv2StructureFactory, baseline_id
from xarray_ms.backend.msv2.table_factory import TableFactory


@pytest.mark.parametrize("na", [4, 7])
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_baseline_id(na, auto_corrs):
  ant1, ant2 = np.triu_indices(na, 0 if auto_corrs else 1)
  bl_id = baseline_id(ant1, ant2, na, auto_corrs)
  assert_array_equal(np.arange(ant1.size), bl_id)


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
def test_structure_factory(simmed_ms):
  partition_columns = ["FIELD_ID", "DATA_DESC_ID", "OBSERVATION_ID"]
  table_factory = TableFactory(Table.from_filename, simmed_ms)
  structure_factory = MSv2StructureFactory(table_factory, partition_columns)
  assert pickle.loads(pickle.dumps(structure_factory)) == structure_factory

  structure_factory2 = MSv2StructureFactory(table_factory, partition_columns)
  assert structure_factory() is structure_factory2()

  keys = tuple(k for kv in structure_factory().keys() for k, _ in kv)
  assert tuple(sorted(partition_columns)) == keys
