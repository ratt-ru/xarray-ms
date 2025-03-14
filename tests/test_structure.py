import concurrent.futures as cf
import pickle

import numpy as np
import pyarrow as pa
import pytest
from arcae.lib.arrow_tables import Table
from numpy.testing import assert_array_equal, assert_equal

from xarray_ms.backend.msv2.structure import (
  MSv2StructureFactory,
  TablePartitioner,
  baseline_id,
)
from xarray_ms.backend.msv2.table_factory import TableFactory


@pytest.mark.parametrize("na", [4, 7])
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_baseline_id(na, auto_corrs):
  ant1, ant2 = np.triu_indices(na, 0 if auto_corrs else 1)
  bl_id = baseline_id(ant1, ant2, na, auto_corrs)
  assert_array_equal(np.arange(ant1.size), bl_id)


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
@pytest.mark.parametrize("epoch", ["abcdef"])
def test_structure_factory(simmed_ms, epoch):
  partition_schema = ["FIELD_ID", "DATA_DESC_ID", "OBSERVATION_ID", "OBS_MODE"]
  table_factory = TableFactory(Table.from_filename, simmed_ms)
  structure_factory = MSv2StructureFactory(table_factory, partition_schema, epoch)
  assert pickle.loads(pickle.dumps(structure_factory)) == structure_factory

  structure_factory2 = MSv2StructureFactory(table_factory, partition_schema, epoch)
  assert structure_factory() is structure_factory2()

  keys = tuple(k for kv in structure_factory().keys() for k, _ in kv)
  assert tuple(sorted(partition_schema)) == keys


def test_table_partitioner():
  table = pa.Table.from_pydict(
    {
      "DATA_DESC_ID": pa.array([1, 1, 0, 0, 0], pa.int32()),
      "FIELD_ID": pa.array([1, 0, 0, 1, 0], pa.int32()),
      "TIME": pa.array([0, 1, 4, 3, 2], pa.float64()),
      "ANTENNA1": pa.array([0, 0, 0, 0, 0], pa.int32()),
      "ANTENNA2": pa.array([1, 1, 1, 1, 1], pa.int32()),
    }
  )

  partitioner = TablePartitioner(
    ["DATA_DESC_ID", "FIELD_ID"], ["TIME", "ANTENNA1", "ANTENNA2"], ["row"]
  )

  with cf.ThreadPoolExecutor(max_workers=4) as pool:
    groups = partitioner.partition(table, pool)

  assert_equal(
    groups,
    {
      (("DATA_DESC_ID", 0), ("FIELD_ID", 0)): {
        "DATA_DESC_ID": [0, 0],
        "FIELD_ID": [0, 0],
        "TIME": [2.0, 4.0],
        "ANTENNA1": [0, 0],
        "ANTENNA2": [1, 1],
        "row": [4, 2],
      },
      (("DATA_DESC_ID", 0), ("FIELD_ID", 1)): {
        "DATA_DESC_ID": [0],
        "FIELD_ID": [1],
        "TIME": [3.0],
        "ANTENNA1": [0],
        "ANTENNA2": [1],
        "row": [3],
      },
      (("DATA_DESC_ID", 1), ("FIELD_ID", 0)): {
        "DATA_DESC_ID": [1],
        "FIELD_ID": [0],
        "TIME": [1.0],
        "ANTENNA1": [0],
        "ANTENNA2": [1],
        "row": [1],
      },
      (("DATA_DESC_ID", 1), ("FIELD_ID", 1)): {
        "DATA_DESC_ID": [1],
        "FIELD_ID": [1],
        "TIME": [0.0],
        "ANTENNA1": [0],
        "ANTENNA2": [1],
        "row": [0],
      },
    },
  )


def test_epoch(simmed_ms):
  partition_schema = ["FIELD_ID", "DATA_DESC_ID", "OBSERVATION_ID"]
  table_factory = TableFactory(Table.from_filename, simmed_ms)
  structure_factory = MSv2StructureFactory(table_factory, partition_schema, "abc")
  structure_factory2 = MSv2StructureFactory(table_factory, partition_schema, "abc")

  assert structure_factory() is structure_factory2()

  structure_factory3 = MSv2StructureFactory(table_factory, partition_schema, "def")

  assert structure_factory() is not structure_factory3()
