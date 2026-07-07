import concurrent.futures as cf
import pickle
from itertools import product

import numpy as np
import pyarrow as pa
import pytest
import xarray
from arcae.lib.arrow_tables import Table
from numpy.testing import assert_array_equal, assert_equal
from rarg_python_patterns.multiton import Multiton

from xarray_ms.backend.msv2.structure import (
  MainTableFactory,
  MSv2Structure,
  SubtableFactory,
  TablePartitioner,
  baseline_id,
)


@pytest.mark.parametrize("na", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_baseline_id(na, auto_corrs):
  ant1, ant2 = np.triu_indices(na, 0 if auto_corrs else 1)
  bl_id = baseline_id(ant1, ant2, na, auto_corrs)
  assert_array_equal(np.arange(ant1.size), bl_id)


@pytest.mark.parametrize("simmed_ms", [{"name": "proxy.ms"}], indirect=True)
@pytest.mark.parametrize("epoch", ["abcdef"])
def test_structure_factory(simmed_ms, epoch):
  partition_schema = [
    "FIELD_ID",
    "DATA_DESC_ID",
    "PROCESSOR_ID",
    "OBSERVATION_ID",
    "OBS_MODE",
  ]
  table_factory: MainTableFactory = Multiton(Table.from_filename, simmed_ms)
  from xarray_ms.backend.msv2.entrypoint_utils import subtable_factory

  subtables: dict[str, SubtableFactory] = {
    st: Multiton(subtable_factory, f"{simmed_ms}::{st}")
    for st in ("DATA_DESCRIPTION", "FEED", "FIELD", "STATE")
  }
  structure_factory = Multiton(
    MSv2Structure, table_factory, subtables, partition_schema, epoch, True
  )
  assert pickle.loads(pickle.dumps(structure_factory)) == structure_factory

  structure_factory2 = Multiton(
    MSv2Structure, table_factory, subtables, partition_schema, epoch, True
  )
  assert structure_factory.instance is structure_factory2.instance

  keys = tuple(k for kv in structure_factory.instance.keys() for k, _ in kv)
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
  table_factory: MainTableFactory = Multiton(Table.from_filename, simmed_ms)
  from xarray_ms.backend.msv2.entrypoint_utils import subtable_factory

  subtables: dict[str, SubtableFactory] = {
    st: Multiton(subtable_factory, f"{simmed_ms}::{st}")
    for st in ("DATA_DESCRIPTION", "FEED", "FIELD", "STATE")
  }

  structure_factory = Multiton(
    MSv2Structure, table_factory, subtables, partition_schema, "abc", True
  )
  structure_factory2 = Multiton(
    MSv2Structure, table_factory, subtables, partition_schema, "abc", True
  )

  assert structure_factory.instance is structure_factory2.instance

  structure_factory3 = Multiton(
    MSv2Structure, table_factory, subtables, partition_schema, "def", True
  )

  assert structure_factory.instance is not structure_factory3.instance


def test_msv2_structure_release(simmed_ms):
  def ncached_structures():
    return sum(
      isinstance(obj, MSv2Structure)
      for obj, _, _, _ in Multiton._INSTANCE_CACHE.values()
    )

  assert ncached_structures() == 0

  with xarray.open_datatree(simmed_ms):
    assert ncached_structures() > 0

  assert ncached_structures() == 0


def test_memory_footprint(simmed_ms):
  time_chunk = 10
  freq_chunk = 10

  with xarray.open_datatree(simmed_ms, engine="xarray-ms:msv2") as dt:
    cache_size_after_open = len(Multiton._INSTANCE_CACHE)
    assert cache_size_after_open > 0

    for node in dt.children.values():
      ntime = node.sizes["time"]
      nfreq = node.sizes["frequency"]
      for t, f in product(
        range(0, ntime, time_chunk),
        range(0, nfreq, freq_chunk),
      ):
        sub = node.isel(
          time=slice(t, min(t + time_chunk, ntime)),
          frequency=slice(f, min(f + freq_chunk, nfreq)),
        )
        sub.load()

    assert len(Multiton._INSTANCE_CACHE) == cache_size_after_open
