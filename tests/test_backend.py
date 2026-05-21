from contextlib import ExitStack

import numpy as np
import pytest
import xarray
import xarray.testing as xt
from numpy.testing import assert_array_equal

from xarray_ms.backend.msv2.entrypoint import MSv2EntryPoint
from xarray_ms.testing.utils import id_string, prune_datetime_structures


def test_entrypoint(simmed_ms):
  # The entrypoint thinks it can open the MS
  entrypoint = MSv2EntryPoint()
  assert entrypoint.guess_can_open(simmed_ms) is True


@pytest.mark.filterwarnings("ignore:.*?matched multiple partitions")
@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
    }
  ],
  indirect=True,
)
def test_open_dataset(simmed_ms):
  distributed = pytest.importorskip("dask.distributed")
  Client = distributed.Client
  LocalCluster = distributed.LocalCluster

  # Works with xarray default load mechanism
  with ExitStack() as stack:
    mem_ds = stack.enter_context(xarray.open_dataset(simmed_ms))
    mem_ds.load()
    prune_datetime_structures(mem_ds)
    assert isinstance(mem_ds.VISIBILITY.data, np.ndarray)

  chunks = {"time": 2, "frequency": 2}

  # Works with default dask scheduler
  with ExitStack() as stack:
    ds = stack.enter_context(xarray.open_dataset(simmed_ms, chunks=chunks))
    assert prune_datetime_structures(ds).identical(mem_ds)

  # Works with a LocalCluster
  with ExitStack() as stack:
    cluster = stack.enter_context(LocalCluster(processes=True, n_workers=4))
    stack.enter_context(Client(cluster))
    ds = stack.enter_context(xarray.open_dataset(simmed_ms, chunks=chunks))
    assert prune_datetime_structures(ds).identical(mem_ds)


@pytest.mark.filterwarnings("ignore:.*?matched multiple partitions")
@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nfield": 2,
      "data_description": [
        (8, ["XX", "XY", "YX", "YY"]),
        (4, ["RR", "LL"]),
        (16, ["RR", "RL", "LR", "LL"]),
      ],
    }
  ],
  indirect=True,
)
@pytest.mark.parametrize(
  "partition_key, pols, nfreq",
  [
    # Resolve to DATA_DESC_ID 0, FIELD_ID 0
    (None, ["XX", "XY", "YX", "YY"], 8),
    # Resolve to FIELD_ID 0
    ((("DATA_DESC_ID", 0),), ["XX", "XY", "YX", "YY"], 8),
    ((("DATA_DESC_ID", 1),), ["RR", "LL"], 4),
    ((("DATA_DESC_ID", 2),), ["RR", "RL", "LR", "LL"], 16),
    ("D=0", ["XX", "XY", "YX", "YY"], 8),
    ("D=1", ["RR", "LL"], 4),
    ("D=2", ["RR", "RL", "LR", "LL"], 16),
    # Resolves to DATA_DESC_ID 0
    ((("FIELD_ID", 0),), ["XX", "XY", "YX", "YY"], 8),
    ((("FIELD_ID", 1),), ["XX", "XY", "YX", "YY"], 8),
    ("F=0", ["XX", "XY", "YX", "YY"], 8),
    ("F=1", ["XX", "XY", "YX", "YY"], 8),
    # Full key resolution
    ((("DATA_DESC_ID", 0), ("FIELD_ID", 0)), ["XX", "XY", "YX", "YY"], 8),
    ((("DATA_DESC_ID", 1), ("FIELD_ID", 0)), ["RR", "LL"], 4),
    ((("DATA_DESC_ID", 2), ("FIELD_ID", 0)), ["RR", "RL", "LR", "LL"], 16),
    ((("DATA_DESC_ID", 0), ("FIELD_ID", 1)), ["XX", "XY", "YX", "YY"], 8),
    ((("DATA_DESC_ID", 1), ("FIELD_ID", 1)), ["RR", "LL"], 4),
    ((("DATA_DESC_ID", 2), ("FIELD_ID", 1)), ["RR", "RL", "LR", "LL"], 16),
    # Reverse order
    ((("FIELD_ID", 0), ("DATA_DESC_ID", 0)), ["XX", "XY", "YX", "YY"], 8),
    ((("FIELD_ID", 0), ("DATA_DESC_ID", 1)), ["RR", "LL"], 4),
    ((("FIELD_ID", 0), ("DATA_DESC_ID", 2)), ["RR", "RL", "LR", "LL"], 16),
    ((("FIELD_ID", 1), ("DATA_DESC_ID", 0)), ["XX", "XY", "YX", "YY"], 8),
    ((("FIELD_ID", 1), ("DATA_DESC_ID", 1)), ["RR", "LL"], 4),
    ((("FIELD_ID", 1), ("DATA_DESC_ID", 2)), ["RR", "RL", "LR", "LL"], 16),
    # Short column name
    ((("F", 0), ("D", 0)), ["XX", "XY", "YX", "YY"], 8),
    ((("D", 0), ("F", 0)), ["XX", "XY", "YX", "YY"], 8),
    # String keys
    ("", ["XX", "XY", "YX", "YY"], 8),
    ("D=0,F=0", ["XX", "XY", "YX", "YY"], 8),
    ("D=1,F=0", ["RR", "LL"], 4),
    ("D=2,F=0", ["RR", "RL", "LR", "LL"], 16),
    ("D=0,F=1", ["XX", "XY", "YX", "YY"], 8),
    ("D=1,F=1", ["RR", "LL"], 4),
    ("D=2,F=1", ["RR", "RL", "LR", "LL"], 16),
  ],
  ids=id_string,
)
@pytest.mark.parametrize(
  "partition_schema", [["DATA_DESC_ID", "OBSERVATION_ID", "FIELD_ID"]]
)
def test_open_dataset_partition_keys(
  simmed_ms, partition_schema, partition_key, pols, nfreq
):
  ds = xarray.open_dataset(
    simmed_ms, partition_schema=partition_schema, partition_key=partition_key
  )
  assert_array_equal(ds.polarization.values, pols)
  assert {("frequency", nfreq), ("polarization", len(pols))}.issubset(ds.sizes.items())


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
    }
  ],
  indirect=True,
)
def test_open_datatree(simmed_ms):
  distributed = pytest.importorskip("dask.distributed")
  Client = distributed.Client
  LocalCluster = distributed.LocalCluster

  # Works with xarray default load mechanism
  with ExitStack() as stack:
    mem_dt = prune_datetime_structures(
      stack.enter_context(xarray.open_datatree(simmed_ms))
    )
    mem_dt.load()

    for node in mem_dt.subtree:
      if node.attrs.get("type") == "visibility":
        assert isinstance(node.VISIBILITY.data, np.ndarray)

  chunks = {"time": 2, "frequency": 2}

  # Works with default dask scheduler
  with ExitStack() as stack:
    dt = stack.enter_context(xarray.open_datatree(simmed_ms, preferred_chunks=chunks))
    xt.assert_identical(prune_datetime_structures(dt), mem_dt)

  # Works with a LocalCluster
  with ExitStack() as stack:
    cluster = stack.enter_context(LocalCluster(processes=True, n_workers=4))
    stack.enter_context(Client(cluster))
    dt = stack.enter_context(xarray.open_datatree(simmed_ms, preferred_chunks=chunks))
    xt.assert_identical(prune_datetime_structures(dt), mem_dt)


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
    }
  ],
  indirect=True,
)
def test_open_datatree_chunking(simmed_ms):
  """Test opening a datatree with both uniform
  and partition-specific chunking"""

  ncdt = prune_datetime_structures(xarray.open_datatree(simmed_ms, auto_corrs=True))
  ncdt.load()

  dt = xarray.open_datatree(
    simmed_ms,
    auto_corrs=True,
    chunks={},
    preferred_chunks={"time": 3, "frequency": 2},
  )

  assert dict(dt["backend_partition_000"].ds.chunks) == {
    "time": (3, 2),
    "baseline_id": (6,),
    "frequency": (2, 2, 2, 2),
    "polarization": (4,),
    "uvw_label": (3,),
  }

  assert dict(dt["backend_partition_001"].ds.chunks) == {
    "time": (3, 2),
    "baseline_id": (6,),
    "frequency": (2, 2),
    "polarization": (2,),
    "uvw_label": (3,),
  }

  assert ncdt.identical(prune_datetime_structures(dt))

  dt = xarray.open_datatree(
    simmed_ms,
    auto_corrs=True,
    chunks={},
    preferred_chunks={
      "D=0": {"time": 2, "baseline_id": 2},
      "D=1": {"time": 3, "frequency": 2},
    },
  )

  assert dict(dt["backend_partition_000"].ds.chunks) == {
    "time": (2, 2, 1),
    "baseline_id": (2, 2, 2),
    "frequency": (8,),
    "polarization": (4,),
    "uvw_label": (3,),
  }
  assert dict(dt["backend_partition_001"].ds.chunks) == {
    "time": (3, 2),
    "baseline_id": (6,),
    "frequency": (2, 2),
    "polarization": (2,),
    "uvw_label": (3,),
  }

  assert ncdt.identical(prune_datetime_structures(dt))
