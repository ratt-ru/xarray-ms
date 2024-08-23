import numpy as np
import pytest
from arcae.lib.arrow_tables import Table

from xarray_ms.backend.msv2.structure import MSv2StructureFactory
from xarray_ms.backend.msv2.table_factory import TableFactory


@pytest.fixture(params=[{"auto_corrs": True, "na": 4}])
def irregular_ms(tmp_path, request):
  fn = str(tmp_path / "irregular.ms")
  auto_corrs = request.param.get("auto_corrs", True)
  na = request.param.get("na", 4)

  ddids = []
  times = []
  ant1s = []
  ant2s = []

  for ddid in range(0, 3):
    time = np.linspace((ddid + 1) * 0.0, (ddid + 1) * 10.0, 5)
    ant1, ant2 = map(np.int32, np.triu_indices(na, 0 if auto_corrs else 1))
    ddid = np.array([ddid], np.int32)

    ddid, time, ant1, ant2 = map(
      np.ravel,
      np.broadcast_arrays(
        ddid[:, None, None],
        time[None, :, None],
        ant1[None, None, :],
        ant2[None, None, :],
      ),
    )

    ddids.append(ddid)
    times.append(time)
    ant1s.append(ant1)
    ant2s.append(ant2)

  ddid = np.concatenate(ddids)
  time = np.concatenate(times)
  ant1 = np.concatenate(ant1s)
  ant2 = np.concatenate(ant2s)

  create_table_query = f"""
    CREATE TABLE {fn}
    [ANTENNA1 I4,
    ANTENNA2 I4,
    DATA_DESC_ID I4,
    UVW R8 [NDIM=1, SHAPE=[3]],
    TIME R8,
    DATA C8 [NDIM=2, SHAPE=[4, 16]]]
    LIMIT {ddid.size}
  """

  rng = np.random.default_rng()
  data_shape = (len(time), 16, 4)
  data = rng.random(data_shape) + rng.random(data_shape) * 1j
  uvw = rng.random((len(time), 3)).astype(np.float64)

  # Create the table
  with Table.from_taql(create_table_query) as ms:
    ms.putcol("DATA_DESC_ID", ddid)
    ms.putcol("TIME", time)
    ms.putcol("ANTENNA1", ant1)
    ms.putcol("ANTENNA2", ant2)
    ms.putcol("UVW", uvw)
    ms.putcol("DATA", data)

  yield fn


@pytest.mark.skip(reason="Redo with MSv2Structure")
@pytest.mark.parametrize("na", [4])
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_row_mapping(irregular_ms, na, auto_corrs):
  table_factory = TableFactory(Table.from_filename, irregular_ms)
  structure_factory = MSv2StructureFactory(table_factory, auto_corrs=auto_corrs)
  structure = structure_factory()

  ddid = table_factory().getcol("DATA_DESC_ID")
  ant1 = table_factory().getcol("ANTENNA1")
  ant2 = table_factory().getcol("ANTENNA2")
  time = table_factory().getcol("TIME")

  for ddid in np.unique(ddid):
    if auto_corrs:
      mask = ddid == 0
    else:
      mask = np.logical_and(ddid == 0, ant1 != ant2)

    key = (
      ("DATA_DESC_ID", ddid),
      ("FEED1", 0),
      ("FEED2", 0),
      ("FIELD_ID", 0),
      ("PROCESSOR_ID", 0),
    )
    utime = np.unique(time[mask])
    ubl = np.unique(np.stack([ant1[mask], ant2[mask]], axis=1), axis=0)
    rowmap = structure.row_map(key)
    assert rowmap.shape == (utime.size, ubl.shape[0])
