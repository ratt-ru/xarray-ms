from contextlib import ExitStack

import arcae
import numpy as np
import pytest
import xarray
from arcae.lib.arrow_tables import ms_descriptor
from rarg_python_patterns.multiton import Multiton
from xarray import DataTree

import xarray_ms
from xarray_ms.backend.msv2.writes import (
  DataVariableInfo,
  generate_column_descriptor,
  msv2_store_from_dataset,
)
from xarray_ms.errors import MismatchedWriteRegion, NonCanonicalColumnWarning
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


def test_write_support():
  assert xarray_ms.multithreaded_writes()


@pytest.mark.parametrize("simmed_ms", [{"name": "test_store.ms"}], indirect=True)
def test_store(simmed_ms):
  read = written = False

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    # Overwrite UVW coordinates with zeroes
    # Add a CORRECTED column
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert not np.all(node.UVW == 0)
        node.UVW[:] = 0
        assert len(node.encoding) > 0
        ds = node.ds.assign(CORRECTED=xarray.full_like(node.VISIBILITY, 2 + 3j))
        xdt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    xdt.sync_msv2()
    xdt.to_msv2()
    written = written or True

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        np.testing.assert_array_equal(node.UVW, 0)
        np.testing.assert_array_equal(node.CORRECTED, 2 + 3j)
        read = read or True

  assert read
  assert written

  # We can check that CORRECTED has been written correctly
  with arcae.table(simmed_ms) as T:
    np.testing.assert_array_equal(T.getcol("UVW"), 0)
    np.testing.assert_array_equal(T.getcol("CORRECTED"), 2 + 3j)


@pytest.mark.parametrize("simmed_ms", [{"name": "test_store_region.ms"}], indirect=True)
def test_store_region(simmed_ms):
  region = {"time": slice(0, 2), "frequency": slice(2, 4)}

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    # Add a CORRECTED column
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        ds = node.ds.assign(CORRECTED=xarray.zeros_like(node.VISIBILITY))
        xdt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    # Create the new MS columns
    xdt.sync_msv2()

    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        sizes = node.sizes
        ds = ds.isel(**region)
        ds = ds.assign(CORRECTED=xarray.full_like(ds.CORRECTED, 1 + 2j))
        # Now write it out
        ds.to_msv2(compute=False, region=region)

  # We can check that CORRECTED has been written correctly
  with arcae.table(simmed_ms) as T:
    corrected = T.getcol("CORRECTED")
    nt, nbl, nf, npol = (
      sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    corrected = corrected.reshape((nt, nbl, nf, npol))
    ts, fs = (region[d] for d in ("time", "frequency"))
    mask = np.full(corrected.shape, False, np.bool_)
    mask[ts, :, fs, :] = True
    np.testing.assert_array_equal(corrected[mask], 1 + 2j)
    np.testing.assert_array_equal(corrected[~mask], 0 + 0j)


@pytest.mark.parametrize("chunks", [{"time": 2, "frequency": 2}])
@pytest.mark.parametrize("simmed_ms", [{"name": "distributed-write.ms"}], indirect=True)
@pytest.mark.parametrize("nworkers", [4])
@pytest.mark.parametrize("processes", [True, False])
def test_distributed_write(simmed_ms, processes, nworkers, chunks):
  da = pytest.importorskip("dask.array")
  distributed = pytest.importorskip("dask.distributed")

  with ExitStack() as stack:
    cluster = stack.enter_context(
      distributed.LocalCluster(processes=processes, n_workers=nworkers)
    )
    stack.enter_context(distributed.Client(cluster))
    dt = stack.enter_context(
      xarray.open_datatree(simmed_ms, chunks=chunks, auto_corrs=True)
    )
    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        vis = node.VISIBILITY
        sizes = node.sizes
        corrected = da.arange(np.prod(vis.shape), dtype=np.int32)
        corrected = corrected.reshape(vis.shape).rechunk(vis.data.chunks)
        ds = node.ds.assign(CORRECTED=(vis.dims, corrected))
        dt[node.path] = DataTree(ds)
        assert len(node.encoding) > 0

    # Create the new MS columns
    dt.sync_msv2()
    dt.to_msv2(compute=False)

    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        node.ds.to_msv2(compute=True)

  with arcae.table(simmed_ms) as T:
    corrected = T.getcol("CORRECTED")
    shape = tuple(
      sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    expected = np.arange(np.prod(vis.shape), dtype=np.int32)
    expected = expected.reshape((-1,) + shape[2:])
    np.testing.assert_array_equal(corrected, expected)


@pytest.mark.parametrize("simmed_ms", [{"name": "indexed-write.ms"}], indirect=True)
def test_indexed_write(simmed_ms):
  """Check that we throw if we select a variable out with an integer index
  and then try write that sub-selection out"""
  dt = xarray.open_datatree(simmed_ms)
  assert len(dt.children) == 1

  for node in dt.subtree:
    if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
      ds = node.ds.assign(CORRECTED=xarray.full_like(node.VISIBILITY, 1 + 2j))
      dt[node.path] = DataTree(ds)

  dt.sync_msv2()
  dt.to_msv2(compute=False)

  for node in dt.subtree:
    if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
      ds = node.ds.isel(time=slice(0, 2), baseline_id=slice(0, 2), frequency=1)
      with pytest.raises(MismatchedWriteRegion):
        ds.to_msv2(compute=True)


@pytest.mark.parametrize("simmed_ms", [{"name": "canonical-column.ms"}], indirect=True)
def test_sync_canonical_column(simmed_ms):
  """Columns in the canonical MAIN descriptor that are absent from the
  table must still be created https://github.com/ratt-ru/xarray-ms/issues/171"""
  columns = {"MODEL_DATA": "MODEL_DATA", "MODEL_DATA_CUSTOM": "MODEL_DATA_CUSTOM"}
  canonical_desc = ms_descriptor("MAIN", complete=True)
  assert "MODEL_DATA" in canonical_desc
  assert "MODEL_DATA_CUSTOM" not in canonical_desc

  with arcae.table(simmed_ms) as T:
    assert not set(columns).intersection(T.columns())

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        vis = node.VISIBILITY
        sizes = node.sizes
        # sync_msv2 only reads dims, shape and dtype,
        # so a zero-strided placeholder suffices
        placeholder = np.broadcast_to(np.array(0, vis.dtype), vis.shape)
        ds = node.ds.assign({c: (vis.dims, placeholder) for c in columns})
        xdt[node.path] = DataTree(ds)

    xdt.sync_msv2(write_map=columns)
    # Creation is idempotent
    xdt.sync_msv2(write_map=columns)

  with arcae.table(simmed_ms) as T:
    assert set(columns).issubset(T.columns())
    table_desc = T.tabledesc()

    for column in columns:
      column_desc = table_desc[column]
      # A fixed shape, tiled column, not the variably shaped
      # StandardStMan column of the canonical descriptor
      assert column_desc["dataManagerType"] == "TiledColumnStMan"
      assert column_desc["ndim"] == 2
      assert list(column_desc["shape"]) == [sizes["frequency"], sizes["polarization"]]
      assert column_desc["valueType"] == "complex"

    # Descriptive metadata is inherited from the canonical descriptor
    assert (
      table_desc["MODEL_DATA"]["comment"] == canonical_desc["MODEL_DATA"]["comment"]
    )
    assert table_desc["MODEL_DATA_CUSTOM"]["comment"] == ""


@pytest.mark.parametrize(
  "simmed_ms, fail_second",
  [
    ({"name": "schema-refresh.ms"}, False),
    ({"name": "schema-refresh-fail.ms"}, True),
  ],
  indirect=["simmed_ms"],
)
def test_sync_refreshes_main_table_only_after_schema_change(
  simmed_ms, monkeypatch, fail_second
):
  events = []
  original_release = Multiton.release
  original_instance = Multiton.instance.fget
  released = False
  add_count = 0

  class ObservedTable:
    def __init__(self, table):
      self.table = table

    def __getattr__(self, name):
      return getattr(self.table, name)

    def close(self):
      events.append("close")
      self.table.close()

    def addcols(self, *args, **kwargs):
      nonlocal add_count
      add_count += 1
      if fail_second and add_count == 2:
        raise RuntimeError("second column failed")
      return self.table.addcols(*args, **kwargs)

  def record_instance(multiton):
    nonlocal released
    table = original_instance(multiton)
    if multiton == factory:
      if released:
        events.append("reopen")
        released = False
      return ObservedTable(table)
    return table

  def record_release(multiton):
    nonlocal released
    if multiton == factory:
      events.append("release")
      released = True
    return original_release(multiton)

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as dt:
    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        vis = node.VISIBILITY
        new_columns = {"MODEL_DATA": xarray.zeros_like(vis)}
        if fail_second:
          new_columns["MODEL_DATA_CUSTOM"] = xarray.zeros_like(vis)
        dt[node.path] = DataTree(node.ds.assign(new_columns))
        correlated_ds = dt[node.path].ds

    factory = msv2_store_from_dataset(correlated_ds).table_factory
    monkeypatch.setattr(Multiton, "instance", property(record_instance))
    monkeypatch.setattr(Multiton, "release", record_release)

    if fail_second:
      with pytest.raises(RuntimeError, match="second column failed"):
        dt.sync_msv2()
    else:
      dt.sync_msv2()
    expected_events = (
      ["close", "release"] if fail_second else ["close", "release", "reopen"]
    )
    assert events == expected_events
    assert "MODEL_DATA" in factory.instance.columns()
    assert events == ["close", "release", "reopen"]

    dt.sync_msv2()
    if fail_second:
      assert "MODEL_DATA_CUSTOM" in factory.instance.columns()
      assert events == ["close", "release", "reopen"] * 2
    else:
      assert events == ["close", "release", "reopen"]


@pytest.mark.parametrize(
  "simmed_ms", [{"name": "schema-refresh-close.ms"}], indirect=True
)
def test_sync_propagates_close_failure_inside_outer_except(simmed_ms, monkeypatch):
  original_instance = Multiton.instance.fget
  original_release = Multiton.release
  events = []

  class FailingCloseTable:
    def __init__(self, table):
      self.table = table

    def __getattr__(self, name):
      return getattr(self.table, name)

    def addcols(self, *args, **kwargs):
      result = self.table.addcols(*args, **kwargs)
      events.append("addcols")
      return result

    def close(self):
      self.table.close()
      events.append("close")
      raise RuntimeError("closing MAIN failed")

  def record_instance(multiton):
    table = original_instance(multiton)
    return FailingCloseTable(table) if multiton == factory else table

  def record_release(multiton):
    if multiton == factory:
      events.append("release")
    return original_release(multiton)

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as dt:
    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        ds = node.ds.assign(MODEL_DATA=xarray.zeros_like(node.VISIBILITY))
        dt[node.path] = DataTree(ds)
        correlated_ds = dt[node.path].ds

    factory = msv2_store_from_dataset(correlated_ds).table_factory
    monkeypatch.setattr(Multiton, "instance", property(record_instance))
    monkeypatch.setattr(Multiton, "release", record_release)

    try:
      raise ValueError("outer exception")
    except ValueError:
      with pytest.raises(RuntimeError, match="closing MAIN failed"):
        dt.sync_msv2()

    assert events == ["addcols", "close", "release"]
    assert "MODEL_DATA" in factory.instance.columns()


@pytest.mark.parametrize(
  "simmed_ms", [{"name": "canonical-column-region.ms"}], indirect=True
)
def test_canonical_column_region_write(simmed_ms):
  """A canonical column created by sync_msv2 must support partial writes.
  The canonical descriptor itself is variably shaped and does not"""
  region = {"time": slice(0, 2), "frequency": slice(2, 4)}

  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        vis = node.VISIBILITY
        sizes = node.sizes
        placeholder = np.broadcast_to(np.array(0, vis.dtype), vis.shape)
        xdt[node.path] = DataTree(node.ds.assign(MODEL_DATA=(vis.dims, placeholder)))

    xdt.sync_msv2(write_map={"MODEL_DATA": "MODEL_DATA"})

    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        ds = node.ds.isel(**region)
        ds = ds.assign(MODEL_DATA=xarray.full_like(ds.MODEL_DATA, 1 + 2j))
        ds.to_msv2(write_map={"MODEL_DATA": "MODEL_DATA"}, region=region)

  with arcae.table(simmed_ms) as T:
    nt, nbl, nf, npol = (
      sizes[d] for d in ("time", "baseline_id", "frequency", "polarization")
    )
    model = T.getcol("MODEL_DATA").reshape((nt, nbl, nf, npol))
    ts, fs = (region[d] for d in ("time", "frequency"))
    mask = np.full(model.shape, False, np.bool_)
    mask[ts, :, fs, :] = True
    np.testing.assert_array_equal(model[mask], 1 + 2j)
    np.testing.assert_array_equal(model[~mask], 0 + 0j)


@pytest.mark.parametrize(
  "simmed_ms", [{"name": "non-canonical-type.ms"}], indirect=True
)
def test_canonical_column_type_deviation(simmed_ms):
  """The variable data type wins over the canonical column type,
  but the deviation is warned about"""
  with xarray.open_datatree(simmed_ms, auto_corrs=True) as xdt:
    for node in xdt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        vis = node.VISIBILITY
        placeholder = np.broadcast_to(np.array(0, np.complex128), vis.shape)
        xdt[node.path] = DataTree(node.ds.assign(MODEL_DATA=(vis.dims, placeholder)))

    with pytest.warns(NonCanonicalColumnWarning, match="MODEL_DATA"):
      xdt.sync_msv2(write_map={"MODEL_DATA": "MODEL_DATA"})

  with arcae.table(simmed_ms) as T:
    assert T.tabledesc()["MODEL_DATA"]["valueType"] == "dcomplex"


def test_generate_existing_column_descriptor():
  """Columns present on the table are not re-created"""
  var_map = {("VISIBILITY", "DATA"): DataVariableInfo(1, {(8, 4)}, {np.complex64})}
  table_desc = {"DATA": {"valueType": "complex", "ndim": 2}}
  column_descs, dminfo = generate_column_descriptor(table_desc, var_map)
  assert column_descs == {}
  assert dminfo == {}


def test_generate_canonical_column_descriptor():
  """Canonical columns absent from the table are created as
  fixed shape, tiled columns"""
  var_map = {
    ("MODEL_DATA", "MODEL_DATA"): DataVariableInfo(1, {(8, 4)}, {np.complex64})
  }
  column_descs, dminfo = generate_column_descriptor({}, var_map)
  column_desc = column_descs["MODEL_DATA"]
  assert column_desc["dataManagerType"] == "TiledColumnStMan"
  assert column_desc["ndim"] == 2
  # Column descriptor shapes are FORTRAN ordered
  assert column_desc["shape"] == [4, 8]
  assert column_desc["option"] & 4
  assert column_desc["comment"] == "The model data column"
  assert [g["COLUMNS"] for g in dminfo.values()] == [["MODEL_DATA"]]


def test_generate_variably_shaped_column_descriptor():
  """Variably shaped columns are created as TiledShapeStMan columns
  of fixed dimensionality"""
  var_map = {("MODEL", "MODEL"): DataVariableInfo(2, {(8, 4), (16, 2)}, {np.complex64})}
  column_descs, dminfo = generate_column_descriptor({}, var_map)
  column_desc = column_descs["MODEL"]
  assert column_desc["dataManagerType"] == "TiledShapeStMan"
  assert column_desc["ndim"] == 2
  assert "shape" not in column_desc
  assert column_desc["option"] == 0
  (group,) = dminfo.values()
  assert group["TYPE"] == "TiledShapeStMan"


def test_generate_ragged_column_descriptor():
  """Variables whose trailing shapes disagree on dimensionality
  have no column representation"""
  var_map = {("MODEL", "MODEL"): DataVariableInfo(2, {(8, 4), (8,)}, {np.complex64})}
  with pytest.raises(ValueError, match="differing dimensionality"):
    generate_column_descriptor({}, var_map)
