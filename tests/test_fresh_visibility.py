"""Explicit additional visibility columns share DATA's schema and row selections."""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr
from arcae.lib.arrow_tables import Table

from tests.test_fresh_plan import make_tree
from xarray_ms import write_fresh_msv2
from xarray_ms.backend.msv2 import fresh_write
from xarray_ms.errors import FreshMSv2ValidationError, NonCanonicalColumnWarning
from xarray_ms.testing.fresh import make_complete_tree

MAPPING = {"model": "MODEL_DATA", "residual": "RESIDUAL_DATA"}


def add_groups(tree):
  for node in tree.subtree:
    if node.attrs.get("type") != "visibility":
      continue
    ds = node.to_dataset().copy(deep=True)
    ds["MODEL"] = ds.VISIBILITY * (2 + 1j)
    ds["RESIDUAL"] = ds.VISIBILITY - ds.MODEL
    ds.attrs["data_groups"]["model"] = {"correlated_data": "MODEL"}
    ds.attrs["data_groups"]["residual"] = {"correlated_data": "RESIDUAL"}
    node.ds = ds
  return tree


def assert_clone(table, column):
  base = table.getcoldesc("DATA")
  extra = table.getcoldesc(column)
  base.pop("dataManagerGroup")
  extra.pop("dataManagerGroup")
  assert extra == base
  managers = table.getdminfo()
  base = deepcopy(next(m for m in managers.values() if list(m["COLUMNS"]) == ["DATA"]))
  extra = deepcopy(next(m for m in managers.values() if list(m["COLUMNS"]) == [column]))
  for m in (base, extra):
    for key in ("NAME", "COLUMNS", "SEQNR"):
      m.pop(key, None)
    m["SPEC"].pop("SEQNR", None)
  assert extra == base


@pytest.mark.parametrize("mapping", [None, {}])
def test_omitted_extra_groups_are_not_written(tmp_path, mapping):
  target = tmp_path / "only.ms"
  write_fresh_msv2(
    add_groups(make_complete_tree()),
    target,
    row_grid="complete",
    additional_visibility=mapping,
  )
  with Table.from_filename(str(target)) as table:
    assert not {"MODEL_DATA", "RESIDUAL_DATA", "MODEL", "RESIDUAL"} & set(
      table.columns()
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_extra_payloads_descriptors_and_reordered_asymmetric_cells(tmp_path, dtype):
  tree = make_complete_tree()
  ds = tree["part"].to_dataset().isel(baseline_id=[2, 0, 1])
  ds["VISIBILITY"] = ds.VISIBILITY.astype(dtype)
  tree["part"].ds = ds
  add_groups(tree)
  target = tmp_path / "extras.ms"
  kwargs = dict(row_grid="complete", rows_per_batch=2, additional_visibility=MAPPING)
  if dtype == np.complex128:
    with pytest.warns(NonCanonicalColumnWarning):
      write_fresh_msv2(tree, target, **kwargs)
  else:
    write_fresh_msv2(tree, target, **kwargs)
  with Table.from_filename(str(target)) as table:
    for column, var in (
      ("DATA", "VISIBILITY"),
      ("MODEL_DATA", "MODEL"),
      ("RESIDUAL_DATA", "RESIDUAL"),
    ):
      expected = tree["part"][var].isel(baseline_id=[1, 2, 0]).values.reshape(12, 8, 2)
      np.testing.assert_array_equal(table.getcol(column), expected)
      if column != "DATA":
        assert_clone(table, column)
  if dtype == np.complex64:
    with xr.open_datatree(target) as reopened:
      node = next(n for n in reopened.subtree if n.attrs.get("type") == "visibility")
      np.testing.assert_array_equal(
        node["MODEL_DATA"], tree["part"].MODEL.isel(baseline_id=[1, 2, 0])
      )


def test_selected_group_support_columns_and_extra_base(tmp_path):
  tree = add_groups(make_complete_tree())
  ds = tree["part"].to_dataset()
  ds["MODEL_FLAG"] = ~ds.FLAG
  ds["MODEL_WEIGHT"] = ds.WEIGHT * 4
  ds["MODEL_UVW"] = ds.UVW * 2
  ds.attrs["data_groups"]["model"] = {
    **ds.attrs["data_groups"]["base"],
    "correlated_data": "MODEL",
    "flag": "MODEL_FLAG",
    "weight": "MODEL_WEIGHT",
    "uvw": "MODEL_UVW",
  }
  tree["part"].ds = ds
  target = tmp_path / "selected.ms"
  write_fresh_msv2(
    tree,
    target,
    row_grid="complete",
    data_group="model",
    additional_visibility={"base": "CUSTOM_VIS"},
  )
  with Table.from_filename(str(target)) as table:
    for column, var in (
      ("DATA", "MODEL"),
      ("CUSTOM_VIS", "VISIBILITY"),
      ("FLAG", "MODEL_FLAG"),
      ("WEIGHT_SPECTRUM", "MODEL_WEIGHT"),
      ("UVW", "MODEL_UVW"),
    ):
      values = ds[var].values
      np.testing.assert_array_equal(
        table.getcol(column), values.reshape((12,) + values.shape[2:])
      )
    assert_clone(table, "CUSTOM_VIS")


def test_variable_cells_across_partitions(tmp_path):
  tree = make_tree(("/a", "/b"))
  for name in ("a", "b"):
    ds = tree[name].to_dataset()
    ds.time.attrs["integration_time"] = {
      "attrs": {"type": "quantity", "units": "s"},
      "data": 1.0,
    }
    if name == "b":
      ds = ds.isel(frequency=slice(0, 1), polarization=slice(0, 1))
      ds.frequency.attrs["spectral_window_name"] = "other"
      antenna = tree["b/antenna_xds"]
      antenna.ds = antenna.to_dataset().isel(receptor_label=slice(0, 1))
    tree[name].ds = ds
  add_groups(tree)
  target = tmp_path / "ragged.ms"
  write_fresh_msv2(
    tree, target, row_grid="complete", additional_visibility=MAPPING, rows_per_batch=1
  )
  with Table.from_filename(str(target)) as table:
    for column, var in (
      ("DATA", "VISIBILITY"),
      ("MODEL_DATA", "MODEL"),
      ("RESIDUAL_DATA", "RESIDUAL"),
    ):
      assert table.getcoldesc(column)["dataManagerType"] == "TiledShapeStMan"
      for offset, name in ((0, "a"), (2, "b")):
        expected = tree[name][var].values
        actual = table.getcol(column, index=(slice(offset, offset + 2),))
        np.testing.assert_array_equal(
          actual, expected.reshape((2,) + expected.shape[2:])
        )
      if column != "DATA":
        assert_clone(table, column)


@pytest.mark.parametrize(
  "mapping",
  [
    {"base": "MODEL_DATA"},
    {"model": "DATA"},
    {"model": "UVW"},
    {"missing": "MODEL_DATA"},
    {"model": "MODEL_DATA", "residual": "model_data"},
    {"model": "bad name"},
    [],
  ],
)
def test_invalid_mapping_before_transaction(tmp_path, monkeypatch, mapping):
  def forbidden(*args):
    pytest.fail("Invalid mappings must fail before staging")

  monkeypatch.setattr(fresh_write, "_transaction", forbidden)
  with pytest.raises(FreshMSv2ValidationError):
    write_fresh_msv2(
      add_groups(make_complete_tree()),
      tmp_path / "bad.ms",
      row_grid="complete",
      additional_visibility=mapping,
    )


def test_lazy_eager_bounded_selection_and_no_addcols(tmp_path, monkeypatch):
  tree = add_groups(make_complete_tree())
  targets = [tmp_path / "eager.ms", tmp_path / "lazy.ms"]
  original_isel = xr.DataArray.isel
  calls = []

  def bounded(self, *args, **kwargs):
    if self.name in ("MODEL", "RESIDUAL"):
      assert kwargs["time"].stop - kwargs["time"].start == 1
      assert len(kwargs["baseline_id"]) <= 2
      calls.append(self.name)
    return original_isel(self, *args, **kwargs)

  monkeypatch.setattr(xr.DataArray, "isel", bounded)

  def forbidden(*args, **kwargs):
    pytest.fail("Fresh materialization must not add columns")

  class Handle:
    def __init__(self, handle):
      self.handle = handle

    def __enter__(self):
      self.handle.__enter__()
      return self

    def __exit__(self, *args):
      return self.handle.__exit__(*args)

    def __getattr__(self, name):
      return forbidden if name == "addcols" else getattr(self.handle, name)

  class GuardedTable:
    @staticmethod
    def from_filename(*args, **kwargs):
      assert kwargs["ninstances"] == 1
      return Handle(Table.from_filename(*args, **kwargs))

    @staticmethod
    def ms_from_descriptor(*args, **kwargs):
      assert kwargs["ninstances"] == 1
      return Handle(Table.ms_from_descriptor(*args, **kwargs))

  from xarray_ms.backend.msv2 import fresh_create

  monkeypatch.setattr(fresh_write, "Table", GuardedTable)
  monkeypatch.setattr(fresh_create, "Table", GuardedTable)
  for index, target in enumerate(targets):
    if index:
      tree["part"].ds = tree["part"].to_dataset().chunk({"time": 2, "baseline_id": 3})
    write_fresh_msv2(
      tree, target, row_grid="complete", additional_visibility=MAPPING, rows_per_batch=2
    )
  assert len(calls) == 64
  with (
    Table.from_filename(str(targets[0])) as eager,
    Table.from_filename(str(targets[1])) as lazy,
  ):
    for column in ("DATA", "MODEL_DATA", "RESIDUAL_DATA"):
      np.testing.assert_array_equal(eager.getcol(column), lazy.getcol(column))


@pytest.mark.parametrize("failure", ["late_payload", "corruption"])
def test_extra_failure_cleans_staging(tmp_path, monkeypatch, failure):
  target = tmp_path / "failed.ms"
  if failure == "late_payload":
    original = xr.DataArray.isel

    def broken(self, *args, **kwargs):
      if self.name == "MODEL" and kwargs["time"].start == 1:
        raise RuntimeError("extra payload failed")
      return original(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "isel", broken)
  else:
    original = fresh_write._verify_populated

    def corrupt(tree, plan, grids, staging, *args):
      with Table.from_filename(staging, readonly=False, ninstances=1) as table:
        values = table.getcol("MODEL_DATA", index=(slice(0, 1),))
        table.putcol("MODEL_DATA", values + 1, index=(slice(0, 1),))
      original(tree, plan, grids, staging, *args)

    monkeypatch.setattr(fresh_write, "_verify_populated", corrupt)
  with pytest.raises(
    (RuntimeError, ValueError), match="extra payload|MAIN::MODEL_DATA"
  ):
    write_fresh_msv2(
      add_groups(make_complete_tree()),
      target,
      row_grid="complete",
      additional_visibility=MAPPING,
      rows_per_batch=2,
    )
  assert not target.exists()
  assert not list(tmp_path.glob(".*.staging"))
