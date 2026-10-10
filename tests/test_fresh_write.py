"""Admission, bounded writes and transactional failure checks."""

import numpy as np
import pytest
from arcae.lib.arrow_tables import Table

from tests.test_fresh_plan import make_tree
from xarray_ms import create_fresh_msv2, plan_fresh_msv2, write_fresh_msv2
from xarray_ms.backend.msv2 import fresh_create, fresh_write
from xarray_ms.errors import FreshMSv2ValidationError
from xarray_ms.testing.fresh import make_complete_tree


def complete_small():
  tree = make_tree()
  tree["part"].ds.time.attrs["integration_time"] = {
    "attrs": {"type": "quantity", "units": "s"},
    "data": 1.0,
  }
  return tree


def test_asymmetric_skeleton_and_rows(tmp_path):
  tree = make_complete_tree()
  skeleton = tmp_path / "skeleton.ms"
  create_fresh_msv2(plan_fresh_msv2(tree, skeleton))
  with Table.from_filename(str(skeleton)) as table:
    descriptor = table.getcoldesc("DATA")
    assert descriptor["shape"] == ([8, 2] if descriptor.get("_c_order") else [2, 8])
  path = tmp_path / "rows.ms"
  write_fresh_msv2(tree, path, row_grid="complete", rows_per_batch=2)
  with Table.from_filename(str(path)) as table:
    assert table.getcol("DATA").shape == (12, 8, 2)
    np.testing.assert_array_equal(
      table.getcol("DATA"), tree["part"].VISIBILITY.values.reshape(12, 8, 2)
    )
    np.testing.assert_array_equal(table.getcol("WEIGHT"), np.full((12, 2), 2.0))
    np.testing.assert_allclose(table.getcol("SIGMA"), 1 / np.sqrt(2))
    assert table.getcolkeywords("TIME_CENTROID") == table.getcolkeywords("TIME")


@pytest.mark.parametrize(
  "mutation",
  [
    "time",
    "uvw_label",
    "reverse",
    "duplicate",
    "incomplete",
    "interval",
    "marker",
    "contradictory",
    "flags",
    "weights",
    "overflow",
    "uvw",
    "exposure",
    "centroid",
    "irregular",
  ],
)
def test_invalid_sources_leave_no_target(tmp_path, mutation):
  tree = make_complete_tree()
  node = tree["part"]
  ds = node.to_dataset().copy(deep=True)
  if mutation == "time":
    ds = ds.assign_coords(time=("time", [1.0, 1.0, 2.0, 3.0], dict(ds.time.attrs)))
  elif mutation == "uvw_label":
    ds = ds.assign_coords(uvw_label=["w", "v", "u"])
  elif mutation == "reverse":
    ds = ds.assign_coords(
      baseline_antenna1_name=("baseline_id", ["B", "A", "B"]),
      baseline_antenna2_name=("baseline_id", ["A", "C", "C"]),
    )
  elif mutation == "duplicate":
    ds = ds.isel(baseline_id=[0, 0, 2])
  elif mutation == "incomplete":
    ds = ds.isel(baseline_id=[0, 1])
  elif mutation == "interval":
    ds.time.attrs["integration_time"]["data"] = np.nan
  elif mutation == "marker":
    ds.attrs["xarray_ms_row_grid"] = {"status": "padded", "missing_rows": 1}
  elif mutation == "contradictory":
    ds.attrs["xarray_ms_row_grid"] = {"status": "complete", "missing_rows": 1}
  elif mutation == "flags":
    ds["FLAG"] = ds.FLAG.astype("uint8") + 2
  elif mutation in ("weights", "overflow"):
    ds["WEIGHT"] = ds.WEIGHT.astype("float64") * (
      -1 if mutation == "weights" else 1e100
    )
  elif mutation == "uvw":
    ds.UVW.values[0, 0, 0] = np.nan
  elif mutation == "exposure":
    ds["EFFECTIVE_INTEGRATION_TIME"] = (
      "time",
      [11.0] * 4,
      {"type": "quantity", "units": "s"},
    )
  elif mutation == "centroid":
    ds["TIME_CENTROID"] = ("time", ds.time.values, {**ds.time.attrs, "scale": "tai"})
  elif mutation == "irregular":
    ds["TIME"] = (("time", "baseline_id"), np.zeros((4, 3)))
  node.ds = ds
  with pytest.raises(FreshMSv2ValidationError):
    write_fresh_msv2(tree, tmp_path / "bad.ms", row_grid="complete", rows_per_batch=2)
  assert not (tmp_path / "bad.ms").exists()
  assert not list(tmp_path.glob(".*.staging"))


@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_invalid_batch(tmp_path, bound):
  with pytest.raises(FreshMSv2ValidationError):
    write_fresh_msv2(
      complete_small(), tmp_path / "bad.ms", row_grid="complete", rows_per_batch=bound
    )


def test_assertion_required(tmp_path):
  with pytest.raises(TypeError):
    write_fresh_msv2(complete_small(), tmp_path / "bad.ms")


def test_overlap_rejected(tmp_path):
  tree = make_tree(("/a", "/b"))
  for part in ("a", "b"):
    tree[part].ds.time.attrs["integration_time"] = {
      "attrs": {"type": "quantity", "units": "s"},
      "data": 1.0,
    }
  with pytest.raises(FreshMSv2ValidationError, match="overlapping"):
    write_fresh_msv2(tree, tmp_path / "overlap.ms", row_grid="complete")


def test_payload_order_and_zero_weight(tmp_path):
  tree = make_complete_tree()
  tree["part"].ds = tree["part"].to_dataset().isel(baseline_id=[2, 0, 1])
  tree["part"]["WEIGHT"] = tree["part"].WEIGHT * 0
  path = tmp_path / "zero.ms"
  write_fresh_msv2(tree, path, row_grid="complete", rows_per_batch=1)
  with Table.from_filename(str(path)) as table:
    np.testing.assert_array_equal(table.getcol("ANTENNA1"), [0, 0, 1] * 4)
    assert not table.getcol("SIGMA").any()


def test_selected_group(tmp_path):
  tree = complete_small()
  ds = tree["part"].to_dataset()
  ds["OTHER"] = ds.VISIBILITY * 3
  ds.attrs["data_groups"]["other"] = {
    **ds.attrs["data_groups"]["base"],
    "correlated_data": "OTHER",
  }
  tree["part"].ds = ds
  path = tmp_path / "selected.ms"
  write_fresh_msv2(tree, path, row_grid="complete", data_group="other")
  with Table.from_filename(str(path)) as table:
    assert (table.getcol("DATA") == 3).all()
    assert "OTHER" not in table.columns()


@pytest.mark.parametrize("stage", ["partial", "verify", "publish"])
def test_failures_cleanup_attempt(tmp_path, monkeypatch, stage):
  tree = make_complete_tree()
  target = tmp_path / "failed.ms"

  def fail(*args, **kwargs):
    raise KeyboardInterrupt("injected failure")

  if stage == "partial":
    original = fresh_write.row_batches

    def broken(*args):
      yield next(original(*args))
      raise KeyboardInterrupt("injected partial write")

    monkeypatch.setattr(fresh_write, "row_batches", broken)
  elif stage == "verify":
    monkeypatch.setattr(fresh_write, "_verify_populated", fail)
  else:
    monkeypatch.setattr(fresh_create, "_publish_no_replace", fail)
  with pytest.raises(KeyboardInterrupt):
    write_fresh_msv2(tree, target, row_grid="complete", rows_per_batch=2)
  assert not target.exists()
  assert not list(tmp_path.glob(".*.staging"))


def test_bounded_batches(tmp_path, monkeypatch):
  calls = []
  original = fresh_write.row_batches

  def bounded(*args):
    for batch in original(*args):
      calls.append(len(batch["TIME"]))
      assert len(batch["TIME"]) <= 2
      yield batch

  monkeypatch.setattr(fresh_write, "row_batches", bounded)
  write_fresh_msv2(
    make_complete_tree(), tmp_path / "bounded.ms", row_grid="complete", rows_per_batch=2
  )
  assert calls == [2, 1] * 8


def test_eager_dask_equivalence(tmp_path):
  tree = make_complete_tree()
  paths = [tmp_path / "eager.ms", tmp_path / "lazy.ms"]
  write_fresh_msv2(tree, paths[0], row_grid="complete", rows_per_batch=2)
  tree["part"].ds = tree["part"].to_dataset().chunk({"time": 1, "baseline_id": 1})
  write_fresh_msv2(tree, paths[1], row_grid="complete", rows_per_batch=2)
  with (
    Table.from_filename(str(paths[0])) as eager,
    Table.from_filename(str(paths[1])) as lazy,
  ):
    for column in ("DATA", "FLAG", "UVW", "TIME", "WEIGHT_SPECTRUM"):
      np.testing.assert_array_equal(eager.getcol(column), lazy.getcol(column))


def test_autocorrelations(tmp_path):
  tree = complete_small()
  ds = tree["part"].to_dataset().isel(baseline_id=[0, 0, 0])
  ds = ds.assign_coords(
    baseline_id=[0, 1, 2],
    baseline_antenna1_name=("baseline_id", ["A", "A", "B"]),
    baseline_antenna2_name=("baseline_id", ["A", "B", "B"]),
  )
  tree["part"].ds = ds
  path = tmp_path / "autos.ms"
  write_fresh_msv2(tree, path, row_grid="complete", rows_per_batch=2)
  with Table.from_filename(str(path)) as table:
    assert table.nrow() == 6
    np.testing.assert_array_equal(table.getcol("ANTENNA1"), [0, 0, 1] * 2)


def test_tai_centroid_exposure_and_observation_range(tmp_path):
  tree = make_complete_tree()
  ds = tree["part"].to_dataset()
  ds.time.attrs["scale"] = "tai"
  ds["TIME_CENTROID"] = ("time", ds.time.values + 1.0, dict(ds.time.attrs))
  ds["EFFECTIVE_INTEGRATION_TIME"] = (
    "time",
    [8.0] * 4,
    {"type": "quantity", "units": "s"},
  )
  tree["part"].ds = ds
  path = tmp_path / "tai.ms"
  write_fresh_msv2(tree, path, row_grid="complete")
  with Table.from_filename(str(path)) as main:
    assert main.getcolkeywords("TIME_CENTROID")["MEASINFO"]["Ref"] == "TAI"
    np.testing.assert_array_equal(
      main.getcol("TIME_CENTROID"), main.getcol("TIME") + 1.0
    )
    np.testing.assert_array_equal(main.getcol("EXPOSURE"), np.full(12, 8.0))
  with Table.from_filename(str(path / "OBSERVATION")) as obs:
    assert obs.getcolkeywords("TIME_RANGE")["MEASINFO"]["Ref"] == "TAI"
    from xarray_ms.backend.msv2.measures_encoders import EpochCoder

    np.testing.assert_array_equal(
      obs.getcol("TIME_RANGE"),
      [EpochCoder.encode_array(ds.time.values[[0, -1]]) + [-5, 5]],
    )


def test_target_race_preserves_other_target(tmp_path, monkeypatch):
  target = tmp_path / "race.ms"
  original = fresh_write._verify_populated

  def race(*args):
    original(*args)
    target.mkdir()
    (target / "owner").write_text("concurrent target")

  monkeypatch.setattr(fresh_write, "_verify_populated", race)
  from xarray_ms.errors import FreshMSv2TargetError

  with pytest.raises(FreshMSv2TargetError):
    write_fresh_msv2(make_complete_tree(), target, row_grid="complete")
  assert (target / "owner").read_text() == "concurrent target"
  assert not list(tmp_path.glob(".*.staging"))


def test_close_failure_cleanup(tmp_path, monkeypatch):
  original = fresh_write.Table

  class Handle:
    def __init__(self, handle):
      self.handle = handle

    def __enter__(self):
      return self.handle.__enter__()

    def __exit__(self, *args):
      self.handle.__exit__(*args)
      raise RuntimeError("injected close failure")

  class FailingTable:
    @staticmethod
    def from_filename(*args, **kwargs):
      return Handle(original.from_filename(*args, **kwargs))

  monkeypatch.setattr(fresh_write, "Table", FailingTable)
  target = tmp_path / "close.ms"
  with pytest.raises(RuntimeError, match="close failure"):
    write_fresh_msv2(make_complete_tree(), target, row_grid="complete")
  assert not target.exists()
  assert not list(tmp_path.glob(".*.staging"))


def test_vector_measure_units_match_native_components(tmp_path):
  target = tmp_path / "vectors.ms"
  write_fresh_msv2(make_complete_tree(), target, row_grid="complete")
  with Table.from_filename(str(target)) as table:
    assert table.getcolkeywords("UVW")["QuantumUnits"] == ["m"] * 3
  with Table.from_filename(str(target / "ANTENNA")) as table:
    assert table.getcolkeywords("POSITION")["QuantumUnits"] == ["m"] * 3
  with Table.from_filename(str(target / "FIELD")) as table:
    for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
      keywords = table.getcolkeywords(column)
      assert keywords["QuantumUnits"] == ["rad"] * 2
      assert keywords["MEASINFO"]["Ref"] == "ICRS"


@pytest.mark.parametrize(
  "values", [["1", "2", "3", "4"], [1j, 2j, 3j, 4j], [0.0, 1e-10, 2e-10, 3e-10]]
)
def test_time_dtype_and_casa_precision(tmp_path, values):
  tree = make_complete_tree()
  ds = tree["part"].to_dataset()
  ds = ds.assign_coords(time=("time", values, dict(ds.time.attrs)))
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError):
    write_fresh_msv2(tree, tmp_path / "bad.ms", row_grid="complete")


def test_no_casacore_or_source_path_writer_imports():
  import ast
  from pathlib import Path

  import xarray_ms.backend.msv2.fresh_rows as rows

  for module in (fresh_create, fresh_write, rows):
    syntax = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(syntax):
      if isinstance(node, ast.Import):
        assert all("casacore" not in item.name for item in node.names)
      elif isinstance(node, ast.ImportFrom):
        assert "casacore" not in (node.module or "")


@pytest.mark.parametrize("offset", [1e-8, np.nextafter(1.0, 2.0) - 1.0])
def test_cross_partition_native_time_overlap_rejected_before_staging(
  tmp_path, monkeypatch, offset
):
  tree = make_tree(("/a", "/b"))
  for path in ("a", "b"):
    ds = tree[path].to_dataset().isel(time=slice(0, 1))
    attrs = dict(ds.time.attrs)
    attrs["integration_time"] = {
      "attrs": {"type": "quantity", "units": "s"},
      "data": 1.0,
    }
    ds = ds.assign_coords(
      time=("time", [1.0 + (offset if path == "b" else 0.0)], attrs)
    )
    tree[path].ds = ds
  assert tree["a"].time.item() != tree["b"].time.item()

  def unexpected_staging(*args):
    pytest.fail("Overlapping native times must be rejected before staging")

  monkeypatch.setattr(fresh_write, "_transaction", unexpected_staging)
  target = tmp_path / "overlap.ms"
  with pytest.raises(FreshMSv2ValidationError, match="overlapping"):
    write_fresh_msv2(tree, target, row_grid="complete")
  assert not target.exists()
  assert not list(tmp_path.glob(".*.staging"))
