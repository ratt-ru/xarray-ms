"""Real arcae table checks of fresh-MS skeleton creation."""

import sys

import numpy as np
import pytest
from arcae.lib.arrow_tables import Table

from tests.test_fresh_plan import make_tree
from xarray_ms import create_fresh_msv2, plan_fresh_msv2
from xarray_ms.backend.msv2 import fresh_create
from xarray_ms.errors import FreshMSv2TargetError, FreshMSv2ValidationError


def test_fresh_skeleton_publishes_columns_metadata_and_keywords(tmp_path):
  tree = make_tree(extra=True)
  target = tmp_path / "new.ms"
  plan = plan_fresh_msv2(
    tree, target, additional_visibility={"corrected": "CORRECTED_DATA"}
  )
  assert plan.visibility_columns == (
    ("DATA", np.dtype("complex64").str, ((2, 2),)),
    ("CORRECTED_DATA", np.dtype("complex64").str, ((2, 2),)),
  )
  assert create_fresh_msv2(plan) == str(target)
  with Table.from_filename(str(target)) as main:
    assert main.nrow() == 0
    assert {"DATA", "CORRECTED_DATA", "FLAG", "WEIGHT_SPECTRUM", "UVW"} <= set(
      main.columns()
    )
    for column, shape, kind in (
      ("DATA", [2, 2], "complex"),
      ("CORRECTED_DATA", [2, 2], "complex"),
      ("FLAG", [2, 2], "boolean"),
      ("WEIGHT_SPECTRUM", [2, 2], "float"),
      ("UVW", [3], "double"),
    ):
      descriptor = main.getcoldesc(column)
      assert descriptor["shape"] == shape
      assert descriptor["valueType"] == kind
      assert descriptor["dataManagerType"] == "TiledColumnStMan"
    assert main.getcolkeywords("TIME")["MEASINFO"]["Ref"] == "UTC"
    assert main.getcolkeywords("UVW")["MEASINFO"]["Ref"] == "J2000"
    assert set(fresh_create.SUBTABLES) <= set(main.getkeywords())
  with Table.from_filename(str(target / "FIELD")) as field:
    assert field.getcol("SOURCE_ID").tolist() == [0]
    assert field.getcol("NAME").tolist() == ["field"]
    assert field.getcolkeywords("PHASE_DIR")["MEASINFO"]["Ref"] == "ICRS"
  with Table.from_filename(str(target / "SOURCE")) as source:
    assert source.getcol("SOURCE_ID").tolist() == [0]
    assert source.getcol("NAME").tolist() == ["source"]
    assert source.getcolkeywords("DIRECTION")["MEASINFO"]["Ref"] == "ICRS"
  with Table.from_filename(str(target / "DATA_DESCRIPTION")) as dd:
    assert dd.getcol("SPECTRAL_WINDOW_ID").tolist() == [0]
    assert dd.getcol("POLARIZATION_ID").tolist() == [0]
  assert not list(tmp_path.glob("*.staging"))


def test_ragged_spectral_polarization_and_feed_cells(tmp_path):
  tree = make_tree(("/a", "/b"))
  second = tree["b"]
  second.ds = second.to_dataset().isel(frequency=slice(0, 1), polarization=slice(0, 1))
  second.ds.frequency.attrs["spectral_window_name"] = "other_spw"
  antenna = tree["b/antenna_xds"]
  antenna.ds = antenna.to_dataset().isel(receptor_label=slice(0, 1))
  target = tmp_path / "ragged.ms"
  plan = plan_fresh_msv2(tree, target)

  create_fresh_msv2(plan)

  with Table.from_filename(str(target)) as main:
    assert main.nrow() == 0
    assert set(fresh_create.SUBTABLES) <= set(main.getkeywords())
    for column in ("DATA", "FLAG", "WEIGHT_SPECTRUM"):
      assert main.getcoldesc(column)["dataManagerType"] == "TiledShapeStMan"
  with Table.from_filename(str(target / "SPECTRAL_WINDOW")) as spw:
    assert spw.getcol("NUM_CHAN").tolist() == [
      len(row.channel_frequency) for row in plan.metadata.spectral_window_rows
    ]
    assert sorted(spw.getcol("NUM_CHAN").tolist()) == [1, 2]
    assert spw.getcol("MEAS_FREQ_REF").tolist() == [5, 5]
    assert [
      spw.getcol("CHAN_FREQ", index=(slice(i, i + 1),)).shape[1] for i in range(2)
    ] == spw.getcol("NUM_CHAN").tolist()
  with Table.from_filename(str(target / "POLARIZATION")) as pol:
    assert sorted(pol.getcol("NUM_CORR").tolist()) == [1, 2]
    assert [
      pol.getcol("CORR_TYPE", index=(slice(i, i + 1),)).shape[1] for i in range(2)
    ] == pol.getcol("NUM_CORR").tolist()
  with Table.from_filename(str(target / "FEED")) as feed:
    assert sorted(feed.getcol("NUM_RECEPTORS").tolist()) == [1, 1, 2, 2]
    assert [
      feed.getcol("POL_RESPONSE", index=(slice(i, i + 1),)).shape[1:] for i in range(4)
    ] == [(n, n) for n in feed.getcol("NUM_RECEPTORS")]
  with Table.from_filename(str(target / "DATA_DESCRIPTION")) as dd:
    assert dd.getcol("SPECTRAL_WINDOW_ID").tolist() == [
      row.spectral_window_id for row in plan.metadata.data_description_rows
    ]
    assert dd.getcol("POLARIZATION_ID").tolist() == [
      row.polarization_id for row in plan.metadata.data_description_rows
    ]
    assert sorted(dd.getcol("SPECTRAL_WINDOW_ID").tolist()) == [0, 1]
    assert sorted(dd.getcol("POLARIZATION_ID").tolist()) == [0, 1]


def test_channel_frame_code_is_independent_of_reference_frequency_frame(tmp_path):
  tree = make_tree()
  tree["part"].ds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "lsrk"
  target = tmp_path / "independent-frame.ms"
  create_fresh_msv2(plan_fresh_msv2(tree, target))
  with Table.from_filename(str(target / "SPECTRAL_WINDOW")) as spw:
    assert spw.getcol("MEAS_FREQ_REF").tolist() == [5]
    assert spw.getcolkeywords("REF_FREQUENCY")["MEASINFO"]["Ref"] == "LSRK"


def test_feed_beam_offset_uses_direction_by_receptor_axis_order(tmp_path):
  tree = make_tree()
  part = tree["part"]
  part.ds = part.to_dataset().isel(polarization=slice(0, 1))
  antenna = tree["part/antenna_xds"]
  antenna.ds = antenna.to_dataset().isel(receptor_label=slice(0, 1))
  target = tmp_path / "single-receptor.ms"

  create_fresh_msv2(plan_fresh_msv2(tree, target))

  with Table.from_filename(str(target)) as main:
    for column in ("DATA", "FLAG", "WEIGHT_SPECTRUM"):
      assert main.getcoldesc(column)["shape"] == [2, 1]
    assert main.getcoldesc("UVW")["shape"] == [3]
  with Table.from_filename(str(target / "FEED")) as feed:
    assert feed.getcol("BEAM_OFFSET").shape == (2, 2, 1)


def test_existing_target_is_untouched(tmp_path):
  tree = make_tree()
  target = tmp_path / "new.ms"
  plan = plan_fresh_msv2(tree, target)
  target.mkdir()
  (target / "owned").write_text("keep")
  with pytest.raises(FreshMSv2TargetError):
    create_fresh_msv2(plan)
  assert (target / "owned").read_text() == "keep"


def test_cross_partition_dtype_conflict_is_preflight_error(tmp_path):
  tree = make_tree(("/a", "/b"))
  tree["b"]["VISIBILITY"] = tree["b"].ds["VISIBILITY"].astype(np.complex128)
  with pytest.raises(FreshMSv2ValidationError, match="incompatible partition dtypes"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert not (tmp_path / "new.ms").exists()


def test_verification_failure_removes_only_attempt_staging(tmp_path, monkeypatch):
  target = tmp_path / "new.ms"
  other = tmp_path / ".other.ms.staging"
  other.mkdir()
  plan = plan_fresh_msv2(make_tree(), target)

  def fail_verification(*args):
    raise ValueError("verification failed")

  monkeypatch.setattr(fresh_create, "_verify", fail_verification)
  with pytest.raises(ValueError, match="verification failed"):
    create_fresh_msv2(plan)
  assert not target.exists()
  assert list(tmp_path.iterdir()) == [other]


def test_target_appearing_during_verification_is_preserved(tmp_path, monkeypatch):
  target = tmp_path / "new.ms"
  plan = plan_fresh_msv2(make_tree(), target)
  original_verify = fresh_create._verify

  def competing_target(*args):
    original_verify(*args)
    target.mkdir()
    (target / "owned").write_text("keep")

  monkeypatch.setattr(fresh_create, "_verify", competing_target)
  with pytest.raises(FreshMSv2TargetError):
    create_fresh_msv2(plan)
  assert (target / "owned").read_text() == "keep"
  assert not list(tmp_path.glob(".*.staging"))


@pytest.mark.skipif(
  sys.platform not in ("linux", "darwin"), reason="atomic no-clobber rename"
)
def test_publish_is_atomic_no_clobber(tmp_path):
  staged = tmp_path / "private.staging"
  target = tmp_path / "new.ms"
  staged.mkdir()
  target.mkdir()
  with pytest.raises(FreshMSv2TargetError):
    fresh_create._publish_no_replace(str(staged), str(target))
  assert staged.is_dir() and target.is_dir()


def test_one_source_with_distinct_field_directions_is_rejected(tmp_path):
  tree = make_tree(("/a", "/b"))
  # Distinct FIELD names avoid the separate duplicate-FIELD conflict.
  field_node = tree["b/field_and_source_base_xds"]
  field_ds = field_node.to_dataset().copy(deep=True)
  field_ds["FIELD_PHASE_CENTER_DIRECTION"].data[:] = [[0.3, 0.4]]
  field_node.ds = field_ds.assign_coords(field_name=["other"])
  node = tree["b"]
  node.ds = node.to_dataset().assign_coords(field_name=("time", ["other", "other"]))
  with pytest.raises(FreshMSv2ValidationError, match="one fixed SOURCE direction"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_optional_focus_length_is_not_silently_dropped(tmp_path):
  tree = make_tree()
  antenna = tree["part/antenna_xds"]
  ds = antenna.to_dataset().copy()
  ds["ANTENNA_FOCUS_LENGTH"] = (
    "antenna_name",
    [1.0, 1.0],
    {"type": "quantity", "units": "m"},
  )
  antenna.ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="FEED::FOCUS_LENGTH"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert not (tmp_path / "new.ms").exists()
