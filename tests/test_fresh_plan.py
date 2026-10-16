import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

from xarray_ms import plan_fresh_msv2
from xarray_ms.errors import (
  FreshMSv2TargetError,
  FreshMSv2ValidationError,
  MeasureReferenceColumnRequired,
)

DIMS = ("time", "baseline_id", "frequency", "polarization")
COORDS = {
  "time": [1, 2],
  "baseline_id": [0],
  "frequency": [10, 20],
  "polarization": ["XX", "YY"],
  "uvw_label": ["u", "v", "w"],
  "baseline_antenna1_name": ("baseline_id", ["A"]),
  "baseline_antenna2_name": ("baseline_id", ["B"]),
  "field_name": ("time", ["field", "field"]),
  "scan_name": ("time", ["1", "1"], {"scan_intents": ["OBSERVE_TARGET"]}),
}


def make_tree(paths=("/part",), extra=False):
  datasets = {}
  for path in paths:
    groups = {
      "base": {
        "correlated_data": "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
        "field_and_source": "field_and_source_base_xds",
      }
    }
    variables = {
      "VISIBILITY": (DIMS, np.ones((2, 1, 2, 2), dtype=np.complex64)),
      "FLAG": (DIMS, np.zeros((2, 1, 2, 2), dtype=bool)),
      "WEIGHT": (DIMS, np.ones((2, 1, 2, 2))),
      "UVW": (
        ("time", "baseline_id", "uvw_label"),
        np.zeros((2, 1, 3)),
        {"type": "uvw", "units": "m", "frame": "fk5"},
      ),
    }
    if extra:
      variables["CORRECTED"] = (DIMS, np.ones((2, 1, 2, 2), dtype=np.complex64))
      groups["corrected"] = {"correlated_data": "CORRECTED"}
    datasets[path] = xr.Dataset(
      variables,
      coords={
        **COORDS,
        "time": (
          "time",
          [1, 2],
          {"type": "time", "units": "s", "format": "unix", "scale": "utc"},
        ),
        "frequency": (
          "frequency",
          [10, 20],
          {
            "type": "spectral_coord",
            "units": "Hz",
            "observer": "TOPO",
            "spectral_window_name": "spw",
            "spectral_window_intents": ["science"],
            "reference_frequency": {
              "attrs": {"type": "spectral_coord", "units": "Hz", "observer": "TOPO"},
              "data": 10.0,
            },
            "channel_width": {
              "attrs": {"type": "quantity", "units": "Hz"},
              "data": 10.0,
            },
          },
        ),
      },
      attrs={
        "type": "visibility",
        "data_groups": groups,
        "observation_info": {
          "observer": ["Observer"],
          "project_UID": "project",
          "release_date": "2020-01-01T00:00:00+00:00",
        },
        "processor_info": {"type": "CORRELATOR", "sub_type": "test"},
      },
    )
    datasets[f"{path}/field_and_source_base_xds"] = xr.Dataset(
      {
        "FIELD_PHASE_CENTER_DIRECTION": (
          ("field_name", "sky_dir_label"),
          [[0.1, 0.2]],
          {"type": "sky_coord", "units": "rad", "frame": "icrs"},
        )
      },
      coords={
        "field_name": ["field"],
        "sky_dir_label": ["ra", "dec"],
        "source_name": ("field_name", ["source"]),
      },
      attrs={"type": "field_and_source"},
    )
    datasets[f"{path}/antenna_xds"] = xr.Dataset(
      {
        "ANTENNA_POSITION": (
          ("antenna_name", "cartesian_pos_label"),
          [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
          {"type": "location", "units": "m", "frame": "ITRS"},
        ),
        "ANTENNA_DISH_DIAMETER": (
          "antenna_name",
          [12.0, 12.0],
          {"type": "quantity", "units": "m"},
        ),
        "ANTENNA_EFFECTIVE_DISH_DIAMETER": (
          "antenna_name",
          [12.0, 12.0],
          {"type": "quantity", "units": "m"},
        ),
        "ANTENNA_RECEPTOR_ANGLE": (
          ("antenna_name", "receptor_label"),
          [[0.0, 0.0], [0.0, 0.0]],
          {"type": "quantity", "units": "rad"},
        ),
      },
      coords={
        "antenna_name": ["A", "B"],
        "station_name": ("antenna_name", ["A", "B"]),
        "mount": ("antenna_name", ["ALT-AZ", "ALT-AZ"]),
        "telescope_name": ("antenna_name", ["scope", "scope"]),
        "cartesian_pos_label": ["x", "y", "z"],
        "receptor_label": ["pol_0", "pol_1"],
        "polarization_type": (
          ("antenna_name", "receptor_label"),
          [["X", "Y"], ["X", "Y"]],
        ),
      },
      attrs={"type": "antenna", "overall_telescope_name": "scope"},
    )
  return xr.DataTree.from_dict(datasets)


def test_single_partition_is_pure_and_immutable(tmp_path):
  target = tmp_path / "new.ms"
  plan = make_tree().plan_msv2(target)
  assert plan.target == str(target)
  assert plan.data_group == "base"
  assert plan.visibility_mappings == (("base", "DATA"),)
  assert plan.partitions[0].path == "/part"
  assert plan.partitions[0].field_and_source == "/part/field_and_source_base_xds"
  assert plan.partitions[0].antenna == "/part/antenna_xds"
  assert plan.partitions[0].correlated_data == "VISIBILITY"
  assert [(m.column, m.frame) for m in plan.partitions[0].measures] == [
    ("MAIN::TIME", "UTC"),
    ("SPECTRAL_WINDOW::CHAN_FREQ", "TOPO"),
    ("MAIN::UVW", "J2000"),
    ("SPECTRAL_WINDOW::REF_FREQUENCY", "TOPO"),
    ("ANTENNA::POSITION", "ITRF"),
    ("FIELD::PHASE_DIR", "ICRS"),
  ]
  assert not target.exists()
  with pytest.raises(FrozenInstanceError):
    plan.target = "elsewhere"
  with pytest.raises(FrozenInstanceError):
    plan.partitions[0].path = "elsewhere"


def test_processing_set_shaped_tree_has_deterministic_partition_order(tmp_path):
  tree = make_tree(("/processing_set/b/part", "/processing_set/a/part"))
  plan = plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert tuple(p.path for p in plan.partitions) == (
    "/processing_set/a/part",
    "/processing_set/b/part",
  )


def test_cross_partition_reference_conflict_requires_column(tmp_path):
  tree = make_tree(("/one", "/two"))
  tree["two"].ds["UVW"].attrs["frame"] = "icrs"
  with pytest.raises(
    MeasureReferenceColumnRequired, match=r"/one.*UVW.*/two.*UVW.*reference column"
  ):
    plan_fresh_msv2(tree, tmp_path / "fresh.ms")
  assert not (tmp_path / "fresh.ms").exists()


def test_reference_frequency_has_independent_fixed_frame(tmp_path):
  tree = make_tree()
  tree["part"].ds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "lsrk"
  plan = plan_fresh_msv2(tree, tmp_path / "fresh.ms")
  assert plan.metadata.spectral_window_rows[0].frame == "TOPO"
  assert plan.metadata.spectral_window_rows[0].reference_frame == "LSRK"
  assert ("SPECTRAL_WINDOW::REF_FREQUENCY", "LSRK") in [
    (measure.column, measure.frame) for measure in plan.partitions[0].measures
  ]


def test_cross_partition_reference_frequency_conflict(tmp_path):
  tree = make_tree(("/one", "/two"))
  tree["two"].ds.frequency.attrs["reference_frequency"]["attrs"]["observer"] = "lsrk"
  with pytest.raises(
    MeasureReferenceColumnRequired,
    match=r"/one.*reference_frequency.*/two.*reference_frequency.*reference column",
  ):
    plan_fresh_msv2(tree, tmp_path / "fresh.ms")


def test_optional_metadata_measures_are_encoded(tmp_path):
  tree = make_tree()
  measures = plan_fresh_msv2(tree, tmp_path / "fresh.ms").partitions[0].measures
  assert [(m.column, m.frame) for m in measures[-2:]] == [
    ("ANTENNA::POSITION", "ITRF"),
    ("FIELD::PHASE_DIR", "ICRS"),
  ]


def test_field_reference_root_relative_and_absolute(tmp_path):
  tree = make_tree(("/processing_set/part",))
  group = tree["/processing_set/part"].attrs["data_groups"]["base"]
  group["field_and_source"] = "processing_set/part/field_and_source_base_xds"
  assert plan_fresh_msv2(tree, tmp_path / "new.ms").partitions[0].field_and_source == (
    "/processing_set/part/field_and_source_base_xds"
  )
  group["field_and_source"] = "/processing_set/part/field_and_source_base_xds"
  assert plan_fresh_msv2(tree, tmp_path / "new.ms").partitions[0].field_and_source == (
    "/processing_set/part/field_and_source_base_xds"
  )


def test_existing_target_and_invalid_target(tmp_path):
  with pytest.raises(FreshMSv2TargetError, match="already exists"):
    plan_fresh_msv2(make_tree(), tmp_path)
  with pytest.raises(FreshMSv2TargetError, match="nonempty"):
    plan_fresh_msv2(make_tree(), "")
  with pytest.raises(FreshMSv2TargetError, match="nonempty"):
    plan_fresh_msv2(make_tree(), 123)
  with pytest.raises(FreshMSv2TargetError, match="parent is missing"):
    plan_fresh_msv2(make_tree(), tmp_path / "missing" / "new.ms")
  parent_file = tmp_path / "file"
  parent_file.write_text("not a directory")
  with pytest.raises(FreshMSv2TargetError, match="not a directory"):
    plan_fresh_msv2(make_tree(), parent_file / "new.ms")


def test_missing_correlated_dataset_group_role_and_reference(tmp_path):
  target = tmp_path / "new.ms"
  with pytest.raises(FreshMSv2ValidationError, match="No correlated datasets"):
    plan_fresh_msv2(xr.DataTree(), target)
  tree = make_tree()
  with pytest.raises(FreshMSv2ValidationError, match="no group"):
    plan_fresh_msv2(tree, target, data_group="missing")
  del tree["part"].attrs["data_groups"]["base"]["weight"]
  with pytest.raises(FreshMSv2ValidationError, match="lacks role 'weight'"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part"].attrs["data_groups"]["base"]["flag"] = "MISSING"
  with pytest.raises(FreshMSv2ValidationError, match="missing data variable"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part"].attrs["data_groups"]["base"]["field_and_source"] = "missing"
  with pytest.raises(FreshMSv2ValidationError, match="node 'missing' does not exist"):
    plan_fresh_msv2(tree, target)


def test_canonical_coordinates_dimensions_and_dtype(tmp_path):
  target = tmp_path / "new.ms"
  tree = make_tree()
  tree["part"].ds = tree["part"].ds.drop_vars("frequency")
  with pytest.raises(FreshMSv2ValidationError, match="coordinate 'frequency'"):
    plan_fresh_msv2(tree, target)
  for coord in (
    "baseline_antenna1_name",
    "baseline_antenna2_name",
    "field_name",
    "scan_name",
  ):
    tree = make_tree()
    tree["part"].ds = tree["part"].ds.drop_vars(coord)
    with pytest.raises(FreshMSv2ValidationError, match=coord):
      plan_fresh_msv2(tree, target)
  tree = make_tree()
  ds = tree["part"].to_dataset().drop_vars("scan_name")
  ds = ds.assign_coords(scan_name=("baseline_id", ["scan"]))
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="scan_name.*time"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds["VISIBILITY"] = ds.VISIBILITY.transpose(
    "frequency", "time", "baseline_id", "polarization"
  )
  tree["part"].ds = ds
  with pytest.raises(
    FreshMSv2ValidationError, match="base visibility requires dimensions"
  ):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds["VISIBILITY"] = ds.VISIBILITY.real
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="complex64 or complex128 dtype"):
    plan_fresh_msv2(tree, target)


def test_additional_visibility_and_mapping_validation(tmp_path):
  target = tmp_path / "new.ms"
  tree = make_tree(extra=True)
  plan = plan_fresh_msv2(
    tree, target, additional_visibility={"corrected": "corrected_data"}
  )
  assert plan.visibility_mappings == (("base", "DATA"), ("corrected", "corrected_data"))
  assert plan.partitions[0].additional_correlated_data == (("corrected", "CORRECTED"),)
  for mappings, message in (
    ({"corrected": "DATA"}, "writer-controlled"),
    ({"corrected": "flag"}, "writer-controlled"),
    ({"corrected": "UVW"}, "writer-controlled"),
    ({"corrected": "FLAG"}, "writer-controlled"),
    ({"corrected": "WEIGHT_SPECTRUM"}, "writer-controlled"),
    ({"corrected": "corrected_data", "absent": "OTHER_DATA"}, "no group"),
  ):
    with pytest.raises(FreshMSv2ValidationError, match=message):
      plan_fresh_msv2(tree, target, additional_visibility=mappings)
  ds = tree["part"].to_dataset()
  ds["CORRECTED"] = ds.CORRECTED.real
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="must match the base"):
    plan_fresh_msv2(tree, target, additional_visibility={"corrected": "CORRECTED_DATA"})


def test_additional_destination_collision(tmp_path):
  tree = make_tree(extra=True)
  tree["part"].attrs["data_groups"]["other"] = {
    **tree["part"].attrs["data_groups"]["corrected"]
  }
  with pytest.raises(FreshMSv2ValidationError, match="duplicated"):
    plan_fresh_msv2(
      tree,
      tmp_path / "new.ms",
      additional_visibility={"corrected": "MODEL_DATA", "other": "model_data"},
    )


@pytest.mark.parametrize("column", ["ANTENNA3", "CORRECTED_WEIGHT_SPECTRUM"])
def test_canonical_non_visibility_destination_is_rejected_in_preflight(
  tmp_path, column
):
  with pytest.raises(FreshMSv2ValidationError, match="writer-controlled"):
    plan_fresh_msv2(
      make_tree(extra=True),
      tmp_path / "new.ms",
      additional_visibility={"corrected": column},
    )
  assert not (tmp_path / "new.ms").exists()


def test_missing_additional_visibility_reference_and_invalid_mapping_name(tmp_path):
  tree = make_tree(extra=True)
  target = tmp_path / "new.ms"
  with pytest.raises(FreshMSv2ValidationError, match="names must be strings"):
    plan_fresh_msv2(tree, target, additional_visibility={1: "MODEL_DATA"})
  tree["part"].attrs["data_groups"]["corrected"]["correlated_data"] = "MISSING"
  with pytest.raises(FreshMSv2ValidationError, match="missing data variable"):
    plan_fresh_msv2(tree, target, additional_visibility={"corrected": "MODEL_DATA"})
  assert not target.exists()


def test_required_metadata_nodes_and_field_type(tmp_path):
  target = tmp_path / "new.ms"
  for name in ("observation_info", "processor_info"):
    tree = make_tree()
    tree["part"].attrs.pop(name)
    with pytest.raises(FreshMSv2ValidationError, match=name):
      plan_fresh_msv2(tree, target)
    tree["part"].attrs[name] = "not a mapping"
    with pytest.raises(FreshMSv2ValidationError, match=name):
      plan_fresh_msv2(tree, target)
  tree = make_tree()
  del tree["part"]["antenna_xds"]
  with pytest.raises(FreshMSv2ValidationError, match="antenna_xds"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part/antenna_xds"].attrs["type"] = "other"
  with pytest.raises(FreshMSv2ValidationError, match="antenna_xds"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part/field_and_source_base_xds"].attrs["type"] = "other"
  with pytest.raises(FreshMSv2ValidationError, match="invalid type"):
    plan_fresh_msv2(tree, target)
  tree["part/field_and_source_base_xds"].attrs["type"] = "field_and_source_ephemeris"
  with pytest.raises(
    FreshMSv2ValidationError, match="invalid type|unsupported optional"
  ):
    plan_fresh_msv2(tree, target)


@pytest.mark.parametrize(
  ("role", "replacement"),
  [
    ("FLAG", lambda ds: ds.FLAG.astype("int8")),
    (
      "FLAG",
      lambda ds: ds.FLAG.transpose("frequency", "time", "baseline_id", "polarization"),
    ),
    ("WEIGHT", lambda ds: ds.WEIGHT.astype("int64")),
    (
      "WEIGHT",
      lambda ds: ds.WEIGHT.transpose(
        "frequency", "time", "baseline_id", "polarization"
      ),
    ),
    ("UVW", lambda ds: ds.UVW.astype("complex64")),
    ("UVW", lambda ds: ds.UVW.transpose("uvw_label", "time", "baseline_id")),
  ],
)
def test_malformed_base_roles(tmp_path, role, replacement):
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds[role] = replacement(ds)
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match=role):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


@pytest.mark.parametrize(
  ("role", "dtype", "accepted"),
  [
    ("VISIBILITY", "complex64", True),
    ("VISIBILITY", "complex128", True),
    ("VISIBILITY", "clongdouble", False),
    ("FLAG", "bool", True),
    ("FLAG", "uint8", True),
    ("FLAG", "int8", False),
    ("FLAG", "uint16", False),
    ("WEIGHT", "float32", True),
    ("WEIGHT", "float64", True),
    ("WEIGHT", "float16", False),
    ("WEIGHT", "complex64", False),
    ("UVW", "float32", True),
    ("UVW", "float64", True),
    ("UVW", "float16", False),
    ("UVW", "complex64", False),
  ],
)
def test_base_role_supported_dtypes(tmp_path, role, dtype, accepted):
  if dtype == "clongdouble" and np.dtype(dtype) == np.dtype("complex128"):
    pytest.skip("platform long complex has complex128 precision")
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds[role] = ds[role].astype(dtype)
  tree["part"].ds = ds
  if accepted:
    assert plan_fresh_msv2(tree, tmp_path / "new.ms").partitions[0].path == "/part"
  else:
    with pytest.raises(
      FreshMSv2ValidationError,
      match=role if role != "VISIBILITY" else "base visibility",
    ):
      plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_real_backend_tree_can_be_planned_without_creation(simmed_ms, tmp_path):
  target = tmp_path / "fresh.ms"
  with xr.open_datatree(simmed_ms, auto_corrs=True) as tree:
    for partition in tree.children.values():
      del partition["phased_array_xds"]
    plan = tree.plan_msv2(target)
    assert plan.target == str(target)
    assert plan.partitions
    assert all(p.correlated_data == "VISIBILITY" for p in plan.partitions)
  assert not target.exists()


def test_real_backend_phased_array_is_refused(simmed_ms, tmp_path):
  target = tmp_path / "fresh.ms"
  with xr.open_datatree(simmed_ms, auto_corrs=True) as tree:
    with pytest.raises(
      FreshMSv2ValidationError,
      match="unsupported optional metadata.*phased_array_xds",
    ):
      tree.plan_msv2(target)
  assert not target.exists()


def test_missing_uvw_label_coordinate(tmp_path):
  tree = make_tree()
  tree["part"].ds = tree["part"].ds.drop_vars("uvw_label")
  with pytest.raises(FreshMSv2ValidationError, match="uvw_label"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_lazy_visibility_payloads_are_not_computed(tmp_path):
  import dask.array as da
  from dask import delayed

  @delayed
  def fail_on_compute():
    raise AssertionError("planner computed visibility data")

  tree = make_tree(extra=True)
  ds = tree["part"].to_dataset()
  lazy = da.from_delayed(fail_on_compute(), shape=(2, 1, 2, 2), dtype=np.complex64)
  ds["VISIBILITY"] = (DIMS, lazy)
  ds["CORRECTED"] = (DIMS, lazy)
  tree["part"].ds = ds
  plan = plan_fresh_msv2(
    tree, tmp_path / "new.ms", additional_visibility={"corrected": "MODEL_DATA"}
  )
  assert plan.partitions[0].additional_correlated_data == (("corrected", "CORRECTED"),)


def test_lazy_measure_payload_is_not_computed(tmp_path):
  import dask.array as da
  from dask import delayed

  @delayed
  def fail_on_compute():
    raise AssertionError("planner computed UVW measure data")

  tree = make_tree()
  ds = tree["part"].to_dataset()
  attrs = dict(ds.UVW.attrs)
  ds["UVW"] = (
    ("time", "baseline_id", "uvw_label"),
    da.from_delayed(fail_on_compute(), shape=(2, 1, 3), dtype=np.float64),
    attrs,
  )
  tree["part"].ds = ds
  plan = plan_fresh_msv2(tree, tmp_path / "fresh.ms")
  assert plan.partitions[0].measures[2].frame == "J2000"


def test_metadata_dedup_and_foreign_keys_independent_of_input_order(tmp_path):
  first = plan_fresh_msv2(make_tree(("/z", "/a")), tmp_path / "one.ms")
  second = plan_fresh_msv2(make_tree(("/a", "/z")), tmp_path / "two.ms")
  assert first.metadata == second.metadata
  assert tuple(p.foreign_keys for p in first.partitions) == tuple(
    p.foreign_keys for p in second.partitions
  )
  rows = first.metadata
  assert len(rows.antenna_rows) == len(rows.feed_rows) == 2
  assert len(rows.field_rows) == len(rows.source_rows) == 1
  assert len(rows.spectral_window_rows) == len(rows.polarization_rows) == 1
  assert len(rows.data_description_rows) == len(rows.observation_rows) == 1
  assert len(rows.processor_rows) == len(rows.state_rows) == 1
  for partition in first.partitions:
    keys = partition.foreign_keys
    assert keys.antenna1_ids == (0,)
    assert keys.antenna2_ids == (1,)
    assert keys.field_ids == (0, 0)
    assert keys.scan_numbers == (1, 1)
    for ids, count in (
      (keys.antenna1_ids + keys.antenna2_ids, len(rows.antenna_rows)),
      (keys.field_ids, len(rows.field_rows)),
      ((keys.data_desc_id,), len(rows.data_description_rows)),
      ((keys.observation_id,), len(rows.observation_rows)),
      ((keys.processor_id,), len(rows.processor_rows)),
      ((keys.state_id,), len(rows.state_rows)),
    ):
      assert all(0 <= value < count for value in ids)
    feed_pairs = {(row.antenna_id, row.feed_id) for row in rows.feed_rows}
    assert set(zip(keys.antenna1_ids, keys.feed1_ids, strict=True)) <= feed_pairs
    assert set(zip(keys.antenna2_ids, keys.feed2_ids, strict=True)) <= feed_pairs
  assert rows.field_rows[0].source_id == 0
  assert rows.feed_rows[0].spectral_window_id == 0


def test_distinct_spw_polarization_and_feed_configs(tmp_path):
  tree = make_tree(("/a", "/b"))
  ds = tree["b"].to_dataset()
  ds.frequency.attrs["spectral_window_name"] = "other"
  ds = ds.assign_coords(polarization=["RR", "LL"])
  tree["b"].ds = ds
  antenna = tree["b/antenna_xds"].to_dataset()
  antenna["ANTENNA_RECEPTOR_ANGLE"] = antenna.ANTENNA_RECEPTOR_ANGLE + 0.5
  antenna = antenna.assign_coords(
    polarization_type=(
      ("antenna_name", "receptor_label"),
      [["R", "L"], ["R", "L"]],
    )
  )
  tree["b/antenna_xds"].ds = antenna
  plan = plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert len(plan.metadata.spectral_window_rows) == 2
  assert len(plan.metadata.polarization_rows) == 2
  assert len(plan.metadata.data_description_rows) == 2
  assert len(plan.metadata.feed_rows) == 4
  assert (
    plan.partitions[0].foreign_keys.data_desc_id
    != plan.partitions[1].foreign_keys.data_desc_id
  )
  assert {r.feed_id for r in plan.metadata.feed_rows} == {0}
  assert {
    (r.antenna_id, r.feed_id, r.spectral_window_id) for r in plan.metadata.feed_rows
  } == {
    (0, 0, 0),
    (0, 0, 1),
    (1, 0, 0),
    (1, 0, 1),
  }
  assert {p.foreign_keys.feed1_ids[0] for p in plan.partitions} == {0}
  assert {p.foreign_keys.feed2_ids[0] for p in plan.partitions} == {0}


@pytest.mark.parametrize("kind", ["antenna", "field"])
def test_cross_partition_semantic_conflicts(tmp_path, kind):
  tree = make_tree(("/a", "/b"))
  if kind == "antenna":
    node = tree["b/antenna_xds"]
    ds = node.to_dataset()
    ds["ANTENNA_DISH_DIAMETER"] = (
      "antenna_name",
      [13.0, 12.0],
      {"type": "quantity", "units": "m"},
    )
    ds["ANTENNA_EFFECTIVE_DISH_DIAMETER"] = (
      "antenna_name",
      [13.0, 12.0],
      {"type": "quantity", "units": "m"},
    )
    node.ds = ds
    pattern = "antenna 'A'.*conflicts with partition /a"
  else:
    node = tree["b/field_and_source_base_xds"]
    ds = node.to_dataset()
    ds["source_name"] = ("field_name", ["another"])
    node.ds = ds
    pattern = "field 'field'.*conflicts with partition /a"
  with pytest.raises(FreshMSv2ValidationError, match=pattern):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_missing_relationships_and_scan_number(tmp_path):
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds = ds.assign_coords(baseline_antenna1_name=("baseline_id", ["missing"]))
  tree["part"].ds = ds
  with pytest.raises(
    FreshMSv2ValidationError, match="baseline antenna 'missing'.*antenna_xds"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds = ds.assign_coords(field_name=("time", ["absent", "field"]))
  tree["part"].ds = ds
  with pytest.raises(
    FreshMSv2ValidationError, match="field 'absent'.*field_and_source"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds = ds.assign_coords(
    scan_name=("time", ["1", "1.5"], {"scan_intents": ["OBSERVE_TARGET"]})
  )
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="scan_name '1.5'.*integral"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_effective_diameter_must_not_be_dropped(tmp_path):
  tree = make_tree()
  node = tree["part/antenna_xds"]
  ds = node.to_dataset()
  ds["ANTENNA_EFFECTIVE_DISH_DIAMETER"] = (
    "antenna_name",
    [11.0, 12.0],
    {"type": "quantity", "units": "m"},
  )
  node.ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="EFFECTIVE_DISH_DIAMETER differs"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_unknown_source_and_irregular_channel_width(tmp_path):
  tree = make_tree()
  field = tree["part/field_and_source_base_xds"]
  fd = field.to_dataset()
  fd["source_name"] = ("field_name", ["UNKNOWN"])
  field.ds = fd
  node = tree["part"]
  ds = node.to_dataset()
  ds.frequency.attrs["channel_width"]["data"] = np.nan
  ds["CHANNEL_WIDTH"] = (
    "frequency",
    [9.0, 11.0],
    {"type": "quantity", "units": "Hz"},
  )
  ds["EFFECTIVE_CHANNEL_WIDTH"] = (
    "frequency",
    [8.0, 10.0],
    {"type": "quantity", "units": "Hz"},
  )
  ds.frequency.attrs["effective_channel_width"] = "EFFECTIVE_CHANNEL_WIDTH"
  node.ds = ds
  rows = plan_fresh_msv2(tree, tmp_path / "new.ms").metadata
  assert rows.source_rows == ()
  assert rows.field_rows[0].source_id == -1
  assert rows.spectral_window_rows[0].channel_width == (9.0, 11.0)
  assert rows.spectral_window_rows[0].effective_channel_width == (8.0, 10.0)


def test_metadata_unit_validation(tmp_path):
  tree = make_tree()
  node = tree["part/antenna_xds"]
  ds = node.to_dataset()
  ds.ANTENNA_RECEPTOR_ANGLE.attrs["units"] = "deg"
  node.ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="RECEPTOR_ANGLE.*'rad'"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_polarization_products_use_feed_receptor_order(tmp_path):
  tree = make_tree()
  node = tree["part"]
  ds = node.to_dataset().isel(polarization=[1])
  ds = ds.assign_coords(polarization=["YY"])
  node.ds = ds
  plan = plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert plan.metadata.polarization_rows[0].corr_product == ((1, 1),)


def test_inconsistent_feed_receptor_order_is_rejected(tmp_path):
  tree = make_tree()
  node = tree["part/antenna_xds"]
  ds = node.to_dataset()
  ds = ds.assign_coords(
    polarization_type=(
      ("antenna_name", "receptor_label"),
      [["X", "Y"], ["Y", "X"]],
    )
  )
  node.ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="same receptor ordering"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_optional_metadata_is_not_silently_dropped(tmp_path):
  tree = make_tree()
  tree["part/pointing_xds"] = xr.DataTree(
    dataset=xr.Dataset(attrs={"type": "pointing"})
  )
  with pytest.raises(
    FreshMSv2ValidationError, match="unsupported optional metadata.*pointing_xds"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_ephemeris_sibling_is_not_silently_dropped(tmp_path):
  tree = make_tree()
  tree["part/ephemeris"] = xr.DataTree(
    dataset=xr.Dataset(attrs={"type": "field_and_source_ephemeris"})
  )
  with pytest.raises(
    FreshMSv2ValidationError, match="unsupported optional metadata.*ephemeris"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


@pytest.mark.parametrize(
  ("node_path", "variable"),
  [
    ("part/antenna_xds", "ANTENNA_POSITION"),
    ("part/field_and_source_base_xds", "FIELD_PHASE_CENTER_DIRECTION"),
  ],
)
def test_required_measure_metadata_must_be_data_variables(
  tmp_path, node_path, variable
):
  tree = make_tree()
  node = tree[node_path]
  ds = node.to_dataset().set_coords(variable)
  node.ds = ds
  with pytest.raises(FreshMSv2ValidationError, match=rf"{variable!s}.*data variable"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_unsupported_source_content_is_not_silently_dropped(tmp_path):
  tree = make_tree()
  node = tree["part/field_and_source_base_xds"]
  ds = node.to_dataset()
  ds["SOURCE_DIRECTION"] = (
    ("field_name", "sky_dir_label"),
    [[0.1, 0.2]],
    {"type": "sky_coord", "units": "rad", "frame": "icrs"},
  )
  node.ds = ds
  with pytest.raises(
    FreshMSv2ValidationError, match="unsupported FIELD/SOURCE.*SOURCE_DIRECTION"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_dangling_effective_channel_width_is_rejected(tmp_path):
  tree = make_tree()
  tree["part"].ds.frequency.attrs["effective_channel_width"] = "MISSING"
  with pytest.raises(
    FreshMSv2ValidationError, match="effective_channel_width.*missing variable"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_repeated_antenna_feed_configs_are_explicitly_unsupported(tmp_path):
  tree = make_tree()
  node = tree["part/antenna_xds"]
  ds = node.to_dataset().assign_coords(antenna_name=["A", "A"])
  node.ds = ds
  with pytest.raises(
    FreshMSv2ValidationError, match="multiple FEED configurations.*ambiguous"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_equivalent_release_date_offsets_deduplicate(tmp_path):
  tree = make_tree(("/one", "/two"))
  tree["two"].attrs["observation_info"]["release_date"] = "2020-01-01T02:00:00+02:00"
  plan = plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert len(plan.metadata.observation_rows) == 1


def test_reference_frequency_data_must_be_scalar(tmp_path):
  tree = make_tree()
  tree["part"].ds.frequency.attrs["reference_frequency"]["data"] = [10.0]
  with pytest.raises(
    FreshMSv2ValidationError, match="reference_frequency data must be scalar"
  ):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_plan_method_survives_legacy_writer_import_failure():
  script = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'xarray_ms.backend.msv2.writes':
        raise ImportError('simulated legacy writer failure')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import xarray_ms
from xarray import DataTree
assert not xarray_ms.HAS_WRITE_SUPPORT
assert callable(DataTree.plan_msv2)
assert callable(xarray_ms.plan_fresh_msv2)
"""
  result = subprocess.run(
    [sys.executable, "-c", script], capture_output=True, text=True
  )
  assert result.returncode == 0, result.stderr
