"""Fresh metadata keeps each native spectral window's antenna/feed grid scoped."""

import warnings

import numpy as np
import pytest
import xarray as xr
from arcae.lib.arrow_tables import Table

from xarray_ms import plan_fresh_msv2, write_fresh_msv2
from xarray_ms.errors import IrregularBaselineGridWarning
from xarray_ms.testing.fresh import make_complete_tree


def cohort_tree(mode, order=("a", "b")):
  source = make_complete_tree()
  datasets = {}
  for label in order:
    different_ants = mode in ("distinct_spw", "same_spw") and label == "b"
    ds = (
      source["part"]
      .to_dataset()
      .isel(baseline_id=[2 if different_ants else 0])
      .copy(deep=True)
    )
    ds = ds.assign_coords(
      time=("time", ds.time.values + (100 if label == "b" else 0), dict(ds.time.attrs))
    )
    # Distinct payloads make comparisons detect accidental merging or reordering.
    ds["VISIBILITY"] = ds.VISIBILITY * (3 if label == "b" else 1)
    ds["WEIGHT"] = ds.WEIGHT * (4 if label == "b" else 1)
    if mode == "distinct_spw" and label == "b":
      ds.frequency.attrs["spectral_window_name"] = "other"
    antenna = (
      source["part/antenna_xds"]
      .to_dataset(inherit=False)
      .isel(antenna_name=[1, 2] if different_ants else [0, 1])
      .copy(deep=True)
    )
    if mode == "distinct_feeds" and label == "b":
      angle = antenna.ANTENNA_RECEPTOR_ANGLE
      antenna["ANTENNA_RECEPTOR_ANGLE"] = (
        angle.dims,
        angle.values + 0.5,
        dict(angle.attrs),
      )
    datasets[f"/{label}"] = ds
    datasets[f"/{label}/antenna_xds"] = antenna
    datasets[f"/{label}/field_and_source_base_xds"] = (
      source["part/field_and_source_base_xds"].to_dataset(inherit=False).copy(deep=True)
    )
  return xr.DataTree.from_dict(datasets)


@pytest.mark.parametrize("mode", ["distinct_spw", "same_spw", "distinct_feeds"])
def test_scoped_spectral_windows_roundtrip_exact_grids(tmp_path, mode):
  source = cohort_tree(mode)
  target = tmp_path / "scoped.ms"
  plan = plan_fresh_msv2(source, target)
  assert len(plan.metadata.spectral_window_rows) == 2
  assert len(plan.metadata.data_description_rows) == 2
  assert len(plan.metadata.feed_rows) == 4
  assert {r.feed_id for r in plan.metadata.feed_rows} == {0}
  if mode != "distinct_spw":
    assert (
      plan.metadata.spectral_window_rows[0] == plan.metadata.spectral_window_rows[1]
    )
  write_fresh_msv2(source, target, row_grid="complete", rows_per_batch=2)
  with (
    Table.from_filename(str(target)) as main,
    Table.from_filename(str(target / "FEED")) as feed,
    Table.from_filename(str(target / "DATA_DESCRIPTION")) as dd,
  ):
    assert main.nrow() == 8
    spws = dd.getcol("SPECTRAL_WINDOW_ID")
    actual_feeds = set(
      zip(
        feed.getcol("ANTENNA_ID"),
        feed.getcol("FEED_ID"),
        feed.getcol("SPECTRAL_WINDOW_ID"),
        strict=True,
      )
    )
    for part in plan.partitions:
      k = part.foreign_keys
      spw = spws[k.data_desc_id]
      for ant, fid in zip(
        k.antenna1_ids + k.antenna2_ids, k.feed1_ids + k.feed2_ids, strict=True
      ):
        assert (ant, fid, spw) in actual_feeds
      row_indices = np.flatnonzero(main.getcol("DATA_DESC_ID") == k.data_desc_id)
      expected = source[part.path].to_dataset()
      np.testing.assert_array_equal(
        main.getcol("DATA", index=(row_indices,)),
        expected.VISIBILITY.values.reshape(4, 8, 2),
      )
      np.testing.assert_array_equal(
        main.getcol("WEIGHT_SPECTRUM", index=(row_indices,)),
        expected.WEIGHT.values.reshape(4, 8, 2),
      )
      np.testing.assert_array_equal(
        main.getcol("UVW", index=(row_indices,)), expected.UVW.values.reshape(4, 3)
      )
  with warnings.catch_warnings():
    warnings.simplefilter("error", IrregularBaselineGridWarning)
    with xr.open_datatree(target) as reopened:
      parts = [n for n in reopened.subtree if n.attrs.get("type") == "visibility"]
      assert len(parts) == 2
      for actual in parts:
        original = source[
          "b" if actual.time.values[0] > source["a"].time.values[0] else "a"
        ]
        assert actual.attrs["xarray_ms_row_grid"] == {
          "status": "complete",
          "missing_rows": 0,
        }
        assert actual.sizes["baseline_id"] == 1
        assert set(actual["antenna_xds"].antenna_name.values) == set(
          original["antenna_xds"].antenna_name.values
        )
        for name in (
          "VISIBILITY",
          "WEIGHT",
          "UVW",
          "baseline_antenna1_name",
          "baseline_antenna2_name",
        ):
          np.testing.assert_array_equal(actual[name].values, original[name].values)
        np.testing.assert_array_equal(
          actual["antenna_xds"].ANTENNA_RECEPTOR_ANGLE.values,
          original["antenna_xds"].ANTENNA_RECEPTOR_ANGLE.values,
        )


@pytest.mark.parametrize("mode", ["same_spw", "distinct_feeds", "equal"])
def test_cohort_ids_dedup_and_order_stability(tmp_path, mode):
  first = plan_fresh_msv2(cohort_tree(mode), tmp_path / "first.ms")
  second = plan_fresh_msv2(cohort_tree(mode, ("b", "a")), tmp_path / "second.ms")
  assert first.metadata == second.metadata
  assert tuple(p.foreign_keys for p in first.partitions) == tuple(
    p.foreign_keys for p in second.partitions
  )
  expected = 1 if mode == "equal" else 2
  assert len(first.metadata.spectral_window_rows) == expected
  assert len(first.metadata.data_description_rows) == expected
  assert len(first.metadata.feed_rows) == 2 * expected
