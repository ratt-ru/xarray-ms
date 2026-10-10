"""Source-independent qualification of the generated complete-grid profile."""

import shutil

import numpy as np
import pytest
import xarray as xr
from arcae.lib.arrow_tables import Table

from xarray_ms import write_fresh_msv2
from xarray_ms.errors import FreshMSv2ValidationError, IrregularBaselineGridWarning
from xarray_ms.testing.fresh import make_complete_tree


def _visibility(tree):
  return next(n for n in tree.subtree if n.attrs.get("type") == "visibility")


def _average(data, flags, weights):
  usable = np.where(flags, 0, weights)
  return np.sum(data * usable, axis=(0, 1)) / np.sum(usable, axis=(0, 1))


def test_generated_profile_from_reopened_zarr(tmp_path):
  tree = make_complete_tree()
  expected = _visibility(tree).to_dataset().load().copy(deep=True)
  store = tmp_path / "complete.zarr"
  tree.to_zarr(store, consolidated=True)
  tree.close()
  del tree
  target = tmp_path / "fresh.ms"
  with xr.open_datatree(store, engine="zarr") as reopened:
    write_fresh_msv2(reopened, target, row_grid="complete", rows_per_batch=2)

  with Table.from_filename(str(target), ninstances=1) as table:
    shape = expected.VISIBILITY.shape
    assert table.nrow() == shape[0] * shape[1]
    data = table.getcol("DATA").reshape(shape)
    flags = table.getcol("FLAG").reshape(shape)
    weights = table.getcol("WEIGHT_SPECTRUM").reshape(shape)
    np.testing.assert_array_equal(data, expected.VISIBILITY)
    np.testing.assert_array_equal(flags, expected.FLAG)
    np.testing.assert_allclose(weights, expected.WEIGHT)
    np.testing.assert_allclose(
      _average(data, flags, weights),
      _average(
        expected.VISIBILITY.values, expected.FLAG.values, expected.WEIGHT.values
      ),
    )
  with xr.open_datatree(target) as result:
    actual = _visibility(result)
    for name in ("VISIBILITY", "FLAG", "WEIGHT", "UVW", "time", "frequency"):
      np.testing.assert_allclose(actual[name].values, expected[name].values)
    for name in ("polarization", "baseline_antenna1_name", "baseline_antenna2_name"):
      np.testing.assert_array_equal(actual[name].values, expected[name].values)


def test_ms_backed_zarr_survives_removing_original_ms(tmp_path):
  source = tmp_path / "source.ms"
  write_fresh_msv2(make_complete_tree(), source, row_grid="complete")
  store = tmp_path / "detached.zarr"
  with xr.open_datatree(source) as tree:
    expected = _visibility(tree).to_dataset().load().copy(deep=True)
    tree.to_zarr(store, consolidated=True)
  shutil.rmtree(source)
  assert not source.exists()

  target = tmp_path / "restored.ms"
  with xr.open_datatree(store, engine="zarr") as tree:
    write_fresh_msv2(tree, target, row_grid="complete", rows_per_batch=2)
  with xr.open_datatree(target) as tree:
    actual = _visibility(tree)
    for name in ("VISIBILITY", "FLAG", "WEIGHT", "UVW", "time", "frequency"):
      np.testing.assert_allclose(actual[name].values, expected[name].values)


def test_reader_padding_remains_refused_after_zarr(tmp_path):
  source = tmp_path / "crosses.ms"
  write_fresh_msv2(make_complete_tree(), source, row_grid="complete")
  store = tmp_path / "padded.zarr"
  with pytest.warns(IrregularBaselineGridWarning):
    tree = xr.open_datatree(source, auto_corrs=True)
  with tree:
    marker = _visibility(tree).attrs["xarray_ms_row_grid"]
    assert marker == {"status": "padded", "missing_rows": 12}
    tree.to_zarr(store, consolidated=True)
  shutil.rmtree(source)
  target = tmp_path / "refused.ms"
  with xr.open_datatree(store, engine="zarr") as reopened:
    assert _visibility(reopened).attrs["xarray_ms_row_grid"] == marker
    with pytest.raises(FreshMSv2ValidationError, match="padded"):
      write_fresh_msv2(reopened, target, row_grid="complete")
  assert not target.exists()
  assert not list(tmp_path.glob("*.staging"))
