"""Values written to baselines stored as ANTENNA1 > ANTENNA2 are
converted back to the stored orientation.

https://github.com/ratt-ru/xarray-ms/issues/199
"""

import arcae
import numpy as np
import pytest
import xarray
from numpy.testing import assert_array_equal
from xarray import DataTree

from tests import test_baseline_orientation as orientation_tests
from tests.test_baseline_orientation import (
  UNKNOWN_COLUMN_DESC,
  _add_unknown_column,
  _hermitian,
  _open,
)
from xarray_ms.errors import BaselineOrientationError, BaselineOrientationWarning
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES

# Share the reference and flipped Measurement Set fixture
reference_and_flipped = orientation_tests.reference_and_flipped

# XX, XY, YX, YY -> XX, YX, XY, YY
PERM = [0, 2, 1, 3]


def _modify(ds):
  """Returns a copy of ``ds`` with known changes to the
  variables that transform under baseline reversal"""
  return ds.assign(
    VISIBILITY=ds.VISIBILITY * (2 - 1j) + (1 + 3j),
    UVW=ds.UVW * 2 + 1,
    WEIGHT=ds.WEIGHT + xarray.DataArray(np.arange(4.0), dims="polarization"),
    FLAG=ds.FLAG | xarray.DataArray([False, True, False, False], dims="polarization"),
  )


def _write_all(ms, transform, **kw):
  """Opens ``ms`` as a DataTree, applies ``transform`` to each
  correlated dataset and writes the result back"""
  with xarray.open_datatree(ms, **kw) as dt:
    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        dt[node.path] = DataTree(transform(node.ds))
    dt.sync_msv2()
    dt.to_msv2()


def _raw(ms, column):
  with arcae.table(ms) as T:
    return T.getcol(column), T.getcol("ANTENNA1"), T.getcol("ANTENNA2")


def test_write_round_trip(reference_and_flipped):
  """Writing to a flipped MS stores the reversed-baseline values,
  and reading them back recovers the written values"""
  reference, flipped = reference_and_flipped
  expected = _modify(_open(flipped).load())
  variables = ("VISIBILITY", "UVW", "WEIGHT", "FLAG")

  # The reference stores the upper-triangular values written to flipped
  _write_all(reference, lambda ds: ds.assign({v: expected[v] for v in variables}))
  _write_all(flipped, _modify)

  ds = _open(flipped).load()
  for v in variables:
    assert_array_equal(ds[v].values, expected[v].values)

  # The flipped MS has the reference's rows, with the antennas of FLIPPED_PAIRS
  # swapped. Its flipped rows hold the reversed-baseline values
  for column, reverse in (
    ("DATA", lambda v: _hermitian(v, PERM)),
    ("WEIGHT_SPECTRUM", lambda v: v[..., PERM]),
    ("FLAG", lambda v: v[..., PERM]),
    ("UVW", lambda v: -v),
  ):
    ref, _, _ = _raw(reference, column)
    raw, ant1, ant2 = _raw(flipped, column)
    rows = ant1 > ant2
    assert rows.sum() > 0
    assert_array_equal(raw[~rows], ref[~rows])
    assert_array_equal(raw[rows], reverse(ref[rows]))


def test_write_unmodified_is_identity(reference_and_flipped):
  """Writing back unmodified data leaves the stored values unchanged"""
  _, flipped = reference_and_flipped
  before = {c: _raw(flipped, c)[0] for c in ("DATA", "UVW", "FLAG")}
  _write_all(flipped, lambda ds: ds)
  for column, values in before.items():
    assert_array_equal(_raw(flipped, column)[0], values)


def test_write_partial_polarization_region(reference_and_flipped):
  """Writing only XY stores YX on flipped baselines"""
  _, flipped = reference_and_flipped
  before, ant1, ant2 = _raw(flipped, "DATA")
  rows = ant1 > ant2
  # Auto-correlations are not on the default grid
  cross = ant1 != ant2

  region = {"time": slice(1, 3), "polarization": slice(1, 2)}
  with xarray.open_datatree(flipped) as dt:
    (node,) = [n for n in dt.subtree if n.attrs.get("type") in CORRELATED_DATASET_TYPES]
    ds = node.ds.isel(**region)
    ds = ds[["VISIBILITY"]].assign(VISIBILITY=xarray.full_like(ds.VISIBILITY, 5 + 7j))
    ds.to_msv2(region=region)

  after, _, _ = _raw(flipped, "DATA")
  with arcae.table(flipped) as T:
    time = T.getcol("TIME")
  utime = np.unique(time)
  in_time = np.isin(time, utime[region["time"]])
  written = after != before

  # Unflipped rows in the region change only XY, flipped rows only YX
  for sel, pol, value in (
    (in_time & cross & ~rows, 1, 5 + 7j),
    (in_time & rows, 2, 5 - 7j),
  ):
    assert sel.sum() > 0
    others = [p for p in range(4) if p != pol]
    assert np.all(written[sel][..., pol])
    assert not np.any(written[sel][..., others])
    assert_array_equal(after[sel][..., pol], value)
  assert not np.any(written[~in_time])

  ds = _open(flipped).load()
  # Reading back presents XY on every baseline
  assert_array_equal(ds.VISIBILITY.values[region["time"], :, :, 1], 5 + 7j)


@pytest.mark.parametrize(
  "reference_and_flipped",
  [{"transform_data": _add_unknown_column, "table_desc": UNKNOWN_COLUMN_DESC}],
  indirect=True,
)
def test_write_secondary_columns(reference_and_flipped):
  """Known and declared secondary columns are converted on write.
  Columns of unknown orientation cannot be written to flipped baselines"""
  _, flipped = reference_and_flipped

  def add_columns(ds):
    return ds.assign(
      CORRECTED_DATA=ds.VISIBILITY * 3,
      RESIDUAL=ds.VISIBILITY - 1,
    )

  orientation = {"BASELINE_PHASE": "antisymmetric", "RESIDUAL": "hermitian"}
  expected = add_columns(_open(flipped, baseline_orientation=orientation).load())
  _write_all(flipped, add_columns, baseline_orientation=orientation)

  ds = _open(flipped, baseline_orientation=orientation).load()
  for v in ("CORRECTED_DATA", "RESIDUAL", "BASELINE_PHASE"):
    assert_array_equal(ds[v].values, expected[v].values)

  raw, ant1, ant2 = _raw(flipped, "RESIDUAL")
  data, _, _ = _raw(flipped, "DATA")
  cross = ant1 != ant2
  assert (ant1 > ant2).sum() > 0
  # DATA and RESIDUAL are stored in the same orientation.
  # Auto-correlations are not on the default grid
  assert_array_equal(raw[cross], data[cross] - 1)

  # Without a declared orientation, a new column cannot be written
  with pytest.raises(BaselineOrientationError, match="UNKNOWN"):
    _write_all(
      flipped,
      lambda ds: ds.assign(UNKNOWN=ds.VISIBILITY),
      baseline_orientation=orientation,
    )


@pytest.mark.parametrize(
  "reference_and_flipped",
  [{"data_description": [(4, ["XX", "XY"])]}],
  indirect=True,
)
def test_write_unpaired_correlation(reference_and_flipped):
  """Correlations whose partner is absent cannot be written to flipped baselines"""
  _, flipped = reference_and_flipped

  with pytest.warns(BaselineOrientationWarning):
    with pytest.raises(BaselineOrientationError, match="reversed baselines"):
      _write_all(flipped, lambda ds: ds[["VISIBILITY"]])

  # Parallel hands can still be written
  region = {"polarization": slice(0, 1)}
  with pytest.warns(BaselineOrientationWarning):
    with xarray.open_datatree(flipped) as dt:
      (node,) = [
        n for n in dt.subtree if n.attrs.get("type") in CORRELATED_DATASET_TYPES
      ]
      ds = node.ds[["VISIBILITY"]].isel(**region)
      ds.assign(VISIBILITY=xarray.full_like(ds.VISIBILITY, 1 + 1j)).to_msv2(
        region=region
      )

  with pytest.warns(BaselineOrientationWarning):
    ds = _open(flipped).load()
  assert_array_equal(ds.VISIBILITY.values[..., 0], 1 + 1j)


def test_write_encodes_baseline_orientation(reference_and_flipped):
  """baseline_orientation survives into the encoding used to write"""
  _, flipped = reference_and_flipped
  ds = _open(flipped, baseline_orientation={"RESIDUAL": "hermitian"})
  assert ds.encoding["baseline_orientation"] == {"RESIDUAL": "hermitian"}
  assert _open(flipped).encoding["baseline_orientation"] == {}
