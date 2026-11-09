"""Baselines stored as ANTENNA1 > ANTENNA2 are presented on the
upper-triangular baseline, with their data converted to that orientation.

https://github.com/ratt-ru/xarray-ms/issues/199
"""

import numpy as np
import pytest
import xarray
from numpy.testing import assert_array_equal

from xarray_ms.backend.msv2.entrypoint_utils import CommonStoreArgs
from xarray_ms.backend.msv2.measures_encoders import EpochCoder
from xarray_ms.backend.msv2.structure import MSv2Structure
from xarray_ms.casa_types import Polarisations
from xarray_ms.errors import BaselineOrientationWarning, PartitioningError
from xarray_ms.testing.simulator import DEFAULT_SIM_PARAMS, MSStructureSimulator

# (ANTENNA1, ANTENNA2) pairs stored the other way round
FLIPPED_PAIRS = [(0, 2), (1, 3)]
NANTENNA = 4


def _simulate(path, **params):
  params = {**DEFAULT_SIM_PARAMS, "nantenna": NANTENNA, **params}
  MSStructureSimulator(**params).simulate_ms(str(path))
  return str(path)


def _pair_mask(ant1, ant2, pairs):
  return np.logical_or.reduce([(ant1 == p) & (ant2 == q) for p, q in pairs])


def _flip_pairs(pairs, time_filter=None):
  """Returns a data transform storing ``pairs`` as ``(ANTENNA2, ANTENNA1)``,
  leaving the data untouched"""

  def transform(chunk_desc, data_dict):
    (dims1, ant1), (dims2, ant2) = data_dict["ANTENNA1"], data_dict["ANTENNA2"]
    mask = _pair_mask(ant1, ant2, pairs)
    if time_filter is not None:
      mask &= time_filter(data_dict["TIME"][1])
    data_dict = data_dict.copy()
    data_dict["ANTENNA1"] = (dims1, np.where(mask, ant2, ant1))
    data_dict["ANTENNA2"] = (dims2, np.where(mask, ant1, ant2))
    return data_dict

  return transform


def _compose(*transforms):
  def transform(chunk_desc, data_dict):
    for t in transforms:
      data_dict = t(chunk_desc, data_dict)
    return data_dict

  return transform


def _open(ms, **kw):
  return xarray.open_dataset(ms, engine="xarray-ms:msv2", **kw)


def _baseline_mask(ds, pairs):
  names = [f"ANTENNA-{a}" for a in range(NANTENNA)]
  a1 = ds.baseline_antenna1_name.values
  a2 = ds.baseline_antenna2_name.values
  return np.logical_or.reduce([(a1 == names[p]) & (a2 == names[q]) for p, q in pairs])


def _hermitian(values, pol_perm):
  """Reference baseline reversal of (..., polarization) values"""
  values = values[..., pol_perm]
  return np.conj(values) if np.iscomplexobj(values) else values


@pytest.fixture
def reference_and_flipped(tmp_path, request):
  params = getattr(request, "param", {})
  extra = params.pop("transform_data", None)
  flip = _flip_pairs(FLIPPED_PAIRS)
  reference = _simulate(tmp_path / "reference.ms", transform_data=extra, **params)
  flipped = _simulate(
    tmp_path / "flipped.ms",
    transform_data=flip if extra is None else _compose(extra, flip),
    **params,
  )
  return reference, flipped


@pytest.mark.parametrize("auto_corrs", [True, False])
def test_flipped_baselines(reference_and_flipped, auto_corrs):
  """Flipped baselines keep upper-triangular labels, VISIBILITY is
  conjugated with cross-hands swapped, UVW negated and
  cross-hands of FLAG and WEIGHT swapped"""
  reference, flipped = reference_and_flipped
  ref = _open(reference, auto_corrs=auto_corrs).load()
  ds = _open(flipped, auto_corrs=auto_corrs).load()

  # Labels are unchanged
  assert_array_equal(ds.baseline_antenna1_name, ref.baseline_antenna1_name)
  assert_array_equal(ds.baseline_antenna2_name, ref.baseline_antenna2_name)
  a1 = ds.baseline_antenna1_name.values
  a2 = ds.baseline_antenna2_name.values
  assert np.all(a1 <= a2)

  bl = _baseline_mask(ds, FLIPPED_PAIRS)
  assert bl.sum() == len(FLIPPED_PAIRS)
  # XX, XY, YX, YY -> XX, YX, XY, YY
  perm = [0, 2, 1, 3]

  # Unflipped baselines are unchanged
  for v in ("VISIBILITY", "UVW", "FLAG", "WEIGHT", "TIME_CENTROID"):
    assert_array_equal(ds[v].values[:, ~bl], ref[v].values[:, ~bl])

  # The data is stored for (q, p): the reference holds the stored values
  assert_array_equal(
    ds.VISIBILITY.values[:, bl], _hermitian(ref.VISIBILITY.values[:, bl], perm)
  )
  assert_array_equal(ds.UVW.values[:, bl], -ref.UVW.values[:, bl])
  assert_array_equal(ds.FLAG.values[:, bl], ref.FLAG.values[:, bl][..., perm])
  assert_array_equal(ds.WEIGHT.values[:, bl], ref.WEIGHT.values[:, bl][..., perm])
  assert_array_equal(ds.TIME_CENTROID.values[:, bl], ref.TIME_CENTROID.values[:, bl])

  # Parallel hands are unchanged apart from conjugation
  parallel = [0, 3]
  assert_array_equal(
    ds.VISIBILITY.values[:, bl][..., parallel],
    np.conj(ref.VISIBILITY.values[:, bl][..., parallel]),
  )


def test_partial_polarization_selection(reference_and_flipped):
  """Selecting a cross-hand reads its partner on flipped baselines"""
  reference, flipped = reference_and_flipped
  ref = _open(reference)
  ds = _open(flipped)
  bl = _baseline_mask(ds, FLIPPED_PAIRS)

  # XY on flipped baselines is the conjugate of the stored YX
  for sel, partner in ((1, 2), (2, 1), (0, 0)):
    xy = ds.VISIBILITY.isel(polarization=[sel]).values
    stored = ref.VISIBILITY.isel(polarization=[partner]).values
    assert_array_equal(xy[:, bl], np.conj(stored[:, bl]))
    unflipped = ref.VISIBILITY.isel(polarization=[sel]).values
    assert_array_equal(xy[:, ~bl], unflipped[:, ~bl])

  # Integer and slice selections across time and baseline
  b = int(np.flatnonzero(bl)[0])
  value = ds.VISIBILITY.isel(time=1, baseline_id=b, polarization=1).values
  stored = ref.VISIBILITY.isel(time=1, baseline_id=b, polarization=2).values
  assert_array_equal(value, np.conj(stored))

  flags = ds.FLAG.isel(baseline_id=slice(b, b + 1), polarization=slice(1, 3)).values
  stored = ref.FLAG.isel(baseline_id=slice(b, b + 1)).values[..., [2, 1]]
  assert_array_equal(flags, stored)


def _remove_weight_spectrum_add_weight_sigma(chunk_desc, data_dict):
  data_dict = data_dict.copy()
  del data_dict["WEIGHT_SPECTRUM"]
  ddid = chunk_desc.DATA_DESC_ID.item()
  _, ncorr = map(len, chunk_desc.data_description[ddid])
  nrow = data_dict["DATA_DESC_ID"][-1].size
  shape = (nrow, ncorr)
  ramp = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
  data_dict["WEIGHT"] = ("row", ramp)
  data_dict["SIGMA"] = ("row", ramp + 0.5)
  return data_dict


@pytest.mark.parametrize(
  "reference_and_flipped",
  [
    {
      "transform_data": _remove_weight_spectrum_add_weight_sigma,
      "table_desc": {"__remove_columns__": ["WEIGHT_SPECTRUM"]},
    }
  ],
  indirect=True,
)
def test_flipped_low_resolution_columns(reference_and_flipped):
  """Broadcast WEIGHT and SIGMA columns swap cross-hands on flipped baselines"""
  reference, flipped = reference_and_flipped
  ref = _open(reference).load()
  ds = _open(flipped).load()
  bl = _baseline_mask(ds, FLIPPED_PAIRS)
  perm = [0, 2, 1, 3]

  for v in ("WEIGHT", "SIGMA_SPECTRUM"):
    assert_array_equal(ds[v].values[:, ~bl], ref[v].values[:, ~bl])
    assert_array_equal(ds[v].values[:, bl], ref[v].values[:, bl][..., perm])


@pytest.mark.parametrize(
  "reference_and_flipped",
  [{"data_description": [(4, ["RR", "RL", "LR", "LL"])]}],
  indirect=True,
)
def test_flipped_circular(reference_and_flipped):
  """RL and LR are swapped on flipped baselines"""
  reference, flipped = reference_and_flipped
  ref = _open(reference).load()
  ds = _open(flipped).load()
  bl = _baseline_mask(ds, FLIPPED_PAIRS)
  assert_array_equal(
    ds.VISIBILITY.values[:, bl],
    _hermitian(ref.VISIBILITY.values[:, bl], [0, 2, 1, 3]),
  )


@pytest.mark.parametrize(
  "reference_and_flipped",
  [{"data_description": [(4, ["XX", "XY"])]}],
  indirect=True,
)
def test_flipped_unpaired_correlation(reference_and_flipped):
  """Correlations whose partner is absent are padded on flipped baselines"""
  reference, flipped = reference_and_flipped
  ref = _open(reference).load()

  with pytest.warns(BaselineOrientationWarning, match=r"correlations \['XY'\]"):
    ds = _open(flipped).load()

  bl = _baseline_mask(ds, FLIPPED_PAIRS)
  assert_array_equal(
    ds.VISIBILITY.values[:, bl, :, 0], np.conj(ref.VISIBILITY.values[:, bl, :, 0])
  )
  assert np.all(np.isnan(ds.VISIBILITY.values[:, bl, :, 1]))
  assert np.all(ds.FLAG.values[:, bl, :, 1])
  assert_array_equal(ds.VISIBILITY.values[:, ~bl], ref.VISIBILITY.values[:, ~bl])


@pytest.mark.parametrize(
  "values, expected",
  [
    (["XX", "XY", "YX", "YY"], [0, 2, 1, 3]),
    (["RR", "RL", "LR", "LL"], [0, 2, 1, 3]),
    (["XX", "YY"], [0, 1]),
    (["I", "Q", "U", "V"], [0, 1, 2, 3]),
    (["XY"], [-1]),
    (["RX", "XR", "LY", "YL"], [1, 0, 3, 2]),
    (["PP", "PQ", "QP", "QQ"], [0, 2, 1, 3]),
  ],
)
def test_transpose_permutation(values, expected):
  perm = Polarisations.from_values(values).transpose_permutation()
  assert_array_equal(perm, expected)


def test_mixed_orientation_different_times(tmp_path):
  """A baseline stored in both orientations at different times
  is converted cell by cell"""
  reference = _simulate(tmp_path / "reference.ms")
  ref = _open(reference).load()
  late = ref.time.values[2]  # unix seconds
  mjd_late = EpochCoder.encode_array(np.asarray(late))
  flipped = _simulate(
    tmp_path / "flipped.ms",
    transform_data=_flip_pairs(FLIPPED_PAIRS, lambda t: t >= mjd_late),
  )
  ds = _open(flipped).load()

  bl = _baseline_mask(ds, FLIPPED_PAIRS)
  late = ds.time.values >= late
  assert late.any() and (~late).any()

  vis, ref_vis = ds.VISIBILITY.values, ref.VISIBILITY.values
  assert_array_equal(vis[~late], ref_vis[~late])
  assert_array_equal(vis[late][:, ~bl], ref_vis[late][:, ~bl])
  assert_array_equal(vis[late][:, bl], _hermitian(ref_vis[late][:, bl], [0, 2, 1, 3]))


def _duplicate_reversed(chunk_desc, data_dict):
  """Appends copies of the (0, 2) rows stored as (2, 0)"""
  ant1, ant2 = data_dict["ANTENNA1"][1], data_dict["ANTENNA2"][1]
  rows = np.flatnonzero(_pair_mask(ant1, ant2, [(0, 2)]))
  result = {}
  for column, (dims, data) in data_dict.items():
    extra = data[rows]
    if column == "ANTENNA1":
      extra = ant2[rows]
    elif column == "ANTENNA2":
      extra = ant1[rows]
    result[column] = (dims, np.concatenate([data, extra]))
  return result


def test_mixed_orientation_same_time(tmp_path):
  """A baseline stored in both orientations at the same time
  is reported as such"""
  ms = _simulate(tmp_path / "duplicate.ms", transform_data=_duplicate_reversed)

  with pytest.raises(PartitioningError, match=r"\(0, 2\) at TIME"):
    _open(ms)


def test_structure_flipped(reference_and_flipped):
  """PartitionData.flipped is None when no baselines are flipped,
  and set exactly on the flipped cells otherwise"""
  reference, flipped = reference_and_flipped

  for ms, expect_flipped in ((reference, False), (flipped, True)):
    structure = CommonStoreArgs(ms).structure_factory.instance
    (partition,) = structure.values()
    if not expect_flipped:
      assert partition.flipped is None
      continue

    assert partition.flipped is not None
    assert partition.flipped.shape == partition.row_map.shape
    a1, a2 = partition.antenna_pairs
    bl = _pair_mask(a1, a2, FLIPPED_PAIRS)
    assert np.all(partition.flipped[:, bl])
    assert not np.any(partition.flipped[:, ~bl])


def test_sorted_antennas_not_modified():
  """Placing rows in their upper-triangular slot does not
  modify the partition's antenna columns"""
  ant1 = np.array([2, 0, 3], np.int32)
  ant2 = np.array([0, 1, 1], np.int32)
  a1, a2 = ant1.copy(), ant2.copy()
  row_map = np.full(6, -1, np.int64)
  index, flips = MSv2Structure.gen_row_interval_grids(
    np.zeros(3, np.int32),
    ant1,
    ant2,
    np.ones(3),
    np.arange(3, dtype=np.int32),
    row_map,
    np.full(6, -1.0),
    np.arange(4, dtype=np.int32),
    4,
    6,
    False,
  )
  assert_array_equal(ant1, a1)
  assert_array_equal(ant2, a2)
  assert_array_equal(flips, [True, False, True])


UNKNOWN_COLUMN_DESC = {
  "BASELINE_PHASE": {
    "_c_order": True,
    "comment": "BASELINE_PHASE column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    "valueType": "FLOAT",
  },
  "CORRECTED_DATA": {
    "_c_order": True,
    "comment": "CORRECTED_DATA column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    "valueType": "COMPLEX",
  },
}


def _add_unknown_column(chunk_desc, data_dict):
  data_dict = data_dict.copy()
  dims, data = data_dict["DATA"]
  data_dict["BASELINE_PHASE"] = (dims, np.angle(data))
  data_dict["CORRECTED_DATA"] = (dims, data)
  return data_dict


@pytest.mark.parametrize(
  "reference_and_flipped",
  [{"transform_data": _add_unknown_column, "table_desc": UNKNOWN_COLUMN_DESC}],
  indirect=True,
)
def test_secondary_columns(reference_and_flipped):
  """Secondary columns with an unknown orientation are omitted only from
  partitions with flipped baselines, unless an orientation is declared"""
  reference, flipped = reference_and_flipped
  ref = _open(reference).load()
  assert "BASELINE_PHASE" in ref

  with pytest.warns(BaselineOrientationWarning, match="BASELINE_PHASE"):
    ds = _open(flipped).load()

  assert "BASELINE_PHASE" not in ds
  bl = _baseline_mask(ds, FLIPPED_PAIRS)
  perm = [0, 2, 1, 3]
  # VISIBILITY_CORRECTED has a known orientation
  assert_array_equal(
    ds.VISIBILITY_CORRECTED.values[:, bl],
    _hermitian(ref.VISIBILITY_CORRECTED.values[:, bl], perm),
  )

  ds = _open(flipped, baseline_orientation={"BASELINE_PHASE": "antisymmetric"}).load()
  assert_array_equal(ds.BASELINE_PHASE.values[:, bl], -ref.BASELINE_PHASE.values[:, bl])
  assert_array_equal(
    ds.BASELINE_PHASE.values[:, ~bl], ref.BASELINE_PHASE.values[:, ~bl]
  )

  dt = xarray.open_datatree(
    flipped, baseline_orientation={"BASELINE_PHASE": "antisymmetric"}
  )
  assert "BASELINE_PHASE" in dt[next(iter(dt.children))]

  with pytest.raises(ValueError, match="Invalid baseline_orientation"):
    _open(flipped, baseline_orientation={"BASELINE_PHASE": "sideways"})
