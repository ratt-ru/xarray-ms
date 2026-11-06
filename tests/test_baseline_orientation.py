"""Baselines are labelled in the orientation in which they are stored (#199)"""

import numpy as np
import pytest
import xarray
from arcae.lib.arrow_tables import Table
from numpy.testing import assert_array_equal
from rarg_python_patterns.multiton import Multiton

from xarray_ms.backend.msv2.entrypoint_utils import subtable_factory
from xarray_ms.backend.msv2.structure import MSv2Structure
from xarray_ms.errors import (
  InconsistentBaselineOrientationWarning,
  IrregularBaselineGridWarning,
  PartitioningError,
)
from xarray_ms.testing.simulator import DEFAULT_SIM_PARAMS, MSStructureSimulator


def _pair_mask(a1, a2, pairs):
  mask = np.zeros(a1.shape, np.bool_)
  for p, q in pairs:
    mask |= (a1 == p) & (a2 == q)
  return mask


def _transform(flip=(), flip_obs=None, flip_first_time=False, remove=()):
  """Returns a simulator data transform that

  - removes the rows of the ``(p, q, first_time_only)`` baselines in ``remove``,
  - swaps ANTENNA1 and ANTENNA2 on the ``(p, q)`` baselines in ``flip``,
    optionally restricted to the OBSERVATION_IDs in ``flip_obs``
    or to the first timestep.
  """

  def transform(chunk_desc, data_dict):
    a1 = data_dict["ANTENNA1"][1]
    a2 = data_dict["ANTENNA2"][1]
    time = data_dict["TIME"][1]

    keep = np.ones(a1.shape, np.bool_)
    for p, q, first_time_only in remove:
      mask = _pair_mask(a1, a2, [(p, q)])
      if first_time_only:
        mask &= time == time.min()
      keep &= ~mask

    data_dict = {k: (d, v[keep]) for k, (d, v) in data_dict.items()}
    a1 = data_dict["ANTENNA1"][1].copy()
    a2 = data_dict["ANTENNA2"][1].copy()
    time = data_dict["TIME"][1]

    mask = _pair_mask(a1, a2, flip)
    if flip_obs is not None:
      mask &= np.isin(data_dict["OBSERVATION_ID"][1], flip_obs)
    if flip_first_time:
      mask &= time == time.min()

    a1[mask], a2[mask] = a2[mask], a1[mask]
    data_dict["ANTENNA1"] = (data_dict["ANTENNA1"][0], a1)
    data_dict["ANTENNA2"] = (data_dict["ANTENNA2"][0], a2)
    return data_dict

  return transform


def _simulate(tmp_path, name, **kw):
  ms = str(tmp_path / name)
  MSStructureSimulator(**{**DEFAULT_SIM_PARAMS, **kw}).simulate_ms(ms)
  return ms


def _names(ds):
  return list(
    zip(
      ds.baseline_antenna1_name.values.tolist(),
      ds.baseline_antenna2_name.values.tolist(),
    )
  )


def _ant(i):
  return f"ANTENNA-{i}"


def _assert_same_data(ds, ref):
  for v in ("VISIBILITY", "UVW", "FLAG", "WEIGHT"):
    assert_array_equal(ds[v].values, ref[v].values)


@pytest.mark.parametrize("auto_corrs", [True, False])
def test_all_baselines_flipped(tmp_path, auto_corrs):
  """Every label is reversed and the data is unchanged"""
  na = 4
  pairs = [(p, q) for p in range(na) for q in range(p + 1, na)]
  ref_ms = _simulate(tmp_path, "ref.ms", nantenna=na)
  ms = _simulate(tmp_path, "flip.ms", nantenna=na, transform_data=_transform(pairs))

  ref_dt = xarray.open_datatree(ref_ms, auto_corrs=auto_corrs)
  dt = xarray.open_datatree(ms, auto_corrs=auto_corrs)

  for (_, ref_node), (_, node) in zip(ref_dt.children.items(), dt.children.items()):
    ref, ds = ref_node.ds, node.ds
    _assert_same_data(ds, ref)
    assert _names(ds) == [(n2, n1) if n1 != n2 else (n1, n2) for n1, n2 in _names(ref)]


def test_flipped_baseline_missing_cells(tmp_path):
  """A flipped baseline with missing cells is labelled (q, p) and
  its gaps are imputed"""
  ref_ms = _simulate(tmp_path, "ref.ms")
  ms = _simulate(
    tmp_path,
    "flip.ms",
    transform_data=_transform(flip=[(0, 2)], remove=[(0, 2, True)]),
  )

  with pytest.warns(IrregularBaselineGridWarning):
    dt = xarray.open_datatree(ms)
  ref_dt = xarray.open_datatree(ref_ms)

  for (_, ref_node), (_, node) in zip(ref_dt.children.items(), dt.children.items()):
    ref, ds = ref_node.ds, node.ds
    names = _names(ds)
    k = names.index((_ant(2), _ant(0)))
    assert _names(ref)[k] == (_ant(0), _ant(2))

    # The missing first timestep is imputed
    assert np.all(np.isnan(ds.VISIBILITY.values[0, k]))
    assert np.all(np.isnan(ds.UVW.values[0, k]))
    assert np.all(ds.FLAG.values[0, k] == 1)

    # Everything else holds the stored values
    mask = np.ones(ds.VISIBILITY.shape[:2], np.bool_)
    mask[0, k] = False
    assert_array_equal(ds.VISIBILITY.values[mask], ref.VISIBILITY.values[mask])
    assert_array_equal(ds.UVW.values[mask], ref.UVW.values[mask])


def test_baseline_without_rows(tmp_path):
  """A baseline with no rows is labelled (p, q)"""
  ms = _simulate(
    tmp_path,
    "flip.ms",
    transform_data=_transform(flip=[(0, 2)], remove=[(1, 2, False)]),
  )

  with pytest.warns(IrregularBaselineGridWarning):
    dt = xarray.open_datatree(ms)

  for node in dt.children.values():
    names = _names(node.ds)
    assert (_ant(2), _ant(0)) in names
    k = names.index((_ant(1), _ant(2)))
    assert np.all(np.isnan(node.ds.VISIBILITY.values[:, k]))


def test_mixed_between_partitions(tmp_path):
  """Each partition labels a baseline in its own stored orientation"""
  ref_ms = _simulate(tmp_path, "ref.ms", nobs=2)
  ms = _simulate(
    tmp_path, "flip.ms", nobs=2, transform_data=_transform([(0, 2)], flip_obs=[1])
  )

  with pytest.warns(InconsistentBaselineOrientationWarning, match=r"\(0, 2\)"):
    dt = xarray.open_datatree(ms)
  ref_dt = xarray.open_datatree(ref_ms)

  pairs = [(r.ds, n.ds) for r, n in zip(ref_dt.children.values(), dt.children.values())]
  assert len(pairs) == 2

  for ref, ds in pairs:
    _assert_same_data(ds, ref)

  # Partitions are ordered by OBSERVATION_ID
  (ref0, ds0), (ref1, ds1) = pairs
  assert _names(ds0) == _names(ref0)
  assert (_ant(0), _ant(2)) in _names(ds0)
  assert (_ant(2), _ant(0)) in _names(ds1)
  assert _names(ds1) == [
    (_ant(2), _ant(0)) if n == (_ant(0), _ant(2)) else n for n in _names(ref1)
  ]


def test_mixed_within_partition_different_times(tmp_path):
  ms = _simulate(
    tmp_path, "flip.ms", transform_data=_transform([(0, 2)], flip_first_time=True)
  )

  with pytest.raises(PartitioningError, match=r"\(0, 2\) are stored as both"):
    xarray.open_datatree(ms)


def _add_reversed_duplicate(chunk_desc, data_dict):
  """Adds a reversed copy of the first (0, 1) row, at the same TIME"""
  a1 = data_dict["ANTENNA1"][1]
  a2 = data_dict["ANTENNA2"][1]
  (r, *_) = np.where((a1 == 0) & (a2 == 1))[0]
  data_dict = {
    k: (d, np.concatenate([v, v[r : r + 1]])) for k, (d, v) in data_dict.items()
  }
  data_dict["ANTENNA1"][1][-1] = 1
  data_dict["ANTENNA2"][1][-1] = 0
  return data_dict


def test_mixed_within_partition_same_time(tmp_path):
  """Both orientations at the same TIME are not reported as duplicates"""
  ms = _simulate(tmp_path, "flip.ms", transform_data=_add_reversed_duplicate)

  with pytest.raises(PartitioningError, match=r"\(0, 1\) are stored as both"):
    xarray.open_datatree(ms)


def _structure(ms, auto_corrs):
  subtables = {
    st: Multiton(subtable_factory, f"{ms}::{st}")
    for st in ("DATA_DESCRIPTION", "FEED", "FIELD", "STATE")
  }
  return MSv2Structure(
    Multiton(Table.from_filename, ms), subtables, [], "epoch", auto_corrs
  )


@pytest.mark.parametrize("auto_corrs", [True, False])
def test_structure_baseline_flipped(tmp_path, auto_corrs):
  na = 4
  flip = [(0, 2), (1, 3)]
  ref_ms = _simulate(tmp_path, "ref.ms", nantenna=na)
  ms = _simulate(tmp_path, "flip.ms", nantenna=na, transform_data=_transform(flip))

  for partition in _structure(ref_ms, auto_corrs).values():
    assert not np.any(partition.baseline_flipped)

  for partition in _structure(ms, auto_corrs).values():
    a1, a2 = np.triu_indices(na, 0 if auto_corrs else 1)
    expected = _pair_mask(a1, a2, flip)
    assert_array_equal(partition.baseline_flipped, expected)

    ant1, ant2 = partition.antenna_pairs
    assert_array_equal(ant1, np.where(expected, a2, a1))
    assert_array_equal(ant2, np.where(expected, a1, a2))
