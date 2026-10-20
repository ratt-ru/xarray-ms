import warnings
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from xarray_ms.backend.msv2.antenna_selection import (
  select_partition_antennas,
  unique_antenna_names,
)
from xarray_ms.errors import DuplicateAntennaNameWarning, HeterogeneousFeedWarning


def test_all_unique_no_change():
  names = np.array(["A", "B", "C"])
  result = unique_antenna_names(names)
  npt.assert_array_equal(result, names)


def test_all_unique_no_warning():
  names = np.array(["A", "B", "C"])
  with warnings.catch_warnings():
    warnings.simplefilter("error", DuplicateAntennaNameWarning)
    unique_antenna_names(names)  # should not raise


def test_two_duplicates():
  names = np.array(["A", "A", "B"])
  with pytest.warns(DuplicateAntennaNameWarning):
    result = unique_antenna_names(names)
  npt.assert_array_equal(result, ["A-1", "A-2", "B"])


def test_triple_duplicate():
  names = np.array(["A", "A", "A"])
  with pytest.warns(DuplicateAntennaNameWarning):
    result = unique_antenna_names(names)
  npt.assert_array_equal(result, ["A-1", "A-2", "A-3"])


def test_non_contiguous_duplicates():
  names = np.array(["A", "B", "A", "C", "A"])
  with pytest.warns(DuplicateAntennaNameWarning):
    result = unique_antenna_names(names)
  npt.assert_array_equal(result, ["A-1", "B", "A-2", "C", "A-3"])


def test_unique_name_unchanged_when_mixed():
  names = np.array(["X", "Y", "X"])
  with pytest.warns(DuplicateAntennaNameWarning):
    result = unique_antenna_names(names)
  npt.assert_array_equal(result, ["X-1", "Y", "X-2"])


def test_underscore_in_base_name():
  """Dash separator avoids collision with names already containing underscores."""
  names = np.array(["ANTENNA_1", "ANTENNA_1"])
  with pytest.warns(DuplicateAntennaNameWarning):
    result = unique_antenna_names(names)
  npt.assert_array_equal(result, ["ANTENNA_1-1", "ANTENNA_1-2"])


def test_empty_array():
  names = np.array([], dtype=str)
  result = unique_antenna_names(names)
  npt.assert_array_equal(result, [])


def test_single_element():
  names = np.array(["A"])
  result = unique_antenna_names(names)
  npt.assert_array_equal(result, ["A"])


def test_warning_raised_on_duplicates():
  names = np.array(["A", "A", "B"])
  with pytest.warns(DuplicateAntennaNameWarning, match="Duplicate antenna names"):
    unique_antenna_names(names)


def test_warning_lists_duplicate_names():
  names = np.array(["FOO", "BAR", "FOO"])
  with pytest.warns(DuplicateAntennaNameWarning, match="FOO"):
    unique_antenna_names(names)


def test_result_is_always_unique():
  """Output names must all be distinct regardless of input."""
  names = np.array(["A", "A", "B", "B", "C"])
  with pytest.warns(DuplicateAntennaNameWarning):
    result = unique_antenna_names(names)
  assert len(set(result.tolist())) == len(result)


def _multi_feed_tables(receptor_angle):
  """ANTENNA and FEED tables with three antennas, each with two feeds.
  FEED rows are ordered by feed, then by antenna in the order 2, 0, 1"""
  antenna = pa.table({"NAME": ["A0", "A1", "A2"]})
  feed = pa.table(
    {
      "ANTENNA_ID": pa.array([2, 0, 1, 2, 0, 1], pa.int32()),
      "FEED_ID": pa.array([0, 0, 0, 1, 1, 1], pa.int32()),
      "SPECTRAL_WINDOW_ID": pa.array([-1] * 6, pa.int32()),
      "RECEPTOR_ANGLE": pa.array(receptor_angle, pa.list_(pa.float64())),
      "POLARIZATION_TYPE": pa.array([["X", "Y"]] * 6, pa.list_(pa.string())),
    }
  )
  partition = SimpleNamespace(
    spw_id=0,
    feed_ids=np.array([0, 1], np.int32),
    antenna_ids=np.array([0, 1, 2], np.int32),
  )
  return antenna, feed, partition


def test_select_partition_antennas_multi_feed():
  """Antennas with multiple feeds are selected once, in FEED-table order"""
  antenna, feed, partition = _multi_feed_tables([[0.0, 1.0]] * 6)
  with warnings.catch_warnings():
    warnings.simplefilter("error", HeterogeneousFeedWarning)
    selection = select_partition_antennas(antenna, feed, partition)

  npt.assert_array_equal(selection.antenna_ids, [2, 0, 1])
  npt.assert_array_equal(selection.antenna_names, ["A2", "A0", "A1"])
  npt.assert_array_equal(selection.feed_row_indices, [0, 1, 2])


def test_select_partition_antennas_heterogeneous_feeds():
  """Differing per-feed quantities warn and use the first FEED row"""
  receptor_angle = [[0.0, 1.0]] * 3 + [[0.5, 1.5]] * 3
  antenna, feed, partition = _multi_feed_tables(receptor_angle)
  with pytest.warns(HeterogeneousFeedWarning, match="RECEPTOR_ANGLE"):
    selection = select_partition_antennas(antenna, feed, partition)

  npt.assert_array_equal(selection.antenna_ids, [2, 0, 1])
  npt.assert_array_equal(selection.feed_row_indices, [0, 1, 2])


def test_select_partition_antennas_prefers_partition_spw():
  """FEED rows matching the partition spw are preferred over global spw rows"""
  antenna = pa.table({"NAME": ["A0", "A1"]})
  feed = pa.table(
    {
      "ANTENNA_ID": pa.array([0, 1, 0, 0], pa.int32()),
      "FEED_ID": pa.array([0, 0, 0, 0], pa.int32()),
      "SPECTRAL_WINDOW_ID": pa.array([-1, -1, 1, 0], pa.int32()),
      "RECEPTOR_ANGLE": pa.array(
        [[0.0, 1.0], [0.0, 1.0], [0.25, 1.25], [0.5, 1.5]], pa.list_(pa.float64())
      ),
      "POLARIZATION_TYPE": pa.array([["X", "Y"]] * 4, pa.list_(pa.string())),
    }
  )
  partition = SimpleNamespace(
    spw_id=0,
    feed_ids=np.array([0], np.int32),
    antenna_ids=np.array([0, 1], np.int32),
  )
  with warnings.catch_warnings():
    warnings.simplefilter("error", HeterogeneousFeedWarning)
    selection = select_partition_antennas(antenna, feed, partition)

  # Antenna 0 uses its spw 0 row, antenna 1 falls back to its global spw row
  npt.assert_array_equal(selection.antenna_ids, [1, 0])
  npt.assert_array_equal(selection.feed_row_indices, [1, 3])
