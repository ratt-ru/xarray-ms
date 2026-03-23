import warnings

import numpy as np
import numpy.testing as npt
import pytest

from xarray_ms.backend.msv2.table_utils import unique_antenna_names
from xarray_ms.errors import DuplicateAntennaNameWarning


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
