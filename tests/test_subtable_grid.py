import numpy as np
import numpy.testing as npt

from xarray_ms.backend.msv2.subtable_grid import effective_interval


def test_effective_interval_per_antenna_cadence():
  """Instantaneous samples are valid for the cadence of their own antenna"""
  time = np.concatenate([np.arange(0.0, 60.0, 1.0), np.arange(0.0, 60.0, 30.0)])
  antenna_id = np.repeat([0, 1], [60, 2])
  interval = np.full(time.size, 0.048)
  result = effective_interval(time, interval, antenna_id)
  npt.assert_array_equal(result, np.repeat([1.0, 30.0], [60, 2]))


def test_effective_interval_retains_recorded_intervals():
  """Intervals comparable to the cadence are retained,
  even if shorter than the cadence"""
  time = np.arange(0.0, 80.0, 8.0)
  interval = np.full(time.size, 4.0)
  result = effective_interval(time, interval, np.zeros(time.size, np.int32))
  npt.assert_array_equal(result, interval)


def test_effective_interval_without_cadence():
  """Zero intervals of single samples are valid indefinitely"""
  result = effective_interval(
    np.array([5.0, 7.0]), np.array([0.0, 10.0]), np.array([0, 1])
  )
  npt.assert_array_equal(result, [np.inf, 10.0])
