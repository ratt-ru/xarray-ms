import numpy as np
import numpy.testing as npt
import pytest
import xarray
from arcae.lib.arrow_tables import Table, ms_descriptor

NANTENNA = 4


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "backend.ms",
      "nantenna": NANTENNA,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
    }
  ],
  indirect=True,
)
def test_antenna_feed_join(simmed_ms):
  """Tests a number of cases for the join of the ANTENNA and FEED subtables."""

  def _fill_other_columns(T, index):
    """Fill other feed columns with boilerplate"""
    T.putcol("NUM_RECEPTORS", np.array([2]), index=index)
    T.putcol("BEAM_OFFSET", np.full((1, 2, 2), 0.1), index=index)
    T.putcol("RECEPTOR_ANGLE", np.full((1, 2), 0.1), index=index)
    T.putcol("POL_RESPONSE", np.full((1, 2, 2), 0.1 + 0.1j), index=index)

  feed_table_desc = ms_descriptor("FEED", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "FEED", table_desc=feed_table_desc) as T:
    # Test some antenna, feed, spw combinations
    # Due to the internal simulator logic and the above
    # data_description configuration, we get two partitions
    # with partition 0 containing SPW 0 == FEED 1
    # and partition 1 containing SPW 1 == FEED 0
    T.addrows(7)

    # Add ANTENNA-0 to second partition
    index = (np.array([0]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([1]), index=index)
    T.putcol("FEED_ID", np.array([0]), index=index)
    T.putcol("ANTENNA_ID", np.array([0]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["R", "L"]]), index=index)
    _fill_other_columns(T, index)

    # Add ANTENNA-1 to first partition
    index = (np.array([1]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([0]), index=index)
    T.putcol("FEED_ID", np.array([1]), index=index)
    T.putcol("ANTENNA_ID", np.array([1]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]]), index=index)
    _fill_other_columns(T, index)

    # Add ANTENNA-2 to second partition (SPW_ID = -1 and FEED_ID = 0)
    index = (np.array([2]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([-1]), index=index)
    T.putcol("FEED_ID", np.array([0]), index=index)
    T.putcol("ANTENNA_ID", np.array([2]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["R", "L"]]), index=index)
    _fill_other_columns(T, index)

    # Add ANTENNA-2 to first partition (SPW_ID = -1 and FEED_ID = 1)
    index = (np.array([3]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([-1]), index=index)
    T.putcol("FEED_ID", np.array([1]), index=index)
    T.putcol("ANTENNA_ID", np.array([2]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]]), index=index)
    _fill_other_columns(T, index)

    # Add ANTENNA-3 link to non-existent MAIN table FEED
    index = (np.array([4]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([-1]), index=index)
    T.putcol("FEED_ID", np.array([2]), index=index)
    T.putcol("ANTENNA_ID", np.array([3]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["R", "L"]]), index=index)
    _fill_other_columns(T, index)

    # Add ANTENNA-4 link to non-existent partition (SPW_ID = 1, FEED_ID = 1)
    index = (np.array([5]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([1]), index=index)
    T.putcol("FEED_ID", np.array([1]), index=index)
    T.putcol("ANTENNA_ID", np.array([3]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["R", "L"]]), index=index)
    _fill_other_columns(T, index)

    # Add ANTENNA-5 link to non-existent partition (SPW_ID = 0, FEED_ID = 0)
    index = (np.array([6]),)
    T.putcol("SPECTRAL_WINDOW_ID", np.array([0]), index=index)
    T.putcol("FEED_ID", np.array([0]), index=index)
    T.putcol("ANTENNA_ID", np.array([4]), index=index)
    T.putcol("POLARIZATION_TYPE", np.array([["R", "L"]]), index=index)
    _fill_other_columns(T, index)

  dt = xarray.open_datatree(simmed_ms)
  assert list(dt.children) == ["backend_partition_000", "backend_partition_001"]

  p0 = dt["backend_partition_000"]
  p1 = dt["backend_partition_001"]

  npt.assert_array_equal(p0.polarization, ["XX", "XY", "YX", "YY"])
  npt.assert_array_equal(p1.polarization, ["RR", "LL"])

  npt.assert_array_equal(
    p0["antenna_xds"].antenna_name, np.array(["ANTENNA-1", "ANTENNA-2"], dtype=object)
  )
  npt.assert_array_equal(
    p1["antenna_xds"].antenna_name, np.array(["ANTENNA-0", "ANTENNA-2"], dtype=object)
  )
