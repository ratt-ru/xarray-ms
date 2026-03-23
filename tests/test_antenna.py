import numpy as np
import numpy.testing as npt
import pytest
import xarray
from arcae.lib.arrow_tables import Table, ms_descriptor

from xarray_ms.errors import DuplicateAntennaNameWarning

NANTENNA = 6


def _set_ddid_antenna_ids(desc):
  if desc.DATA_DESC_ID.item() == 0:
    ant_ids = np.array([1, 2], np.int32)
  elif desc.DATA_DESC_ID.item() == 1:
    ant_ids = np.array([0, 2], np.int32)
  else:
    raise NotImplementedError(f"DATA_DESC_ID {desc.DATA_DESC_ID}")

  # k = 0 produces full resolution grid
  ant1, ant2 = (ant_ids[a] for a in np.triu_indices(ant_ids.size, 0))

  desc.ANTENNA1 = ant1
  desc.ANTENNA2 = ant2

  return desc


@pytest.mark.parametrize(
  "simmed_ms",
  [
    {
      "name": "fj.ms",
      "nantenna": NANTENNA,
      "data_description": [(8, ["XX", "XY", "YX", "YY"]), (4, ["RR", "LL"])],
      "transform_chunk_desc": _set_ddid_antenna_ids,
      "auto_corrs": True,
    }
  ],
  indirect=True,
)
@pytest.mark.parametrize("auto_corrs", [True, False])
def test_antenna_feed_join(simmed_ms, auto_corrs):
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

  dt = xarray.open_datatree(simmed_ms, auto_corrs=auto_corrs)

  # There are two partitions one with ANTENNA-0 and the other with ANTENNA-1
  # They both share ANTENNA-2.
  # Feed configuration associated with non-existent partitions are ignored.
  # This means that each partition has 3 baselines
  # with auto correlations and 1 baseline without
  assert list(dt.children) == ["fj_partition_000", "fj_partition_001"]
  dt = dt.load()

  p0 = dt["fj_partition_000"]
  p1 = dt["fj_partition_001"]

  assert len(p0.baseline_id) == (3 if auto_corrs else 1)
  assert len(p1.baseline_id) == (3 if auto_corrs else 1)

  npt.assert_array_equal(p0.polarization, ["XX", "XY", "YX", "YY"])
  npt.assert_array_equal(p1.polarization, ["RR", "LL"])

  npt.assert_array_equal(
    p0["antenna_xds"].antenna_name, np.array(["ANTENNA-1", "ANTENNA-2"], dtype=object)
  )
  npt.assert_array_equal(
    p1["antenna_xds"].antenna_name, np.array(["ANTENNA-0", "ANTENNA-2"], dtype=object)
  )


@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "dup.ms", "nantenna": 3, "auto_corrs": True}],
  indirect=True,
)
def test_duplicate_antenna_names(simmed_ms):
  """When ANTENNA::NAME contains duplicates, names are made unique and a
  warning is emitted.  The names in antenna_xds and the baseline coordinate
  must be consistent with each other."""

  # Overwrite ANTENNA::NAME so that antenna 0 and antenna 1 share a name.
  with Table.from_filename(f"{simmed_ms}::ANTENNA") as T:
    T.putcol("NAME", np.asarray(["SAME", "SAME", "OTHER"]))

  with pytest.warns(DuplicateAntennaNameWarning, match="SAME"):
    dt = xarray.open_datatree(simmed_ms, auto_corrs=True).load()

  partition = dt[list(dt.children)[0]]
  antenna_xds = partition["antenna_xds"]

  # All antenna_name coordinate values must be unique
  ant_names = antenna_xds.antenna_name.values.tolist()
  assert len(ant_names) == len(set(ant_names)), "antenna_name coordinate has duplicates"

  # Duplicated base names get -N suffixes; unique names are unchanged
  assert "SAME-1" in ant_names
  assert "SAME-2" in ant_names
  assert "OTHER" in ant_names

  # Every baseline antenna name must appear in antenna_xds.antenna_name
  ant_name_set = set(ant_names)
  assert set(partition.baseline_antenna1_name.values.tolist()) <= ant_name_set
  assert set(partition.baseline_antenna2_name.values.tolist()) <= ant_name_set


@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "nodup.ms", "nantenna": 3, "auto_corrs": True}],
  indirect=True,
)
def test_unique_antenna_names_no_warning(simmed_ms):
  """When all antenna names are already unique no warning should be emitted."""
  import warnings

  with warnings.catch_warnings():
    warnings.simplefilter("error", DuplicateAntennaNameWarning)
    xarray.open_datatree(simmed_ms, auto_corrs=True).load()
