import os.path

import pytest
import xarray

import xarray_ms  # noqa

pmx = pytest.mark.xfail


@pytest.mark.parametrize(
  "msv4_corpus_dataset",
  [
    "ea25_cal_small_before_fixed.split.ms",
    "ea25_cal_small_after_fixed.split.ms",
    "J1924-2914.ms.calibrated.split.SPW3",
    "AA2-Mid-sim_00000.ms",
    pytest.param(
      "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms",
      marks=pmx(reason="FEED + ANTENNA mismatch"),
    ),
    "Antennae_North.cal.lsrk.ms",
    "Antennae_North.cal.lsrk.split.ms",
    "global_vlbi_gg084b_reduced.ms",
    "VLBA_TL016B_split_lsrk.ms",
    "VLBA_TL016B_split.ms",
    "VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms",
    "ngEHT_E17A10.0.bin0000.source0000_split_lsrk.ms",
    "ngEHT_E17A10.0.bin0000.source0000_split.ms",
    "ska_low_sim_18s.ms",
    "small_meerkat.ms",
    "small_lofar.ms",
    pytest.param(
      "uid___A002_X1015532_X1926f.small.ms", marks=pmx(reason="Single Dish")
    ),
    pytest.param("uid___A002_Xae00c5_X2e6b.small.ms", marks=pmx(reason="Single Dish")),
    pytest.param("uid___A002_Xced5df_Xf9d9.small.ms", marks=pmx(reason="Single Dish")),
    pytest.param("uid___A002_Xe3a5fd_Xe38e.small.ms", marks=pmx(reason="Single Dish")),
    pytest.param("SNR_G55_10s.split.ms", marks=pmx(reason="Only one feed?")),
    "59749_bp_8beams_pattern.ms",
    "59750_altaz_2settings.ms",
    "59754_altaz_2weights_0.ms",
    "59754_altaz_2weights_15.ms",
    "59755_eq_interleave_0.ms",
    "59755_eq_interleave_15.ms",
    pytest.param("gmrt.ms", marks=pmx(reason="TAI Measures conversion")),
  ],
  indirect=True,
)
def test_msv4_corpus(msv4_corpus_dataset):
  name, path = msv4_corpus_dataset
  xarray.open_datatree(f"{path}{os.path.sep}{name}")
