import os.path

import pytest
import xarray

import xarray_ms  # noqa

pmx = pytest.mark.xfail


@pytest.mark.msv4_test_corpus
@pytest.mark.filterwarnings("ignore::xarray_ms.errors.FrameConversionWarning")
@pytest.mark.filterwarnings("ignore::xarray_ms.errors.ImputedMetadataWarning")
@pytest.mark.filterwarnings("ignore::xarray_ms.errors.IrregularTimeGridWarning")
@pytest.mark.filterwarnings("ignore::xarray_ms.errors.IrregularBaselineGridWarning")
@pytest.mark.parametrize(
  "msv4_corpus_dataset, partition_schema",
  [
    # 3 antenna/baselines
    ("ea25_cal_small_before_fixed.split.ms", ([], ["FIELD_ID"])),
    ("ea25_cal_small_after_fixed.split.ms", ([], ["FIELD_ID"])),
    ("J1924-2914.ms.calibrated.split.SPW3", ([], ["FIELD_ID"])),
    ("AA2-Mid-sim_00000.ms", ([], ["FIELD_ID"])),
    pytest.param(
      "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms",
      ([], ["FIELD_ID"]),
      marks=pmx(reason="FEED + ANTENNA mismatch"),
    ),
    ("Antennae_North.cal.lsrk.ms", ([], ["FIELD_ID"])),
    ("Antennae_North.cal.lsrk.split.ms", ([], ["FIELD_ID"])),
    ("global_vlbi_gg084b_reduced.ms", ([], ["FIELD_ID"])),
    ("VLBA_TL016B_split_lsrk.ms", ([], ["FIELD_ID"])),
    ("VLBA_TL016B_split.ms", ([], ["FIELD_ID"])),
    ("VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms", ([],)),
    ("ngEHT_E17A10.0.bin0000.source0000_split_lsrk.ms", ([], ["FIELD_ID"])),
    ("ngEHT_E17A10.0.bin0000.source0000_split.ms", ([], ["FIELD_ID"])),
    ("ska_low_sim_18s.ms", ([], ["FIELD_ID"])),
    ("small_meerkat.ms", ([], ["FIELD_ID"])),
    ("small_lofar.ms", ([], ["FIELD_ID"])),
    pytest.param(
      "uid___A002_X1015532_X1926f.small.ms",
      ([], ["FIELD_ID"]),
      marks=pmx(reason="Single Dish"),
    ),
    pytest.param(
      "uid___A002_Xae00c5_X2e6b.small.ms",
      ([], ["FIELD_ID"]),
      marks=pmx(reason="Single Dish"),
    ),
    pytest.param(
      "uid___A002_Xced5df_Xf9d9.small.ms",
      ([], ["FIELD_ID"]),
      marks=pmx(reason="Single Dish"),
    ),
    pytest.param(
      "uid___A002_Xe3a5fd_Xe38e.small.ms",
      ([], ["FIELD_ID"]),
      marks=pmx(reason="Single Dish"),
    ),
    ("SNR_G55_10s.split.ms", ([], ["FIELD_ID"])),
    ("59749_bp_8beams_pattern.ms", ([], ["FIELD_ID"])),
    ("59750_altaz_2settings.ms", ([], ["FIELD_ID"])),
    ("59754_altaz_2weights_0.ms", ([], ["FIELD_ID"])),
    ("59754_altaz_2weights_15.ms", ([], ["FIELD_ID"])),
    ("59755_eq_interleave_0.ms", ([], ["FIELD_ID"])),
    ("59755_eq_interleave_15.ms", ([], ["FIELD_ID"])),
    ("gmrt.ms", ([], ["FIELD_ID"])),
  ],
  indirect=["msv4_corpus_dataset"],
)
def test_msv4_corpus(msv4_corpus_dataset, partition_schema):
  name, path = msv4_corpus_dataset
  for ps in partition_schema:
    dt = xarray.open_datatree(f"{path}{os.path.sep}{name}", partition_schema=ps)
    dt.load()
