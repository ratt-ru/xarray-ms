import numpy as np
import pytest
import xarray
from arcae.lib.arrow_tables import Table, ms_descriptor
from numpy.testing import assert_array_equal

from xarray_ms.errors import ImputedMetadataWarning
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.ImputedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_obs.ms"}], indirect=True)
def test_imputed_observation_metadata(simmed_ms):
  """Tests that data is imputed if the OBSERVATION subtable is empty"""
  obs_table_desc = ms_descriptor("OBSERVATION", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "OBSERVATION", table_desc=obs_table_desc):
    pass

  with pytest.warns(
    ImputedMetadataWarning,
    match="No row exists in the OBSERVATION table of length 0 for OBSERVATION_ID=0",
  ):
    for node in xarray.open_datatree(simmed_ms).subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert node.observation_info == {
          "observer": ["unknown"],
          "project": "unknown",
          "intents": ["CALIBRATE_AMPL#OFF_SOURCE"],
          "release_date": "1978-10-09T08:00:00+00:00",
        }


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.ImputedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_state.ms"}], indirect=True)
def test_imputed_state_metadata(simmed_ms):
  """Tests that data is imputed if the STATE subtable is empty"""
  state_table_desc = ms_descriptor("STATE", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "STATE", table_desc=state_table_desc):
    pass

  with pytest.warns(
    ImputedMetadataWarning,
    match="No row exists in the STATE table of length 0 for STATE_ID=0",
  ):
    xarray.open_datatree(simmed_ms)


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.ImputedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_field.ms"}], indirect=True)
def test_imputed_field_metadata(simmed_ms):
  """Tests that data is imputed if the FIELD subtable is empty"""
  field_table_desc = ms_descriptor("FIELD", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "FIELD", table_desc=field_table_desc):
    pass

  with pytest.warns(
    ImputedMetadataWarning,
    match="No row exists in the FIELD table of length 0 for FIELD_ID=0",
  ):
    for node in xarray.open_datatree(simmed_ms).subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert_array_equal(
          node.field_name, np.array(["UNKNOWN-0"] * node.sizes["time"], dtype=object)
        )


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.ImputedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_processor.ms"}], indirect=True)
def test_imputed_processor_metadata(simmed_ms):
  """Tests that data is imputed if the PROCESSOR subtable is empty"""
  field_table_desc = ms_descriptor("PROCESSOR", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "PROCESSOR", table_desc=field_table_desc):
    pass

  with pytest.warns(
    ImputedMetadataWarning,
    match="No row exists in the PROCESSOR table of length 0 for PROCESSOR_ID=0",
  ):
    for node in xarray.open_datatree(simmed_ms).subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert node.processor_info == {"type": "unknown", "sub_type": "unknown"}


def _negative_proc_ids(chunk_desc, data_dict):
  data_dict["PROCESSOR_ID"][-1][:] = -1
  return data_dict


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.ImputedMetadataWarning")
@pytest.mark.parametrize(
  "simmed_ms",
  [{"name": "missing_processor.ms", "transform_data": _negative_proc_ids}],
  indirect=True,
)
def test_imputed_processor_metadata_negative_proc_ids(simmed_ms):
  """Tests that data is imputed if the PROCESSOR subtable is empty"""
  field_table_desc = ms_descriptor("PROCESSOR", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "PROCESSOR", table_desc=field_table_desc):
    pass

  with Table.from_filename(simmed_ms) as T:
    assert_array_equal(T.getcol("PROCESSOR_ID"), -1)

  with pytest.warns(
    ImputedMetadataWarning,
    match="No row exists in the PROCESSOR table of length 0 for PROCESSOR_ID=0",
  ):
    for node in xarray.open_datatree(simmed_ms).subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert node.processor_info == {"type": "unknown", "sub_type": "unknown"}
