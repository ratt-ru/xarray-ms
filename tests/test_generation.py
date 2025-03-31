import pytest
import xarray
from arcae.lib.arrow_tables import Table, ms_descriptor

from xarray_ms.errors import GeneratedMetadataWarning
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.GeneratedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_obs.ms"}], indirect=True)
def test_generated_observation_metadata(simmed_ms):
  """Tests that data is generated if the OBSERVATION subtable is empty"""
  obs_table_desc = ms_descriptor("OBSERVATION", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "OBSERVATION", table_desc=obs_table_desc):
    pass

  with pytest.warns(
    GeneratedMetadataWarning,
    match="No row exists in the OBSERVATION table of length 0 for OBSERVATION_ID=0",
  ):
    dt = xarray.open_datatree(simmed_ms)

    for node in dt.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        assert node.observation_info == {"observer": "unknown", "project": "unknown"}


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.GeneratedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_state.ms"}], indirect=True)
def test_generated_state_metadata(simmed_ms):
  """Tests that data is generated if the STATE subtable is empty"""
  state_table_desc = ms_descriptor("STATE", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "STATE", table_desc=state_table_desc):
    pass

  with pytest.warns(
    GeneratedMetadataWarning,
    match="No row exists in the STATE table of length 0 for STATE_ID=0",
  ):
    xarray.open_datatree(simmed_ms)


@pytest.mark.filterwarnings("ignore::xarray_ms.errors.GeneratedMetadataWarning")
@pytest.mark.parametrize("simmed_ms", [{"name": "missing_field.ms"}], indirect=True)
def test_generated_field_metadata(simmed_ms):
  """Tests that data is generated if the FIELD subtable is empty"""
  field_table_desc = ms_descriptor("FIELD", complete=True)
  with Table.ms_from_descriptor(simmed_ms, "FIELD", table_desc=field_table_desc):
    pass

  with pytest.warns(
    GeneratedMetadataWarning,
    match="No row exists in the FIELD table of length 0 for FIELD_ID=0",
  ):
    xarray.open_datatree(simmed_ms)
