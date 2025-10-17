import json
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pytest
import xarray
import xarray.testing as xt
from arcae.lib.arrow_tables import ms_descriptor
from xarray import Variable

from xarray_ms.backend.msv2.array import MainMSv2Array
from xarray_ms.backend.msv2.measures_adapters import MeasuresAdapterFactory
from xarray_ms.backend.msv2.measures_encoders import EpochCoder, MSv2CoderFactory
from xarray_ms.backend.msv2.table_utils import extract_table_desc as extract_table_desc
from xarray_ms.casa_types import FrequencyMeasures
from xarray_ms.errors import PartitioningError


@pytest.fixture(params=[["TOPO", "TOPO", "LSRK"]])
def spw_with_meas_freq_ref(request):
  """Creates a minimal SPECTRAL_WINDOW table with a MEAS_FREQ_REF column
  containing the codes referring to the input measures frames"""
  frames = request.param
  meas_freq_ref = np.array([FrequencyMeasures[f].value for f in frames])
  num_chans = np.arange(1, len(frames) + 1) * 4
  ref_freq = np.full(len(num_chans), (0.856e9 + 2 * 0.856e9) / 2)
  flat_chan_freqs = np.concatenate(
    [np.linspace(0.856e9, 2 * 0.856e9, n) for n in num_chans]
  )
  offsets = np.cumsum(np.concatenate(([0], num_chans)))
  chan_freq = pa.ListArray.from_arrays(offsets, flat_chan_freqs)
  return pa.Table.from_pydict(
    {
      "MEAS_FREQ_REF": meas_freq_ref,
      "NUM_CHAN": num_chans,
      "CHAN_FREQ": chan_freq,
      "REF_FREQUENCY": ref_freq,
    },
  )


@pytest.fixture
def spw_with_meas_freq_ref_lsrk_ref(spw_with_meas_freq_ref):
  """Extends the SPECTRAL_WINDOW table columns to only contain
  a single LSRK frame"""
  return spw_with_meas_freq_ref.replace_schema_metadata(
    metadata={
      "__arcae_metadata__": json.dumps(
        {
          "__casa_descriptor__": {
            "REF_FREQUENCY": {
              "valueType": "double",
              "option": 0,
              "keywords": {
                "QuantumUnits": ["Hz"],
                "MEASINFO": {
                  "type": "frequency",
                  "Ref": "LSRK",
                },
              },
            },
            "CHAN_FREQ": {
              "valueType": "double",
              "option": 0,
              "keywords": {
                "QuantumUnits": ["Hz"],
                "MEASINFO": {
                  "type": "frequency",
                  "Ref": "LSRK",
                },
              },
            },
          }
        }
      )
    },
  )


@pytest.fixture
def spw_with_meas_freq_ref_varcolref(spw_with_meas_freq_ref):
  """Extends the SPECTRAL_WINDOW table columns to contain multiple
  frames via the MEAS_FREQ_REF column referred to by VarRefCol"""
  return spw_with_meas_freq_ref.replace_schema_metadata(
    {
      "__arcae_metadata__": json.dumps(
        {
          "__casa_descriptor__": {
            "MEAS_FREQ_REF": {
              "valueType": "integer",
              "option": 0,
            },
            "REF_FREQUENCY": {
              "valueType": "double",
              "option": 0,
              "keywords": {
                "QuantumUnits": ["Hz"],
                "MEASINFO": {
                  "type": "frequency",
                  "VarRefCol": "MEAS_FREQ_REF",
                  "TabRefTypes": [
                    "REST",
                    "LSRK",
                    "LSRD",
                    "BARY",
                    "GEO",
                    "TOPO",
                    "GALACTO",
                    "LGROUP",
                    "CMB",
                    "Undefined",
                  ],
                  "TabRefCodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 64],
                },
              },
            },
            "CHAN_FREQ": {
              "valueType": "double",
              "option": 0,
              "keywords": {
                "QuantumUnits": ["Hz"],
                "MEASINFO": {
                  "type": "frequency",
                  "VarRefCol": "MEAS_FREQ_REF",
                  "TabRefTypes": [
                    "REST",
                    "LSRK",
                    "LSRD",
                    "BARY",
                    "GEO",
                    "TOPO",
                    "GALACTO",
                    "LGROUP",
                    "CMB",
                    "Undefined",
                  ],
                  "TabRefCodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 64],
                },
              },
            },
          }
        }
      )
    },
  )


def test_spw_measures_table_desc(spw_with_meas_freq_ref_lsrk_ref):
  """Test extraction of measures via the table descriptor only"""

  measures_adapter_factory = MeasuresAdapterFactory.from_table_desc(
    extract_table_desc(spw_with_meas_freq_ref_lsrk_ref)
  )
  chan_freq_adapter = measures_adapter_factory.create("CHAN_FREQ")
  assert chan_freq_adapter.msv2_frame() == "LSRK"
  assert chan_freq_adapter.msv2_type() == "frequency"
  assert chan_freq_adapter.msv4_type() == "spectral_coord"
  assert chan_freq_adapter.quantum_unit() == "Hz"

  ref_freq_adapter = measures_adapter_factory.create("REF_FREQUENCY")
  assert ref_freq_adapter.msv2_frame() == "LSRK"
  assert chan_freq_adapter.msv2_type() == "frequency"
  assert chan_freq_adapter.msv4_type() == "spectral_coord"
  assert chan_freq_adapter.quantum_unit() == "Hz"

  coder_factory = MSv2CoderFactory(measures_adapter_factory)
  frequency_coder = coder_factory.create("CHAN_FREQ")
  frequency = frequency_coder.decode(
    Variable("frequency", spw_with_meas_freq_ref_lsrk_ref["CHAN_FREQ"].to_numpy())
  )
  assert frequency.attrs == {
    "units": "Hz",
    "type": "spectral_coord",
    "observer": "lsrk",
  }
  ref_freq_coder = coder_factory.create("REF_FREQUENCY")
  ref_freq = ref_freq_coder.decode(
    Variable("spw", spw_with_meas_freq_ref_lsrk_ref["REF_FREQUENCY"].to_numpy())
  )
  assert ref_freq.attrs == {"units": "Hz", "type": "spectral_coord", "observer": "lsrk"}


@pytest.mark.parametrize("spw_with_meas_freq_ref", [["TOPO", "TOPO"]], indirect=True)
def test_spw_measures_table(spw_with_meas_freq_ref_varcolref):
  """Test extraction of measures via the table descriptor and table object"""
  measures_adapter_factory = MeasuresAdapterFactory.from_arrow_table(
    spw_with_meas_freq_ref_varcolref
  )
  chan_freq_adapter = measures_adapter_factory.create("CHAN_FREQ")
  assert chan_freq_adapter.msv2_frame() == "TOPO"
  assert chan_freq_adapter.msv2_type() == "frequency"
  assert chan_freq_adapter.msv4_type() == "spectral_coord"
  assert chan_freq_adapter.quantum_unit() == "Hz"

  coder_factory = MSv2CoderFactory(measures_adapter_factory)
  coder = coder_factory.create("CHAN_FREQ")
  var = coder.decode(
    Variable("frequency", spw_with_meas_freq_ref_varcolref["CHAN_FREQ"].to_numpy())
  )
  assert var.attrs == {"units": "Hz", "type": "spectral_coord", "observer": "TOPO"}


@pytest.mark.parametrize("spw_with_meas_freq_ref", [["TOPO", "LSRK"]], indirect=True)
def test_spw_measures_table_multiple_frames(spw_with_meas_freq_ref_varcolref):
  """Test extraction of measures via the table descriptor and table object fails
  if MEAS_FREQ_REF contains multiple frames"""
  measures_adapter_factory = MeasuresAdapterFactory.from_arrow_table(
    spw_with_meas_freq_ref_varcolref
  )
  chan_freq_adapter = measures_adapter_factory.create("CHAN_FREQ")
  with pytest.raises(PartitioningError):
    assert chan_freq_adapter.msv2_frame() == "TOPO"

  coder_factory = MSv2CoderFactory(measures_adapter_factory)
  coder = coder_factory.create("CHAN_FREQ")
  with pytest.raises(PartitioningError):
    coder.decode(
      Variable("frequency", spw_with_meas_freq_ref_varcolref["CHAN_FREQ"].to_numpy())
    )


SECONDS_IN_DAY = 24 * 60 * 60


# Worked examples from https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/DateConv/dateconv.pl
@pytest.mark.parametrize(
  "mjd, utc",
  [
    (51544.0, datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
    (59999.874340277776, datetime(2023, 2, 24, 20, 59, 3, tzinfo=timezone.utc)),
    (60592.87866898148, datetime(2024, 10, 9, 21, 5, 17, tzinfo=timezone.utc)),
  ],
)
@pytest.mark.parametrize("msv2_frame", ["UTC", "TAI"])
def test_utc_time_encoder_roundtrip_ndarray(mjd, utc, msv2_frame):
  """Test conversion of Modified Julian Date
  to UTC/TAI in seconds, and vice versa"""

  table_desc = ms_descriptor("MAIN", True)
  table_desc["TIME"]["keywords"]["MEASINFO"]["Ref"] = msv2_frame
  coder_factory = MSv2CoderFactory.from_table_desc(table_desc)
  time_coder = coder_factory.create("TIME")
  time = Variable(("dummy",), np.array([mjd * SECONDS_IN_DAY]))
  decoded_time = time_coder.decode(time)
  utc_seconds = decoded_time.values[0]
  assert utc == datetime.fromtimestamp(utc_seconds, timezone.utc)
  assert decoded_time.attrs == {
    "type": "time",
    "units": "s",
    "scale": msv2_frame.lower(),
    "format": "unix",
  }
  encoded_time = time_coder.encode(decoded_time)
  xt.assert_equal(encoded_time, time)
  assert encoded_time.attrs == {}


@pytest.mark.parametrize(
  "mjd, utc",
  [
    (51544.0, datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
    (59999.874340277776, datetime(2023, 2, 24, 20, 59, 3, tzinfo=timezone.utc)),
    (60592.87866898148, datetime(2024, 10, 9, 21, 5, 17, tzinfo=timezone.utc)),
  ],
)
def test_utc_time_encoder_roundtrip_backend_array_utc(mjd, utc):
  """Test conversion of Modified Julian Date
  to UTC in seconds, and vice versa"""

  coder_factory = MSv2CoderFactory.from_table_desc(ms_descriptor("MAIN", True))
  time_coder = coder_factory.create("TIME")
  data = np.array([mjd * SECONDS_IN_DAY])[:, None]
  array = MainMSv2Array(None, None, None, "TIME", data.shape, data.dtype, 0.0, None)
  time = Variable(("time", "baseline_Id"), array)
  decoded_time = time_coder.decode(time)
  assert decoded_time._data.transform is EpochCoder.decode_array
  assert utc == datetime.fromtimestamp(
    decoded_time._data.transform(data).item(), timezone.utc
  )
  assert decoded_time.attrs == {
    "type": "time",
    "units": "s",
    "scale": "utc",
    "format": "unix",
  }


def test_standard_conversion_measures(simmed_ms):
  """Test measures conversion of some standard columns on a standard measurement set"""
  dt = xarray.open_datatree(simmed_ms)

  for partition in dt.children:
    # correlated_data_xds
    ds = dt[partition]
    assert ds.time.attrs == {
      "format": "unix",
      "integration_time": {
        "attrs": {
          "type": "quantity",
          "units": "s",
        },
        "data": 8.0,
      },
      "scale": "utc",
      "type": "time",
      "units": "s",
    }
    assert ds.frequency.attrs == {
      "channel_width": {
        "attrs": {
          "type": "quantity",
          "units": "Hz",
        },
        "data": 107000000.0,
      },
      "effective_channel_width": "EFFECTIVE_CHANNEL_WIDTH",
      "observer": "REST",
      "reference_frequency": {
        "attrs": {
          "observer": "REST",
          "type": "spectral_coord",
          "units": "Hz",
        },
        "data": 1712000000.0,
      },
      "spectral_window_intents": ["<Unknown>"],
      "spectral_window_name": "<Unknown>",
      "type": "spectral_coord",
      "units": "Hz",
    }
    assert ds.scan_name.attrs == {
      "scan_intents": ["CALIBRATE_AMPL#OFF_SOURCE"],
    }
    assert ds.TIME_CENTROID.attrs == {
      "format": "unix",
      "scale": "utc",
      "type": "time",
      "units": "s",
    }
    assert ds.EFFECTIVE_INTEGRATION_TIME.attrs == {
      "type": "quantity",
      "units": "s",
    }
    assert ds.UVW.attrs == {"frame": "fk5", "type": "uvw", "units": "m"}
    assert ds.VISIBILITY.attrs == {"type": "quantity", "units": "Jy"}
    assert ds.WEIGHT.attrs == {}
    assert ds.FLAG.attrs == {}

    # antenna_xds
    ant = ds["antenna_xds"]
    assert ant.ANTENNA_POSITION.attrs == {
      "coordinate_system": "geocentric",
      "origin_object_name": "earth",
      "type": "location",
      "units": "m",
      "frame": "ITRS",
    }
    assert ant.ANTENNA_DISH_DIAMETER.attrs == {"type": "quantity", "units": "m"}
    assert ant.ANTENNA_EFFECTIVE_DISH_DIAMETER.attrs == {
      "type": "quantity",
      "units": "m",
    }
    assert ant.ANTENNA_RECEPTOR_ANGLE.attrs == {"type": "quantity", "units": "rad"}

    # field_and_source_xds
    for _, dg in ds.attrs["data_groups"].items():
      fns = dt[dg["field_and_source"]]
      fns.FIELD_PHASE_CENTER_DIRECTION.attrs == {
        "type": "sky_coord",
        "units": "rad",
        "frame": "fk5",
      }
