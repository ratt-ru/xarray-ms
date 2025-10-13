import json

import numpy as np
import pyarrow as pa
import pytest

from xarray_ms.casa_types import FrequencyMeasures


@pytest.fixture(params=[["TOPO", "TOPO", "LSRK"]])
def spw_table(request):
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
    metadata={
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


@pytest.mark.parametrize("spw_table", [["TOPO", "TOPO"]], indirect=True)
def test_spw_measures(spw_table):
  from xarray_ms.backend.msv2.measures_adapters import ArrowTableMeasuresAdapter

  print(spw_table)

  chan_freq_adapter = ArrowTableMeasuresAdapter("CHAN_FREQ", spw_table)
  print(f"CHAN_FREQ frame {chan_freq_adapter.msv2_frame()}")
  print(f"CHAN_FREQ type {chan_freq_adapter.msv4_type()}")
  print(f"CHAN_FREQ unit {chan_freq_adapter.quantum_unit()}")

  pass
