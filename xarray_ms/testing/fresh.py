"""Generated complete-crosscorr-v1 demonstrator, not observational evidence."""

import numpy as np
import xarray as xr


def make_complete_tree():
  """Three realistic antennas, four integrations, eight channels and XX/YY."""
  dims = ("time", "baseline_id", "frequency", "polarization")
  ramp = np.arange(4 * 3 * 8 * 2).reshape(4, 3, 8, 2)
  flags = np.zeros(ramp.shape, bool)
  flags[0, 1, 2, 0] = True
  uvw = np.array([[30.0, 10.0, 4.0], [60.0, -20.0, 8.0], [30.0, -30.0, 4.0]])

  def quantity(unit):
    return {"type": "quantity", "units": unit}

  frequency_attrs = {
    "type": "spectral_coord",
    "units": "Hz",
    "observer": "TOPO",
    "spectral_window_name": "spw",
    "spectral_window_intents": ["science"],
    "reference_frequency": {
      "attrs": {"type": "spectral_coord", "units": "Hz", "observer": "TOPO"},
      "data": 1.4e9,
    },
    "channel_width": {"attrs": quantity("Hz"), "data": 1e6},
  }
  ds = xr.Dataset(
    {
      "VISIBILITY": (dims, (1 + ramp * 0.001 + 0.01j).astype(np.complex64)),
      "FLAG": (dims, flags),
      "WEIGHT": (dims, np.full(ramp.shape, 2.0, np.float32)),
      "UVW": (
        ("time", "baseline_id", "uvw_label"),
        np.stack([uvw + [i, i * 0.5, 0] for i in range(4)]),
        {"type": "uvw", "units": "m", "frame": "fk5"},
      ),
    },
    coords={
      "time": (
        "time",
        1_600_000_000.0 + np.arange(4) * 10,
        {
          "type": "time",
          "units": "s",
          "format": "unix",
          "scale": "utc",
          "integration_time": {"attrs": quantity("s"), "data": 10.0},
        },
      ),
      "baseline_id": np.arange(3),
      "frequency": ("frequency", 1.4e9 + np.arange(8) * 1e6, frequency_attrs),
      "polarization": ["XX", "YY"],
      "uvw_label": ["u", "v", "w"],
      "baseline_antenna1_name": ("baseline_id", ["A", "A", "B"]),
      "baseline_antenna2_name": ("baseline_id", ["B", "C", "C"]),
      "field_name": ("time", ["field"] * 4),
      "scan_name": ("time", ["1"] * 4, {"scan_intents": ["OBSERVE_TARGET"]}),
    },
    attrs={
      "type": "visibility",
      "generated_demonstrator": "complete-crosscorr-v1",
      "data_groups": {
        "base": {
          "correlated_data": "VISIBILITY",
          "flag": "FLAG",
          "weight": "WEIGHT",
          "uvw": "UVW",
          "field_and_source": "field_and_source_base_xds",
        }
      },
      "observation_info": {
        "observer": ["Observer"],
        "project_UID": "project",
        "release_date": "2020-01-01T00:00:00+00:00",
      },
      "processor_info": {"type": "CORRELATOR", "sub_type": "generated"},
    },
  )
  field = xr.Dataset(
    {
      "FIELD_PHASE_CENTER_DIRECTION": (
        ("field_name", "sky_dir_label"),
        [[0.1, 0.2]],
        {"type": "sky_coord", "units": "rad", "frame": "icrs"},
      )
    },
    coords={
      "field_name": ["field"],
      "sky_dir_label": ["ra", "dec"],
      "source_name": ("field_name", ["source"]),
    },
    attrs={"type": "field_and_source"},
  )
  antenna = xr.Dataset(
    {
      "ANTENNA_POSITION": (
        ("antenna_name", "cartesian_pos_label"),
        [
          [5109342.0, 2005241.0, -3239939.0],
          [5109372.0, 2005251.0, -3239935.0],
          [5109402.0, 2005221.0, -3239931.0],
        ],
        {"type": "location", "units": "m", "frame": "ITRS"},
      ),
      "ANTENNA_DISH_DIAMETER": ("antenna_name", [12.0] * 3, quantity("m")),
      "ANTENNA_EFFECTIVE_DISH_DIAMETER": ("antenna_name", [12.0] * 3, quantity("m")),
      "ANTENNA_RECEPTOR_ANGLE": (
        ("antenna_name", "receptor_label"),
        [[0.0, 0.0]] * 3,
        quantity("rad"),
      ),
    },
    coords={
      "antenna_name": ["A", "B", "C"],
      "station_name": ("antenna_name", ["A", "B", "C"]),
      "mount": ("antenna_name", ["ALT-AZ"] * 3),
      "telescope_name": ("antenna_name", ["scope"] * 3),
      "cartesian_pos_label": ["x", "y", "z"],
      "receptor_label": ["pol_0", "pol_1"],
      "polarization_type": (("antenna_name", "receptor_label"), [["X", "Y"]] * 3),
    },
    attrs={"type": "antenna", "overall_telescope_name": "scope"},
  )
  return xr.DataTree.from_dict(
    {
      "/part": ds,
      "/part/field_and_source_base_xds": field,
      "/part/antenna_xds": antenna,
    }
  )
