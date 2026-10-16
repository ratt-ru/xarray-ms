"""Fixed measure metadata is semantic, strict and payload-independent."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from xarray import Variable

from xarray_ms.backend.msv2.measure_encoding import encode_fixed_measure
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory
from xarray_ms.errors import MeasureEncodingError, MeasureReferenceColumnRequired


@pytest.mark.parametrize(
  ("kind", "unit", "key", "value", "frame", "extra"),
  [
    ("time", "s", "scale", "utc", "UTC", {"format": "unix"}),
    ("time", "s", "scale", "tai", "TAI", {"format": "unix"}),
    *[
      ("spectral_coord", "Hz", "observer", value, frame, {})
      for value, frame in (
        ("REST", "REST"),
        ("BARY", "BARY"),
        ("TOPO", "TOPO"),
        ("lsrk", "LSRK"),
        ("lsrd", "LSRD"),
        ("gcrs", "GEO"),
      )
    ],
    *[
      ("sky_coord", "rad", "frame", value, frame, {})
      for value, frame in (
        ("fk5", "J2000"),
        ("icrs", "ICRS"),
        ("altaz", "AZELGEO"),
      )
    ],
    ("location", "m", "frame", "ITRS", "ITRF", {}),
    *[
      ("uvw", "m", "frame", value, frame, {})
      for value, frame in (
        ("fk5", "J2000"),
        ("icrs", "ICRS"),
        ("APP", "APP"),
      )
    ],
  ],
)
def test_exact_keywords_and_semantic_round_trip(kind, unit, key, value, frame, extra):
  attrs = {"type": kind, "units": unit, key: value, **extra}
  variable = Variable(("row",), np.array([1.0]), attrs)
  result = encode_fixed_measure(variable, "/partition", "SOURCE", "COLUMN")
  assert result.frame == frame
  assert result.keywords == {
    "MEASINFO": {"type": result.msv2_type, "Ref": frame},
    "QuantumUnits": (unit,),
  }
  with pytest.raises(TypeError):
    result.keywords["MEASINFO"]["Ref"] = "other"
  with pytest.raises(FrozenInstanceError):
    result.frame = "other"
  descriptor_keywords = result.to_column_keywords()
  assert descriptor_keywords == {
    "MEASINFO": {"type": result.msv2_type, "Ref": frame},
    "QuantumUnits": [unit],
  }
  assert json.loads(json.dumps(descriptor_keywords)) == descriptor_keywords
  descriptor_keywords["MEASINFO"]["Ref"] = "CHANGED"
  descriptor_keywords["QuantumUnits"].append("extra")
  assert result.to_column_keywords()["MEASINFO"]["Ref"] == frame
  assert result.to_column_keywords()["QuantumUnits"] == [unit]
  assert result.keywords["MEASINFO"]["Ref"] == frame
  factory = MSv2CoderFactory.from_table_desc(
    {
      "COLUMN": {
        "valueType": "double",
        "option": 0,
        "keywords": result.to_column_keywords(),
      }
    }
  )
  coder = factory.create("COLUMN")
  decoded = coder.decode(coder.encode(variable))
  assert decoded.attrs == attrs
  np.testing.assert_allclose(decoded.values, variable.values)


@pytest.mark.parametrize(
  "attrs",
  [
    {"type": "time", "units": "s", "format": "mjd", "scale": "utc"},
    {"type": "time", "units": "ms", "format": "unix", "scale": "utc"},
    {"type": "time", "units": "s", "format": "unix", "scale": "tt"},
    {"type": "spectral_coord", "units": "Hz", "observer": "CMB"},
    {"type": "sky_coord", "units": "deg", "frame": "fk5"},
    {"type": "location", "units": "m", "frame": "WGS84"},
    {"type": "uvw", "units": "m", "frame": "ITRF"},
  ],
)
def test_invalid_fixed_metadata_is_contextual(attrs):
  with pytest.raises(MeasureEncodingError, match="/part variable 'X' -> COL"):
    encode_fixed_measure(Variable(("row",), [1], attrs), "/part", "X", "COL")


@pytest.mark.parametrize("reference", [["fk5", "icrs"], np.array(["fk5"])])
def test_variable_reference_requires_column(reference):
  with pytest.raises(MeasureReferenceColumnRequired, match="reference column required"):
    encode_fixed_measure(
      Variable(("row",), [1], {"type": "uvw", "units": "m", "frame": reference}),
      "/part",
      "UVW",
      "MAIN::UVW",
    )


@pytest.mark.parametrize("reference", [None, 42, np.array("fk5")])
def test_missing_or_malformed_scalar_reference_is_encoding_error(reference):
  with pytest.raises(
    MeasureEncodingError, match="/part variable 'UVW' -> MAIN::UVW: invalid fixed frame"
  ) as exc:
    encode_fixed_measure(
      Variable(("row",), [1], {"type": "uvw", "units": "m", "frame": reference}),
      "/part",
      "UVW",
      "MAIN::UVW",
    )
  assert not isinstance(exc.value, MeasureReferenceColumnRequired)


def test_absent_reference_is_encoding_error():
  with pytest.raises(MeasureEncodingError, match="invalid fixed frame None") as exc:
    encode_fixed_measure(
      Variable(("row",), [1], {"type": "uvw", "units": "m"}),
      "/part",
      "UVW",
      "MAIN::UVW",
    )
  assert not isinstance(exc.value, MeasureReferenceColumnRequired)
