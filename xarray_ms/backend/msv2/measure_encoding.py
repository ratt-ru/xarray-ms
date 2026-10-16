"""Pure, fixed-reference MSv4 measure metadata for fresh MSv2 writes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from xarray import Variable

from xarray_ms.backend.msv2.measures_encoders import (
  DirectionCoder,
  EpochCoder,
  FrequencyCoder,
  PositionCoder,
  UvwCoder,
)
from xarray_ms.errors import MeasureEncodingError, MeasureReferenceColumnRequired

# These are the reader's mappings, inverted only after validating exact metadata.
_SUPPORTED = {
  "time": ("epoch", "s", "scale", EpochCoder.MSV2_TO_MSV4_FRAME),
  "spectral_coord": ("frequency", "Hz", "observer", FrequencyCoder.MSV2_TO_MSV4_FRAME),
  "sky_coord": ("direction", "rad", "frame", DirectionCoder.MSV2_TO_MSV4_FRAME),
  "location": ("position", "m", "frame", PositionCoder.MSV2_TO_MSV4_FRAME),
  "uvw": ("uvw", "m", "frame", UvwCoder.MSV2_TO_MSV4_FRAME),
}


@dataclass(frozen=True)
class FixedMeasureEncoding:
  """A descriptor for one source variable and one destination column."""

  path: str
  variable: str
  column: str
  msv2_type: str
  frame: str
  unit: str

  @property
  def keywords(self) -> Mapping[str, Any]:
    """Immutable CASA column keywords; freshly materialized for each caller."""
    return MappingProxyType(
      {
        "MEASINFO": MappingProxyType({"type": self.msv2_type, "Ref": self.frame}),
        "QuantumUnits": (self.unit,),
      }
    )

  def to_column_keywords(self) -> dict[str, Any]:
    """Fresh, plain CASA descriptor keywords for a table-creation caller."""
    return {
      "MEASINFO": {"type": self.msv2_type, "Ref": self.frame},
      "QuantumUnits": [self.unit],
    }


def encode_fixed_measure(
  variable: Variable, path: str, name: str, column: str
) -> FixedMeasureEncoding:
  """Validate MSv4 attrs without reading the variable's payload."""
  attrs = variable.attrs
  context = f"{path} variable {name!r} -> {column}"
  kind = attrs.get("type")
  if not isinstance(kind, str) or kind not in _SUPPORTED:
    raise MeasureEncodingError(f"{context}: unsupported measure type {kind!r}")
  msv2_type, unit, key, reader_frames = _SUPPORTED[kind]
  if kind == "time" and attrs.get("format") != "unix":
    raise MeasureEncodingError(f"{context}: time format must be 'unix'")
  if attrs.get("units") != unit:
    raise MeasureEncodingError(f"{context}: units must be {unit!r}")
  frame = attrs.get(key)
  if (isinstance(frame, Sequence) and not isinstance(frame, (str, bytes))) or (
    isinstance(frame, np.ndarray) and frame.ndim > 0
  ):
    raise MeasureReferenceColumnRequired(
      f"{context}: non-scalar {key} {frame!r}; MSv2 reference column required"
    )
  if not isinstance(frame, str):
    raise MeasureEncodingError(f"{context}: invalid fixed {key} {frame!r}")
  reverse = {msv4: msv2 for msv2, msv4 in reader_frames.items()}
  if frame not in reverse:
    raise MeasureEncodingError(f"{context}: unsupported fixed {key} {frame!r}")
  return FixedMeasureEncoding(path, name, column, msv2_type, reverse[frame], unit)


def check_shared_reference(
  previous: FixedMeasureEncoding, current: FixedMeasureEncoding
) -> None:
  """One MSv2 column descriptor cannot express two fixed references."""
  if (previous.msv2_type, previous.frame, previous.unit) != (
    current.msv2_type,
    current.frame,
    current.unit,
  ):
    raise MeasureReferenceColumnRequired(
      f"{previous.path} variable {previous.variable!r} and "
      f"{current.path} variable {current.variable!r} -> {current.column}: "
      "different fixed references require an MSv2 reference column"
    )
