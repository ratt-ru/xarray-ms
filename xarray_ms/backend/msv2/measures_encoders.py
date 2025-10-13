from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Dict

import numpy as np
import numpy.typing as npt
from xarray import Variable
from xarray.coding.variables import (
  VariableCoder,
  unpack_for_decoding,
  unpack_for_encoding,
)

from xarray_ms.backend.msv2.array import MSv2Array
from xarray_ms.backend.msv2.measures_adapters import AbstractMeasuresAdapter
from xarray_ms.errors import FrameConversionWarning

if TYPE_CHECKING:
  from xarray.coding.variables import T_Name


def msv2_to_msv4_frame(frame_map: Dict[str, str], frame: str) -> str:
  try:
    return frame_map[frame]
  except KeyError:
    warnings.warn(
      f"No conversion exists from MSv2 frame {frame}. "
      f"Available mappings: {frame_map}. "
      f"{frame} will be used as the MSv4 frame.",
      FrameConversionWarning,
    )
    return frame


class BaseCasaCoder(VariableCoder):
  _measures: AbstractMeasuresAdapter

  def __init__(self, measures_adapter: AbstractMeasuresAdapter):
    self._measures = measures_adapter


class TemporaryCoder(BaseCasaCoder):
  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(super().encode(variable, name))
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    return Variable(dims, data, attrs, encoding, fastpath=True)


class UvwCoder(BaseCasaCoder):
  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(super().encode(variable, name))
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "units", "frame"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    attrs["type"] = self._measures.msv4_type()
    if units := self._measures.quantum_unit():
      attrs["units"] = units

    if (msv2_frame := self._measures.msv2_frame()) == "J2000":
      attrs["frame"] = "fk5"
    elif msv2_frame == "APP":
      attrs["frame"] = msv2_frame
    elif msv2_frame == "ITRF":
      # NOTE: ITRF is not a valid UVW frame
      # but some CASA tasks will set it
      # This should be modified in post-processing
      # where an appropriate frame can be selected
      # from the field_and_source dataset
      attrs["frame"] = msv2_frame
    else:
      raise NotImplementedError(f"UVW frame {msv2_frame}")

    return Variable(dims, data, attrs, encoding, fastpath=True)


class EpochCoder(BaseCasaCoder):
  """Encode/Decode MJD UTC and TAI to unix UTC/TAI"""

  MJD_EPOCH: datetime = datetime(1858, 11, 17)
  UTC_EPOCH: datetime = datetime(1970, 1, 1)
  MJD_OFFSET_SECONDS: float = (UTC_EPOCH - MJD_EPOCH).total_seconds()

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(super().encode(variable, name))

    if isinstance(data, MSv2Array):
      data.transform = EpochCoder.encode_array
    elif isinstance(data, np.ndarray):
      data = EpochCoder.encode_array(data)
    else:
      raise NotImplementedError(f"Encoding of {type(data)}")

    attrs = {
      k: v for k, v in attrs.items() if k not in {"type", "format", "units", "scale"}
    }

    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    attrs["type"] = self._measures.msv4_type()
    if units := self._measures.quantum_unit():
      attrs["units"] = units
    attrs["format"] = "unix"
    if (msv2_frame := self._measures.msv2_frame()) == "UTC":
      attrs["scale"] = "utc"
    elif msv2_frame == "TAI":
      attrs["scale"] == "tai"
    else:
      raise NotImplementedError(f"Epoch frame {msv2_frame}")

    if isinstance(data, MSv2Array):
      data.transform = EpochCoder.decode_array
    elif isinstance(data, np.ndarray):
      data = EpochCoder.decode(data)
    else:
      raise NotImplementedError(f"Decoding of {type(data)}")

    return Variable(dims, data, attrs, encoding, fastpath=True)

  @staticmethod
  def encode_array(data: npt.NDArray) -> npt.NDArray:
    return data + EpochCoder.MJD_OFFSET_SECONDS

  @staticmethod
  def decode_array(data: npt.NDArray) -> npt.NDArray:
    return data - EpochCoder.MJD_OFFSET_SECONDS


class DirectionCoder(BaseCasaCoder):
  MSV2_TO_MSV4_FRAME = {
    "AZELGEO": "altaz",
    "ICRS": "icrs",
    "J2000": "fk5",
  }

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(super().encode(variable, name))
    attrs = {k: v for k, v in attrs.items() if k not in {"frame", "units", "type"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    attrs["type"] = self._measures.msv4_type()
    if units := self._measures.quantum_unit():
      attrs["units"] = units
    msv2_frame = self._measures.msv2_frame()
    assert msv2_frame is not None
    attrs["frame"] = msv2_to_msv4_frame(self.MSV2_TO_MSV4_FRAME, msv2_frame)
    return Variable(dims, data, attrs, encoding, fastpath=True)


class PositionCoder(BaseCasaCoder):
  MSV2_TO_MSV4_FRAME = {"ITRF": "ITRS"}

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(super().encode(variable, name))
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "frame", "units"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    attrs["type"] = self._measures.msv4_type()
    if units := self._measures.quantum_unit():
      attrs["units"] = units
    if msv2_frame := self._measures.msv2_frame():
      attrs["frame"] = msv2_to_msv4_frame(self.MSV2_TO_MSV4_FRAME, msv2_frame)
    return Variable(dims, data, attrs, encoding, fastpath=True)


class FrequencyCoder(BaseCasaCoder):
  MSV2_TO_MSV4_FRAME = {
    "BARY": "BARY",
    "REST": "REST",
    "TOPO": "TOPO",
    "LSRK": "lsrk",
    "LSRD": "lsrd",
    "GEO": "gcrs",
  }

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(super().encode(variable, name))
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "observer", "units"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    attrs["type"] = self._measures.msv4_type()
    if units := self._measures.quantum_unit():
      attrs["units"] = units
    if msv2_frame := self._measures.msv2_frame():
      attrs["observer"] = msv2_to_msv4_frame(self.MSV2_TO_MSV4_FRAME, msv2_frame)
    return Variable(dims, data, attrs, encoding, fastpath=True)


CASA_MEASURES_CODERS_MAP = {
  "uvw": UvwCoder,
  "epoch": EpochCoder,
  "frequency": FrequencyCoder,
  "direction": DirectionCoder,
  "position": PositionCoder,
}
