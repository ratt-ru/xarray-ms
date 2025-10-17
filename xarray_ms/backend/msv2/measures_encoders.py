from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from xarray import Variable
from xarray.coding.variables import (
  VariableCoder,
  unpack_for_decoding,
  unpack_for_encoding,
)

from xarray_ms.backend.msv2.array import MSv2Array
from xarray_ms.backend.msv2.measures_adapters import (
  AbstractMeasuresAdapter,
  MeasuresAdapterFactory,
)
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


class MSv2CoderFactory:
  """MSv2Coder Factory"""

  __slots__ = "_measures_adapter_factory"
  _measures_adapter_factory: MeasuresAdapterFactory

  def __init__(self, measures_adapter_factory: MeasuresAdapterFactory):
    self._measures_adapter_factory = measures_adapter_factory

  @staticmethod
  def from_table_desc(table_desc: Dict[str, Any]):
    """Constructs a factory from a table descriptor"""
    return MSv2CoderFactory(MeasuresAdapterFactory.from_table_desc(table_desc))

  @staticmethod
  def from_arrow_table(table: pa.Table):
    """Constructs a factory from an arrow table"""
    return MSv2CoderFactory(MeasuresAdapterFactory.from_arrow_table(table))

  def create(self, column_name: str) -> MSv2Coder:
    """Creates a coder for the given column"""
    measures_adapter = self._measures_adapter_factory.create(column_name)

    # Instantiate the appropriate coder object for a given measures type
    if (msv2_type := measures_adapter.msv2_type(on_missing="none")) is not None:
      if msv2_type == "epoch":
        return EpochCoder(measures_adapter)
      elif msv2_type == "frequency":
        return FrequencyCoder(measures_adapter)
      elif msv2_type == "uvw":
        return UvwCoder(measures_adapter)
      elif msv2_type == "position":
        return PositionCoder(measures_adapter)
      elif msv2_type == "direction":
        return DirectionCoder(measures_adapter)
      else:
        raise NotImplementedError(f"{msv2_type} measures")
    # Otherwise this may only be a quantity
    elif (unit := measures_adapter.quantum_unit(on_missing="none")) is not None:
      return SuppliedQuantityCoder(unit)
    else:
      return NoopCoder()


class MSv2Coder(VariableCoder):
  """Base class for encoding/decoding MSv2 variables"""


class NoopCoder(MSv2Coder):
  """This coder does not modify variables during encoding and decoding"""

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    return variable

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    return variable


class MeasuresCoder(MSv2Coder):
  """Base class for encoding/decoding Casa Measures
  using actual measures data"""

  __slots__ = "measures_adapter"
  measures_adapter: AbstractMeasuresAdapter

  def __init__(self, measures_adapter: AbstractMeasuresAdapter):
    self.measures_adapter = measures_adapter


class QuantityCoder(MeasuresCoder):
  """Encodes a quantum unit"""

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "units"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = "quantity"
    attrs["units"] = self.measures_adapter.quantum_unit("raise")
    return Variable(dims, data, attrs, encoding, fastpath=True)


class SuppliedAttributesCoder(MSv2Coder):
  """Adds and removes the supplied attributes during decoding and encoding"""

  __slots__ = "_attrs"
  _attrs: Dict[str, Any]

  def __init__(self, attrs: Dict[str, Any]):
    self._attrs = attrs

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs = {k: v for k, v in attrs.items() if k not in self._attrs}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs.update(self._attrs)
    return Variable(dims, data, attrs, encoding, fastpath=True)


class SuppliedQuantityCoder(SuppliedAttributesCoder):
  """Encodes a quantum unit supplied in the constructor"""

  def __init__(self, units: str):
    super().__init__({"type": "quantity", "units": units})


class VisiblityCoder(SuppliedQuantityCoder):
  """Visibility coder"""

  def __init__(self):
    super().__init__("Jy")


class DimensionlessCoder(SuppliedQuantityCoder):
  """Dimensionless quantity coder"""

  def __init__(self):
    super().__init__("dimensionless")


class UvwCoder(MeasuresCoder):
  """Encodes UVW coordinate measures"""

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "units", "frame"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = self.measures_adapter.msv4_type("raise")
    attrs["units"] = self.measures_adapter.quantum_unit("raise")
    if (msv2_frame := self.measures_adapter.msv2_frame("raise")) == "J2000":
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


class EpochCoder(MeasuresCoder):
  """Encode/Decode MJD UTC and TAI to unix UTC/TAI"""

  MJD_EPOCH: datetime = datetime(1858, 11, 17)
  UTC_EPOCH: datetime = datetime(1970, 1, 1)
  MJD_OFFSET_SECONDS: float = (UTC_EPOCH - MJD_EPOCH).total_seconds()

  @staticmethod
  def encode_array(data: npt.NDArray) -> npt.NDArray:
    """Converts from unix UTC/TAI to MJD UTC/TAI"""
    return data + EpochCoder.MJD_OFFSET_SECONDS

  @staticmethod
  def decode_array(data: npt.NDArray) -> npt.NDArray:
    """Converts from MJD UTC/TAI to unix UTC/TAI"""
    return data - EpochCoder.MJD_OFFSET_SECONDS

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)

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
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = self.measures_adapter.msv4_type()
    attrs["units"] = self.measures_adapter.quantum_unit("raise")
    attrs["format"] = "unix"
    if (msv2_frame := self.measures_adapter.msv2_frame("raise")) == "UTC":
      attrs["scale"] = "utc"
    elif msv2_frame == "TAI":
      attrs["scale"] = "tai"
    else:
      raise NotImplementedError(f"Epoch frame {msv2_frame}")

    if isinstance(data, MSv2Array):
      data.transform = EpochCoder.decode_array
    elif isinstance(data, np.ndarray):
      data = EpochCoder.decode_array(data)
    else:
      raise NotImplementedError(f"Decoding of {type(data)}")

    return Variable(dims, data, attrs, encoding, fastpath=True)


class DirectionCoder(MeasuresCoder):
  """Encodes direction measures.

  This encompasses celestial directions or, directions in the sky"""

  MSV2_TO_MSV4_FRAME = {
    "AZELGEO": "altaz",
    "ICRS": "icrs",
    "J2000": "fk5",
  }

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs = {k: v for k, v in attrs.items() if k not in {"frame", "units", "type"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = self.measures_adapter.msv4_type("raise")
    attrs["units"] = self.measures_adapter.quantum_unit("raise")
    attrs["frame"] = msv2_to_msv4_frame(
      self.MSV2_TO_MSV4_FRAME, self.measures_adapter.msv2_frame("raise")
    )
    return Variable(dims, data, attrs, encoding, fastpath=True)


class PositionCoder(MeasuresCoder):
  """Encodes position measures.

  This encompasses terrestrial locations or, locations on the ground"""

  MSV2_TO_MSV4_FRAME = {"ITRF": "ITRS"}

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "frame", "units"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = self.measures_adapter.msv4_type("raise")
    attrs["units"] = self.measures_adapter.quantum_unit("raise")
    attrs["frame"] = msv2_to_msv4_frame(
      self.MSV2_TO_MSV4_FRAME, self.measures_adapter.msv2_frame("raise")
    )
    return Variable(dims, data, attrs, encoding, fastpath=True)


class FrequencyCoder(MeasuresCoder):
  """Encodes frequency measures"""

  MSV2_TO_MSV4_FRAME = {
    "BARY": "BARY",
    "REST": "REST",
    "TOPO": "TOPO",
    "LSRK": "lsrk",
    "LSRD": "lsrd",
    "GEO": "gcrs",
  }

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs = {k: v for k, v in attrs.items() if k not in {"type", "observer", "units"}}
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = self.measures_adapter.msv4_type("raise")
    attrs["units"] = self.measures_adapter.quantum_unit("raise")
    attrs["observer"] = msv2_to_msv4_frame(
      self.MSV2_TO_MSV4_FRAME, self.measures_adapter.msv2_frame("raise")
    )
    return Variable(dims, data, attrs, encoding, fastpath=True)
