"""
TODO(sjperkins): This implementation is incomplete and
will need refactoring.

In particular, the reference columns and codes for more complex Measures
are not yet handled.

This logic can be found here:

casacore/measures/TableMeasRefDesc/TableMeasRefDesc.cc
TableMeasRefDesc::TableMeasRefDesc
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
  if sys.version_info >= (3, 11):
    from typing import Self
  else:
    from typing_extensions import Self


import numpy as np
import numpy.testing as npt
from xarray import Variable
from xarray.coding.variables import (
  VariableCoder,
  unpack_for_decoding,
  unpack_for_encoding,
)

from xarray_ms.backend.msv2.array import MSv2Array
from xarray_ms.errors import MissingMeasuresInfo, MissingQuantumUnits

if TYPE_CHECKING:
  from xarray.coding.variables import T_Name

  from xarray_ms.casa_types import ColumnDesc


class CasaCoder(VariableCoder):
  """Base class for CASA Measures Coders"""

  _column: str
  _column_descs: Dict[str, ColumnDesc]

  def __init__(self, column: str, column_descs: Dict[str, ColumnDesc]):
    assert column in column_descs
    self._column = column
    self._column_descs = column_descs

  @property
  def column(self) -> str:
    """Returns the column"""
    return self._column

  @property
  def column_descs(self) -> Dict[str, ColumnDesc]:
    """Returns the column descriptors"""
    return self._column_descs

  @property
  def column_desc(self) -> ColumnDesc:
    """Returns the column descriptor"""
    try:
      return self._column_descs[self._column]
    except KeyError:
      raise KeyError(f"No Column Descriptor exist for {self.column}")

  @property
  def measinfo(self) -> Dict[str, Any]:
    """Returns the MEASINFO keyword in the column descriptor"""
    kw = self.column_desc.keywords
    try:
      return kw["MEASINFO"]
    except KeyError:
      raise MissingMeasuresInfo(f"No MEASINFO found for {self.column}")

  @property
  def quantum_units(self) -> List[str]:
    """REturns the QuantumUnits keyword in the column descriptor"""
    kw = self.column_desc.keywords
    try:
      return kw["QuantumUnits"]
    except KeyError:
      raise MissingQuantumUnits(f"No QuantumUnits found for {self.column}")


class QuantityCoder(CasaCoder):
  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    attrs.pop("units", None)
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = "quantity"
    attrs["units"] = self.quantum_units
    return Variable(dims, data, attrs, encoding, fastpath=True)


class TimeCoder(CasaCoder):
  """Dispatches encoding functionality to sub-classes"""

  @classmethod
  def from_time_coder(cls, time_coder: TimeCoder) -> Self:
    return cls(time_coder._column, time_coder._column_descs)

  def dispatched_coder(self) -> TimeCoder:
    measures = self.measinfo
    assert measures["type"] == "epoch"
    ref = measures["Ref"].upper()
    if ref == "UTC":
      cls = UTCCoder
    else:
      raise NotImplementedError(measures["Ref"])

    return cls.from_time_coder(self)

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    return self.dispatched_coder().encode(variable, name)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    return self.dispatched_coder().decode(variable, name)


class UTCCoder(TimeCoder):
  """Encode MJD UTC"""

  MJD_EPOCH: datetime = datetime(1858, 11, 17)
  UTC_EPOCH: datetime = datetime(1970, 1, 1)
  MJD_OFFSET_SECONDS: float = (UTC_EPOCH - MJD_EPOCH).total_seconds()

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    """Convert UTC in seconds to Modified Julian Date"""
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    attrs.pop("units", None)
    attrs.pop("scale", None)
    attrs.pop("format", None)

    if isinstance(data, MSv2Array):
      data.set_transform(UTCCoder.encode_array)
    elif isinstance(data, np.ndarray):
      data = UTCCoder.encode_array(data)
    else:
      raise TypeError(f"Unknown data type {type(data)}")

    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    """Convert Modified Julian Date in seconds to UTC in seconds"""
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = "time"
    attrs["units"] = self.quantum_units
    attrs["scale"] = "utc"
    attrs["format"] = "unix"

    if isinstance(data, MSv2Array):
      data.set_transform(UTCCoder.decode_array)
    elif isinstance(data, np.ndarray):
      data = UTCCoder.decode_array(data)
    else:
      raise TypeError(f"Unknown data type {type(data)}")

    return Variable(dims, data, attrs, encoding, fastpath=True)

  @staticmethod
  def encode_array(data: npt.NDArray) -> npt.NDArray:
    return data + UTCCoder.MJD_OFFSET_SECONDS

  @staticmethod
  def decode_array(data: npt.NDArray) -> npt.NDArray:
    return data - UTCCoder.MJD_OFFSET_SECONDS


class SpectralCoordCoder(CasaCoder):
  """Encode Measures Spectral Coordinates"""

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    measures = self.measinfo
    assert measures["type"] == "frequency"
    attrs["type"] = "spectral_coord"
    # TODO(sjperkins): topo is hard-coded here and will almost
    # certainly need extra work to support other frames
    attrs["frame"] = "topo"
    attrs["units"] = self.quantum_units
    return Variable(dims, data, attrs, encoding, fastpath=True)
