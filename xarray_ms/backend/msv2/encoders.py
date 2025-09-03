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
import warnings
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
from xarray_ms.casa_types import (
  ColumnDesc,
  DirectionMeasures,
  FrequencyMeasures,
)
from xarray_ms.errors import (
  FrameConversionWarning,
  MissingMeasuresInfo,
  MissingQuantumUnits,
  MultipleQuantumUnits,
  PartitioningError,
)

if TYPE_CHECKING:
  from xarray.coding.variables import T_Name


class CasaCoder(VariableCoder):
  """Base class for CASA Measures Coders"""

  _column_desc: ColumnDesc
  _var_ref_cols: Dict[str, npt.NDArray]

  def __init__(self, column_desc: ColumnDesc):
    self._column_desc = column_desc
    self._var_ref_cols = {}

  def with_var_ref_cols(self, fn) -> CasaCoder:
    """Calls ``fn`` with the name of the desired VarRefCol,
    fn should return it."""
    try:
      measinfo = self.measinfo
    except MissingMeasuresInfo:
      return self

    if "VarRefCol" in measinfo:
      var_ref_col = measinfo["VarRefCol"]
      self._var_ref_cols[var_ref_col] = fn(var_ref_col)

    return self

  @property
  def column(self) -> str:
    """Returns the column"""
    return self._column_desc.name

  @property
  def column_desc(self) -> ColumnDesc:
    """Returns the column descriptor"""
    return self._column_desc

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
    """Returns the QuantumUnits keyword in the column descriptor"""
    try:
      return self.column_desc.keywords["QuantumUnits"]
    except KeyError:
      raise MissingQuantumUnits(f"No QuantumUnits found for {self.column}")

  @property
  def quantum_unit(self) -> str:
    """Attempts to return a single QuantumUnit associated with the column"""
    units = self.quantum_units
    set_units = set(units)
    if len(set_units) != 1:
      raise MultipleQuantumUnits(
        f"Unable to derive a single QuantumUnit for {self.column} "
        f"as multiple QuantumUnits {set_units} types were present. "
      )

    return next(iter(set_units))


class QuantityCoder(CasaCoder):
  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    attrs.pop("units", None)
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = "quantity"
    attrs["units"] = self.quantum_unit
    return Variable(dims, data, attrs, encoding, fastpath=True)


class VisibilityCoder(CasaCoder):
  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs.pop("units", None)
    attrs.pop("type", None)
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = "quantity"
    attrs["units"] = "Jy"
    return Variable(dims, data, attrs, encoding, fastpath=True)


class UVWCoder(CasaCoder):
  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs.pop("type", None)
    attrs.pop("units", None)
    attrs.pop("scale", None)
    attrs.pop("format", None)
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    measures = self.measinfo
    assert measures["type"] == "uvw"
    ref = measures["Ref"].upper()
    attrs["type"] = "uvw"
    attrs["units"] = self.quantum_unit

    if ref == "J2000":
      attrs["frame"] = "fk5"
    elif ref == "APP":
      attrs["frame"] = ref
    elif ref == "ITRF":
      # NOTE: ITRF is not a valid UVW frame
      # but some CASA tasks will set it
      # This should be modified in post-processing
      # where an appropriate frame can be selected
      # from the field_and_source dataset
      attrs["frame"] = ref
    else:
      raise NotImplementedError(f"UVW frame {ref}")

    return Variable(dims, data, attrs, encoding, fastpath=True)


class TimeCoder(CasaCoder):
  """Dispatches encoding functionality to sub-classes"""

  @classmethod
  def from_time_coder(cls, time_coder: TimeCoder) -> Self:
    return cls(time_coder.column_desc)

  def dispatched_coder(self) -> TimeCoder:
    measures = self.measinfo
    assert measures["type"] == "epoch"
    ref = measures["Ref"].upper()
    if ref == "UTC":
      cls = UTCCoder
    elif ref == "TAI":
      cls = TAICoder
    else:
      raise NotImplementedError(f"Epoch frame {measures['Ref']}")

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
      data.transform = UTCCoder.encode_array
    elif isinstance(data, np.ndarray):
      data = UTCCoder.encode_array(data)
    else:
      raise TypeError(f"Unknown data type {type(data)}")

    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    """Convert Modified Julian Date in seconds to UTC in seconds"""
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    attrs["type"] = "time"
    attrs["units"] = self.quantum_unit
    attrs["scale"] = "utc"
    attrs["format"] = "unix"

    if isinstance(data, MSv2Array):
      data.transform = UTCCoder.decode_array
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


class TAICoder(UTCCoder):
  """TAI is UTC + ~37 seconds"""

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    """Convert Modified Julian Date in seconds to TAI in seconds"""
    dims, data, attrs, encoding = unpack_for_decoding(super().decode(variable, name))
    attrs["scale"] = "tai"
    return Variable(dims, data, attrs, encoding, fastpath=True)


class SpectralCoordCoder(CasaCoder):
  """Encode Spectral Coordinate Measures"""

  CASA_TO_ASTROPY = {
    "BARY": "BARY",
    "REST": "REST",
    "TOPO": "TOPO",
    "LSRK": "lsrk",
    "LSRD": "lsrd",
    "GEO": "gcrs",
  }

  @staticmethod
  def casa_to_astropy(frame: str) -> str:
    try:
      return SpectralCoordCoder.CASA_TO_ASTROPY[frame]
    except KeyError:
      warnings.warn(
        f"No specific conversion exists from CASA frame {frame}. "
        f"{frame} will be used as is.",
        FrameConversionWarning,
      )
      return frame

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    measures = self.measinfo
    assert measures["type"] == "frequency"
    attrs["type"] = "spectral_coord"

    if "Ref" in measures:
      frame = self.casa_to_astropy(measures["Ref"])
    elif "VarRefCol" in measures:
      var_ref_col = measures["VarRefCol"]
      if var_ref_col not in self._var_ref_cols:
        raise RuntimeError(
          f"Use coder.with_var_ref_cols(...) to register {var_ref_col}"
        )
      if not (var_ref_data := np.unique(self._var_ref_cols[var_ref_col])).size == 1:
        raise PartitioningError(
          f"Multiple distinct measures codes were found in {var_ref_col}. "
          f"This usually stems from not partitioning the Measurement Set "
          f"finely enough."
        )

      measure_enum = FrequencyMeasures(var_ref_data.item())
      frame = self.casa_to_astropy(measure_enum.name)
    else:
      raise NotImplementedError(
        f"Decoding {measures['type']} measures in {self.column} "
        f"without a Ref or VarRefCol entry: {measures}"
      )

    attrs["observer"] = frame
    attrs["units"] = self.quantum_unit
    return Variable(dims, data, attrs, encoding, fastpath=True)


class DirectionCoder(CasaCoder):
  """Encode Direction Measures"""

  CASA_TO_ASTROPY = {
    "AZELGEO": "altaz",
    "ICRS": "icrs",
    "J2000": "fk5",
  }

  @staticmethod
  def casa_to_astropy(frame: str) -> str:
    try:
      return DirectionCoder.CASA_TO_ASTROPY[frame]
    except KeyError:
      warnings.warn(
        f"No specific conversion exists from CASA frame {frame}. "
        f"{frame} will be used as is.",
        FrameConversionWarning,
      )
      return frame

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    attrs.pop("frame")
    attrs.pop("units")
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    measures = self.measinfo
    assert measures["type"] == "direction"
    attrs["type"] = "sky_coord"
    if "Ref" in measures:
      frame = self.casa_to_astropy(measures["Ref"])
    elif "VarRefCol" in measures:
      var_ref_col = measures["VarRefCol"]
      if var_ref_col not in self._var_ref_cols:
        raise RuntimeError(
          f"Use coder.with_var_ref_cols(...) to register {var_ref_col}"
        )
      if not (var_ref_data := np.unique(self._var_ref_cols[var_ref_col])).size == 1:
        raise PartitioningError(
          f"Multiple distinct measures codes were found in {var_ref_col}. "
          f"This usually stems from not partitioning the Measurement Set "
          f"finely enough. Consider including FIELD_ID and maybe SOURCE_ID "
          f"in your partioning strategy"
        )

      measure_enum = DirectionMeasures(var_ref_data.item())
      frame = self.casa_to_astropy(measure_enum.name)
    else:
      raise NotImplementedError(
        f"Decoding {measures['type']} measures in {self.column} "
        f"without a Ref or VarRefCol entry: {measures}"
      )

    attrs["frame"] = frame
    attrs["units"] = self.quantum_unit
    return Variable(dims, data, attrs, encoding, fastpath=True)


class PositionCoder(CasaCoder):
  """Encode Direction Measures"""

  CASA_TO_ASTROPY = {
    "ITRF": "ITRS",
  }

  @staticmethod
  def casa_to_astropy(frame: str) -> str:
    try:
      return PositionCoder.CASA_TO_ASTROPY[frame]
    except KeyError:
      warnings.warn(
        f"No specific conversion exists from CASA frame {frame}. "
        f"{frame} will be used as is.",
        FrameConversionWarning,
      )
      return frame

  def encode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_encoding(variable)
    attrs.pop("type", None)
    attrs.pop("frame")
    attrs.pop("units")
    return Variable(dims, data, attrs, encoding, fastpath=True)

  def decode(self, variable: Variable, name: T_Name = None) -> Variable:
    dims, data, attrs, encoding = unpack_for_decoding(variable)
    measures = self.measinfo
    assert measures["type"] == "position"
    attrs["type"] = "location"
    if "Ref" in measures:
      frame = self.casa_to_astropy(measures["Ref"])
    elif "VarRefCol" in measures:
      var_ref_col = measures["VarRefCol"]
      if var_ref_col not in self._var_ref_cols:
        raise RuntimeError(
          f"Use coder.with_var_ref_cols(...) to register {var_ref_col}"
        )
      if not (var_ref_data := np.unique(self._var_ref_cols[var_ref_col])).size == 1:
        raise PartitioningError(
          f"Multiple distinct measures codes were found in {var_ref_col}. "
          f"This is unusual for position measures"
        )

      measure_enum = DirectionMeasures(var_ref_data.item())
      frame = self.casa_to_astropy(measure_enum.name)
    else:
      raise NotImplementedError(
        f"Decoding {measures['type']} measures in {self.column} "
        f"without a Ref or VarRefCol entry: {measures}"
      )

    attrs["frame"] = frame
    attrs["units"] = self.quantum_unit
    return Variable(dims, data, attrs, encoding, fastpath=True)
