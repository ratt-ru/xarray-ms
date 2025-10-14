from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Collection, Dict, Sequence

import numpy as np

from xarray_ms.backend.msv2.table_utils import table_desc
from xarray_ms.casa_types import (
  ColumnDesc,
  DirectionMeasures,
  EpochMeasures,
  FrequencyMeasures,
  PositionMeasures,
  UvwMeasures,
)
from xarray_ms.errors import (
  InvalidMeasurementSet,
  MultipleQuantumUnits,
  PartitioningError,
)

if TYPE_CHECKING:
  import pyarrow as pa


@dataclass
class MeasuresData:
  msv4_type: str
  enum: Any


# {casa-measures-type: (msv4-measures-type, casa-measure-enum)}
CASA_MEASURES_MAP = {
  "uvw": MeasuresData("uvw", UvwMeasures),
  "epoch": MeasuresData("time", EpochMeasures),
  "frequency": MeasuresData("spectral_coord", FrequencyMeasures),
  "direction": MeasuresData("sky_coord", DirectionMeasures),
  "position": MeasuresData("location", PositionMeasures),
}


class ColumnInspectionMixin:
  """Adds common methods for extracting measure information from
  Column Descriptors"""

  def extract_quantum_unit(self, column_desc: ColumnDesc) -> str | None:
    """Attempts to extract a single quantum unit from the keywords
    of the column descriptor"""
    if (quantum_units := column_desc.keywords.get("QuantumUnits")) is None:
      return None

    if len(quantum_unit_set := set(quantum_units)) == 1:
      return next(iter(quantum_unit_set))

    if len(quantum_unit_set) == 0:
      return None

    raise MultipleQuantumUnits(
      f"Multiple quantum units {quantum_units} found "
      f"for column {getattr(self, 'column_name', '<unknown>')}. "
      f"MSv4 doesn't strictly cater for this unusual case, "
      f"but please log an issue for it"
    )

  def extract_msv2_type(self, column_desc: ColumnDesc) -> str | None:
    """Returns the msv2 measures type, derived from the 'type' keyword
    in the column descriptor keywords"""
    return column_desc.keywords.get("MEASINFO", {}).get("type")

  def extract_msv4_type(self, column_desc: ColumnDesc) -> str | None:
    """Returns the msv4 measures type, derived from the 'type' keyword
    in the column descriptor keywords"""
    if (casa_type := self.extract_msv2_type(column_desc)) is None:
      return None

    try:
      return CASA_MEASURES_MAP[casa_type].msv4_type
    except KeyError:
      raise NotImplementedError(f"{casa_type} CASA Measures")


class AbstractMeasuresAdapter(ABC):
  @property
  @abstractmethod
  def column_name(self) -> str:
    raise NotImplementedError

  @abstractmethod
  def msv2_frame(self) -> str | None:
    raise NotImplementedError

  @abstractmethod
  def msv2_type(self) -> str | None:
    raise NotImplementedError

  @abstractmethod
  def msv4_type(self) -> str | None:
    raise NotImplementedError

  @abstractmethod
  def quantum_unit(self) -> str | None:
    raise NotImplementedError


class SimpleMeasuresAdapter(AbstractMeasuresAdapter, ColumnInspectionMixin):
  _column_desc: ColumnDesc

  def __init__(self, column_desc: ColumnDesc):
    self._column_desc = column_desc

  @property
  def column_name(self) -> str:
    return self._column_desc.name

  def quantum_unit(self) -> str | None:
    return self.extract_quantum_unit(self._column_desc)

  def msv2_type(self):
    return self.extract_msv2_type(self._column_desc)

  def msv4_type(self) -> str | None:
    return self.extract_msv4_type(self._column_desc)

  def msv2_frame(self) -> str | None:
    return self._column_desc.keywords.get("MEASINFO", {}).get("Ref")


class ArrowTableMeasuresAdapter(AbstractMeasuresAdapter, ColumnInspectionMixin):
  _table: pa.Table
  _table_desc: Dict[str, Collection[str]]
  _column_desc: ColumnDesc

  def __init__(self, column_name: str, table: pa.Table):
    self._table = table
    self._table_desc = td = table_desc(table)
    self._column_desc = ColumnDesc.from_descriptor(column_name, td)

  @property
  def column_name(self) -> str:
    return self._column_desc.name

  def quantum_unit(self) -> str | None:
    return self.extract_quantum_unit(self._column_desc)

  def msv2_type(self):
    return self.extract_msv2_type(self._column_desc)

  def msv4_type(self) -> str | None:
    return self.extract_msv4_type(self._column_desc)

  def msv2_frame(self) -> str | None:
    measinfo = self._column_desc.keywords.get("MEASINFO", {})

    if not isinstance(measure_type := measinfo.get("type"), str):
      return None

    try:
      MeasuresEnum = CASA_MEASURES_MAP[measure_type].enum
    except KeyError:
      raise NotImplementedError(f"{measure_type} measures")

    # If there's a VarRefCol the reference frame can vary per row
    # Attempt to find a unique frame
    if isinstance(var_ref_col_name := measinfo.get("VarRefCol"), str):
      code_frame_map: Dict[int, str] = {}
      var_ref_col = self._table[var_ref_col_name].to_numpy()

      if not np.issubdtype(var_ref_col.dtype, np.integer):
        raise InvalidMeasurementSet(
          f"{var_ref_col_name} is not integral {var_ref_col.dtype}"
        )

      # The measures information may contain it's own reference types and codes
      if isinstance(tab_ref_types := measinfo.get("TabRefTypes"), Sequence):
        if isinstance(tab_ref_codes := measinfo.get("TabRefCodes"), Sequence):
          assert len(tab_ref_types) == len(tab_ref_codes)
        else:
          tab_ref_codes = [MeasuresEnum[t].value for t in tab_ref_types]

        code_frame_map = {c: t for c, t in zip(tab_ref_codes, tab_ref_types)}
      else:
        # Create a default code map for the measure type
        code_frame_map = {m.value: m.name for m in MeasuresEnum}

      # Try and find a unique frame
      if len(frames := {code_frame_map[c] for c in var_ref_col}) == 0:
        return None

      if len(frames) > 1:
        raise PartitioningError(
          f"Multiple measures frames {list(frames)} found in {self.column_name} "
          f"This can occur if data partitioning is too coarse. "
          f"Consider adding more partitioning columns to your partition_schema."
        )

      return next(iter(frames))

    # There's a single Reference frame
    elif isinstance(frame := measinfo.get("Ref"), str):
      return frame
    else:
      return None
