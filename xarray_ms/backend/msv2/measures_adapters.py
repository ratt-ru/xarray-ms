from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Sequence, overload

import numpy as np

from xarray_ms.backend.msv2.table_utils import extract_table_desc as extract_table_desc
from xarray_ms.casa_types import (
  ColumnDesc,
  DirectionMeasures,
  EpochMeasures,
  FrequencyMeasures,
  PositionMeasures,
  UvwMeasures,
)
from xarray_ms.errors import (
  ComplexMeasuremetSet,
  InvalidMeasurementSet,
  MissingMeasuresInfo,
  MissingQuantumUnits,
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


def raise_invalid_on_missing(on_missing: Literal["none", "raise"]):
  if on_missing not in {"none", "raise"}:
    raise ValueError(f"'on_missing' {on_missing} not in {{'none', 'raise'}}")


def raise_on_measinfo_indirection(column_name: str, measinfo: Dict[str, Any]):
  """Raises if the MEASINFO requires indirection to other columns in the
  table in order to fully resolve the Measures

  i.e. if it has a ``VarRefCol`` for defining a per-row frame or
  ``RefOff*`` keywords for defining offsets.
  """

  def check() -> bool:
    for k in measinfo:
      if "VarRefCol" in k:
        return True

      if k.startswith("RefOff"):
        return True

    return False

  if check():
    raise ComplexMeasuremetSet(
      f"The MEASINFO in column {column_name} {measinfo} "
      f"contains indirection in the form of `VarRefCol` "
      f"or `RefCol*` entries. "
      f"A more complex measures adapter is required to "
      f"handle this case"
    )


class MeasuresMixin:
  """Adds common methods for extracting measure information from
  Column Descriptors"""

  @overload
  def extract_measinfo(
    self, column_desc: ColumnDesc, on_missing: Literal["none"]
  ) -> Dict[str, Any] | None: ...

  @overload
  def extract_measinfo(
    self, column_desc: ColumnDesc, on_missing: Literal["raise"]
  ) -> Dict[str, Any]: ...

  def extract_measinfo(
    self, column_desc: ColumnDesc, on_missing: Literal["none", "raise"] = "none"
  ) -> Dict[str, Any] | None:
    """Extracts the MEASINFO keyword from the column descriptor, if present"""
    try:
      return column_desc.keywords["MEASINFO"]
    except KeyError:
      if on_missing == "none":
        return None
      elif on_missing == "raise":
        raise MissingMeasuresInfo(
          f"{column_desc.name} keywords does not contain MEASINFO"
        )
      raise_invalid_on_missing(on_missing)

  @overload
  def extract_quantum_unit(
    self, column_desc: ColumnDesc, on_missing: Literal["none"]
  ) -> str | None: ...

  @overload
  def extract_quantum_unit(
    self, column_desc: ColumnDesc, on_missing: Literal["raise"]
  ) -> str: ...

  def extract_quantum_unit(
    self, column_desc: ColumnDesc, on_missing: Literal["none", "raise"] = "none"
  ) -> str | None:
    """Attempts to extract a single quantum unit from the keywords
    of the column descriptor.

    if multiple QuantumUnits values are present then they must all be the same
    """
    try:
      quantum_units = column_desc.keywords["QuantumUnits"]
    except KeyError:
      if on_missing == "none":
        return None
      elif on_missing == "raise":
        raise MissingQuantumUnits(
          f"QuantumUnits is not present in {column_desc.name} keywords"
        )
      raise_invalid_on_missing(on_missing)

    if len(quantum_unit_set := set(quantum_units)) == 1:
      return next(iter(quantum_unit_set))

    if len(quantum_unit_set) == 0:
      if on_missing == "none":
        return None
      elif on_missing == "raise":
        raise MissingQuantumUnits(f"Empty QuantumUnits in {column_desc.name} keywords")
      raise_invalid_on_missing(on_missing)

    raise MultipleQuantumUnits(
      f"Multiple quantum units {list(quantum_unit_set)} found "
      f"for column {column_desc.name}. "
      f"MSv4 doesn't strictly cater for this unusual case, "
      f"but please log an issue for it"
    )

  @overload
  def extract_msv2_type(
    self, column_desc: ColumnDesc, on_missing: Literal["none"]
  ) -> str | None: ...

  @overload
  def extract_msv2_type(
    self, column_desc: ColumnDesc, on_missing: Literal["raise"]
  ) -> str: ...

  def extract_msv2_type(
    self, column_desc: ColumnDesc, on_missing: Literal["none", "raise"] = "none"
  ) -> str | None:
    """Returns the msv2 measures type, derived from the 'type' keyword
    in the column descriptor keywords"""
    try:
      if (
        measinfo := self.extract_measinfo(column_desc, on_missing=on_missing)
      ) is None:
        return None
    except MissingMeasuresInfo:
      if on_missing == "none":
        return None
      elif on_missing == "raise":
        raise
      raise_invalid_on_missing(on_missing)

    try:
      return typing.cast(str, measinfo["type"])
    except KeyError:
      if on_missing == "none":
        return None
      elif on_missing == "raise":
        raise MissingMeasuresInfo(
          f"'type' was not present in {column_desc.name} MEASINFO {measinfo}"
        )
      raise_invalid_on_missing(on_missing)

  @overload
  def extract_msv4_type(
    self, column_desc: ColumnDesc, on_missing: Literal["none"]
  ) -> str | None: ...

  @overload
  def extract_msv4_type(
    self, column_desc: ColumnDesc, on_missing: Literal["raise"]
  ) -> str: ...

  def extract_msv4_type(
    self, column_desc: ColumnDesc, on_missing: Literal["none", "raise"] = "none"
  ) -> str | None:
    """Returns the msv4 measures type, derived from the 'type' keyword
    in the column descriptor keywords"""
    if (
      casa_type := self.extract_msv2_type(column_desc, on_missing=on_missing)
    ) is None:
      return None

    try:
      return CASA_MEASURES_MAP[casa_type].msv4_type
    except KeyError:
      raise NotImplementedError(f"{casa_type} CASA Measures")


class AbstractMeasuresAdapter(ABC):
  """Abstract base class presenting a uniform interface
  for extracting measures information from an underlying data source

  Five methods should be implemented:

  - ``column_name()``: Returns the column for this measures
  - ``msv2_type()``: Returns the MSv2 measures type.
    For example, ``epoch``, ``frequency`` or ``location``.
  - ``msv2_frame()``: Returns the MSv2 measures frame associated with a measures type.
    For example ``UTC`` or ``TAI`` for a ``epoch`` measures.
  - ``msv4_frame()``: Returns the MSv4 measures type. For example
    ``time`` and ``spectral_coord`` if the MSv2 type is
    ``epoch`` and ``frequency`` respectively,
  - ``quantum_unit()``: Returns a single quantum unit associated with the measure.


  """

  @property
  @abstractmethod
  def column_name(self) -> str:
    """Returns the column name associated with this measures"""
    raise NotImplementedError

  @overload
  def msv2_frame(self, on_missing: Literal["none"]) -> str | None: ...

  @overload
  def msv2_frame(self, on_missing: Literal["raise"]) -> str: ...

  @abstractmethod
  def msv2_frame(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    """Returns the MSv2 Measures frame.

    This corresponds to ``MEASINFO['Ref']`` or possibly ``MEASINFO['VarRefCol']``"""
    raise NotImplementedError

  @abstractmethod
  def msv2_type(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    """Returns the MSv2 Measures type.

    This corresponds to ``MEASINFO['type']`` in the column descriptor keywords"""
    raise NotImplementedError

  @abstractmethod
  def msv4_type(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    """Return the MSv4 Measures type"""
    raise NotImplementedError

  @abstractmethod
  def quantum_unit(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    """Returns a single unit associated with this measure.

    This corresponds to ``MEASINFO['QuantumUnits']`` in the
    column descriptor keywords"""
    raise NotImplementedError


class ColumnDescMeasuresAdapter(AbstractMeasuresAdapter, MeasuresMixin):
  """Measures Adapter extracting information from a Column Descriptor only

  This adapter only extracts information from a MEASINFO that contains no indirection.
  This means that the Column Descriptor is sufficient to define measure completely.
  More concrently the frame should be constant per row
  (i.e. ``MEASINFO["Ref"]`` is defined)

  More generally, MEASINFO can contain indirection that
  refers to other columns in the table in cases where the frame,
  offsets and values vary per row.
  For example, ``MEASINFO["VarRefCol"]`` refers to a table column
  which defines the per-row frame codes.
  """

  _column_desc: ColumnDesc

  def __init__(self, column_desc: ColumnDesc):
    self._column_desc = column_desc

  @property
  def column_name(self) -> str:
    return self._column_desc.name

  def quantum_unit(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    return self.extract_quantum_unit(self._column_desc, on_missing=on_missing)

  def msv2_type(self, on_missing: Literal["none", "raise"] = "none"):
    return self.extract_msv2_type(self._column_desc, on_missing=on_missing)

  def msv4_type(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    return self.extract_msv4_type(self._column_desc, on_missing=on_missing)

  @overload
  def msv2_frame(self, on_missing: Literal["none"]) -> str | None: ...

  @overload
  def msv2_frame(self, on_missing: Literal["raise"]) -> str: ...

  def msv2_frame(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    if (
      measinfo := self.extract_measinfo(self._column_desc, on_missing=on_missing)
    ) is None:
      return None

    try:
      return typing.cast(str, measinfo["Ref"])
    except KeyError:
      raise_on_measinfo_indirection(self.column_name, measinfo)
      if on_missing == "none":
        return None
      elif on_missing == "raise":
        raise MissingMeasuresInfo(
          f"'Ref' was not present in {self.column_name} MEASINFO {measinfo}"
        )
      raise_invalid_on_missing(on_missing)


class ArrowTableMeasuresAdapter(ColumnDescMeasuresAdapter, MeasuresMixin):
  """Measures adapter extracting measures information from an Arrow table.

  This adapter can handle indirection in the MEASINFO of a column"""

  _table: pa.Table
  _table_desc: Dict[str, Any]

  def __init__(
    self,
    column_name: str,
    table: pa.Table,
    table_desc: Dict[str, Any] | None = None,
  ):
    self._table = table
    self._table_desc = extract_table_desc(table) if table_desc is None else table_desc
    self._column_desc = ColumnDesc.from_descriptor(column_name, self._table_desc)
    super().__init__(self._column_desc)

  @overload
  def msv2_frame(self, on_missing: Literal["none"]) -> str | None: ...

  @overload
  def msv2_frame(self, on_missing: Literal["raise"]) -> str: ...

  def msv2_frame(self, on_missing: Literal["none", "raise"] = "none") -> str | None:
    if (
      measinfo := self.extract_measinfo(self._column_desc, on_missing=on_missing)
    ) is None:
      return None

    if (
      measure_type := self.extract_msv2_type(self._column_desc, on_missing=on_missing)
    ) is None:
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

    # Otherwise attempt to defer to "Ref" in the column descriptor
    return super().msv2_frame(on_missing=on_missing)


class MeasuresAdapterFactory:
  _table: pa.Table | None
  _table_desc: Dict[str, Any] | None

  def __init__(
    self,
    table: pa.Table | None = None,
    table_desc: Dict[str, Any] | None = None,
  ):
    self._table = table
    self._table_desc = table_desc

  @staticmethod
  def from_arrow_table(table: pa.Table):
    return MeasuresAdapterFactory(table=table, table_desc=extract_table_desc(table))

  @staticmethod
  def from_table_desc(table_desc: Dict[str, Any]):
    return MeasuresAdapterFactory(table_desc=table_desc)

  def create(self, column_name: str) -> AbstractMeasuresAdapter:
    if self._table is not None:
      return ArrowTableMeasuresAdapter(column_name, self._table, self._table_desc)
    elif self._table_desc is not None:
      return ColumnDescMeasuresAdapter(
        ColumnDesc.from_descriptor(column_name, self._table_desc)
      )
    else:
      raise ValueError(
        "MeasuresAdapterFactory was not configured with a column or table source"
      )
