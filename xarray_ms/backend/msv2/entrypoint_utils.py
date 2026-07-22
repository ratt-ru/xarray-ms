import os.path
from typing import Dict, List, Literal

import pyarrow as pa
from arcae.lib.arrow_tables import Table
from rarg_python_patterns.multiton import Multiton

from xarray_ms.backend.msv2.structure import (
  DEFAULT_PARTITION_COLUMNS,
  MainTableFactory,
  MSv2Structure,
  MSv2StructureFactory,
  SubtableFactory,
)

# These tables should always be present on an MS
DEFAULT_SUBTABLES = [
  "ANTENNA",
  "DATA_DESCRIPTION",
  "FEED",
  "FIELD",
  "FLAG_CMD",
  "HISTORY",
  "OBSERVATION",
  "POINTING",
  "POLARIZATION",
  "PROCESSOR",
  "SPECTRAL_WINDOW",
  "STATE",
]

EXTRA_SUBTABLES = [
  "SOURCE",
]


def subtable_factory(
  name: str, on_missing: Literal["raise", "empty_table"] = "empty_table"
) -> pa.Table:
  try:
    return Table.from_filename(name, ninstances=1, readonly=True).to_arrow()
  except pa.lib.ArrowInvalid as e:
    e_str = str(e)
    if "subtable" in e_str and "is invalid" in e_str:
      if on_missing == "raise":
        raise
      else:
        return pa.Table.from_pydict({})


class CommonStoreArgs:
  """Holds common xarray store arguments, but also provides
  a common mechanism for initialising the same default
  values from multiple locations"""

  ms: str
  ninstances: int
  auto_corrs: bool
  epoch: str
  partition_schema: List[str]
  preferred_chunks: Dict[str, int]
  ms_factory: MainTableFactory
  subtable_factories: Dict[str, SubtableFactory]
  structure_factory: MSv2StructureFactory

  __slots__ = (
    "ms",
    "ninstances",
    "auto_corrs",
    "epoch",
    "partition_schema",
    "preferred_chunks",
    "ms_factory",
    "subtable_factories",
    "structure_factory",
  )

  def __init__(
    self,
    ms: str,
    ninstances: int = 1,
    auto_corrs: bool = True,
    epoch: str | None = None,
    partition_schema: List[str] | None = None,
    preferred_chunks: Dict[str, int] | None = None,
    ms_factory: MainTableFactory | None = None,
    subtable_factories: Dict[str, SubtableFactory] | None = None,
    structure_factory: MSv2StructureFactory | None = None,
  ):
    if not os.path.exists(ms):
      raise FileNotFoundError(f"{ms} does not exist")

    self.ms = ms
    self.ninstances = ninstances
    self.auto_corrs = auto_corrs
    self.epoch = epoch if epoch is not None else ""
    self.partition_schema = partition_schema or DEFAULT_PARTITION_COLUMNS
    self.preferred_chunks = preferred_chunks or {}
    self.ms_factory = ms_factory or Multiton(
      Table.from_filename, self.ms, ninstances=self.ninstances, readonly=True
    )
    self.subtable_factories = subtable_factories or {
      subtable: Multiton(subtable_factory, f"{ms}::{subtable}")
      for subtable in (DEFAULT_SUBTABLES + EXTRA_SUBTABLES)
    }
    self.structure_factory = structure_factory or Multiton(
      MSv2Structure,
      self.ms_factory,
      self.subtable_factories,
      self.partition_schema,
      self.epoch,
      auto_corrs=self.auto_corrs,
    )
