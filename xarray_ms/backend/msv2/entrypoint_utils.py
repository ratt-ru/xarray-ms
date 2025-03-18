import os.path
from typing import Dict, List
from uuid import uuid4

import pyarrow as pa
from arcae.lib.arrow_tables import Table

from xarray_ms.backend.msv2.structure import (
  DEFAULT_PARTITION_COLUMNS,
  MSv2StructureFactory,
)
from xarray_ms.multiton import Multiton

DEFAULT_SUBTABLES = [
  "ANTENNA",
  "DATA_DESCRIPTION",
  "FEED",
  "FIELD",
  "SPECTRAL_WINDOW",
  "STATE",
  "POLARIZATION",
  "OBSERVATION",
]


def subtable_factory(name: str) -> pa.Table:
  return Table.from_filename(
    name, ninstances=1, readonly=True, lockoptions="nolock"
  ).to_arrow()


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
  ms_factory: Multiton
  subtable_factories: Dict[str, Multiton]
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
    ms_factory: Multiton | None = None,
    subtable_factories: Dict[str, Multiton] | None = None,
    structure_factory: MSv2StructureFactory | None = None,
  ):
    if not os.path.exists(ms):
      raise FileNotFoundError(f"{ms} does not exist")

    self.ms = ms
    self.ninstances = ninstances
    self.auto_corrs = auto_corrs
    self.epoch = epoch or uuid4().hex[:8]
    self.partition_schema = partition_schema or DEFAULT_PARTITION_COLUMNS
    self.preferred_chunks = preferred_chunks or {}
    self.ms_factory = ms_factory or Multiton(
      Table.from_filename,
      self.ms,
      ninstances=self.ninstances,
      readonly=True,
      lockoptions="nolock",
    )
    self.subtable_factories = subtable_factories or {
      subtable: Multiton(subtable_factory, f"{ms}::{subtable}")
      for subtable in DEFAULT_SUBTABLES
    }
    self.structure_factory = structure_factory or MSv2StructureFactory(
      self.ms_factory,
      self.subtable_factories,
      self.partition_schema,
      self.epoch,
      auto_corrs=self.auto_corrs,
    )
