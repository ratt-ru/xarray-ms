import os.path
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Literal

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

#: Backend drivers currently supported by xarray-ms.
SUPPORTED_DRIVERS = ("arcae",)
#: Default keyword arguments passed to the backend driver.
DEFAULT_DRIVER_KWARGS: Dict[str, Any] = {"cache_size": 256}
#: Default number of instances opened for the main table.
DEFAULT_MAIN_NINSTANCES = 8
#: Reserved key in ``driver_kwargs`` holding per-table override dictionaries.
TABLE_OVERRIDES = "table_overrides"
#: Reserved key in ``driver_kwargs[TABLE_OVERRIDES]`` addressing the main table.
MAIN_TABLE = "MAIN"

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


def resolve_driver_kwargs(
  driver: str,
  driver_kwargs: Dict[str, Any] | None,
  ninstances: int | None = None,
) -> Dict[str, Any]:
  """Validate ``driver`` and materialise the effective ``driver_kwargs``.

  The deprecated ``ninstances`` argument, when supplied, is folded into the
  main table's per-table overrides and a :class:`DeprecationWarning` is raised.
  The returned dictionary is a deep copy that is safe to mutate downstream.
  """
  if driver not in SUPPORTED_DRIVERS:
    raise ValueError(
      f"Unsupported driver {driver!r}. Supported drivers: {SUPPORTED_DRIVERS}"
    )

  dk = deepcopy(DEFAULT_DRIVER_KWARGS if driver_kwargs is None else driver_kwargs)

  if ninstances is not None:
    warnings.warn(
      "The 'ninstances' argument is deprecated and will not be respected in "
      "a future release. Pass driver-specific options via "
      "driver_kwargs={'table_overrides': {'MAIN': {'ninstances': N}}} instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    # Historically ninstances only affected the main table. An explicit
    # driver_kwargs override for the main table takes precedence.
    dk.setdefault(TABLE_OVERRIDES, {}).setdefault(MAIN_TABLE, {}).setdefault(
      "ninstances", ninstances
    )

  return dk


def table_driver_kwargs(driver_kwargs: Dict[str, Any], name: str) -> Dict[str, Any]:
  """Resolve the effective driver kwargs for the table identified by ``name``.

  ``name`` is :data:`MAIN_TABLE` for the main table, or a subtable name (e.g.
  ``"POINTING"``). Flat kwargs apply to every table; per-table overrides under
  the reserved :data:`TABLE_OVERRIDES` key take precedence. The main table
  defaults to :data:`DEFAULT_MAIN_NINSTANCES` instances when unspecified.
  """
  base = {k: v for k, v in driver_kwargs.items() if k != TABLE_OVERRIDES}
  resolved = {**base, **driver_kwargs.get(TABLE_OVERRIDES, {}).get(name, {})}
  if name == MAIN_TABLE:
    resolved.setdefault("ninstances", DEFAULT_MAIN_NINSTANCES)
  return resolved


def subtable_factory(
  name: str,
  on_missing: Literal["raise", "empty_table"] = "empty_table",
  **driver_kwargs: Any,
) -> pa.Table:
  # Subtables are read once via to_arrow(); a single instance is sufficient.
  driver_kwargs.setdefault("ninstances", 1)
  try:
    return Table.from_filename(name, **driver_kwargs).to_arrow()
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
  driver: str
  driver_kwargs: Dict[str, Any]
  auto_corrs: bool
  epoch: str
  partition_schema: List[str]
  preferred_chunks: Dict[str, int]
  ms_factory: MainTableFactory
  subtable_factories: Dict[str, SubtableFactory]
  structure_factory: MSv2StructureFactory

  __slots__ = (
    "ms",
    "driver",
    "driver_kwargs",
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
    driver: str = "arcae",
    driver_kwargs: Dict[str, Any] | None = None,
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
    self.driver = driver
    self.driver_kwargs = (
      deepcopy(DEFAULT_DRIVER_KWARGS) if driver_kwargs is None else driver_kwargs
    )
    self.auto_corrs = auto_corrs
    self.epoch = epoch if epoch is not None else ""
    self.partition_schema = partition_schema or DEFAULT_PARTITION_COLUMNS
    self.preferred_chunks = preferred_chunks or {}
    self.ms_factory = ms_factory or Multiton(
      Table.from_filename,
      self.ms,
      **table_driver_kwargs(self.driver_kwargs, MAIN_TABLE),
    )
    self.subtable_factories = subtable_factories or {
      subtable: Multiton(
        subtable_factory,
        f"{ms}::{subtable}",
        **table_driver_kwargs(self.driver_kwargs, subtable),
      )
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
