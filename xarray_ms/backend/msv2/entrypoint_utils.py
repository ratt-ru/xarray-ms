import os.path
import warnings
from copy import deepcopy
from typing import (
  Any,
  Dict,
  FrozenSet,
  Iterable,
  List,
  Literal,
  Mapping,
  Tuple,
  TypeGuard,
)

import pyarrow as pa
from arcae.lib.arrow_tables import Table
from rarg_python_patterns.multiton import Multiton

from xarray_ms.backend.msv2.factories.correlated import (
  DATA_GROUP_KEYS,
  DATA_GROUP_VARIABLE_KEYS,
  STANDARD_DATA_GROUPS,
  DataGroupColumnT,
  DataGroupVariableT,
  data_group_renames,
)
from xarray_ms.backend.msv2.structure import (
  DEFAULT_PARTITION_COLUMNS,
  MainTableFactory,
  MSv2Structure,
  MSv2StructureFactory,
  SubtableFactory,
)
from xarray_ms.backend.msv2.subtable_index import (
  DEFAULT_SUBTABLE_NINSTANCES,
  SubtableIndex,
  SubtableIndexFactory,
  SubtableTableFactory,
)
from xarray_ms.casa_types import BaselineOrientation

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
  "PHASED_ARRAY",
  "ASDM_STATION",
]

#: Subtables keyed on ``(ANTENNA_ID, TIME)`` that are indexed,
#: rather than read in full, mapped to their partitioning
#: and (candidate) key columns
INDEXED_SUBTABLES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
  "POINTING": ((), ("ANTENNA_ID",)),
  "SYSCAL": (("SPECTRAL_WINDOW_ID",), ("ANTENNA_ID",)),
  "WEATHER": ((), ("NS_WX_STATION_ID", "ANTENNA_ID")),
}

#: Sub-datasets whose time axis can be associated
#: with the time axis of the correlated dataset
INTERPOLATABLE_DATASETS = ("pointing", "sys_cal", "weather")


def resolve_interpolate(interpolate: str | Iterable[str] | None) -> FrozenSet[str]:
  """Normalise the ``interpolate`` argument into a set of sub-dataset names"""
  if interpolate is None:
    return frozenset()

  names = frozenset([interpolate] if isinstance(interpolate, str) else interpolate)

  if invalid := names - set(INTERPOLATABLE_DATASETS):
    raise ValueError(
      f"Invalid interpolate values {sorted(invalid)}. "
      f"Valid values are {list(INTERPOLATABLE_DATASETS)}"
    )

  return names


def resolve_baseline_orientation(
  baseline_orientation: Mapping[
    str, Literal["invariant", "hermitian", "antisymmetric"] | BaselineOrientation
  ]
  | None,
) -> Dict[str, BaselineOrientation]:
  """Normalise the ``baseline_orientation`` argument into a
  ``{column: BaselineOrientation}`` dictionary"""
  if baseline_orientation is None:
    return {}

  if not isinstance(baseline_orientation, Mapping):
    raise TypeError(
      f"baseline_orientation {baseline_orientation} must be a "
      f"mapping of column names to baseline orientations"
    )

  valid = [o.value for o in BaselineOrientation]
  result = {}

  for column, orientation in baseline_orientation.items():
    try:
      result[column] = BaselineOrientation(orientation)
    except ValueError:
      raise ValueError(
        f"Invalid baseline_orientation {orientation!r} for column {column}. "
        f"Valid values are {valid}"
      ) from None

  return result


def _is_data_group_pair(value: Any) -> TypeGuard[DataGroupColumnT]:
  """Is this a ``(data_group_name, correlated_data_column)`` tuple?"""
  return (
    isinstance(value, tuple)
    and len(value) == 2
    and all(isinstance(v, str) for v in value)
  )


def resolve_data_groups(
  data_groups: str
  | DataGroupColumnT
  | Iterable[str | DataGroupColumnT]
  | Mapping[str, Mapping[str, DataGroupVariableT]]
  | None,
) -> Dict[str, Dict[str, DataGroupVariableT]]:
  """Normalise the ``data_groups`` argument into a
  ``{data_group_name: {data_group_key: variable}}`` dictionary.

  A string names a custom data group whose correlated data
  is the column of the same name.
  A ``(data_group_name, correlated_data_column)`` tuple names a data group
  whose correlated data is the given column.
  An iterable of these specifies multiple data groups.
  Standard data groups are always present if their column is present,
  but can be overridden with the tuple or mapping forms.
  In the mapping form, a variable is either a variable name,
  or a ``(source, destination)`` tuple renaming the non-standard
  ``source`` column to the ``destination`` variable.
  """
  if data_groups is None:
    return {}

  if isinstance(data_groups, str) or _is_data_group_pair(data_groups):
    data_groups = [data_groups]

  if isinstance(data_groups, Mapping):
    items = list(data_groups.items())
  elif isinstance(data_groups, Iterable):
    items = []

    for entry in data_groups:
      if isinstance(entry, str):
        if entry in STANDARD_DATA_GROUPS:
          raise ValueError(
            f"{entry!r} is a standard data group, which is present if "
            f"its column is present. It can be overridden with a "
            f"(data_group_name, correlated_data_column) tuple or a mapping, "
            f"e.g. data_groups={{{entry!r}: {{'weight': 'WEIGHT'}}}}"
          )
        items.append((entry, {"correlated_data": entry}))
      elif _is_data_group_pair(entry):
        name, column = entry
        items.append((name, {"correlated_data": column}))
      else:
        raise TypeError(
          f"data_groups entry {entry!r} must be a string or a "
          f"(data_group_name, correlated_data_column) tuple of strings"
        )

    names = [n for n, _ in items]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
      raise ValueError(f"Data groups {duplicates} are specified multiple times")
  else:
    raise TypeError(
      f"data_groups {data_groups} must be a string, "
      f"a (data_group_name, correlated_data_column) tuple, "
      f"an iterable of these, or a mapping of data group names "
      f"onto mappings of data group keys and variable names"
    )

  result = {}

  for name, group in items:
    if not isinstance(name, str) or not isinstance(group, Mapping):
      raise TypeError(
        f"data_groups entry {name!r}: {group!r} must map a data group name "
        f"onto a mapping of data group keys and variable names"
      )

    if invalid := set(group.keys()) - set(DATA_GROUP_KEYS):
      raise ValueError(
        f"Invalid keys {sorted(invalid)} in data group {name!r}. "
        f"Valid keys are {list(DATA_GROUP_KEYS)}"
      )

    resolved: Dict[str, DataGroupVariableT] = {}

    for key, value in group.items():
      if isinstance(value, str):
        resolved[key] = value
      elif (
        key in DATA_GROUP_VARIABLE_KEYS
        and isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, str) for v in value)
      ):
        resolved[key] = (value[0], value[1])
      elif key in DATA_GROUP_VARIABLE_KEYS:
        raise TypeError(
          f"Data group {name!r} {key}={value!r} must be a variable name "
          f"or a (source, destination) tuple of strings"
        )
      else:
        raise TypeError(f"Data group {name!r} {key}={value!r} must be a string")

    if name not in STANDARD_DATA_GROUPS and "correlated_data" not in group:
      raise ValueError(f"Data group {name!r} must specify its correlated_data")

    result[name] = resolved

  # Check for conflicting renames
  data_group_renames(result)
  return result


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
    with Table.from_filename(
      name, readonly=True, lockoptions="nolock", **driver_kwargs
    ) as table:
      return table.to_arrow()
  except pa.lib.ArrowInvalid as e:
    e_str = str(e)
    if "subtable" in e_str and "is invalid" in e_str:
      if on_missing == "raise":
        raise
      else:
        return pa.Table.from_pydict({})
    # re-raise any other ArrowInvalid errors
    raise


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
  subtable_table_factories: Dict[str, SubtableTableFactory]
  subtable_index_factories: Dict[str, SubtableIndexFactory]
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
    "subtable_table_factories",
    "subtable_index_factories",
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
      readonly=True,
      lockoptions="nolock",
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
    self.subtable_table_factories = {
      subtable: Multiton(
        Table.from_filename,
        f"{ms}::{subtable}",
        readonly=True,
        lockoptions="nolock",
        **{
          "ninstances": DEFAULT_SUBTABLE_NINSTANCES,
          **table_driver_kwargs(self.driver_kwargs, subtable),
        },
      )
      for subtable in INDEXED_SUBTABLES
    }
    self.subtable_index_factories = {
      subtable: Multiton(
        SubtableIndex,
        self.subtable_table_factories[subtable],
        partition_columns,
        key_columns,
      )
      for subtable, (partition_columns, key_columns) in INDEXED_SUBTABLES.items()
    }
    self.structure_factory = structure_factory or Multiton(
      MSv2Structure,
      self.ms_factory,
      self.subtable_factories,
      self.partition_schema,
      self.epoch,
      auto_corrs=self.auto_corrs,
    )
