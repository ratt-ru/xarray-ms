from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import logging
import multiprocessing as mp
import os.path
from collections import defaultdict
from functools import partial
from numbers import Integral
from typing import (
  Any,
  ClassVar,
  Dict,
  Iterator,
  List,
  Mapping,
  Sequence,
  Set,
  Tuple,
)

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from arcae.lib.arrow_tables import Table, merge_np_partitions
from cacheout import Cache
from xarray.core.utils import FrozenDict

from xarray_ms.backend.msv2.table_factory import TableFactory
from xarray_ms.casa_types import ColumnDesc, FrequencyMeasures, Polarisations
from xarray_ms.errors import InvalidMeasurementSet, InvalidPartitionKey

logger = logging.getLogger(__name__)


def nr_of_baselines(na: int, auto_corrs: bool = True) -> int:
  """Returns the number of baselines, given the number of antenna

  Args:
    na: Number of antenna
    auto_corrs: Include auto_correlations

  Returns:
    The number of baselines
  """
  return na * (na + (1 if auto_corrs else -1)) // 2


def baseline_id(
  ant1: npt.NDArray[np.integer],
  ant2: npt.NDArray[np.integer],
  na: int,
  auto_corrs: bool = True,
) -> npt.NDArray[np.int64]:
  """Generates a baseline identity from antenna pairs

  Args:
    ant1: First Antenna ID.
    ant2: Second Antenna ID.
    na: Total number of antenna.
    auto_corrs: Include auto correlations.

  Returns:
    An array of baseline IDs.
  """
  # TODO: Not depending on auto_corrs works
  # because it's included in the expression below
  # Refactor this
  nbl = nr_of_baselines(na, auto_corrs=False)

  # if auto_corrs:
  #   return nbl - ant1 + ant2 + na - (na - ant1) * (na - ant1 + 1) // 2
  # else:
  #   return nbl - ant1 + ant2 - 1 - (na - ant1) * (na - ant1 - 1) // 2

  if not (ant1.shape == ant1.shape and ant1.dtype == ant2.dtype):
    raise ValueError("ant1 and ant2 differ in shape or dtype")

  result = np.full_like(ant1, nbl, dtype=np.int64)
  result[:] -= ant1
  result[:] += ant2
  result[:] += na if auto_corrs else -1

  terms = np.empty((2, ant1.shape[0]), ant1.dtype)
  terms[0, :] = na
  terms[0, :] -= ant1

  terms[1, :] = terms[0, :]
  terms[1, :] += 1 if auto_corrs else -1

  terms[0, :] *= terms[1, :]
  terms[0, :] //= 2

  result[:] -= terms[0, :]
  return result


def is_partition_key(key: PartitionKeyT) -> bool:
  return (
    isinstance(key, tuple)
    and len(key) == 2
    and isinstance(key[0], str)
    and isinstance(key[1], int)
  )


def partition_args(data: npt.NDArray, chunk: int) -> List[npt.NDArray]:
  return [data[i : i + chunk] for i in range(0, len(data), chunk)]


DEFAULT_PARTITION_COLUMNS: List[str] = [
  "DATA_DESC_ID",
  "OBS_MODE",
  "OBSERVATION_ID",
]  #: Default Partitioning Column Schema


SHORT_TO_LONG_PARTITION_COLUMNS: Dict[str, str] = {
  "D": "DATA_DESC_ID",
  "F": "FIELD_ID",
  "O": "OBSERVATION_ID",
  "P": "PROCESSOR_ID",
  "S": "SOURCE_ID",
  "SI": "STATE_ID",
  "SN": "SCAN_NUMBER",
  "OM": "OBS_MODE",
  "SSN": "SUB_SCAN_NUMBER",
}

# MAIN table columns
VALID_MAIN_PARTITION_COLUMNS: List[str] = [
  "DATA_DESC_ID",
  "OBSERVATION_ID",
  "FIELD_ID",
  "SCAN_NUMBER",
  "STATE_ID",
]

VALID_FIELD_PARTITION_COLUMNS: List[str] = ["SOURCE_ID"]

# STATE table columns
VALID_STATE_PARTITION_COLUMNS: List[str] = [
  "OBS_MODE",
  "SUB_SCAN_NUMBER",
]

#: Valid partitioning columns
VALID_PARTITION_COLUMNS: List[str] = (
  VALID_MAIN_PARTITION_COLUMNS
  + VALID_FIELD_PARTITION_COLUMNS
  + VALID_STATE_PARTITION_COLUMNS
)

SORT_COLUMNS: List[str] = ["TIME", "ANTENNA1", "ANTENNA2"]


@dataclasses.dataclass
class PartitionData:
  """Dataclass describing data unique to a partition"""

  time: npt.NDArray[np.float64]
  interval: npt.NDArray[np.float64]
  chan_freq: npt.NDArray[np.float64]
  chan_width: npt.NDArray[np.float64]
  corr_type: npt.NDArray[np.int32]
  field_names: List[str]
  line_names: List[str]
  source_names: List[str]
  intents: List[str]
  spw_name: str
  spw_freq_group_name: str
  spw_ref_freq: float
  spw_frame: str
  scan_numbers: List[int]
  sub_scan_numbers: List[int]

  row_map: npt.NDArray[np.int64]


PartitionKeyT = Tuple[Tuple[str, int | str], ...]


class TablePartitioner:
  """Partitions and sorts MSv2 indexing columns"""

  _partitionby: List[str]
  _sortby: List[str]
  _other: List[str]

  def __init__(
    self, partitionby: Sequence[str], sortby: Sequence[str], other: Sequence[str]
  ):
    self._partitionby = list(partitionby)
    self._sortby = list(sortby)
    self._other = list(other)

  def partition(
    self, index: pa.Table, pool: cf.ThreadPoolExecutor
  ) -> Dict[PartitionKeyT, Dict[str, npt.NDArray]]:
    other = set(self._other)

    try:
      other.remove("row")
      index = index.append_column(
        "row", pa.array(np.arange(len(index), dtype=np.int64))
      )
    except KeyError:
      pass

    nrow = len(index)
    nworkers = pool._max_workers
    chunk = (nrow + nworkers - 1) // nworkers

    # Order columns by
    #
    # 1. Partitioning columns
    # 2. Sorting columns
    # 3. Others (such as row and INTERVAL)
    # 4. Remaining columns
    #
    # 4 is needed for the merge_np_partitions to work
    ordered_columns = self._partitionby + self._sortby + self._other
    ordered_columns += list(set(index.column_names) - set(ordered_columns))

    # Create a dictionary out of the pyarrow table
    table_dict = {k: index[k].to_numpy() for k in ordered_columns}
    # Partition the data over the workers in the pool
    partitions = [
      {k: v[s : s + chunk] for k, v in table_dict.items()}
      for s in range(0, nrow, chunk)
    ]

    # Sort each partition in parallel
    def sort_partition(p):
      sort_arrays = tuple(p[k] for k in reversed(ordered_columns))
      indices = np.lexsort(sort_arrays)
      return {k: v[indices] for k, v in p.items()}

    partitions = list(pool.map(sort_partition, partitions))
    # Merge partitions
    merged = merge_np_partitions(partitions)

    # Find the edges of the group partitions in parallel by
    # partitioning the sorted merged values into chunks, including
    # the starting value of the next chunk.
    starts = list(range(0, nrow, chunk))
    group_values = [
      {k: v[s : s + chunk + 1] for k, v in merged.items() if k in self._partitionby}
      for s in starts
    ]
    assert len(starts) == len(group_values)

    # Find the group start and end points in parallel by finding edges
    def find_edges(p, s):
      diffs = [np.diff(p[v]) > 0 for v in self._partitionby]
      return np.where(np.logical_or.reduce(diffs))[0] + s + 1

    edges = list(pool.map(find_edges, group_values, starts))
    group_offsets = np.concatenate([[0]] + edges + [[nrow]])

    # Create the grouped partitions
    groups: Dict[PartitionKeyT, Dict[str, npt.NDArray]] = {}

    for start, end in zip(group_offsets[:-1], group_offsets[1:]):
      key = tuple(sorted((k, merged[k][start].item()) for k in self._partitionby))
      groups[key] = {k: v[start:end] for k, v in merged.items()}

    return groups


def on_get_keep_alive(key, value, exists):
  """Reinsert to refresh the TTL for the entry on get operations"""
  if exists:
    MSv2StructureFactory._STRUCTURE_CACHE._set(key, value)


class MSv2StructureFactory:
  """Hashable, callable and picklable factory class
  for creating and caching an MSv2Structure"""

  _ms_factory: TableFactory
  _partition_schema: List[str]
  _epoch: str
  _auto_corrs: bool
  _STRUCTURE_CACHE: ClassVar[Cache] = Cache(
    maxsize=100, ttl=60, on_get=on_get_keep_alive
  )

  def __init__(
    self,
    ms: TableFactory,
    partition_schema: List[str],
    epoch: str,
    auto_corrs: bool = True,
  ):
    self._ms_factory = ms
    self._partition_schema = partition_schema
    self._epoch = epoch
    self._auto_corrs = auto_corrs

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, MSv2StructureFactory):
      return NotImplemented

    return (
      self._ms_factory == other._ms_factory
      and self._partition_schema == other._partition_schema
      and self._epoch == other._epoch
      and self._auto_corrs == other._auto_corrs
    )

  def __hash__(self):
    return hash(
      (self._ms_factory, tuple(self._partition_schema), self._epoch, self._auto_corrs)
    )

  def __reduce__(self):
    return (
      MSv2StructureFactory,
      (self._ms_factory, self._partition_schema, self._epoch, self._auto_corrs),
    )

  def __call__(self, *args, **kw) -> MSv2Structure:
    assert not args and not kw
    return self._STRUCTURE_CACHE.get(
      self,
      lambda self: MSv2Structure(
        self._ms_factory, self._partition_schema, self._auto_corrs
      ),
    )


class MSv2Structure(Mapping):
  """Holds structural information about an MSv2 dataset"""

  _ms_factory: TableFactory
  _auto_corrs: bool
  _partition_columns: List[str]
  _partitions: Mapping[PartitionKeyT, PartitionData]
  _column_descs: Dict[str, Dict[str, ColumnDesc]]
  _ant: pa.Table
  _ddid: pa.Table
  _feed: pa.Table
  _spw: pa.Table
  _pol: pa.Table
  _field: pa.Table
  _state: pa.Table

  def __getitem__(self, key: PartitionKeyT) -> PartitionData:
    return self._partitions[key]

  def __iter__(self) -> Iterator[Any]:
    return iter(self._partitions)

  def __len__(self) -> int:
    return len(self._partitions)

  @property
  def auto_corrs(self) -> bool:
    """Are auto-correlations included"""
    return self._auto_corrs

  @property
  def column_descs(self) -> Dict[str, Dict[str, ColumnDesc]]:
    """Return the per-table column descriptors.
    Outer key is "MAIN", "SPECTRAL_WINDOW", inner key
    is the column name such as "FLAG" and "WEIGHT_SPECTRUM"
    """
    return self._column_descs

  @property
  def na(self) -> int:
    """Number of antenna in the Measurement Set"""
    return len(self._ant)

  @property
  def antenna_pairs(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Return default per-baseline antenna pairs"""
    return tuple(map(np.int32, np.triu_indices(self.na, 0 if self.auto_corrs else 1)))

  @property
  def nbl(self) -> int:
    """Number of baselines in the Measurement Set"""
    return nr_of_baselines(self.na, self.auto_corrs)

  @staticmethod
  def parse_partition_key(key: str) -> PartitionKeyT:
    """Parses a "DATA_DESC_ID=0, FIELD_ID_1,..." style string
    into a tuple of (column, id) tuples"""
    pairs = []

    for component in [k.strip() for k in key.split(",")]:
      try:
        column, value = component.split("=")
      except ValueError as e:
        raise InvalidPartitionKey(f"Invalid partition key {key!r}") from e
      else:
        pairs.append((column, int(value)))

    return tuple(sorted(pairs))

  def resolve_key(self, key: str | PartitionKeyT | None) -> List[PartitionKeyT]:
    """Given a possibly incomplete key, resolves to a list of matching partition keys"""
    if not key:
      return list(self.keys())

    if isinstance(key, str):
      key = self.parse_partition_key(key)

    column_set = set(self._partition_columns) | set(self._subtable_partition_columns)

    # Check that the key columns and values are valid
    new_key: List[Tuple[str, int | str]] = []
    for column, value in key:
      column = column.upper()
      column = SHORT_TO_LONG_PARTITION_COLUMNS.get(column, column)
      if column not in column_set:
        raise InvalidPartitionKey(
          f"{column} is not valid a valid partition column "
          f"{self._partition_columns}"
        )
      if not isinstance(value, (str, Integral)):
        raise InvalidPartitionKey(
          f"{value} is an invalid partition key value. "
          f"Should be an integer or (rarely) a string"
        )
      new_key.append((column, value))

    key_set = set(new_key)
    partition_keys = list(self._partitions.keys())
    partition_key_sets = list(map(set, partition_keys))
    matches = []

    for i, partition_key_set in enumerate(partition_key_sets):
      if key_set.issubset(partition_key_set):
        matches.append(partition_keys[i])

    return matches

  def maybe_get_source_id(
    self, pool: cf.Executor, ncpus: int, field_id: npt.NDArray[np.int32]
  ) -> npt.NDArray[np.int32] | None:
    """Constructs a SOURCE_ID array from MAIN.FIELD_ID
    broadcast against FIELD.SOURCE_ID"""
    if hasattr(self, "_field") and hasattr(self, "_source") and len(self._source) != 0:
      field_source_id = self._field["SOURCE_ID"].to_numpy()
      source_id = np.empty_like(field_id)
      chunk = (len(source_id) + ncpus - 1) // ncpus

      def par_copy(source, field):
        source[:] = field_source_id[field]

      pool.map(
        par_copy, partition_args(source_id, chunk), partition_args(field_id, chunk)
      )

      return source_id

    return None

  def partition_columns_from_schema(
    self, partition_schema: List[str]
  ) -> Tuple[List[str], List[str]]:
    """Given a partitioning schema, produce

    1. a list of partitioning columns of the MAIN table
    2. a list of subtable partitioning columns.
       i.e. `FIELD.SOURCE_ID` and `STATE.OBS_MODE`
    """
    schema: Set[str] = set(partition_schema)

    # Always partition by these columns
    columns: Dict[str, None] = {c: None for c in DEFAULT_PARTITION_COLUMNS}

    for column in VALID_MAIN_PARTITION_COLUMNS:
      if column in schema and column not in columns:
        columns[column] = None

    # Add FIELD_ID if partitioning by FIELD columns
    if (
      len(set(VALID_FIELD_PARTITION_COLUMNS).intersection(schema)) > 0
      and "FIELD_ID" not in columns
    ):
      columns["FIELD_ID"] = None

    # Add STATE_ID if partitioning by STATE columns
    # and the STATE table is present and populated
    if (
      hasattr(self, "_state")
      and len(self._state) != 0
      and len(set(VALID_STATE_PARTITION_COLUMNS).intersection(schema)) > 0
      and "STATE_ID" not in columns
    ):
      columns["STATE_ID"] = None

    subtable_columns: List[str] = []
    MAIN_COLUMN_SET: Set[str] = set(VALID_MAIN_PARTITION_COLUMNS)

    for column in list(columns.keys()):
      if column not in MAIN_COLUMN_SET:
        subtable_columns.append(column)
        del columns[column]

    return list(columns.keys()), subtable_columns

  @staticmethod
  def par_unique(pool, ncpus, data, return_inverse=False):
    """Parallel unique function using the associated threadpool"""
    chunk_size = (len(data) + ncpus - 1) // ncpus
    data_chunks = partition_args(data, chunk_size)
    if return_inverse:
      unique_fn = partial(np.unique, return_inverse=True)
      udatas, indices = zip(*pool.map(unique_fn, data_chunks))
      udata = np.unique(np.concatenate(udatas))

      def inv_fn(data, idx):
        return np.searchsorted(udata, data)[idx]

      def par_assign(target, data):
        target[:] = data

      data_ids = pool.map(inv_fn, udatas, indices)
      inverse = np.empty(len(data), dtype=indices[0].dtype)
      pool.map(par_assign, partition_args(inverse, chunk_size), data_ids)

      return udata, inverse
    else:
      udata = list(pool.map(np.unique, data_chunks))
      return np.unique(np.concatenate(udata))

  @staticmethod
  def read_subtables(
    table_name: str,
  ) -> Tuple[Dict[str, pa.Table], Dict[str, Dict[str, ColumnDesc]]]:
    subtables: Dict[str, pa.Table] = {}
    coldescs: Dict[str, Dict[str, ColumnDesc]] = defaultdict(dict)
    SUBTABLES: List[Tuple[str, str, bool]] = [
      ("ANTENNA", "_ant", True),
      ("DATA_DESCRIPTION", "_ddid", True),
      ("SPECTRAL_WINDOW", "_spw", True),
      ("POLARIZATION", "_pol", True),
      ("FEED", "_feed", True),
      ("FIELD", "_field", False),
      ("SOURCE", "_source", False),
      ("STATE", "_state", False),
    ]

    for subtable_name, attribute, required in SUBTABLES:
      subtable_path = os.path.join(table_name, subtable_name)
      if not os.path.exists(subtable_path):
        if required:
          raise FileNotFoundError(f"Required subtable {subtable_name} does not exist")
        else:
          continue

      with Table.from_filename(
        f"{table_name}::{subtable_name}", lockoptions="nolock"
      ) as subtable:
        subtables[attribute] = subtable.to_arrow()
        coldescs[subtable_path] = FrozenDict(
          {
            c: ColumnDesc.from_descriptor(c, subtable.tabledesc())
            for c in subtable.columns()
          }
        )

    return subtables, coldescs

  def partition_data_factory(
    self,
    name: str,
    auto_corrs: bool,
    key: PartitionKeyT,
    value: Dict[str, npt.NDArray],
    pool: cf.Executor,
    ncpus: int,
  ) -> Tuple[PartitionKeyT, PartitionData]:
    """Generate an updated partition key and
    `PartitionData` object.

    The partition key is updated with subtable partitioning keys
    (primarily `FIELD.SOURCE_ID` and `STATE.OBSMODE`).

    The `PartitionData` object represents a summary of the
    partition data passed in via arguments.
    """
    time = value["TIME"]
    interval = value["INTERVAL"]
    ant1 = value["ANTENNA1"]
    ant2 = value["ANTENNA2"]
    rows = value["row"]

    # Compute the unique times and their inverse index
    utime, time_ids = self.par_unique(pool, ncpus, time, return_inverse=True)

    try:
      ddid = next(int(i) for (c, i) in key if c == "DATA_DESC_ID")
    except StopIteration:
      raise KeyError(f"DATA_DESC_ID must be present in partition key {key}")

    if ddid >= len(self._ddid):
      raise InvalidMeasurementSet(
        f"DATA_DESC_ID {ddid} does not exist in {name}::DATA_DESCRIPTION"
      )

    # Extract field and source names
    field_id = value.get("FIELD_ID")
    field_names: List[str] = []
    source_names: List[str] = []
    line_names: List[str] = []

    if field_id is not None and len(self._field) > 0:
      ufield_ids = self.par_unique(pool, ncpus, field_id)
      fields = self._field.take(ufield_ids)
      field_names = fields["NAME"].to_pylist()
      source_ids = fields["SOURCE_ID"].to_pylist()

      if "SOURCE_ID" in self._subtable_partition_columns:
        assert len(source_ids) == 1
        key += (("SOURCE_ID", source_ids[0]),)

      # Select out SOURCES if we have the table
      if hasattr(self, "_source") and len(self._source) > 0:
        sources = self._source.take(source_ids)
        source_names = sources["NAME"].to_pylist()
        if "TRANSITION" in self._source.column_names:
          line_names = sources["TRANSITION"].to_pylist()

    # Extract scan numbers
    scan_number = value.get("SCAN_NUMBER")
    scan_numbers: List[int] = []

    if scan_number is not None:
      scan_numbers = self.par_unique(pool, ncpus, scan_number).tolist()

    # Extract intents and sub scan numbers
    state_id = value.get("STATE_ID")
    intents: List[str] = []
    sub_scan_numbers: List[int] = []

    if state_id is not None and len(self._state) > 0:
      ustate_ids = self.par_unique(pool, ncpus, state_id)
      states = self._state.take(ustate_ids)
      intents = states["OBS_MODE"].to_numpy(zero_copy_only=False).tolist()
      sub_scan_numbers = states["SUB_SCAN"].to_numpy(zero_copy_only=False).tolist()

      if "OBS_MODE" in self._subtable_partition_columns:
        assert len(intents) == 1
        key += (("OBS_MODE", intents[0]),)

    # Extract polarization information
    pol_id = self._ddid["POLARIZATION_ID"][ddid].as_py()

    if pol_id >= len(self._pol):
      raise InvalidMeasurementSet(
        f"POLARIZATION_ID {pol_id} does not exist in {name}::POLARIZATION"
      )

    corr_type = Polarisations.from_values(self._pol["CORR_TYPE"][pol_id].as_py())

    # Extract spectral window information
    spw_id = self._ddid["SPECTRAL_WINDOW_ID"][ddid].as_py()

    if spw_id >= len(self._spw):
      raise InvalidMeasurementSet(
        f"SPECTRAL_WINDOW_ID {spw_id} does not exist in {name}::SPECTRAL_WINDOW"
      )

    chan_freq = self._spw["CHAN_FREQ"][spw_id].as_py()
    uchan_width = np.unique(self._spw["CHAN_WIDTH"][spw_id].as_py())
    spw_name = self._spw["NAME"][spw_id].as_py()
    spw_freq_group_name = self._spw["FREQ_GROUP_NAME"][spw_id].as_py()
    spw_ref_freq = self._spw["REF_FREQUENCY"][spw_id].as_py()
    spw_meas_freq_ref = self._spw["MEAS_FREQ_REF"][spw_id].as_py()
    spw_frame = FrequencyMeasures(spw_meas_freq_ref).name.lower()

    # Generate the row map and interval grid in parallel
    row_map = np.full(utime.size * self.nbl, -1.0, dtype=np.int64)
    interval_grid = np.full(utime.size * self.nbl, -1.0, dtype=np.float64)
    chunk_size = (len(rows) + ncpus - 1) // ncpus

    def gen_row_map(time_ids, ant1, ant2, ints, rows):
      assert len(ints) == len(rows) == len(ant1) == len(ant2) == len(time_ids)
      bl_ids = baseline_id(ant1, ant2, self.na, auto_corrs=auto_corrs)
      idx = time_ids.copy()
      idx *= self.nbl
      idx += bl_ids
      row_map[idx] = rows
      interval_grid[idx] = ints

    s = partial(partition_args, chunk=chunk_size)
    list(pool.map(gen_row_map, s(time_ids), s(ant1), s(ant2), s(interval), s(rows)))

    # In the case of averaged datasets, intervals in the last timestep
    # may differ from the rest of the interval, remove it and try to find
    # a unique interval
    interval_grid = interval_grid.reshape(utime.size, self.nbl)

    if interval_grid.shape[0] > 1:
      interval_grid = interval_grid[:-1, :]

    uinterval = self.par_unique(pool, ncpus, interval_grid.ravel())
    uinterval = uinterval[uinterval >= 0]

    partition_data = PartitionData(
      time=utime,
      interval=uinterval,
      chan_freq=chan_freq,
      chan_width=uchan_width,
      corr_type=corr_type.to_str(),
      intents=intents,
      field_names=field_names,
      line_names=line_names,
      source_names=source_names,
      spw_name=spw_name,
      spw_freq_group_name=spw_freq_group_name,
      spw_ref_freq=spw_ref_freq,
      spw_frame=spw_frame,
      row_map=row_map.reshape(utime.size, self.nbl),
      scan_numbers=scan_numbers,
      sub_scan_numbers=sub_scan_numbers,
    )

    return tuple(sorted(key)), partition_data

  def __init__(
    self, ms: TableFactory, partition_schema: List[str], auto_corrs: bool = True
  ):
    import time as modtime

    start = modtime.time()

    partition_columns, subtable_columns = self.partition_columns_from_schema(
      partition_schema
    )

    self._ms_factory = ms
    self._partition_columns = partition_columns
    self._subtable_partition_columns = subtable_columns
    self._auto_corrs = auto_corrs

    ms_table = ms()
    name = ms_table.name()
    table_desc = ms_table.tabledesc()
    subtables, coldescs = self.read_subtables(name)

    for a, subtable in subtables.items():
      setattr(self, a, subtable)

    coldescs["MAIN"] = FrozenDict(
      {c: ColumnDesc.from_descriptor(c, table_desc) for c in ms_table.columns()}
    )

    self._column_descs = FrozenDict(coldescs)

    other_columns = ["INTERVAL"]
    read_columns = (
      set(VALID_MAIN_PARTITION_COLUMNS) | set(SORT_COLUMNS) | set(other_columns)
    )
    arrow_table = ms_table.to_arrow(columns=read_columns)

    ncpus = mp.cpu_count()
    with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:
      source_id = self.maybe_get_source_id(
        pool, ncpus, arrow_table["FIELD_ID"].to_numpy()
      )
      if source_id is not None:
        arrow_table = arrow_table.append_column("SOURCE_ID", source_id[None, :])

      partitions = TablePartitioner(
        partition_columns, SORT_COLUMNS, other_columns + ["row"]
      ).partition(arrow_table, pool)
      self._partitions = {}

      for k, v in partitions.items():
        key, partition = self.partition_data_factory(
          name, auto_corrs, k, v, pool, ncpus
        )
        self._partitions[key] = partition

    logger.info("Reading %s structure in took %fs", name, modtime.time() - start)
