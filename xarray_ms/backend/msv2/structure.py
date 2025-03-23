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
  Literal,
  Mapping,
  Sequence,
  Set,
  Tuple,
)

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from arcae.lib.arrow_tables import Table
from cacheout import Cache

from xarray_ms.backend.msv2.partition import PartitionKeyT, TablePartitioner
from xarray_ms.errors import (
  InvalidMeasurementSet,
  InvalidPartitionKey,
  PartitioningError,
)
from xarray_ms.multiton import Multiton

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


def partition_args(data: npt.NDArray, chunk: int) -> List[npt.NDArray]:
  return [data[i : i + chunk] for i in range(0, len(data), chunk)]


DEFAULT_MAIN_PARTITION_COLUMNS: List[str] = ["DATA_DESC_ID", "OBSERVATION_ID"]

DEFAULT_SUBTABLE_PARTITION_COLUMNS: List[str] = ["OBS_MODE"]

DEFAULT_PARTITION_COLUMNS: List[str] = (
  DEFAULT_MAIN_PARTITION_COLUMNS + DEFAULT_SUBTABLE_PARTITION_COLUMNS
)  #: Default Partitioning Column Schema


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
  """Dataclass describing data unique to a partition

  As `DATA_DESC_ID`, `OBSERVATION_ID` and `STATE::OBS_MODE` are
  always part of the partitioning schema, values related to them
  are singleton values.

  For other cases, multiple values are generally assumed.
  """

  # Main table
  time: npt.NDArray[np.float64]  # Unique timesteps
  interval: npt.NDArray[np.float64]  # Unique intervals
  obs_id: int  # unique from OBSERVATION_ID
  spw_id: int  # unique from DATA_DESC_ID
  pol_id: int  # unique from DATA_DESC_ID
  # Multiple values per partition
  antenna_ids: npt.NDArray[np.int32]
  feed_ids: npt.NDArray[np.int32]
  field_ids: npt.NDArray[np.int32]
  state_ids: npt.NDArray[np.int32]
  scan_numbers: npt.NDArray[np.int32]
  # FIELD subtable
  source_ids: npt.NDArray[np.int32]
  # STATE subtable
  obs_mode: str  # unique from STATE::OBS_MODE
  sub_scan_numbers: npt.NDArray[np.int32]

  # Row to baseline map
  row_map: npt.NDArray[np.int64]

  @property
  def ntime(self) -> int:
    """Number of timesteps"""
    return self.row_map.shape[0]

  @property
  def nbl(self) -> int:
    """Number of baselines"""
    return self.row_map.shape[1]

  @property
  def na(self) -> int:
    """Number of antenna"""
    return len(self.antenna_ids)

  @property
  def auto_corrs(self) -> bool:
    """Returns true if auto-correlations are included"""
    if self.na * (self.na + 1) // 2 == self.nbl:
      return True
    elif self.na * (self.na - 1) // 2 == self.nbl:
      return False
    else:
      raise RuntimeError(f"Invalid antenna {self.na} baseline {self.nbl} relation")

  @property
  def antenna_pairs(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Return per-baseline antenna pairs"""
    a1, a2 = np.triu_indices(self.na, 0 if self.auto_corrs else 1)
    aids = np.asarray(self.antenna_ids, np.int32)
    return aids[a1], aids[a2]


def on_get_keep_alive(key, value, exists):
  """Reinsert to refresh the TTL for the entry on get operations"""
  if exists:
    MSv2StructureFactory._STRUCTURE_CACHE._set(key, value)


class MSv2StructureFactory:
  """Hashable, callable and picklable factory class
  for creating and caching an MSv2Structure"""

  _ms_factory: Multiton
  _subtable_factories: Dict[str, Multiton]
  _partition_schema: List[str]
  _epoch: str
  _auto_corrs: bool
  _STRUCTURE_CACHE: ClassVar[Cache] = Cache(
    maxsize=100, ttl=60, on_get=on_get_keep_alive
  )

  def __init__(
    self,
    ms: Multiton,
    subtables: Dict[str, Multiton],
    partition_schema: List[str],
    epoch: str,
    auto_corrs: bool,
  ):
    self._ms_factory = ms
    self._subtable_factories = subtables
    self._partition_schema = partition_schema
    self._epoch = epoch
    self._auto_corrs = auto_corrs

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, MSv2StructureFactory):
      return NotImplemented

    return (
      self._ms_factory == other._ms_factory
      and self._subtable_factories == other._subtable_factories
      and self._partition_schema == other._partition_schema
      and self._epoch == other._epoch
      and self._auto_corrs == other._auto_corrs
    )

  def __hash__(self):
    return hash(
      (
        self._ms_factory,
        frozenset(self._subtable_factories.items()),
        tuple(self._partition_schema),
        self._epoch,
        self._auto_corrs,
      )
    )

  def __reduce__(self):
    return (
      MSv2StructureFactory,
      (
        self._ms_factory,
        self._subtable_factories,
        self._partition_schema,
        self._epoch,
        self._auto_corrs,
      ),
    )

  @staticmethod
  def _create_instance(self):
    return MSv2Structure(
      self._ms_factory,
      self._subtable_factories,
      self._partition_schema,
      self._auto_corrs,
    )

  @property
  def instance(self) -> MSv2Structure:
    return self._STRUCTURE_CACHE.get(self, self._create_instance)

  def release(self) -> bool:
    return self._STRUCTURE_CACHE.delete(self) > 0


class MSv2Structure(Mapping):
  """Holds structural information about an MSv2 dataset"""

  _ms_factory: Multiton
  _subtable_factories: Dict[str, Multiton]
  _partition_columns: List[str]
  _subtable_partition_columns: List[str]
  _partitions: Mapping[PartitionKeyT, PartitionData]

  def __getitem__(self, key: PartitionKeyT) -> PartitionData:
    return self._partitions[key]

  def __iter__(self) -> Iterator[Any]:
    return iter(self._partitions)

  def __len__(self) -> int:
    return len(self._partitions)

  @staticmethod
  def parse_partition_key(key: str) -> PartitionKeyT:
    """Parses a "DATA_DESC_ID=0, FIELD_ID_1,..." style string
    into a tuple of (column, id) tuples"""
    pairs: List[Tuple[str, int | str]] = []

    for component in [k.strip() for k in key.split(",")]:
      try:
        column, str_value = component.split("=")
      except ValueError as e:
        raise InvalidPartitionKey(f"Invalid partition key {key!r}") from e
      else:
        try:
          int_value = int(str_value)
        except ValueError:
          pairs.append((column, str_value))
        else:
          pairs.append((column, int_value))

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

  @staticmethod
  def broadcast_source_id(
    pool: cf.Executor,
    ncpus: int,
    field: pa.Table,
    field_id: npt.NDArray[np.int32],
  ) -> npt.NDArray[np.int32]:
    """Constructs a SOURCE_ID array from MAIN.FIELD_ID
    broadcast against FIELD.SOURCE_ID"""
    field_source_id = field["SOURCE_ID"].to_numpy()
    source_id = np.empty_like(field_id)
    chunk = (len(source_id) + ncpus - 1) // ncpus

    def par_assign(sid, fid):
      sid[:] = field_source_id[fid]

    list(
      pool.map(
        par_assign, partition_args(source_id, chunk), partition_args(field_id, chunk)
      )
    )
    return source_id

  @staticmethod
  def broadcast_sub_scan_number(
    pool: cf.Executor,
    ncpus: int,
    state: pa.Table,
    state_id: npt.NDArray[npt.int32],
  ) -> npt.NDArray[np.int32]:
    """Constructs a SUB_SCAN_NUMBER array from MAIN.STATE_ID
    broadcast against STATE.SUB_SCAN_NUMBER"""
    state_ssn = state["SUB_SCAN"].to_numpy()
    subscan_nr = np.empty_like(state_id)
    chunk = (len(state_id) + ncpus - 1) // ncpus

    def par_assign(ssn, sid):
      ssn[:] = state_ssn[sid]

    list(
      pool.map(
        par_assign, partition_args(subscan_nr, chunk), partition_args(state_id, chunk)
      )
    )
    return subscan_nr

  @staticmethod
  def broadcast_obsmode_id(
    pool: cf.Executor,
    ncpus: int,
    state: pa.Table,
    state_id: npt.NDArray[npt.int32],
  ) -> Tuple[npt.NDArray[np.int32], Dict[str, List[int]]]:
    """Constructs an OBS_MODE_ID array from MAIN.STATE_ID broadcast
    against unique entries in STATE.OBS_MODE"""
    obs_mode = state["OBS_MODE"].to_numpy()

    # Map unique observation modes to state_ids
    obs_mode_state_id_map: Mapping[str, List[int]] = defaultdict(list)
    for sid, obs_mode in enumerate(obs_mode):
      obs_mode_state_id_map[obs_mode].append(sid)

    # Generate the reverse mapping of state id's to unique observation mode id's
    state_id_obs_mode_id_map = np.empty(len(state), np.int32)

    for o, (obs_mode, state_ids) in enumerate(obs_mode_state_id_map.items()):
      for sid in state_ids:
        state_id_obs_mode_id_map[sid] = o

    # Broadcast generated observation mode id's against MAIN.STATE_ID
    obs_mode_id = np.empty_like(state_id)
    chunk = (len(state_id) + ncpus - 1) // ncpus

    def par_assign(oid, sid):
      oid[:] = state_id_obs_mode_id_map[sid]

    list(
      pool.map(
        par_assign, partition_args(obs_mode_id, chunk), partition_args(state_id, chunk)
      )
    )
    return obs_mode_id, dict(obs_mode_state_id_map)

  def partition_columns_from_schema(
    self, partition_schema: List[str]
  ) -> Tuple[List[str], List[str]]:
    """Given a partitioning schema, produce

    1. a list of partitioning columns of the MAIN table
    2. a list of subtable partitioning columns.
       i.e. `FIELD.SOURCE_ID` and `STATE.OBS_MODE`
    """
    schema: Set[str] = set(partition_schema)

    # Always partition by default columns
    columns: Dict[str, None] = {c: None for c in DEFAULT_MAIN_PARTITION_COLUMNS}
    subtable_columns: Dict[str, None] = {
      c: None for c in DEFAULT_SUBTABLE_PARTITION_COLUMNS
    }

    for column in VALID_MAIN_PARTITION_COLUMNS:
      if column in schema and column not in columns:
        columns[column] = None

    for column in VALID_FIELD_PARTITION_COLUMNS:
      if column in schema and column not in subtable_columns:
        subtable_columns[column] = None

    for column in VALID_STATE_PARTITION_COLUMNS:
      if column in schema and column not in subtable_columns:
        subtable_columns[column] = None

    return list(columns.keys()), list(subtable_columns.keys())

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

      data_ids = list(pool.map(inv_fn, udatas, indices))
      inverse = np.empty(len(data), dtype=indices[0].dtype)
      list(pool.map(par_assign, partition_args(inverse, chunk_size), data_ids))

      return udata, inverse
    else:
      udata = list(pool.map(np.unique, data_chunks))
      return np.unique(np.concatenate(udata))

  @staticmethod
  def feed_antennas(
    feed: pa.Table, spw_id: int, feed_ids: List[int]
  ) -> npt.NDArray[np.int32]:
    """Returns the unique ANTENNA_ID's associated with
    the given SPECTRAL_WINDOW_ID and FEED_IDS"""
    feed_spw_ids = feed["SPECTRAL_WINDOW_ID"].to_numpy()
    feed_feed_ids = feed["FEED_ID"].to_numpy()
    antenna_ids = feed["ANTENNA_ID"].to_numpy()
    mask = np.logical_or(feed_spw_ids == -1, feed_spw_ids == spw_id)
    np.logical_and(mask, np.isin(feed_feed_ids, feed_ids), out=mask)
    return np.unique(antenna_ids[mask])

  @staticmethod
  def gen_row_interval_grids(
    time_ids: npt.NDArray[np.int32],
    antenna1: npt.NDArray[np.int32],
    antenna2: npt.NDArray[np.int32],
    intervals: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int32],
    row_map: npt.NDArray[np.int64],
    interval_grid: npt.NDArray[np.float64],
    feed_antennas: npt.NDArray[npt.int32],
    na: int,
    nbl: int,
    auto_corrs: bool,
  ) -> None:
    """Populate the row map and interval grids"""

    # If the row_map contains no auto-correlations
    # they must be stripped out of the data
    if not auto_corrs and not np.all(mask := antenna1 != antenna2):
      time_ids = time_ids[mask]
      antenna1 = antenna1[mask]
      antenna2 = antenna2[mask]
      rows = rows[mask]
      intervals = intervals[mask]

    # Normalise the baseline direction
    if np.any(mask := antenna1 > antenna2):
      antenna1[mask], antenna2[mask] = antenna2[mask], antenna1[mask]

    # Maybe normalise the antenna id's to np.arange(feed_antennas)
    normed_antennas = np.arange(feed_antennas.size, dtype=feed_antennas.dtype)
    if not np.all(feed_antennas == normed_antennas):
      antenna1 = np.searchsorted(feed_antennas, antenna1)
      antenna2 = np.searchsorted(feed_antennas, antenna2)

    index = time_ids * nbl
    index += baseline_id(antenna1, antenna2, na, auto_corrs)
    row_map[index] = rows
    interval_grid[index] = intervals

    return index

  @staticmethod
  def read_subtable(
    table_name: str,
    subtable_name: str,
    missing: Literal["raise", "none"] = "raise",
    columns: Sequence[str] = (),
  ) -> pa.Table | None:
    subtable_path = os.path.join(table_name, subtable_name)

    if not os.path.exists(subtable_path):
      if missing == "raise":
        raise FileNotFoundError(f"Required subtable {subtable_name} does not exist")
      else:
        return None

    with Table.from_filename(subtable_path, lockoptions="nolock") as T:
      return T.to_arrow(columns=list(columns))

  def __init__(
    self,
    ms: Multiton,
    subtable_factories: Dict[str, Multiton],
    partition_schema: List[str],
    auto_corrs: bool,
  ):
    import time as modtime

    start = modtime.time()

    partition_columns, subtable_columns = self.partition_columns_from_schema(
      partition_schema
    )

    self._ms_factory = ms
    self._subtable_factories = subtable_factories
    self._partition_columns = partition_columns
    self._subtable_partition_columns = subtable_columns

    ms_table = ms.instance
    ms_name = ms_table.name()

    try:
      data_description = subtable_factories["DATA_DESCRIPTION"].instance
      feed = subtable_factories["FEED"].instance
      state = subtable_factories["STATE"].instance
      field = subtable_factories["FIELD"].instance
    except KeyError as e:
      raise InvalidMeasurementSet(
        f"Measurement Set {ms_name} is missing required subtable {e}"
      )

    other_columns = ["FEED1", "FEED2", "INTERVAL"]
    read_columns = (
      set(VALID_MAIN_PARTITION_COLUMNS) | set(SORT_COLUMNS) | set(other_columns)
    )
    arrow_table = ms_table.to_arrow(columns=read_columns)

    ncpus = mp.cpu_count()
    with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:

      def subtable_column_group(s):
        """Return the group that the subtable column should be assigned to"""
        return partition_columns if s in subtable_columns else other_columns

      def get_uid_column(column, dkey, ids) -> npt.NDArray:
        """Get the unique values for the given column, preferably from the
        partition key or failing that, from `ids`. Generally should be used with
        ID columns"""
        try:
          return np.array([dkey[column]])
        except KeyError:
          return self.par_unique(pool, ncpus, ids)

      def time_coord(column, dkey, ids, utime, time_ids) -> npt.NDArray:
        try:
          value = dkey[column]
        except KeyError:
          result = np.empty(utime.shape, dtype=ids.dtype)
          result[time_ids] = ids
          return result
        else:
          return np.full(utime.shape, value)

      # Broadcast and add FIELD.SOURCE_ID column
      field_id = arrow_table["FIELD_ID"].to_numpy()
      source_id = self.broadcast_source_id(pool, ncpus, field, field_id)
      arrow_table = arrow_table.append_column("SOURCE_ID", source_id[None, :])
      subtable_column_group("SOURCE_ID").append("SOURCE_ID")

      # Broadcast and add STATE.OBS_MODE and STATE.SUB_SCAN_NUMBER columns
      state_id = arrow_table["STATE_ID"].to_numpy()
      obs_mode_id, om_to_sid_map = self.broadcast_obsmode_id(
        pool, ncpus, state, state_id
      )
      subscan_nr = self.broadcast_sub_scan_number(pool, ncpus, state, state_id)
      arrow_table = arrow_table.append_column("OBS_MODE_ID", obs_mode_id[None, :])
      arrow_table = arrow_table.append_column("SUB_SCAN_NUMBER", subscan_nr[None, :])
      subtable_column_group("SUB_SCAN_NUMBER").append("SUB_SCAN_NUMBER")
      # Substitute OBS_MODE for OBS_MODE_ID
      partition_columns.append("OBS_MODE_ID")

      # Perform partitioning
      partitions = TablePartitioner(
        partition_columns, SORT_COLUMNS, other_columns + ["row"]
      ).partition(arrow_table, pool)

      # Generate a PartitionData variable per partition
      self._partitions = {}

      for key, partition in partitions.items():
        dkey = dict(key)

        # The following should always be part of the partioning key
        try:
          ddid = int(dkey["DATA_DESC_ID"])
          obs_id = int(dkey["OBSERVATION_ID"])
          obs_mode_id = int(dkey.pop("OBS_MODE_ID"))
        except KeyError as e:
          raise KeyError(f"{e} must be present in partition key {key}")

        dkey["OBS_MODE"] = tuple(om_to_sid_map.keys())[obs_mode_id]

        if ddid >= len(data_description):
          raise InvalidMeasurementSet(
            f"DATA_DESC_ID {ddid} does not exist in {ms_name}::DATA_DESCRIPTION"
          )

        spw_id = data_description["SPECTRAL_WINDOW_ID"][ddid].as_py()
        pol_id = data_description["POLARIZATION_ID"][ddid].as_py()
        antenna1 = partition["ANTENNA1"]
        antenna2 = partition["ANTENNA2"]
        interval = partition["INTERVAL"]
        rows = partition["row"]

        # Unique sorting/other column values
        utime, time_ids = self.par_unique(
          pool, ncpus, partition["TIME"], return_inverse=True
        )

        # Unique partition key values
        ufield_ids = time_coord(
          "FIELD_ID", dkey, partition["FIELD_ID"], utime, time_ids
        )
        usubscan_nrs = time_coord(
          "SUB_SCAN_NUMBER", dkey, partition["SUB_SCAN_NUMBER"], utime, time_ids
        )
        uscan_nrs = time_coord(
          "SCAN_NUMBER", dkey, partition["SCAN_NUMBER"], utime, time_ids
        )
        ustate_ids = get_uid_column("STATE_ID", dkey, partition["STATE_ID"])
        usource_ids = get_uid_column("SOURCE_ID", dkey, partition["SOURCE_ID"])

        uantenna1 = self.par_unique(pool, ncpus, antenna1)
        uantenna2 = self.par_unique(pool, ncpus, antenna2)
        uantennas = np.union1d(uantenna1, uantenna2)
        ufeed1s = self.par_unique(pool, ncpus, partition["FEED1"])
        ufeed2s = self.par_unique(pool, ncpus, partition["FEED2"])
        ufeeds = np.union1d(ufeed1s, ufeed2s)

        # Query the FEED table to discover all canonical
        # antennas in this partition
        feed_antennas = self.feed_antennas(feed, spw_id, ufeeds)
        if not np.all(np.isin(uantennas, feed_antennas)):
          raise InvalidMeasurementSet(
            f"Unique ANTENNA1 and ANTENNA2 values {uantennas} "
            f"are not a subset of FEED::ANTENNA_ID {feed_antennas} "
            f"for SPECTRAL_WINDOW_ID={spw_id} and feeds={ufeeds}"
          )

        na = len(feed_antennas)
        nbl = nr_of_baselines(na, auto_corrs)
        chunk = (len(rows) + ncpus - 1) // ncpus

        # Populate row map and interval grids
        row_map = np.full(utime.size * nbl, -1, dtype=np.int64)
        interval_grid = np.full(utime.size * nbl, -1.0, dtype=np.float64)

        index_list = list(
          pool.map(
            partial(
              self.gen_row_interval_grids,
              row_map=row_map,
              interval_grid=interval_grid,
              feed_antennas=feed_antennas,
              na=na,
              nbl=nbl,
              auto_corrs=auto_corrs,
            ),
            partition_args(time_ids, chunk),
            partition_args(antenna1, chunk),
            partition_args(antenna2, chunk),
            partition_args(interval, chunk),
            partition_args(rows, chunk),
          )
        )

        indices = np.concatenate(index_list)

        if np.any(np.bincount(indices) > 1):
          msg = []
          if len(ufeeds) > 1:
            msg.append(f"Multiple feeds {ufeeds} are present")
          if len(ustate_ids) > 1:
            msg.append(f"Multiple STATE_ID {ustate_ids} are present")
          if len(ufield_ids) > 1:
            msg.append(f"Multiple FIELD_ID {ufield_ids}  are present")

          msg_str = "\n  ".join(msg)

          raise PartitioningError(
            f"Multiple occurences of a (TIME, ANTENNA1, ANTENNA2) "
            f"tuple are present in partition {key}. "
            f"It is not possible to establish a unique "
            f"(time, baseline) grid for this partition. "
            f"Generally, the solution is to partition more finely by adding "
            f"columns to the partition_schema.\n"
            f"{msg_str}"
          )

        # In the case of averaged datasets, intervals in the last timestep
        # may differ from other intervals. Remove it and try to find
        # a unique interval
        interval_grid = interval_grid.reshape(utime.size, nbl)

        if interval_grid.shape[0] > 1:
          interval_grid = interval_grid[:-1, :]

        uinterval = self.par_unique(pool, ncpus, interval_grid.ravel())
        uinterval = uinterval[uinterval >= 0]

        self._partitions[tuple(sorted(dkey.items()))] = PartitionData(
          time=utime,
          interval=uinterval,
          obs_id=obs_id,
          spw_id=spw_id,
          pol_id=pol_id,
          antenna_ids=feed_antennas,
          feed_ids=ufeeds,
          field_ids=ufield_ids,
          scan_numbers=uscan_nrs,
          source_ids=usource_ids,
          state_ids=ustate_ids,
          obs_mode=str(dkey["OBS_MODE"]),
          sub_scan_numbers=usubscan_nrs,
          row_map=row_map.reshape(utime.size, nbl),
        )

    logger.info("Reading %s structure in took %fs", ms_name, modtime.time() - start)
