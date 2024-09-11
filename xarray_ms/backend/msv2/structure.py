from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import logging
import multiprocessing as mp
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
  Tuple,
)

import arcae
import numpy as np
import numpy.typing as npt
import pyarrow as pa
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


DEFAULT_PARTITION_COLUMNS: List[str] = [
  "DATA_DESC_ID",
  "FIELD_ID",
  "OBSERVATION_ID",
]


SHORT_TO_LONG_PARTITION_COLUMNS: Dict[str, str] = {
  "D": "DATA_DESC_ID",
  "F": "FIELD_ID",
  "O": "OBSERVATION_ID",
  "P": "PROCESSOR_ID",
  "F1": "FEED1",
  "F2": "FEED2",
}


SORT_COLUMNS: List[str] = ["TIME", "ANTENNA1", "ANTENNA2"]


@dataclasses.dataclass
class PartitionData:
  """Dataclass described data unique to a partition"""

  time: npt.NDArray[np.float64]
  interval: npt.NDArray[np.float64]
  chan_freq: npt.NDArray[np.float64]
  chan_width: npt.NDArray[np.float64]
  corr_type: npt.NDArray[np.int32]
  spw_name: str
  spw_freq_group_name: str
  spw_ref_freq: float
  spw_frame: str
  row_map: npt.NDArray[np.int64]


PartitionKeyT = Tuple[Tuple[str, int], ...]


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

  def partition(self, index: pa.Table) -> Dict[PartitionKeyT, pa.Table]:
    sortby = set(self._sortby)
    other = set(self._other)

    try:
      other.remove("row")
      index = index.append_column(
        "row", pa.array(np.arange(len(index), dtype=np.int64))
      )
    except KeyError:
      pass

    maybe_row = {"row"} if "row" in index.column_names else set()
    read_columns = sortby | other
    if not read_columns.issubset(set(index.column_names) - maybe_row):
      raise ValueError(f"{read_columns} is not a subset of {index.column_names}")

    agg_cmd = [
      (c, "list") for c in (maybe_row | set(read_columns) - set(self._partitionby))
    ]
    partitions = index.group_by(self._partitionby).aggregate(agg_cmd)
    renames = {f"{c}_list": c for c, _ in agg_cmd}
    partitions = partitions.rename_columns(
      renames.get(c, c) for c in partitions.column_names
    )

    partition_map: Dict[PartitionKeyT, pa.Table] = {}

    for p in range(len(partitions)):
      key: PartitionKeyT = tuple(
        sorted((c, int(partitions[c][p].as_py())) for c in self._partitionby)
      )
      table_dict = {c: partitions[c][p].values for c in read_columns | maybe_row}
      partition_table = pa.Table.from_pydict(table_dict)
      if sortby:
        partition_table = partition_table.sort_by([(c, "ascending") for c in sortby])
      partition_map[key] = partition_table

    return partition_map


def on_get_keep_alive(key, value, exists):
  """Reinsert to refresh the TTL for the entry on get operations"""
  if exists:
    MSv2StructureFactory._STRUCTURE_CACHE._set(key, value)


class MSv2StructureFactory:
  """Hashable, callable and picklable factory class
  for creating and caching an MSv2Structure"""

  _ms_factory: TableFactory
  _partition_columns: List[str]
  _auto_corrs: bool
  _STRUCTURE_CACHE: ClassVar[Cache] = Cache(
    maxsize=100, ttl=60, on_get=on_get_keep_alive
  )

  def __init__(
    self, ms: TableFactory, partition_columns: List[str], auto_corrs: bool = True
  ):
    self._ms_factory = ms
    self._partition_columns = partition_columns
    self._auto_corrs = auto_corrs

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, MSv2StructureFactory):
      return NotImplemented

    return (
      self._ms_factory == other._ms_factory
      and self._partition_columns == other._partition_columns
      and self._auto_corrs == other._auto_corrs
    )

  def __hash__(self):
    return hash((self._ms_factory, tuple(self._partition_columns), self._auto_corrs))

  def __reduce__(self):
    return (
      MSv2StructureFactory,
      (self._ms_factory, self._partition_columns, self._auto_corrs),
    )

  def __call__(self, *args, **kw) -> MSv2Structure:
    assert not args and not kw
    return self._STRUCTURE_CACHE.get(
      self,
      lambda self: MSv2Structure(
        self._ms_factory, self._partition_columns, self._auto_corrs
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

  def resolve_key(self, key: str | PartitionKeyT) -> List[PartitionKeyT]:
    """Given a possibly incomplete key, resolves to a list of matching partition keys"""
    if isinstance(key, str):
      key = self.parse_partition_key(key)

    column_set = set(self._partition_columns)

    # Check that the key columns and values are valid
    new_key: List[Tuple[str, int]] = []
    for column, value in key:
      column = column.upper()
      column = SHORT_TO_LONG_PARTITION_COLUMNS.get(column, column)
      if column not in column_set:
        raise InvalidPartitionKey(
          f"{column} is not valid a valid partition column "
          f"{self._partition_columns}"
        )
      if not isinstance(value, Integral):
        raise InvalidPartitionKey(f"{value} is not a valid partition value")
      new_key.append((column, value))

    key_set = set(new_key)
    partition_keys = list(self._partitions.keys())
    partition_key_sets = list(map(set, partition_keys))
    matches = []

    for i, partition_key_set in enumerate(partition_key_sets):
      if key_set.issubset(partition_key_set):
        matches.append(partition_keys[i])

    return matches

  def __init__(
    self, ms: TableFactory, partition_columns: List[str], auto_corrs: bool = True
  ):
    import time as modtime

    start = modtime.time()

    if "DATA_DESC_ID" not in partition_columns:
      raise ValueError("DATA_DESC_ID must be included as a partitioning column")

    self._ms_factory = ms
    self._partition_columns = partition_columns
    self._auto_corrs = auto_corrs

    table = ms()
    name = table.name()
    table_desc = table.tabledesc()
    col_descs: Dict[str, Dict[str, ColumnDesc]] = defaultdict(dict)
    col_descs["MAIN"] = FrozenDict(
      {c: ColumnDesc.from_descriptor(c, table_desc) for c in table.columns()}
    )

    with arcae.table(f"{name}::ANTENNA", lockoptions="nolock") as A:
      self._ant = A.to_arrow()
      table_desc = A.tabledesc()
      col_descs["ANTENNA"] = FrozenDict(
        {c: ColumnDesc.from_descriptor(c, table_desc) for c in A.columns()}
      )

    with arcae.table(f"{name}::FEED", lockoptions="nolock") as F:
      self._feed = F.to_arrow()
      table_desc = F.tabledesc()
      col_descs["FEED"] = FrozenDict(
        {c: ColumnDesc.from_descriptor(c, table_desc) for c in F.columns()}
      )

    with arcae.table(f"{name}::DATA_DESCRIPTION", lockoptions="nolock") as D:
      self._ddid = D.to_arrow()
      table_desc = D.tabledesc()
      col_descs["DATA_DESCRIPTION"] = FrozenDict(
        {c: ColumnDesc.from_descriptor(c, table_desc) for c in D.columns()}
      )

    with arcae.table(f"{name}::SPECTRAL_WINDOW", lockoptions="nolock") as S:
      self._spw = S.to_arrow()
      table_desc = S.tabledesc()
      col_descs["SPECTRAL_WINDOW"] = FrozenDict(
        {c: ColumnDesc.from_descriptor(c, table_desc) for c in S.columns()}
      )

    with arcae.table(f"{name}::POLARIZATION", lockoptions="nolock") as P:
      self._pol = P.to_arrow()
      table_desc = P.tabledesc()
      col_descs["POLARIZATION"] = FrozenDict(
        {c: ColumnDesc.from_descriptor(c, table_desc) for c in P.columns()}
      )

    self._column_descs = FrozenDict(col_descs)

    other_columns = ["INTERVAL"]
    read_columns = set(partition_columns) | set(SORT_COLUMNS) | set(other_columns)
    index = table.to_arrow(columns=read_columns)
    partitions = TablePartitioner(
      partition_columns, SORT_COLUMNS, other_columns + ["row"]
    ).partition(index)

    self._partitions = {}

    ncpus = mp.cpu_count()
    unique_inv_fn = partial(np.unique, return_inverse=True)

    with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:
      for k, v in partitions.items():
        time = v["TIME"].to_numpy()
        interval = v["INTERVAL"].to_numpy()
        ant1 = v["ANTENNA1"].to_numpy()
        ant2 = v["ANTENNA2"].to_numpy()
        rows = v["row"].to_numpy()

        # Compute the unique times and their inverse index
        chunk_size = len(time) // ncpus
        time_chunks = [
          time[i : i + chunk_size] for i in range(0, len(time), chunk_size)
        ]
        utimes, indices = zip(*pool.map(unique_inv_fn, time_chunks))
        utime = np.unique(np.concatenate(utimes))
        inv_fn = partial(np.searchsorted, utime)
        time_ids = pool.map(lambda t, i: inv_fn(t)[i], utimes, indices)
        time_id = np.concatenate(list(time_ids))

        # Compute unique intervals
        interval_chunks = [
          interval[i : i + chunk_size] for i in range(0, len(interval), chunk_size)
        ]
        uintervals = list(pool.map(np.unique, interval_chunks))
        uinterval = np.unique(np.concatenate(uintervals))

        try:
          ddid = next(i for (c, i) in k if c == "DATA_DESC_ID")
        except StopIteration:
          raise KeyError(f"DATA_DESC_ID must be present in partition key {k}")

        if ddid >= len(self._ddid):
          raise InvalidMeasurementSet(
            f"DATA_DESC_ID {ddid} does not exist in {name}::DATA_DESCRIPTION"
          )

        spw_id = self._ddid["SPECTRAL_WINDOW_ID"][ddid].as_py()
        pol_id = self._ddid["POLARIZATION_ID"][ddid].as_py()

        if spw_id >= len(self._spw):
          raise InvalidMeasurementSet(
            f"SPECTRAL_WINDOW_ID {spw_id} does not exist in {name}::SPECTRAL_WINDOW"
          )

        if pol_id >= len(self._pol):
          raise InvalidMeasurementSet(
            f"POLARIZATION_ID {pol_id} does not exist in {name}::POLARIZATION"
          )

        corr_type = Polarisations.from_values(self._pol["CORR_TYPE"][pol_id].as_py())
        chan_freq = self._spw["CHAN_FREQ"][spw_id].as_py()
        uchan_width = np.unique(self._spw["CHAN_WIDTH"][spw_id].as_py())

        spw_name = self._spw["NAME"][spw_id].as_py()
        spw_freq_group_name = self._spw["FREQ_GROUP_NAME"][spw_id].as_py()
        spw_ref_freq = self._spw["REF_FREQUENCY"][spw_id].as_py()
        spw_meas_freq_ref = self._spw["MEAS_FREQ_REF"][spw_id].as_py()
        spw_frame = FrequencyMeasures(spw_meas_freq_ref).name.lower()

        nbl = self.nbl
        bl_id = baseline_id(ant1, ant2, self.na, auto_corrs=auto_corrs)
        row_map = np.full(utime.size * nbl, -1, dtype=np.int64)
        row_map[time_id * nbl + bl_id] = rows

        self._partitions[k] = PartitionData(
          time=utime,
          interval=uinterval,
          chan_freq=chan_freq,
          chan_width=uchan_width,
          corr_type=corr_type.to_str(),
          spw_name=spw_name,
          spw_freq_group_name=spw_freq_group_name,
          spw_ref_freq=spw_ref_freq,
          spw_frame=spw_frame,
          row_map=row_map.reshape(utime.size, nbl),
        )

    logger.info("Reading %s structure in took %fs", name, modtime.time() - start)
