from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Dict, Iterator, List, Mapping, Sequence, Tuple

import arcae
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from cacheout import Cache

from xarray_ms.backend.msv2.table_factory import TableFactory
from xarray_ms.casa_types import Polarisations


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
  #   return nbl - ant1 + ant2 + na - (na - ant1 + 1) * (na - ant1) // 2
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


class IrregularGridError(ValueError):
  """Raised when the intervals associated with each timestep are not homogenous"""

  pass


class InvalidMeasurementSet(ValueError):
  """Raised when the Measurement Set foreign key indexing is invalid"""


PARTITION_COLUMNS = ["FIELD_ID", "PROCESSOR_ID", "DATA_DESC_ID", "FEED1", "FEED2"]


@dataclass
class PartitionCoordinateData:
  """Dataclass containing partition coordinates"""

  time: npt.NDArray[np.float64]
  interval: float
  chan_freq: npt.NDArray[np.float64]
  corr_type: npt.NDArray[np.int32]


PartitionKeyT = Tuple[Tuple[str, int], ...]


class TablePartitioner:
  """Partitions and sorts MSv2 indexing columns"""

  _sortby: List[str]
  _other: List[str]

  def __init__(self, sortby: Sequence[str], other: Sequence[str]):
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
      (c, "list") for c in (maybe_row | set(read_columns) - set(PARTITION_COLUMNS))
    ]
    partitions = index.group_by(PARTITION_COLUMNS).aggregate(agg_cmd)
    renames = {f"{c}_list": c for c, _ in agg_cmd}
    partitions = partitions.rename_columns(
      renames.get(c, c) for c in partitions.column_names
    )

    partition_map: Dict[PartitionKeyT, pa.Table] = {}

    for p in range(len(partitions)):
      key: PartitionKeyT = tuple(
        sorted((c, int(partitions[c][p].as_py())) for c in PARTITION_COLUMNS)
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
  """Hashable, callable and picklable factory class for creating an MSv2Structure"""

  _ms_factory: TableFactory
  _auto_corrs: bool
  _STRUCTURE_CACHE: ClassVar[Cache] = Cache(
    maxsize=100, ttl=60, on_get=on_get_keep_alive
  )

  def __init__(self, ms: TableFactory, auto_corrs: bool = True):
    self._ms_factory = ms
    self._auto_corrs = auto_corrs

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, MSv2StructureFactory):
      return NotImplemented

    return (
      self._ms_factory == other._ms_factory and self._auto_corrs == other._auto_corrs
    )

  def __hash__(self):
    return hash((self._ms_factory, self._auto_corrs))

  def __reduce__(self):
    return (MSv2StructureFactory, (self._ms_factory, self._auto_corrs))

  def __call__(self, *args, **kw) -> MSv2Structure:
    assert not args and not kw
    return self._STRUCTURE_CACHE.get(
      self, lambda self: MSv2Structure(self._ms_factory, self._auto_corrs)
    )


class MSv2Structure(Mapping):
  """Holds structural information about an MSv2 dataset"""

  _ms_factory: TableFactory
  _auto_corrs: bool
  _partitions: Mapping[PartitionKeyT, PartitionCoordinateData]
  _table_desc: Dict[str, Any]
  _columns: List[str]
  _rowmap: Mapping[PartitionKeyT, npt.NDArray[np.int64]]
  _ant: pa.Table
  _ddid: pa.Table
  _feed: pa.Table
  _spw: pa.Table
  _pol: pa.Table

  def __getitem__(self, key: PartitionKeyT) -> PartitionCoordinateData:
    return self._partitions[key]

  def __iter__(self) -> Iterator[Any]:
    return iter(self._partitions)

  def __len__(self) -> int:
    return len(self._partitions)

  def __hash__(self) -> int:
    return hash(self._ms_factory)

  @property
  def auto_corrs(self) -> bool:
    """Are auto-correlations included"""
    return self._auto_corrs

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

  def row_map(self, key: PartitionKeyT) -> npt.NDArray:
    """Returns the (time, baseline) => row map for the key partition"""
    return self._row_map[key]

  def __init__(self, ms: TableFactory, auto_corrs: bool = True):
    import time as modtime

    start = modtime.time()

    self._ms_factory = ms
    self._auto_corrs = auto_corrs
    self._table_desc = ms().tabledesc()
    self._columns = ms().columns()
    name = ms().name()

    with arcae.table(f"{name}::ANTENNA", lockoptions="nolock") as A:
      self._ant = A.to_arrow()

    with arcae.table(f"{name}::FEED", lockoptions="nolock") as F:
      self._feed = F.to_arrow()

    with arcae.table(f"{name}::DATA_DESCRIPTION", lockoptions="nolock") as D:
      self._ddid = D.to_arrow()

    with arcae.table(f"{name}::SPECTRAL_WINDOW", lockoptions="nolock") as S:
      self._spw = S.to_arrow()

    with arcae.table(f"{name}::POLARIZATION", lockoptions="nolock") as P:
      self._pol = P.to_arrow()

    sort_columns = ["TIME", "ANTENNA1", "ANTENNA2"]
    other_columns = ["INTERVAL"]
    read_columns = set(PARTITION_COLUMNS) | set(sort_columns) | set(other_columns)
    index = ms().to_arrow(columns=read_columns)
    partitions = TablePartitioner(sort_columns, other_columns + ["row"]).partition(
      index
    )

    self._row_map = {}
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

        if uinterval.size != 1:
          raise IrregularGridError(
            f"INTERVAL values for partition {k} are not unique {uinterval}"
          )

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

        self._partitions[k] = PartitionCoordinateData(
          time=utime,
          interval=uinterval.item(),
          chan_freq=chan_freq,
          corr_type=corr_type.to_str(),
        )

        nbl = self.nbl
        bl_id = baseline_id(ant1, ant2, self.na, auto_corrs=auto_corrs)
        row_map = np.full(utime.size * nbl, -1, dtype=np.int64)
        row_map[time_id * nbl + bl_id] = rows
        self._row_map[k] = row_map.reshape(utime.size, nbl)

    print(f"Reading {name} structure in {os.getpid()} took {modtime.time() - start}s")
