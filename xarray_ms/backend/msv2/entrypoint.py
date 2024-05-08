from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Any, Tuple
from uuid import uuid4

import arcae
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from arcae.lib.arrow_tables import Table
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.backends.common import AbstractWritableDataStore, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.indexing import (
  IndexingSupport,
  LazilyIndexedArray,
  explicit_indexing_adapter,
)
from xarray.core.utils import FrozenDict, try_read_magic_number_from_file_or_path
from xarray.core.variable import Variable

from xarray_ms.backend.msv2.cacheset import CACHESET
from xarray_ms.backend.msv2.table_factory import TableFactory
from xarray_ms.testing.casa_types import Polarisations

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray.backends.common import AbstractDataStore
  from xarray.core.dataset import Dataset
  from xarray.core.datatree import DataTree


PARTITION_COLUMNS = ["FIELD_ID", "PROCESSOR_ID", "DATA_DESC_ID", "FEED1", "FEED2"]
SORTING_COLUMNS = ["TIME", "ANTENNA1", "ANTENNA2"]


@dataclass
class PartitionData:
  time: npt.NDArray[np.float64]
  antenna1: npt.NDArray[np.int32]
  antenna2: npt.NDArray[np.int32]
  tbl_to_row: npt.NDArray[np.int64]
  row: npt.NDArray[np.int64]
  chan_freq: npt.NDArray[np.float64]
  corr_type: npt.NDArray[np.int32]


PartitionKeyT = Tuple[Tuple[str, Number]]


class MSv2Structure(Mapping):
  """Holds structural information about an MSv2 dataset"""

  _ms: str
  _partitions: Mapping[PartitionKeyT, PartitionData]
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

  def __init__(self, ms):
    self._ms = ms

    with arcae.table(ms, readonly=True, lockoptions="nolock") as T:
      read_columns = set(PARTITION_COLUMNS) | set(SORTING_COLUMNS)
      index = T.to_arrow(columns=read_columns)

    index = index.append_column("row", pa.array(np.arange(len(index), dtype=np.int64)))
    agg_cmd = [(c, "list") for c in (set(read_columns) - set(PARTITION_COLUMNS))]
    agg_cmd += [("row", "list")]
    partitions = index.group_by(PARTITION_COLUMNS).aggregate(agg_cmd)
    renames = {f"{c}_list": c for c, _ in agg_cmd}
    partitions = partitions.rename_columns(
      renames.get(c, c) for c in partitions.column_names
    )

    with arcae.table(f"{ms}::ANTENNA", lockoptions="nolock") as A:
      self._ant = A.to_arrow()

    with arcae.table(f"{ms}::FEED", lockoptions="nolock") as F:
      self._feed = F.to_arrow()

    with arcae.table(f"{ms}::DATA_DESCRIPTION", lockoptions="nolock") as D:
      self._ddid = D.to_arrow()

    with arcae.table(f"{ms}::SPECTRAL_WINDOW", lockoptions="nolock") as S:
      self._spw = S.to_arrow()

    with arcae.table(f"{ms}::POLARIZATION", lockoptions="nolock") as P:
      self._pol = P.to_arrow()

    self._partitions = {}

    # Full resolution baseline map
    ant1, ant2 = np.triu_indices(len(self._ant), 0)
    ant1, ant2 = (a.astype(np.int32) for a in (ant1, ant2))

    for p in range(len(partitions)):
      key = tuple(sorted((c, partitions[c][p].as_py()) for c in PARTITION_COLUMNS))
      sort_column_names = ["TIME", "ANTENNA1", "ANTENNA2", "row"]
      sort_columns = [partitions[c][p].values.to_numpy() for c in sort_column_names]
      search_columns = {n: c for n, c in zip(sort_column_names, sort_columns)}
      utime = np.unique(search_columns["TIME"])[:, None]
      putime, pant1, pant2 = (
        a.ravel() for a in np.broadcast_arrays(utime, ant1[None, :], ant2[None, :])
      )

      sdtype = np.dtype([("t", putime.dtype), ("a1", pant1.dtype), ("a2", pant2.dtype)])

      full_res = np.zeros(putime.size, sdtype)
      full_res["t"] = putime
      full_res["a1"] = pant1
      full_res["a2"] = pant2

      search = np.zeros(sort_columns[0].size, sdtype)
      search["t"] = search_columns["TIME"]
      search["a1"] = search_columns["ANTENNA1"]
      search["a2"] = search_columns["ANTENNA2"]

      idx = np.searchsorted(full_res, search)
      tbl_to_row = np.full(putime.size, -1, np.int64)
      tbl_to_row[idx] = search_columns["row"]

      index = np.lexsort(tuple(reversed(sort_columns[:-1])))
      partition_data = {
        n.lower(): c[index] for n, c in zip(sort_column_names, sort_columns)
      }

      ddid = partitions["DATA_DESC_ID"][p].as_py()
      spw_id = self._ddid["SPECTRAL_WINDOW_ID"][ddid].as_py()
      pol_id = self._ddid["POLARIZATION_ID"][ddid].as_py()
      corr_type = Polarisations.from_values(self._pol["CORR_TYPE"][pol_id].as_py())

      partition_data.update(
        (
          ("tbl_to_row", tbl_to_row.reshape((utime.shape[0], ant1.shape[0]))),
          ("chan_freq", self._spw["CHAN_FREQ"][spw_id].as_py()),
          ("corr_type", corr_type.to_str()),
        )
      )

      self._partitions[key] = PartitionData(**partition_data)


class MSv2Array(BackendArray):
  def __init__(self, factory, partition, column, shape, dtype):
    self._factory = factory
    self._partition = partition
    self._column = column
    self.shape = shape
    self.dtype = np.dtype(dtype)

    assert len(shape) >= 2
    assert shape[:2] == partition.tbl_to_row.shape

  def __getitem__(self, key):
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key):
    rows = self._partition.tbl_to_row[key[:2]]
    xkey = (rows.ravel(),) + key[2:]
    import os

    print(
      os.getpid(),
      CACHESET["tables"].size(),
      self._factory._key,
      self._factory in CACHESET["tables"],
    )
    table = CACHESET["tables"].get(self._factory, lambda k: k())
    row_data = table.getcol(self._column, xkey)
    return row_data.reshape(rows.shape + row_data.shape[1:])


class MSv2Store(AbstractWritableDataStore):
  """Store for reading and writing data via arcae"""

  __slots__ = ("_factory", "_partition", "_epoch", "_structure")

  def __init__(self, factory, partition=None, structure=None, epoch=None):
    if not partition:
      partition = next(iter(structure.keys()))
      warnings.warn(f"No partition was supplied. Randomly selected {partition}")
    elif partition not in structure:
      raise ValueError(f"{partition} not in {list(structure.keys())}")

    self._factory = factory
    self._partition = partition
    self._epoch = epoch or uuid4().hex[:8]
    self._structure = structure

  @classmethod
  def open(cls, ms: str, drop_variables=None, partition=None, epoch=None):
    if not isinstance(ms, str):
      raise ValueError("Measurement Sets paths can only be strings")

    structure = MSv2Structure(ms)
    factory = TableFactory(Table.from_filename, ms, readonly=True, lockoptions="nolock")
    return cls(factory, partition, structure=structure)

  def close(self, **kwargs):
    print("Closing MSv2Store")

  def get_variables(self):
    partition = self._structure[self._partition]
    nfreq = len(partition.chan_freq)
    npol = len(partition.corr_type)
    ant1, ant2 = np.triu_indices(len(self._structure._ant), 0)
    utime = np.unique(partition.time)
    ntime, nbl = partition.tbl_to_row.shape

    uvw = MSv2Array(self._factory, partition, "UVW", (ntime, nbl, 3), np.float64)
    data = MSv2Array(
      self._factory, partition, "DATA", (ntime, nbl, nfreq, npol), np.complex64
    )
    flag = MSv2Array(
      self._factory, partition, "FLAG", (ntime, nbl, nfreq, npol), np.uint8
    )

    data_vars = [
      ("UVW", (("time", "baseline", "uvw_label"), uvw, None)),
      ("DATA", (("time", "baseline", "frequency", "polarization"), data, None)),
      ("FLAG", (("time", "baseline", "frequency", "polarization"), flag, None)),
    ]

    data_vars = [(n, (d, LazilyIndexedArray(v), a)) for n, (d, v, a) in data_vars]

    # Add coordinates
    data_vars += [
      (
        "baseline_id",
        (("baseline",), np.arange(len(ant1)), {"coordinates": "baseline_id"}),
      ),
      ("antenna1_id", (("baseline",), ant1, {"coordinates": "antenna1_id"})),
      ("antenna2_id", (("baseline",), ant2, {"coordinates": "antenna2_id"})),
      (
        "time",
        (
          ("time",),
          utime,
          {
            "type": "time",
            "scale": "utc",
            "units": ["s"] * len(utime),
            "format": "unix",
          },
        ),
      ),
      (
        "frequency",
        (
          ("frequency",),
          partition.chan_freq,
          {"type": "spectral_coord", "units": "Hz", "frame": "lsrk"},
        ),
      ),
      ("polarization", (("polarization",), partition.corr_type, None)),
    ]

    return FrozenDict((n, Variable(d, v, a)) for n, (d, v, a) in data_vars)

  def get_attrs(self):
    return {}

  def get_dimensions(self):
    return None

  def get_encoding(self):
    return {}


class MSv2PartitionEntryPoint(BackendEntrypoint):
  open_dataset_parameters = ["filename_or_obj", "partition"]
  description = "Opens v2 CASA Measurement Sets in Xarray"
  url = "https://link_to/your_backend/documentation"

  def guess_can_open(
    self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore
  ) -> bool:
    """Return true if this is a CASA table"""
    if not isinstance(filename_or_obj, (str, os.PathLike)):
      return False

    path = _normalize_path(filename_or_obj)

    # CASA Tables are directories containing a table.dat file
    dat_path = os.path.join(path, "table.dat")
    if not os.path.exists(dat_path):
      return False

    # Check the magic number
    if magic := try_read_magic_number_from_file_or_path(dat_path, count=4):
      return magic == b"\xbe\xbe\xbe\xbe"

    return False

  def open_dataset(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables=None,
    partition=None,
    epoch=None,
  ) -> Dataset:
    filename_or_obj = _normalize_path(filename_or_obj)
    store = MSv2Store.open(
      filename_or_obj, drop_variables=drop_variables, partition=partition, epoch=epoch
    )
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store)

  def open_datatree(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    **kwargs,
  ) -> DataTree:
    pass
