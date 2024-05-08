from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
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
from xarray_ms.backend.msv2.structure import MSv2Structure
from xarray_ms.backend.msv2.table_factory import TableFactory

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray.backends.common import AbstractDataStore
  from xarray.core.dataset import Dataset
  from xarray.core.datatree import DataTree


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
