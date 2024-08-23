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

from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.backend.msv2.table_factory import TableFactory

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray.backends.common import AbstractDataStore
  from xarray.core.dataset import Dataset
  from xarray.core.datatree import DataTree


class MSv2Array(BackendArray):
  def __init__(self, table_factory, structure_factory, partition, column, shape, dtype):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition = partition
    self._column = column
    self.shape = shape
    self.dtype = np.dtype(dtype)

    assert len(shape) >= 2, "(time, baseline) needed"

  def __getitem__(self, key):
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key):
    rows = self._structure_factory().row_map(self._partition)[key[:2]]
    xkey = (rows.ravel(),) + key[2:]
    row_data = self._table_factory().getcol(self._column, xkey)
    return row_data.reshape(rows.shape + row_data.shape[1:])


class MSv2Store(AbstractWritableDataStore):
  """Store for reading and writing MSv2 data"""

  __slots__ = (
    "_table_factory",
    "_structure_factory",
    "_partition",
    "_epoch",
    "_auto_corrs",
    "_structure",
  )

  _table_factory: TableFactory
  _structure_factory: MSv2StructureFactory
  _partition: PartitionKeyT
  _autocorrs: bool
  _epoch: str

  def __init__(
    self,
    table_factory: TableFactory,
    structure_factory: MSv2StructureFactory,
    partition: PartitionKeyT,
    auto_corrs: bool,
    epoch: str,
  ):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition = partition
    self._auto_corrs = auto_corrs
    self._epoch = epoch

  @classmethod
  def open(
    cls,
    ms: str,
    drop_variables=None,
    partition: PartitionKeyT | None = None,
    auto_corrs: bool = True,
    epoch: str | None = None,
  ):
    if not isinstance(ms, str):
      raise ValueError("Measurement Sets paths must be strings")

    table_factory = TableFactory(
      Table.from_filename, ms, readonly=True, lockoptions="nolock"
    )

    structure_factory = MSv2StructureFactory(table_factory, auto_corrs)
    structure = structure_factory()

    if partition is None:
      partition = next(iter(structure.keys()))
      warnings.warn(f"No partition was supplied. Selected first partition {partition}")
    elif partition not in structure:
      raise ValueError(f"{partition} not in {list(structure.keys())}")

    return cls(
      table_factory,
      structure_factory,
      partition=partition,
      auto_corrs=auto_corrs,
      epoch=epoch or uuid4().hex[:8],
    )

  def close(self, **kwargs):
    pass

  def get_variables(self):
    structure = self._structure_factory()
    partition = structure[self._partition]
    (ntime,) = partition.time.shape
    nfreq = len(partition.chan_freq)
    npol = len(partition.corr_type)
    ant1, ant2 = structure.antenna_pairs
    nbl = structure.nbl
    assert (nbl,) == ant1.shape

    uvw = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition,
      "UVW",
      (ntime, nbl, 3),
      np.float64,
    )
    time_centroid = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition,
      "TIME_CENTROID",
      (ntime, nbl),
      np.float64,
    )
    exposure = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition,
      "EXPOSURE",
      (ntime, nbl),
      np.float64,
    )
    vis = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition,
      "DATA",
      (ntime, nbl, nfreq, npol),
      np.complex64,
    )
    flag = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition,
      "FLAG",
      (ntime, nbl, nfreq, npol),
      np.uint8,
    )
    weight = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition,
      "WEIGHT_SPECTRUM",
      (ntime, nbl, nfreq, npol),
      np.float32,
    )
    data_vars = [
      ("TIME_CENTROID", (("time", "baseline"), time_centroid, None)),
      ("EFFECTIVE_INTEGRATION_TIME", (("time", "baseline"), exposure, None)),
      ("UVW", (("time", "baseline", "uvw_label"), uvw, None)),
      ("VISIBILITY", (("time", "baseline", "frequency", "polarization"), vis, None)),
      ("FLAG", (("time", "baseline", "frequency", "polarization"), flag, None)),
      ("WEIGHT", (("time", "baseline", "frequency", "polarization"), weight, None)),
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
          partition.time,
          {
            "type": "time",
            "scale": "utc",
            "units": ["s"],
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
  open_dataset_parameters = ["filename_or_obj", "partition", "epoch"]
  description = "Opens v2 CASA Measurement Sets in Xarray"
  url = "https://link_to/your_backend/documentation"

  def guess_can_open(
    self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore
  ) -> bool:
    """Return true if this is a CASA table"""
    if not isinstance(filename_or_obj, (str, os.PathLike)):
      return False

    # CASA Tables are directories containing a table.dat file
    table_path = os.path.join(_normalize_path(filename_or_obj), "table.dat")
    if not os.path.exists(table_path):
      return False

    # Check the magic number
    if magic := try_read_magic_number_from_file_or_path(table_path, count=4):
      return magic == b"\xbe\xbe\xbe\xbe"

    return False

  def open_dataset(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables=None,
    partition=None,
    auto_corrs=True,
    epoch=None,
  ) -> Dataset:
    filename_or_obj = _normalize_path(filename_or_obj)
    store = MSv2Store.open(
      filename_or_obj,
      drop_variables=drop_variables,
      partition=partition,
      auto_corrs=auto_corrs,
      epoch=epoch,
    )
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store)

  def open_datatree(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    **kwargs,
  ) -> DataTree:
    pass
