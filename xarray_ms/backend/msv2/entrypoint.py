from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable
from uuid import uuid4

import xarray
from arcae.lib.arrow_tables import Table
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractWritableDataStore, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.datatree import DataTree
from xarray.core.utils import try_read_magic_number_from_file_or_path

from xarray_ms.backend.msv2.antenna_dataset_factory import AntennaDatasetFactory
from xarray_ms.backend.msv2.main_dataset_factory import MainDatasetFactory
from xarray_ms.backend.msv2.structure import (
  MSv2Structure,
  MSv2StructureFactory,
)
from xarray_ms.backend.msv2.table_factory import TableFactory
from xarray_ms.errors import InvalidPartitionKey

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray.backends.common import AbstractDataStore
  from xarray.core.dataset import Dataset
  from xarray.core.datatree import DataTree

  from xarray_ms.backend.msv2.structure import PartitionKeyT


def table_factory_factory(ms: str, ninstances: int) -> TableFactory:
  """
  Ensures consistency when creating a TableFactory.
  Multiple calls to this method with the same argument will
  resolve to the same cached instance.
  """
  return TableFactory(
    Table.from_filename,
    ms,
    ninstances=ninstances,
    readonly=True,
    lockoptions="nolock",
  )


def promote_chunks(
  structure: MSv2Structure, chunks: Dict | None
) -> Dict[PartitionKeyT, Dict[str, int]] | None:
  """Promotes a chunks dictionary into a
  :code:`{partition_key: chunks}` dictionary.
  """
  if chunks is None:
    return None

  # Base case, no chunking
  return_chunks: Dict[PartitionKeyT, Dict[str, int]] = {k: {} for k in structure.keys()}

  if all(isinstance(k, str) for k in chunks.keys()):
    # All keys are strings, try promote them to partition keys
    # keys, may resolve to multiple partition keys
    try:
      crkeys = list(map(structure.resolve_key, chunks.keys()))
    except InvalidPartitionKey:
      # Apply a chunk dictionary to all partitions in the structure
      return {k: chunks for k in structure.keys()}
    else:
      return_chunks.update((k, v) for rk, v in zip(crkeys, chunks.values()) for k in rk)
  else:
    for k, v in chunks.items():
      rkeys = structure.resolve_key(k)
      return_chunks.update((k, v) for k in rkeys)

  return return_chunks


class MSv2Store(AbstractWritableDataStore):
  """Store for reading and writing MSv2 data"""

  __slots__ = (
    "_table_factory",
    "_structure_factory",
    "_partition",
    "_auto_corrs",
    "_ninstances",
    "_epoch",
  )

  _table_factory: TableFactory
  _structure_factory: MSv2StructureFactory
  _partition: PartitionKeyT
  _autocorrs: bool
  _ninstances: int
  _epoch: str

  def __init__(
    self,
    table_factory: TableFactory,
    structure_factory: MSv2StructureFactory,
    partition: PartitionKeyT,
    auto_corrs: bool,
    ninstances: int,
    epoch: str,
  ):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition = partition
    self._auto_corrs = auto_corrs
    self._ninstances = ninstances
    self._epoch = epoch

  @classmethod
  def open(
    cls,
    ms: str,
    drop_variables=None,
    partition: PartitionKeyT | None = None,
    auto_corrs: bool = True,
    ninstances: int = 1,
    epoch: str | None = None,
    structure_factory: MSv2StructureFactory | None = None,
  ):
    if not isinstance(ms, str):
      raise ValueError("Measurement Sets paths must be strings")

    table_factory = table_factory_factory(ms, ninstances)
    epoch = epoch or uuid4().hex[:8]
    structure_factory = structure_factory or MSv2StructureFactory(
      table_factory, auto_corrs
    )
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
      ninstances=ninstances,
      epoch=epoch,
    )

  def close(self, **kwargs):
    pass

  def get_variables(self):
    return MainDatasetFactory(
      self._partition, self._table_factory, self._structure_factory
    ).get_variables()

  def get_attrs(self):
    try:
      ddid = next(iter(v for k, v in self._partition if k == "DATA_DESC_ID"))
    except StopIteration:
      raise KeyError("DATA_DESC_ID not found in partition")

    antenna_factory = AntennaDatasetFactory(self._structure_factory)
    ds = antenna_factory.get_dataset()

    return {
      "antenna_xds": ds,
      "version": "0.0.1",
      "creation_date": datetime.now(timezone.utc).isoformat(),
      "data_description_id": ddid,
    }

  def get_dimensions(self):
    return None

  def get_encoding(self):
    return {}


class MSv2PartitionEntryPoint(BackendEntrypoint):
  open_dataset_parameters = [
    "filename_or_obj",
    "partition",
    "auto_corrs",
    "ninstances",
    "epoch",
    "structure_factory",
  ]
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
    filename_or_obj: str
    | os.PathLike[Any]
    | BufferedIOBase
    | AbstractDataStore
    | TableFactory,
    *,
    drop_variables: str | Iterable[str] | None = None,
    partition=None,
    auto_corrs=True,
    ninstances=8,
    epoch=None,
    structure_factory=None,
  ) -> Dataset:
    filename_or_obj = _normalize_path(filename_or_obj)
    store = MSv2Store.open(
      filename_or_obj,
      drop_variables=drop_variables,
      partition=partition,
      auto_corrs=auto_corrs,
      ninstances=ninstances,
      epoch=epoch,
      structure_factory=structure_factory,
    )
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store)

  def open_datatree(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    auto_corrs=True,
    ninstances=8,
    epoch=None,
    **kwargs,
  ) -> DataTree:
    if isinstance(filename_or_obj, os.PathLike):
      ms = str(filename_or_obj)
    elif isinstance(filename_or_obj, str):
      ms = filename_or_obj
    else:
      raise ValueError("Measurement Set paths must be strings")

    table_factory = table_factory_factory(ms, ninstances)
    structure_factory = MSv2StructureFactory(table_factory, auto_corrs=auto_corrs)
    structure = structure_factory()

    datasets = {}
    chunks = kwargs.pop("chunks", None)
    pchunks = promote_chunks(structure, chunks)

    for i, partition_key in enumerate(structure):
      ds = xarray.open_dataset(
        ms,
        drop_variables=drop_variables,
        partition=partition_key,
        auto_corrs=auto_corrs,
        ninstances=ninstances,
        epoch=epoch,
        structure_factory=structure_factory,
        chunks=None if pchunks is None else pchunks[partition_key],
        **kwargs,
      )
      datasets[str(i)] = ds

    return DataTree.from_dict(datasets)
