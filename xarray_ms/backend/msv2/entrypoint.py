from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple
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


DEFAULT_PARTITION_COLUMNS: List[str] = [
  "DATA_DESC_ID",
  "FIELD_ID",
  "OBSERVATION_ID",
]


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


def initialise_default_args(
  ms: str,
  ninstances: int,
  auto_corrs: bool,
  epoch: str | None,
  table_factory: TableFactory | None,
  partition_columns: List[str] | None,
  partition_key: PartitionKeyT | None,
  structure_factory: MSv2StructureFactory | None,
) -> Tuple[str, TableFactory, List[str], PartitionKeyT, MSv2StructureFactory]:
  """
  Ensures consistency when initialising default arguments from multiple locations
  """
  if not os.path.exists(ms):
    raise ValueError(f"MS {ms} does not exist")

  table_factory = table_factory or TableFactory(
    Table.from_filename,
    ms,
    ninstances=ninstances,
    readonly=True,
    lockoptions="nolock",
  )
  epoch = epoch or uuid4().hex[:8]
  partition_columns = partition_columns or DEFAULT_PARTITION_COLUMNS
  structure_factory = structure_factory or MSv2StructureFactory(
    table_factory, partition_columns, auto_corrs=auto_corrs
  )
  structure = structure_factory()
  if partition_key is None:
    partition_key = next(iter(structure.keys()))
    warnings.warn(
      f"No partition_key was supplied. Selected first partition {partition_key}"
    )
  elif partition_key not in structure:
    raise ValueError(f"{partition_key} not in {list(structure.keys())}")

  return epoch, table_factory, partition_columns, partition_key, structure_factory


class MSv2Store(AbstractWritableDataStore):
  """Store for reading and writing MSv2 data"""

  __slots__ = (
    "_table_factory",
    "_structure_factory",
    "_partition_columns",
    "_partition_key",
    "_auto_corrs",
    "_ninstances",
    "_epoch",
  )

  _table_factory: TableFactory
  _structure_factory: MSv2StructureFactory
  _partition_columns: List[str]
  _partition: PartitionKeyT
  _autocorrs: bool
  _ninstances: int
  _epoch: str

  def __init__(
    self,
    table_factory: TableFactory,
    structure_factory: MSv2StructureFactory,
    partition_columns: List[str],
    partition_key: PartitionKeyT,
    auto_corrs: bool,
    ninstances: int,
    epoch: str,
  ):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition_columns = partition_columns
    self._partition_key = partition_key
    self._auto_corrs = auto_corrs
    self._ninstances = ninstances
    self._epoch = epoch

  @classmethod
  def open(
    cls,
    ms: str,
    drop_variables=None,
    partition_columns: List[str] | None = None,
    partition_key: PartitionKeyT | None = None,
    auto_corrs: bool = True,
    ninstances: int = 1,
    epoch: str | None = None,
    structure_factory: MSv2StructureFactory | None = None,
  ):
    if not isinstance(ms, str):
      raise ValueError("Measurement Sets paths must be strings")

    epoch, table_factory, partition_columns, partition_key, structure_factory = (
      initialise_default_args(
        ms,
        ninstances,
        auto_corrs,
        epoch,
        None,
        partition_columns,
        partition_key,
        structure_factory,
      )
    )

    return cls(
      table_factory,
      structure_factory,
      partition_columns=partition_columns,
      partition_key=partition_key,
      auto_corrs=auto_corrs,
      ninstances=ninstances,
      epoch=epoch,
    )

  def close(self, **kwargs):
    pass

  def get_variables(self):
    return MainDatasetFactory(
      self._partition_key, self._table_factory, self._structure_factory
    ).get_variables()

  def get_attrs(self):
    try:
      ddid = next(iter(v for k, v in self._partition_key if k == "DATA_DESC_ID"))
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
    "partition_columns" "partition_key",
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
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    partition_columns=None,
    partition_key=None,
    auto_corrs=True,
    ninstances=8,
    epoch=None,
    structure_factory=None,
  ) -> Dataset:
    filename_or_obj = _normalize_path(filename_or_obj)
    store = MSv2Store.open(
      filename_or_obj,
      drop_variables=drop_variables,
      partition_columns=partition_columns,
      partition_key=partition_key,
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
    partition_columns=None,
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

    epoch, _, partition_columns, _, structure_factory = initialise_default_args(
      ms, ninstances, auto_corrs, epoch, None, partition_columns, None, None
    )

    structure = structure_factory()
    datasets = {}
    chunks = kwargs.pop("chunks", None)
    pchunks = promote_chunks(structure, chunks)

    for i, partition_key in enumerate(structure):
      ds = xarray.open_dataset(
        ms,
        drop_variables=drop_variables,
        partition_columns=partition_columns,
        partition_key=partition_key,
        auto_corrs=auto_corrs,
        ninstances=ninstances,
        epoch=epoch,
        structure_factory=structure_factory,
        chunks=None if pchunks is None else pchunks[partition_key],
        **kwargs,
      )
      datasets[str(i)] = ds

    return DataTree.from_dict(datasets)
