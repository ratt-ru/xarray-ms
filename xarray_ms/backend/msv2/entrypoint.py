from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from importlib.metadata import version as importlib_version
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple
from uuid import uuid4

import xarray
from arcae.lib.arrow_tables import Table
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractWritableDataStore, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.utils import try_read_magic_number_from_file_or_path

from xarray_ms.backend.msv2.antenna_dataset_factory import AntennaDatasetFactory
from xarray_ms.backend.msv2.main_dataset_factory import MainDatasetFactory
from xarray_ms.backend.msv2.structure import (
  DEFAULT_PARTITION_COLUMNS,
  MSv2Structure,
  MSv2StructureFactory,
)
from xarray_ms.backend.msv2.table_factory import TableFactory
from xarray_ms.errors import InvalidPartitionKey
from xarray_ms.utils import format_docstring

if TYPE_CHECKING:
  from io import BufferedIOBase

  from xarray.backends.common import AbstractDataStore

  from xarray_ms.backend.msv2.structure import DEFAULT_PARTITION_COLUMNS, PartitionKeyT


def promote_chunks(
  structure: MSv2Structure, chunks: Dict | str | None
) -> Dict[PartitionKeyT, Dict[str, int]] | str | None:
  """Promotes a chunks dictionary into a
  :code:`{partition_key: chunks}` dictionary.
  """
  if chunks is None:
    return None

  if isinstance(chunks, str):
    return chunks

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
  partition_schema: List[str] | None,
  structure_factory: MSv2StructureFactory | None,
) -> Tuple[str, TableFactory, List[str], MSv2StructureFactory]:
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
  partition_schema = partition_schema or DEFAULT_PARTITION_COLUMNS
  structure_factory = structure_factory or MSv2StructureFactory(
    table_factory, partition_schema, epoch, auto_corrs=auto_corrs
  )
  return epoch, table_factory, partition_schema, structure_factory


class MSv2Store(AbstractWritableDataStore):
  """Store for reading and writing MSv2 data"""

  __slots__ = (
    "_table_factory",
    "_structure_factory",
    "_partition_schema",
    "_partition_key",
    "_preferred_chunks",
    "_auto_corrs",
    "_ninstances",
    "_epoch",
  )

  _table_factory: TableFactory
  _structure_factory: MSv2StructureFactory
  _partition_schema: List[str]
  _preferred_chunks: Dict[str, int]
  _partition: PartitionKeyT
  _autocorrs: bool
  _ninstances: int
  _epoch: str

  def __init__(
    self,
    table_factory: TableFactory,
    structure_factory: MSv2StructureFactory,
    partition_schema: List[str],
    partition_key: PartitionKeyT,
    preferred_chunks: Dict[str, int],
    auto_corrs: bool,
    ninstances: int,
    epoch: str,
  ):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition_schema = partition_schema
    self._partition_key = partition_key
    self._preferred_chunks = preferred_chunks
    self._auto_corrs = auto_corrs
    self._ninstances = ninstances
    self._epoch = epoch

  @classmethod
  def open(
    cls,
    ms: str,
    drop_variables=None,
    partition_schema: List[str] | None = None,
    partition_key: PartitionKeyT | None = None,
    preferred_chunks: Dict[str, int] | None = None,
    auto_corrs: bool = True,
    ninstances: int = 1,
    epoch: str | None = None,
    structure_factory: MSv2StructureFactory | None = None,
  ):
    if not isinstance(ms, str):
      raise ValueError("Measurement Sets paths must be strings")

    epoch, table_factory, partition_schema, structure_factory = initialise_default_args(
      ms,
      ninstances,
      auto_corrs,
      epoch,
      None,
      partition_schema,
      structure_factory,
    )

    # Resolve the user supplied partition key against actual
    # partition keys
    structure = structure_factory()
    partition_keys = structure.resolve_key(partition_key)
    if len(partition_keys) == 0:
      raise ValueError(f"{partition_key} not in {list(structure.keys())}")
    else:
      first_key = next(iter(partition_keys))
      if len(partition_keys) > 1:
        warnings.warn(
          f"{partition_key} matched multiple partitions. Selected {first_key}"
        )
      partition_key = first_key

    if preferred_chunks is None:
      preferred_chunks = {}

    return cls(
      table_factory,
      structure_factory,
      partition_schema=partition_schema,
      partition_key=partition_key,
      preferred_chunks=preferred_chunks,
      auto_corrs=auto_corrs,
      ninstances=ninstances,
      epoch=epoch,
    )

  def close(self, **kwargs):
    pass

  def get_variables(self):
    factory = MainDatasetFactory(
      self._partition_key,
      self._preferred_chunks,
      self._table_factory,
      self._structure_factory,
    )

    return factory.get_variables()

  def get_attrs(self):
    factory = MainDatasetFactory(
      self._partition_key,
      self._preferred_chunks,
      self._table_factory,
      self._structure_factory,
    )

    attrs = {
      "schema_version": "4.0.0",
      "creation_date": datetime.now(timezone.utc).isoformat(),
      "type": "visibility",
      "xarray_ms_version": importlib_version("xarray-ms"),
    }

    return dict(sorted({**attrs, **factory.get_attrs()}.items()))

  def get_dimensions(self):
    return None

  def get_encoding(self):
    return {}


class MSv2EntryPoint(BackendEntrypoint):
  open_dataset_parameters = [
    "filename_or_obj",
    "partition_schema",
    "partition_key",
    "preferred_chunks",
    "auto_corrs",
    "ninstances",
    "epoch",
    "structure_factory",
  ]
  description = "Opens v2 CASA Measurement Sets in Xarray"
  url = "https://xarray-ms.readthedocs.io/"

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

  @format_docstring(DEFAULT_PARTITION_COLUMNS=DEFAULT_PARTITION_COLUMNS)
  def open_dataset(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    partition_schema: List[str] | None = None,
    partition_key: PartitionKeyT | None = None,
    preferred_chunks: Dict[str, int] | None = None,
    auto_corrs: bool = True,
    ninstances: int = 8,
    epoch: str | None = None,
    structure_factory: MSv2StructureFactory | None = None,
  ) -> Dataset:
    """Create a :class:`~xarray.Dataset` presenting an MSv4 view
    over a partition of a MSv2 CASA Measurement Set

    Args:
      filename_or_obj: The path to the MSv2 CASA Measurement Set file.
      drop_variables: Variables to drop from the dataset.
      partition_schema: The columns to use for partitioning the Measurement set.
        Defaults to :code:`{DEFAULT_PARTITION_COLUMNS}`.
      partition_key: A key corresponding to an individual partition.
        For example :code:`(('DATA_DESC_ID', 0), ('FIELD_ID', 0))`.
        If :code:`None`, the first partition will be opened.
      preferred_chunks: The preferred chunks for each partition.
      auto_corrs: Include/Exclude auto-correlations.
      ninstances: The number of Measurement Set instances to open for parallel I/O.
      epoch: A unique string identifying the creation of this Dataset.
        This should not normally need to be set by the user
      structure_factory: A factory for creating MSv2Structure objects.
        This should not normally need to be set by the user

    Returns:
      A :class:`~xarray.Dataset` referring to the unique
      partition specified by :code:`partition_schema` and :code:`partition_key`.
    """
    filename_or_obj = _normalize_path(filename_or_obj)

    store = MSv2Store.open(
      filename_or_obj,
      drop_variables=drop_variables,
      partition_schema=partition_schema,
      partition_key=partition_key,
      preferred_chunks=preferred_chunks,
      auto_corrs=auto_corrs,
      ninstances=ninstances,
      epoch=epoch,
      structure_factory=structure_factory,
    )
    store_entrypoint = StoreBackendEntrypoint()
    return store_entrypoint.open_dataset(store)

  @format_docstring(DEFAULT_PARTITION_COLUMNS=DEFAULT_PARTITION_COLUMNS)
  def open_datatree(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    preferred_chunks: Dict[str, Any] | None = None,
    drop_variables: str | Iterable[str] | None = None,
    partition_schema: List[str] | None = None,
    auto_corrs: bool = True,
    ninstances: int = 8,
    epoch: str | None = None,
    **kwargs,
  ) -> DataTree:
    """Create a :class:`~xarray.core.datatree.DataTree` presenting an MSv4 view
    over multiple partitions of a MSv2 CASA Measurement Set.

    Args:
      filename_or_obj: The path to the MSv2 CASA Measurement Set file.
      preferred_chunks: Chunk sizes along each dimension,
        e.g. :code:`{{"time": 10, "frequency": 16}}`.
        Individual partitions can be chunked differently by
        partially (or fully) specifying a partition key: e.g.

        .. code-block:: python

          {{  # Applies to all partitions with the relevant DATA_DESC_ID
            (("DATA_DESC_ID", 0),): {{"time": 10, "frequency": 16}},
            (("DATA_DESC_ID", 1),): {{"time": 20, "frequency": 32}},
          }}
          {{  # Applies to all partitions with the relevant DATA_DESC_ID and FIELD_ID
            (("DATA_DESC_ID", 0), ('FIELD_ID', 1)): {{"time": 10, "frequency": 16}},
            (("DATA_DESC_ID", 1), ('FIELD_ID', 0)): {{"time": 20, "frequency": 32}},
          }}
          {{  # String variants
            "DATA_DESC_ID=0,FIELD_ID=0": {{"time": 10, "frequency": 16}},
            "D=0,F=1": {{"time": 20, "frequency": 32}},
          }}

        .. note:: xarray's reserved ``chunks`` argument must be specified in order
          to enable this functionality and enable fine-grained chunking
          in Datasets and DataTrees.
          See xarray's backend documentation on
          `Preferred chunk sizes <preferred_chunk_sizes_>`_
          for more information.

      drop_variables: Variables to drop from the dataset.
      partition_schema: The columns to use for partitioning the Measurement set.
        Defaults to :code:`{DEFAULT_PARTITION_COLUMNS}`.
      auto_corrs: Include/Exclude auto-correlations.
      ninstances: The number of Measurement Set instances to open for parallel I/O.
      epoch: A string uniquely identifying this Dataset.
        This should not normally be set by the user

    Returns:
      An xarray :class:`~xarray.core.datatree.DataTree`

    .. _preferred_chunk_sizes: https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html#preferred-chunk-sizes
    """
    groups_dict = self.open_groups_as_dict(
      filename_or_obj,
      drop_variables=drop_variables,
      partition_schema=partition_schema,
      preferred_chunks=preferred_chunks,
      auto_corrs=auto_corrs,
      ninstances=ninstances,
      epoch=epoch,
      **kwargs,
    )

    return DataTree.from_dict(groups_dict)

  @format_docstring(DEFAULT_PARTITION_COLUMNS=DEFAULT_PARTITION_COLUMNS)
  def open_groups_as_dict(
    self,
    filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    *,
    drop_variables: str | Iterable[str] | None = None,
    partition_schema: List[str] | None = None,
    preferred_chunks: Dict[str, int] | None = None,
    auto_corrs: bool = True,
    ninstances: int = 8,
    epoch: str | None = None,
    structure_factory: MSv2StructureFactory | None = None,
    **kwargs,
  ) -> Dict[str, Dataset]:
    """Create a dictionary of :class:`~xarray.Dataset` presenting an MSv4 view
    over a partition of a MSv2 CASA Measurement Set"""

    if isinstance(filename_or_obj, os.PathLike):
      ms = str(filename_or_obj)
    elif isinstance(filename_or_obj, str):
      ms = filename_or_obj
    else:
      raise ValueError("Measurement Set paths must be strings")

    epoch, _, partition_schema, structure_factory = initialise_default_args(
      ms, ninstances, auto_corrs, epoch, None, partition_schema, None
    )

    # /path/to/some_name.ext -> some_name
    ms_name, _ = os.path.splitext(os.path.basename(ms.rstrip(os.path.sep)))

    structure = structure_factory()
    datasets = {}
    pchunks = promote_chunks(structure, preferred_chunks)

    for p, partition_key in enumerate(structure):
      ds = xarray.open_dataset(
        ms,
        drop_variables=drop_variables,
        engine="xarray-ms:msv2",
        partition_schema=partition_schema,
        partition_key=partition_key,
        preferred_chunks=pchunks[partition_key]
        if isinstance(pchunks, Mapping)
        else pchunks,
        auto_corrs=auto_corrs,
        ninstances=ninstances,
        epoch=epoch,
        structure_factory=structure_factory,
        **kwargs,
      )

      antenna_factory = AntennaDatasetFactory(structure_factory)

      path = f"{ms_name}_partition_{p:03}"
      datasets[path] = ds
      datasets[f"{path}/antenna_xds"] = antenna_factory.get_dataset()

    return datasets
