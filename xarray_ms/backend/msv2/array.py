from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
from xarray.backends import BackendArray
from xarray.core.indexing import IndexingSupport, explicit_indexing_adapter

if TYPE_CHECKING:
  import numpy.typing as npt

  from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
  from xarray_ms.multiton import Multiton

  TransformerT = Callable[[npt.NDArray], npt.NDArray] | None


def slice_length(s, max_len):
  if isinstance(s, np.ndarray):
    if s.ndim != 1:
      raise NotImplementedError("Slicing with non-1D numpy arrays")
    return len(s)

  start, stop, step = s.indices(max_len)
  if step != 1:
    raise NotImplementedError(f"Slicing with steps {s} other than 1 not supported")
  return stop - start


class MSv2Array(BackendArray):
  """Backend array containing functionality for reading an MSv2 column"""

  _table_factory: Multiton
  _structure_factory: MSv2StructureFactory
  _partition: PartitionKeyT
  _column: str
  _shape: Tuple[int, ...]
  _dtype: npt.DTypeLike
  _default: Any | None
  _transform: TransformerT

  def __init__(
    self,
    table_factory: Multiton,
    structure_factory: MSv2StructureFactory,
    partition: PartitionKeyT,
    column: str,
    shape: Tuple[int, ...],
    dtype: npt.DTypeLike,
    default: Any | None = None,
    transform: TransformerT = None,
  ):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition = partition
    self._column = column
    self._default = default
    self._transform = transform
    self.shape = shape
    self.dtype = np.dtype(dtype)

    assert len(shape) >= 2, "(time, baseline_ids) required"

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    assert len(key) == len(self.shape)
    expected_shape = tuple(slice_length(k, s) for k, s in zip(key, self.shape))
    # Map the (time, baseline_id) coordinates onto row indices
    rows = self._structure_factory.instance[self._partition].row_map[key[:2]]
    row_key = (rows.ravel(),) + key[2:]
    row_shape = (rows.size,) + expected_shape[2:]
    result = np.full(row_shape, self._default, dtype=self.dtype)
    self._table_factory.instance.getcol(self._column, row_key, result)
    result = result.reshape(rows.shape + expected_shape[2:])
    return self._transform(result) if self._transform else result

  @property
  def transform(self) -> TransformerT:
    return self._transform

  @transform.setter
  def transform(self, value: TransformerT):
    self._transform = value


class BroadcastMSv2Array(MSv2Array):
  """Backend Array that reads an MSv2 column and
  broadcasts the result up to a desired shape"""

  _low_resolution_shape: Tuple[int, ...]
  _low_resolution_index: Tuple[slice | None]

  def __init__(
    self,
    table_factory: Multiton,
    structure_factory: MSv2StructureFactory,
    partition: PartitionKeyT,
    column: str,
    shape: Tuple[int, ...],
    low_resolution_shape: Tuple[int, ...],
    low_resolution_index: Tuple[slice | None],
    dtype: npt.DTypeLike,
    default: Any | None = None,
    transform: TransformerT = None,
  ):
    super().__init__(
      table_factory,
      structure_factory,
      partition,
      column,
      shape,
      dtype,
      default,
      transform,
    )

    self._low_resolution_shape = low_resolution_shape
    self._low_resolution_index = low_resolution_index
    assert low_resolution_shape[:2] == self.shape[:2]
    assert len(low_resolution_index) == len(self.shape)

  def _getitem(self, key) -> npt.NDArray:
    assert len(key) == len(self.shape)
    low_ndim = len(self._low_resolution_shape)
    high_res_shape = tuple(slice_length(k, s) for k, s in zip(key, self.shape))

    # Map the (time, baseline_id) coordinates onto row indices
    rows = self._structure_factory.instance[self._partition].row_map[key[:2]]
    row_key = (rows.ravel(),) + key[2:low_ndim]
    row_shape = (rows.size,) + self._low_resolution_shape[2:]
    result = np.full(row_shape, self._default, dtype=self.dtype)
    self._table_factory.instance.getcol(self._column, row_key, result)
    result = result.reshape(rows.shape + row_shape[1:])
    # Maybe transform the result
    if self._transform is not None:
      result = self._transform(result)

    # Broadcast to high-resolution shape
    return np.broadcast_to(result[self._low_resolution_index], high_res_shape)
