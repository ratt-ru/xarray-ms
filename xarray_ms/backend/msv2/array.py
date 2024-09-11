from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
from xarray.backends import BackendArray
from xarray.core.indexing import IndexingSupport, explicit_indexing_adapter

if TYPE_CHECKING:
  import numpy.typing as npt

  from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
  from xarray_ms.backend.msv2.table_factory import TableFactory


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

  _table_factory: TableFactory
  _structure_factory: MSv2StructureFactory
  _partition: PartitionKeyT
  _column: str
  _shape: Tuple[int, ...]
  _dtype: npt.DTypeLike
  _default: Any | None
  _transform: Callable[[npt.NDArray], npt.NDArray] | None

  def __init__(
    self,
    table_factory: TableFactory,
    structure_factory: MSv2StructureFactory,
    partition: PartitionKeyT,
    column: str,
    shape: Tuple[int, ...],
    dtype: npt.DTypeLike,
    default: Any | None = None,
    transform: Callable[[npt.NDArray], npt.NDArray] | None = None,
  ):
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition = partition
    self._column = column
    self._default = default
    self._transform = transform
    self.shape = shape
    self.dtype = np.dtype(dtype)

    assert len(shape) >= 2, "(time, baseline) required"

  def __getitem__(self, key) -> npt.NDArray:
    return explicit_indexing_adapter(
      key, self.shape, IndexingSupport.OUTER, self._getitem
    )

  def _getitem(self, key) -> npt.NDArray:
    assert len(key) == len(self.shape)
    expected_shape = tuple(slice_length(k, s) for k, s in zip(key, self.shape))
    # Map the (time, baseline) coordinates onto row indices
    rows = self._structure_factory()[self._partition].row_map[key[:2]]
    xkey = (rows.ravel(),) + key[2:]
    row_shape = (rows.size,) + expected_shape[2:]
    result = np.full(row_shape, self._default, dtype=self.dtype)
    self._table_factory().getcol(self._column, xkey, result)
    result = result.reshape(rows.shape + expected_shape[2:])
    return self._transform(result) if self._transform else result

  def set_transform(self, transform: Callable[[npt.NDArray], npt.NDArray]):
    self._transform = transform
