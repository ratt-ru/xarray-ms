from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
from xarray.backends import BackendArray
from xarray.core.indexing import IndexingSupport, explicit_indexing_adapter

if TYPE_CHECKING:
  import numpy.typing as npt

  from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
  from xarray_ms.multiton import Multiton

  TransformerT = Callable[[npt.NDArray], npt.NDArray]


def slice_length(s: npt.NDArray | slice, max_len) -> int:
  if isinstance(s, np.ndarray):
    if s.ndim != 1:
      raise NotImplementedError("Slicing with non-1D numpy arrays")
    return len(s)

  start, stop, step = s.indices(max_len)
  if step != 1:
    raise NotImplementedError(f"Slicing with steps {s} other than 1 not supported")
  return stop - start


class MSv2Array(BackendArray):
  """Base MSv2Array backend array class,
  containing required shape and data type"""

  __slots__ = ("shape", "dtype")

  shape: Tuple[int, ...]
  dtype: npt.DTypeLike

  def __init__(self, shape: Tuple[int, ...], dtype: npt.DTypeLike):
    self.shape = shape
    self.dtype = dtype

  def __getitem__(self, key) -> npt.NDArray:
    raise NotImplementedError

  @property
  def transform(self) -> TransformerT | None:
    raise NotImplementedError

  @transform.setter
  def transform(self, value: TransformerT) -> None:
    raise NotImplementedError


class MainMSv2Array(MSv2Array):
  """Backend array containing functionality for reading an MSv2 column
  from the MAIN table. Columns are assumed to have ("time", "baseline_id")
  as the first dimensions. These are mapped onto the "row" dimension
  via the partition row map"""

  __slots__ = (
    "_table_factory",
    "_structure_factory",
    "_partition",
    "_column",
    "_default",
    "_transform",
  )

  _table_factory: Multiton
  _structure_factory: MSv2StructureFactory
  _partition: PartitionKeyT
  _column: str
  _default: Any | None
  _transform: TransformerT | None

  def __init__(
    self,
    table_factory: Multiton,
    structure_factory: MSv2StructureFactory,
    partition: PartitionKeyT,
    column: str,
    shape: Tuple[int, ...],
    dtype: npt.DTypeLike,
    default: Any | None = None,
    transform: TransformerT | None = None,
  ):
    super().__init__(shape, dtype)
    self._table_factory = table_factory
    self._structure_factory = structure_factory
    self._partition = partition
    self._column = column
    self._default = default
    self._transform = transform

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
  def transform(self) -> TransformerT | None:
    return self._transform

  @transform.setter
  def transform(self, value: TransformerT) -> None:
    self._transform = value


class BroadcastMSv2Array(MSv2Array):
  """Broadcasts a MAIN table MSv2 Column up to an
  MSv4 column. This can be inefficient for example,
  if multiple frequency chunks are read for the same
  ("time", "baseline_id") range as the same
  low resolution data can be read multiple times.

  However, this should be no worse than reading the
  data for a full resolution column.

  This is primarily useful for falling back to the
  WEIGHT column when WEIGHT_SPECTRUM is missing, or
  FLAG_ROW if FLAG is missing.
  """

  __slots__ = ("_low_res_array", "_low_res_index")

  _low_res_array: MSv2Array
  _low_res_index: Tuple[slice | None, ...]
  shape: Tuple[int, ...]

  def __init__(
    self,
    low_res_array: MSv2Array,
    low_res_index: Tuple[slice | None, ...],
    high_res_shape: Tuple[int, ...],
  ):
    self._low_res_array = low_res_array
    self._low_res_index = low_res_index
    self.shape = high_res_shape

  @property
  def dtype(self):
    return self._low_res_array.dtype

  @property
  def transform(self) -> TransformerT | None:
    return self._low_res_array.transform

  @transform.setter
  def transform(self, value: TransformerT) -> None:
    self._low_res_array.transform = value

  def __getitem__(self, key) -> npt.NDArray:
    low_res_data = self._low_res_array.__getitem__(key)
    low_res_data = low_res_data[self._low_res_index]
    return np.broadcast_to(low_res_data, self.shape)
