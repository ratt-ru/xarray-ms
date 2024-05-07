from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray.core.indexing as indexing
from xarray.backends import BackendArray

if TYPE_CHECKING:
  from xarray.core.types import _ShapeLike

  from xarray_ms.backend.msv2.partition import PartitionIndex
  from xarray_ms.backend.msv2.table_proxy import TableProxy


class MSv4ColumnAdapter(BackendArray):
  _proxy: TableProxy
  _column: str
  _index: PartitionIndex
  _shape: _ShapeLike
  _dtype: npt.DTypeLike

  def __init__(self, proxy: TableProxy, column: str, index: PartitionIndex):
    self._proxy = proxy
    self._column = column
    self._index = index
    self._shape = (20, 14, 64, 4)
    self._dtype = np.complex64

  def __getitem__(self, key: indexing.ExplicitIndexer) -> npt.ArrayLike:
    pass

  def __setitem__(self, key, value: npt.ArrayLike) -> None:
    pass

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape
