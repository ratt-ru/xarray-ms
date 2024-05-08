from __future__ import annotations

import inspect
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple
from weakref import WeakValueDictionary, finalize

from xarray_ms.utils import FrozenKey

if TYPE_CHECKING:
  from arcae import Table as ArcaeTable

  FactoryFunctionT = Callable[..., ArcaeTable]

_TABLE_CACHE: WeakValueDictionary[Any, ArcaeTable] = WeakValueDictionary()
_TABLE_LOCK: Lock = Lock()


class TableProxyMetaClass(type):
  """Multiton pattern https://en.wikipedia.org/wiki/Multiton_pattern"""

  def __call__(cls, factory: FactoryFunctionT, *args: Any, **kw: Any):
    # Normalise positional arguments passed as keyword arguments
    if kw:
      argspec = inspect.getfullargspec(factory)
      for i, argname in enumerate(argspec.args):
        if argname in kw:
          args = args[:i] + (kw.pop(argname),) + args[i:]

    key = FrozenKey(cls, factory, *args, **kw)

    with _TABLE_LOCK:
      try:
        return _TABLE_CACHE[key]
      except KeyError:
        instance = type.__call__(cls, factory, *args, **kw)
        instance._key = key
        _TABLE_CACHE[key] = instance
        return instance


class TableProxy(metaclass=TableProxyMetaClass):
  _factory: FactoryFunctionT
  _args: Tuple[Any, ...]
  _kw: Dict[str, Any]
  _key: FrozenKey
  _table: ArcaeTable
  _lock: Lock

  def __init__(self, factory: FactoryFunctionT, *args: Any, **kw: Any):
    self._factory = factory
    self._args = args
    self._kw = kw
    self._lock = Lock()

  @staticmethod
  def from_reduce_args(
    factory: FactoryFunctionT, args: Tuple[Any, ...], kw: Dict[str, Any]
  ) -> TableProxy:
    return TableProxy(factory, *args, **kw)

  def __reduce__(self) -> Tuple:
    return (self.from_reduce_args, (self._factory, self._args, self._kw))

  @property
  def table(self) -> ArcaeTable:
    with self._lock:
      try:
        return self._table
      except AttributeError:
        self._table = table = self._factory(*self._args, **self._kw)
        finalize(self, table.close)
        return table
