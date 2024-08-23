from __future__ import annotations

import inspect
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Tuple

from cacheout import Cache

from xarray_ms.utils import FrozenKey

if TYPE_CHECKING:
  from arcae import Table as ArcaeTable

  FactoryFunctionT = Callable[..., ArcaeTable]


def on_get_keep_alive(key, value, exists):
  """Re-insert on get to update the TTL"""
  if exists:
    # Re-insert to update the TTL
    TableFactory._TABLE_CACHE.set(key, value)


def on_table_delete(key, value, cause):
  """Close arcae tables on cache eviction"""
  value.close()


class TableFactory:
  """Hashable, callable and pickleable factory class for creating an Arcae Table"""

  _TABLE_CACHE: ClassVar[Cache] = Cache(
    maxsize=100, ttl=5 * 60, on_get=on_get_keep_alive, on_delete=on_table_delete
  )
  _factory: FactoryFunctionT
  _args: Tuple[Any, ...]
  _kw: Dict[str, Any]
  _lock: Lock
  _key: FrozenKey

  def __init__(self, factory: FactoryFunctionT, *args: Any, **kw: Any):
    # Normalise args with any arguments present in kw
    if kw:
      argspec = inspect.getfullargspec(factory)
      for i, argname in enumerate(argspec.args):
        if argname in kw:
          args = args[:i] + (kw.pop(argname),) + args[i:]

    self._factory = factory
    self._args = args
    self._kw = kw
    self._lock = Lock()
    self._key = FrozenKey(factory, *args, **kw)

  @staticmethod
  def from_reduce_args(
    factory: FactoryFunctionT, args: Tuple[Any, ...], kw: Dict[str, Any]
  ) -> TableFactory:
    return TableFactory(factory, *args, **kw)

  def __reduce__(
    self,
  ) -> Tuple[Callable, Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]]:
    return (self.from_reduce_args, (self._factory, self._args, self._kw))

  def __hash__(self) -> int:
    return hash(self._key)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, TableFactory):
      return NotImplemented
    return self._key == other._key

  @staticmethod
  def create_table(table_factory, args, kw) -> ArcaeTable:
    args += table_factory._args
    kw.update(table_factory._kw)
    return table_factory._factory(*args, **kw)

  def __call__(self, *args, **kw) -> ArcaeTable:
    # assert not args and not kw
    return self._TABLE_CACHE.get(self, partial(self.create_table, args=args, kw=kw))
