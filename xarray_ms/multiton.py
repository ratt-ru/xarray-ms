from __future__ import annotations

from typing import Any, Callable, ClassVar, Mapping, Tuple

from cacheout import Cache

from xarray_ms.utils import FrozenKey, normalise_args

FactoryFunctionT = Callable[..., Any]


def on_get_keep_alive(key, value, exists):
  """Re-insert on get to update the TTL"""
  if exists:
    # Re-insert to update the TTL
    Multiton._INSTANCE_CACHE.set(key, value)


def on_instance_delete(key, value, cause):
  """Invoke any instance close methods on deletion"""
  if hasattr(value, "close") and callable(value.close):
    value.close()


class Multiton:
  """Hashable and pickleable factory class
  for creating and caching an object instance"""

  _INSTANCE_CACHE: ClassVar[Cache] = Cache(
    maxsize=100, ttl=5 * 60, on_get=on_get_keep_alive, on_delete=on_instance_delete
  )
  _factory: FactoryFunctionT
  _args: Tuple[Any, ...]
  _kw: Mapping[str, Any]
  _key: FrozenKey

  def __init__(self, factory: FactoryFunctionT, *args: Any, **kw: Any):
    self._factory = factory
    self._args, self._kw = normalise_args(factory, args, kw)
    self._key = FrozenKey(factory, *self._args, **self._kw)

  @staticmethod
  def from_reduce_args(
    factory: FactoryFunctionT, args: Tuple[Any, ...], kw: Mapping[str, Any]
  ) -> Multiton:
    return Multiton(factory, *args, **kw)

  def __reduce__(
    self,
  ) -> Tuple[Callable, Tuple[Callable, Tuple[Any, ...], Mapping[str, Any]]]:
    return (self.from_reduce_args, (self._factory, self._args, self._kw))

  def __hash__(self) -> int:
    return hash(self._key)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Multiton):
      return NotImplemented
    return self._key == other._key

  @staticmethod
  def _create_instance(self) -> Any:
    return self._factory(*self._args, **self._kw)

  @property
  def instance(self) -> Any:
    """Create the object instance represented by the Multiton,
    or retrieved the cache instance"""
    return self._INSTANCE_CACHE.get(self, self._create_instance)

  def release(self) -> bool:
    """Evict any cached instance associated with this Multiton"""
    return self._INSTANCE_CACHE.delete(self) > 0
