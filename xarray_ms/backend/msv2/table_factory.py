from __future__ import annotations

import inspect
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from xarray.core.utils import ReprObject

from xarray_ms.utils import FrozenKey

if TYPE_CHECKING:
  from arcae import Table as ArcaeTable

  FactoryFunctionT = Callable[..., ArcaeTable]


class TableFactory:
  """Hashable Callable for creating an Arcae Table"""

  _factory: FactoryFunctionT
  _args: Tuple[Any, ...]
  _kw: Dict[str, Any]
  _lock: Lock
  _key: FrozenKey
  _table: ArcaeTable

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

  def __call__(self, *args, **kw) -> ArcaeTable:
    with self._lock:
      try:
        return self._table
      except AttributeError:
        args += self._args
        kw.update(self._kw)

        if "mode" in kw:
          mode = kw.pop("mode")
          if not isinstance(mode, ReprObject) and mode.value != "<unused>":
            raise NotImplementedError(f"mode argument '{mode}' is not supported")

        self._table = table = self._factory(*args, **kw)
        return table
