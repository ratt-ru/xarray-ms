import inspect
from collections.abc import Callable, Hashable, Mapping, Sequence, Set
from typing import Any, Tuple

from numpy import ndarray


def freeze(arg: Any) -> Any:
  """Recursively convert argument into an immutable representation"""
  if isinstance(arg, (str, bytes)):
    # str and bytes are sequences, return early to avoid tuplification
    return arg

  if isinstance(arg, Sequence):
    return tuple(map(freeze, arg))
  elif isinstance(arg, Set):
    return frozenset(map(freeze, arg))
  elif isinstance(arg, Mapping):
    return frozenset((k, freeze(v)) for k, v in arg.items())
  elif isinstance(arg, ndarray):
    return (arg.data.tobytes(), arg.shape, arg.dtype.char)
  else:
    return arg


class FrozenKey(Hashable):
  """Converts args and kwargs into an immutable, hashable representation"""

  __slots__ = ("_frozen", "_hashvalue")
  _frozen: Tuple[Any, ...]
  _hashvalue: int

  def __init__(self, *args, **kw):
    self._frozen = freeze(args + (kw,))
    self._hashvalue = hash(self._frozen)

  @property
  def frozen(self) -> Tuple[Any, ...]:
    return self._frozen

  def __hash__(self) -> int:
    return self._hashvalue

  def __eq__(self, other) -> bool:
    if not isinstance(other, FrozenKey):
      return NotImplemented
    return self._hashvalue == other._hashvalue and self._frozen == other._frozen

  def __str__(self) -> str:
    return f"FrozenKey({self._hashvalue})"


def normalise_args(
  factory: Callable, args, kw
) -> Tuple[Tuple[Any, ...], Mapping[str, Any]]:
  """Normalise args and kwargs into a hashable representation

  Args:
    factory: factory function
    args: positional arguments
    kw: keyword arguments

  Returns:
    tuple containing the normalised positional arguments and keyword arguments
  """
  spec = inspect.getfullargspec(factory)
  args = list(args)

  for i, arg in enumerate(spec.args):
    if i < len(args):
      continue
    elif arg in kw:
      args.append(kw.pop(arg))
    elif spec.defaults and len(spec.args) - len(spec.defaults) <= i:
      default = spec.defaults[i - (len(spec.args) - len(spec.defaults))]
      args.append(default)

  return tuple(args), kw


def format_docstring(**subs):
  def decorator(o):
    o.__doc__ = o.__doc__.format(**subs)
    return o

  return decorator
