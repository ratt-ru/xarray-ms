from collections.abc import Collection, Mapping, Set
from typing import Any


def freeze(arg: Any):
  """Convert python types"""
  if isinstance(arg, (str, bytes)):
    return arg
  if isinstance(arg, (Set, Collection)):
    return frozenset(map(freeze, arg))
  elif isinstance(arg, Mapping):
    return frozenset((k, map(freeze, v)) for k, v in arg.items())
  else:
    return arg
