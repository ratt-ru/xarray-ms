from collections.abc import Mapping, Sequence
from typing import Any


def id_string(arg: Any) -> str:
  """Converts an argument into a string that can be used as a pytest identifier."""
  if isinstance(arg, str):
    return arg
  elif isinstance(arg, Sequence):
    return f"[{','.join(list(id_string(v) for v in arg))}]"
  elif isinstance(arg, Mapping):
    bits = ",".join(f"{id_string(k)}={id_string(v)}" for k, v in arg.items())
    return f"[{bits}]"
  else:
    return str(arg)
