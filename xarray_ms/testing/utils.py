from collections.abc import Mapping, Sequence
from typing import Any

from xarray import Dataset, DataTree

from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


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


def prune_datetime_structures(xarray_structure: DataTree | DataTree):
  """Prune date values out of Dataset or DataTrees for equality testing"""

  def prune_vis_dataset(ds):
    ds.attrs.pop("creation_date", None)

    for dg in ds.attrs.get("data_groups", {}).values():
      dg.pop("date", None)

  if isinstance(xarray_structure, Dataset):
    prune_vis_dataset(xarray_structure)
  elif isinstance(xarray_structure, DataTree):
    for node in xarray_structure.subtree:
      if node.attrs.get("type") in CORRELATED_DATASET_TYPES:
        prune_vis_dataset(node.ds)

  return xarray_structure
