from typing import Any

from xarray.core.datatree import DataTree
from xarray.core.extensions import (
  register_datatree_accessor,
)


@register_datatree_accessor("subtable")
class SubTableAccessor:
  def __init__(self, node: DataTree):
    self.node = node

  @property
  def antenna(self) -> DataTree:
    """Returns the antenna dataset"""

    try:
      link = self.node.attrs["antenna_xds_link"]
    except KeyError:
      raise ValueError("antenna_xds_link not found")
    else:
      return self.node.root[link]

  def __getitem__(self, key: Any) -> Any:
    try:
      link = self.node.attrs[key]
    except KeyError:
      raise ValueError(f"{key} link attribute not found")
    else:
      return self.node.root[link]
