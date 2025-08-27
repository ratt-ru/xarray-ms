from typing import Dict

from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.multiton import Multiton


class DatasetFactory:
  _partition_key: PartitionKeyT
  _structure_factory: MSv2StructureFactory
  _subtable_factories: Dict[str, Multiton]

  def __init__(
    self,
    partition_key: PartitionKeyT,
    structure_factory: MSv2StructureFactory,
    subtable_factories: Dict[str, Multiton],
  ):
    self._partition_key = partition_key
    self._structure_factory = structure_factory
    self._subtable_factories = subtable_factories
