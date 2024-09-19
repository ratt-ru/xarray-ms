import dataclasses
import warnings
from typing import Any, Mapping, Tuple, Type

import numpy as np
from xarray import Variable
from xarray.coding.variables import unpack_for_decoding
from xarray.core.indexing import LazilyIndexedArray
from xarray.core.utils import FrozenDict

from xarray_ms.backend.msv2.array import MSv2Array
from xarray_ms.backend.msv2.encoders import (
  CasaCoder,
  QuantityCoder,
  TimeCoder,
)
from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.backend.msv2.table_factory import TableFactory
from xarray_ms.errors import IrregularGridWarning


@dataclasses.dataclass
class MSv2ColumnSchema:
  name: str
  dims: Tuple[str, ...]
  default: Any = None
  coder: Type[CasaCoder] | None = None


MSV4_to_MSV2_COLUMN_SCHEMAS = {
  "TIME": MSv2ColumnSchema("TIME", (), np.nan, TimeCoder),
  "INTEGRATION_TIME": MSv2ColumnSchema("INTERVAL", (), np.nan, QuantityCoder),
  "TIME_CENTROID": MSv2ColumnSchema("TIME_CENTROID", (), np.nan, TimeCoder),
  "EFFECTIVE_INTEGRATION_TIME": MSv2ColumnSchema("EXPOSURE", (), np.nan, QuantityCoder),
  "UVW": MSv2ColumnSchema("UVW", ("uvw_label",), np.nan, None),
  "FLAG": MSv2ColumnSchema("FLAG", ("frequency", "polarization"), 1, None),
  "VISIBILITY": MSv2ColumnSchema(
    "DATA", ("frequency", "polarization"), np.nan + np.nan * 1j, None
  ),
  "WEIGHT": MSv2ColumnSchema(
    "WEIGHT_SPECTRUM", ("frequency", "polarization"), np.nan, None
  ),
}

FIXED_DIMENSION_SIZES = {"uvw_label": 3}


class MainDatasetFactory:
  _partition_key: PartitionKeyT
  _table_factory: TableFactory
  _structure_factory: MSv2StructureFactory

  def __init__(
    self,
    partition_key: PartitionKeyT,
    table_factory: TableFactory,
    structure_factory: MSv2StructureFactory,
  ):
    self._partition_key = partition_key
    self._table_factory = table_factory
    self._structure_factory = structure_factory

  def _variable_from_column(self, column: str) -> Variable:
    """Derive an xarray Variable from the MSv2 column descriptor and schemas"""
    structure = self._structure_factory()
    partition = structure[self._partition_key]
    main_column_descs = structure.column_descs["MAIN"]

    try:
      schema = MSV4_to_MSV2_COLUMN_SCHEMAS[column]
    except KeyError:
      raise KeyError(f"No Column Schema exist for {column}")

    try:
      column_desc = main_column_descs[schema.name]
    except KeyError:
      raise KeyError(f"No Column Descriptor exist for {schema.name}")

    dim_sizes = {
      "time": len(partition.time),
      "baseline": structure.nbl,
      "frequency": len(partition.chan_freq),
      "polarization": len(partition.corr_type),
      **FIXED_DIMENSION_SIZES,
    }

    dims = ("time", "baseline") + schema.dims

    try:
      shape = tuple(dim_sizes[d] for d in dims)
    except KeyError as e:
      raise KeyError(f"No dimension size found for {e.args[0]}")

    default = column_desc.dtype.type(schema.default)

    data = MSv2Array(
      self._table_factory,
      self._structure_factory,
      self._partition_key,
      schema.name,
      shape,
      column_desc.dtype,
      default,
    )

    var = Variable(dims, data)

    # Apply any measures encoding
    if schema.coder:
      coder = schema.coder(schema.name, structure.column_descs["MAIN"])
      var = coder.decode(var)

    dims, data, attrs, encoding = unpack_for_decoding(var)
    return Variable(dims, LazilyIndexedArray(data), attrs, encoding, fastpath=True)

  def get_variables(self) -> Mapping[str, Variable]:
    structure = self._structure_factory()
    partition = structure[self._partition_key]
    ant1, ant2 = structure.antenna_pairs
    nbl = structure.nbl
    assert (nbl,) == ant1.shape

    ant_names = structure._ant["NAME"].to_numpy()
    ant1_names = ant_names[ant1]
    ant2_names = ant_names[ant2]

    row_map = partition.row_map
    missing = np.count_nonzero(row_map == -1)
    if missing > 0:
      warnings.warn(
        f"{missing} / {row_map.size} ({100. * missing / row_map.size:.1f}%) "
        f"rows missing from the full (time, baseline) grid "
        f"in partition {self._partition_key}. "
        f"Dataset variables will be padded",
        IrregularGridWarning,
      )

    data_vars = [
      (n, self._variable_from_column(n))
      for n in (
        "TIME_CENTROID",
        "EFFECTIVE_INTEGRATION_TIME",
        "UVW",
        "VISIBILITY",
        "FLAG",
        "WEIGHT",
      )
    ]

    # Add coordinates indexing coordinates
    coordinates = [
      (
        "baseline_id",
        (("baseline",), np.arange(len(ant1)), {"coordinates": "baseline_id"}),
      ),
      (
        "baseline_antenna1_name",
        (("baseline",), ant1_names, {"coordinates": "baseline_antenna1_name"}),
      ),
      (
        "baseline_antenna2_name",
        (("baseline",), ant2_names, {"coordinates": "baseline_antenna2_name"}),
      ),
      ("polarization", (("polarization",), partition.corr_type, None)),
    ]

    coordinates = [(n, Variable(d, v, a)) for n, (d, v, a) in coordinates]

    # Add time coordinate
    time_coder = TimeCoder("TIME", structure.column_descs["MAIN"])

    if partition.interval.size == 1:
      time_attrs = {"integration_time": partition.interval.item()}
    else:
      warnings.warn(
        f"Multiple intervals {partition.interval} "
        f"found in partition {self._partition_key}. "
        f'Setting time.attrs["integration_time"] = nan and '
        f"adding full resolution TIME and INTERVAL columns. ",
        IrregularGridWarning,
      )
      time_attrs = {"integration_time": np.nan}
      data_vars.extend(
        [
          ("TIME", self._variable_from_column("TIME")),
          ("INTEGRATION_TIME", self._variable_from_column("INTEGRATION_TIME")),
        ]
      )

    coordinates.append(
      ("time", time_coder.decode(Variable("time", partition.time, time_attrs)))
    )

    # Add frequency coordinate
    freq_attrs = {
      "type": "spectral_coord",
      "frame": partition.spw_frame,
      "units": ["Hz"],
      "spectral_window_name": partition.spw_name or "<Unknown>",
      "reference_frequency": partition.spw_ref_freq,
      "effective_channel_width": "EFFECTIVE_CHANNEL_WIDTH",
    }

    if partition.spw_freq_group_name:
      freq_attrs["frequency_group_name"] = partition.spw_freq_group_name

    if partition.chan_width.size == 1:
      freq_attrs["channel_width"] = partition.chan_width.item()
    else:
      freq_attrs["channel_width"] = np.nan
      warnings.warn(
        f"Multiple channel widths {partition.chan_width} "
        f"found in partition {self._partition_key}. "
        f'Setting frequency.attrs["channel_width"] = nan and '
        f"adding full resolution CHANNEL_FREQUENCY column. ",
      )
      raise NotImplementedError(
        "Full resolution CHANNEL_FREQUENCY " " and CHANNEL_WIDTH columns"
      )

    coordinates.append(
      ("frequency", Variable("frequency", partition.chan_freq, freq_attrs))
    )

    return FrozenDict(sorted(data_vars + coordinates))
