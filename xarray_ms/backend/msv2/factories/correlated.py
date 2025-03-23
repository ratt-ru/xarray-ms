import dataclasses
import warnings
from typing import Any, Dict, Mapping, Tuple, Type

import numpy as np
from xarray import Variable
from xarray.coding.variables import unpack_for_decoding
from xarray.core.indexing import LazilyIndexedArray
from xarray.core.utils import FrozenDict

from xarray_ms.backend.msv2.array import (
  BroadcastMSv2Array,
  MainMSv2Array,
  MSv2Array,
)
from xarray_ms.backend.msv2.encoders import (
  CasaCoder,
  QuantityCoder,
  TimeCoder,
)
from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.casa_types import ColumnDesc, FrequencyMeasures, Polarisations
from xarray_ms.errors import IrregularGridWarning
from xarray_ms.multiton import Multiton


@dataclasses.dataclass
class MSv2ColumnSchema:
  name: str
  dims: Tuple[str, ...]
  default: Any = None
  coder: Type[CasaCoder] | None = None
  low_res_dims: Tuple[str, ...] | None = None


MSV4_to_MSV2_COLUMN_SCHEMAS = {
  "TIME": MSv2ColumnSchema("TIME", (), np.nan, TimeCoder),
  "INTEGRATION_TIME": MSv2ColumnSchema("INTERVAL", (), np.nan, QuantityCoder),
  "TIME_CENTROID": MSv2ColumnSchema("TIME_CENTROID", (), np.nan, TimeCoder),
  "EFFECTIVE_INTEGRATION_TIME": MSv2ColumnSchema("EXPOSURE", (), np.nan, QuantityCoder),
  "UVW": MSv2ColumnSchema("UVW", ("uvw_label",), np.nan, None),
  "FLAG_ROW": MSv2ColumnSchema(
    "FLAG_ROW", ("frequency", "polarization"), 1, None, low_res_dims=()
  ),
  "FLAG": MSv2ColumnSchema("FLAG", ("frequency", "polarization"), 1, None),
  "VISIBILITY": MSv2ColumnSchema(
    "DATA", ("frequency", "polarization"), np.nan + np.nan * 1j, None
  ),
  "WEIGHT_ROW": MSv2ColumnSchema(
    "WEIGHT",
    ("frequency", "polarization"),
    np.nan,
    None,
    low_res_dims=("polarization",),
  ),
  "WEIGHT": MSv2ColumnSchema(
    "WEIGHT_SPECTRUM", ("frequency", "polarization"), np.nan, None
  ),
}

FIXED_DIMENSION_SIZES = {"uvw_label": 3}


class CorrelatedDatasetFactory:
  _partition_key: PartitionKeyT
  _preferred_chunks: Dict[str, int]
  _ms_factory: Multiton
  _subtable_factories: Dict[str, Multiton]
  _structure_factory: MSv2StructureFactory
  _antenna_factory: Multiton
  _spw_factory: Multiton
  _pol_factory: Multiton
  _obs_factory: Multiton
  _column_descs: Dict[str, ColumnDesc]

  def __init__(
    self,
    partition_key: PartitionKeyT,
    preferred_chunks: Dict[str, int],
    ms_factory: Multiton,
    subtable_factories: Dict[str, Multiton],
    structure_factory: MSv2StructureFactory,
  ):
    self._partition_key = partition_key
    self._preferred_chunks = preferred_chunks
    self._ms_factory = ms_factory
    self._subtable_factories = subtable_factories
    self._structure_factory = structure_factory

    ms = ms_factory.instance
    ms_table_desc = ms.tabledesc()
    self._main_column_descs = {
      c: ColumnDesc.from_descriptor(c, ms_table_desc) for c in ms.columns()
    }

  def _variable_from_column(self, column: str, dim_sizes: Dict[str, int]) -> Variable:
    """Derive an xarray Variable from the MSv2 column descriptor and schemas"""
    try:
      schema = MSV4_to_MSV2_COLUMN_SCHEMAS[column]
    except KeyError:
      raise KeyError(f"Column {column} was not present")

    try:
      column_desc = self._main_column_descs[schema.name]
    except KeyError:
      raise KeyError(f"No Column Descriptor exist for {schema.name}")

    dims = ("time", "baseline_id") + schema.dims

    try:
      shape = tuple(dim_sizes[d] for d in dims)
    except KeyError as e:
      raise KeyError(f"No dimension size found for {e.args[0]}")

    default = column_desc.dtype.type(schema.default)

    high_res_shape = shape
    low_res_index: Tuple[slice | None, ...] = tuple(slice(None) for _ in shape)

    if schema.low_res_dims:
      low_res_dims = ("time", "baseline_id") + schema.low_res_dims
      high_res_shape = shape
      try:
        shape_map = {d: dim_sizes[d] for d in low_res_dims}
      except KeyError as e:
        raise KeyError(f"No dimension size found for {e.args[0]}")
      low_res_index = tuple(slice(None) if d in shape_map else None for d in dims)
      shape = tuple(shape_map.values())

    array: MSv2Array = MainMSv2Array(
      self._ms_factory,
      self._structure_factory,
      self._partition_key,
      schema.name,
      shape,
      column_desc.dtype,
      default,
    )

    if schema.low_res_dims:
      array = BroadcastMSv2Array(array, low_res_index, high_res_shape)

    var = Variable(dims, array, fastpath=True)

    # Apply any measures encoding
    if schema.coder:
      coder = schema.coder(schema.name, self._main_column_descs)
      var = coder.decode(var)

    dims, data, attrs, encoding = unpack_for_decoding(var)

    if self._preferred_chunks:
      encoding["preferred_chunks"] = self._preferred_chunks

    return Variable(dims, LazilyIndexedArray(data), attrs, encoding, fastpath=True)

  def get_variables(self) -> Mapping[str, Variable]:
    structure = self._structure_factory.instance
    partition = structure[self._partition_key]
    ant1, ant2 = partition.antenna_pairs
    assert (partition.nbl,) == ant1.shape

    antenna = self._subtable_factories["ANTENNA"].instance
    ant_names = antenna["NAME"].to_numpy()
    ant1_names = ant_names[ant1]
    ant2_names = ant_names[ant2]

    spw_id = partition.spw_id
    pol_id = partition.pol_id
    spw = self._subtable_factories["SPECTRAL_WINDOW"].instance
    pol = self._subtable_factories["POLARIZATION"].instance
    field = self._subtable_factories["FIELD"].instance

    chan_freq = spw["CHAN_FREQ"][spw_id].as_py()
    uchan_width = np.unique(spw["CHAN_WIDTH"][spw_id].as_py())
    spw_name = spw["NAME"][spw_id].as_py()
    spw_freq_group_name = spw["FREQ_GROUP_NAME"][spw_id].as_py()
    spw_ref_freq = spw["REF_FREQUENCY"][spw_id].as_py()
    spw_meas_freq_ref = spw["MEAS_FREQ_REF"][spw_id].as_py()
    spw_frame = FrequencyMeasures(spw_meas_freq_ref).name.lower()

    corr_type = Polarisations.from_values(pol["CORR_TYPE"][pol_id].as_py()).to_str()

    dim_sizes = {
      "time": len(partition.time),
      "baseline_id": partition.nbl,
      "frequency": len(chan_freq),
      "polarization": len(corr_type),
      **FIXED_DIMENSION_SIZES,
    }

    row_map = partition.row_map
    missing = np.count_nonzero(row_map == -1)
    if missing > 0:
      warnings.warn(
        f"{missing} / {row_map.size} ({100. * missing / row_map.size:.1f}%) "
        f"rows missing from the full (time, baseline_id) grid "
        f"in partition {self._partition_key}. "
        f"Dataset variables will be padded with nans "
        f"in the case of data variables "
        f"and flags will be set",
        IrregularGridWarning,
      )

    data_vars = [
      (n, self._variable_from_column(n, dim_sizes))
      for n in (
        "TIME_CENTROID",
        "EFFECTIVE_INTEGRATION_TIME",
        "UVW",
        "VISIBILITY",
      )
    ]

    if "FLAG" in self._main_column_descs:
      data_vars.append(("FLAG", self._variable_from_column("FLAG", dim_sizes)))
    else:
      data_vars.append(("FLAG", self._variable_from_column("FLAG_ROW", dim_sizes)))

    if "WEIGHT_SPECTRUM" in self._main_column_descs:
      data_vars.append(("WEIGHT", self._variable_from_column("WEIGHT", dim_sizes)))
    else:
      data_vars.append(("WEIGHT", self._variable_from_column("WEIGHT_ROW", dim_sizes)))

    field_names = field.take(partition.field_ids)["NAME"].to_numpy()

    # Add coordinates indexing coordinates
    coordinates = [
      (
        "baseline_id",
        (("baseline_id",), np.arange(len(ant1)), {"coordinates": "baseline_id"}),
      ),
      (
        "baseline_antenna1_name",
        (("baseline_id",), ant1_names, {"coordinates": "baseline_antenna1_name"}),
      ),
      (
        "baseline_antenna2_name",
        (("baseline_id",), ant2_names, {"coordinates": "baseline_antenna2_name"}),
      ),
      ("polarization", (("polarization",), corr_type, None)),
      ("uvw_label", (("uvw_label",), ["u", "v", "w"], None)),
      ("field_name", ("time", field_names, {"coordinates": "field_name"})),
      ("scan_number", ("time", partition.scan_numbers, {"coordinates": "scan_number"})),
      (
        "sub_scan_number",
        ("time", partition.sub_scan_numbers, {"coordinates": "sub_scan_number"}),
      ),
    ]

    e = {"preferred_chunks": self._preferred_chunks} if self._preferred_chunks else None
    coordinates = [(n, Variable(d, v, a, e)) for n, (d, v, a) in coordinates]

    # Add time coordinate
    time_coder = TimeCoder("TIME", self._main_column_descs)

    if partition.interval.size == 1:
      time_attrs = {"integration_time": partition.interval.item()}
    else:
      warnings.warn(
        f"Missing/Multiple intervals {partition.interval} "
        f"found in partition {self._partition_key}. "
        f'Setting time.attrs["integration_time"] = nan and '
        f"adding full resolution TIME and INTEGRATION_TIME columns. "
        f"{'They contain nans in missing rows' if missing else ''}",
        IrregularGridWarning,
      )
      time_attrs = {"integration_time": np.nan}
      data_vars.extend(
        [
          ("TIME", self._variable_from_column("TIME", dim_sizes)),
          (
            "INTEGRATION_TIME",
            self._variable_from_column("INTEGRATION_TIME", dim_sizes),
          ),
        ]
      )

    coordinates.append(
      ("time", time_coder.decode(Variable("time", partition.time, time_attrs)))
    )

    # Add frequency coordinate
    freq_attrs = {
      "type": "spectral_coord",
      "frame": spw_frame,
      "units": ["Hz"],
      "spectral_window_name": spw_name or "<Unknown>",
      "reference_frequency": spw_ref_freq,
      "effective_channel_width": "EFFECTIVE_CHANNEL_WIDTH",
    }

    if spw_freq_group_name:
      freq_attrs["frequency_group_name"] = spw_freq_group_name

    if uchan_width.size == 1:
      freq_attrs["channel_width"] = uchan_width.item()
    else:
      freq_attrs["channel_width"] = np.nan
      warnings.warn(
        f"Multiple channel widths {uchan_width} "
        f"found in partition {self._partition_key}. "
        f'Setting frequency.attrs["channel_width"] = nan and '
        f"adding full resolution CHANNEL_FREQUENCY column.",
        IrregularGridWarning,
      )
      raise NotImplementedError(
        "Full resolution CHANNEL_FREQUENCY and CHANNEL_WIDTH columns"
      )

    coordinates.append(("frequency", Variable("frequency", chan_freq, freq_attrs)))

    return FrozenDict(sorted(data_vars + coordinates))

  def _observation_info(self) -> Dict[str, Any]:
    structure = self._structure_factory.instance
    partition = structure[self._partition_key]
    obs = self._subtable_factories["OBSERVATION"].instance
    observer = obs["OBSERVER"][partition.obs_id].as_py()
    project = obs["PROJECT"][partition.obs_id].as_py()
    # TODO: A Measures conversions is needed here
    release_date = obs["RELEASE_DATE"][partition.obs_id].as_py()  # noqa: F841

    return dict(
      sorted(
        {
          "observer": observer,
          "project": project,
        }.items()
      )
    )

  def get_attrs(self) -> Dict[Any, Any]:
    return {
      "observation_info": self._observation_info(),
    }
