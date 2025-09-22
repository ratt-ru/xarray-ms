import dataclasses
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Set, Tuple, Type

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
  SpectralCoordCoder,
  TimeCoder,
  UVWCoder,
  VisibilityCoder,
)
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.imputation import (
  maybe_impute_field_table,
  maybe_impute_observation_table,
  maybe_impute_processor_table,
)
from xarray_ms.backend.msv2.structure import MSv2StructureFactory, PartitionKeyT
from xarray_ms.backend.msv2.table_utils import table_desc
from xarray_ms.casa_types import ColumnDesc, Polarisations
from xarray_ms.errors import (
  ColumnShapeImputationWarning,
  FrameConversionWarning,
  IrregularBaselineGridWarning,
  IrregularChannelGridWarning,
  IrregularTimeGridWarning,
)
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
  "UVW": MSv2ColumnSchema("UVW", ("uvw_label",), np.nan, UVWCoder),
  "FLAG_ROW": MSv2ColumnSchema(
    "FLAG_ROW", ("frequency", "polarization"), 1, None, low_res_dims=()
  ),
  "FLAG": MSv2ColumnSchema("FLAG", ("frequency", "polarization"), 1, None),
  "VISIBILITY": MSv2ColumnSchema(
    "DATA", ("frequency", "polarization"), np.nan + np.nan * 1j, VisibilityCoder
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

PARTITIONING_LINK = "https://xarray-ms.readthedocs.io/en/latest/partitioning.html"


class CorrelatedFactory(DatasetFactory):
  """Factory class for generating the main correlated dataset
  for a partition of the Measurement Set"""

  _preferred_chunks: Dict[str, int]
  _ms_factory: Multiton
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
    super().__init__(partition_key, structure_factory, subtable_factories)
    self._preferred_chunks = preferred_chunks
    self._ms_factory = ms_factory

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
      coder = schema.coder(self._main_column_descs[schema.name])
      var = coder.decode(var)

    dims, data, attrs, encoding = unpack_for_decoding(var)

    if self._preferred_chunks:
      encoding["preferred_chunks"] = self._preferred_chunks

    return Variable(dims, LazilyIndexedArray(data), attrs, encoding, fastpath=True)

  def secondary_variables(
    self, processed_columns: Set[str], dim_sizes: Dict[str, int]
  ) -> List[Tuple[str, Variable]]:
    """Add any secondary, non-standard variables"""
    import pyarrow as pa
    import pyarrow.compute as pac
    from arcae.lib.arrow_tables import ms_descriptor

    # Ignore all standard msv2 columns, except
    # for CORRECTED_DATA and CORRECTED_WEIGHT_SPECTRUM
    ignored_msv2_column_set = {
      c for c in ms_descriptor("MAIN", complete=True).keys() if not c.startswith("_")
    }
    ignored_msv2_column_set -= {"CORRECTED_DATA", "CORRECTED_WEIGHT_SPECTRUM"}
    remaining_columns = set(self._ms_factory.instance.columns())
    remaining_columns -= ignored_msv2_column_set
    remaining_columns -= processed_columns

    if len(remaining_columns) == 0:
      return []

    partition = self._structure_factory.instance[self._partition_key]
    # Get the row shapes of all defined rows in the partition
    partition_rows = partition.row_map.ravel()
    partition_rows = partition_rows[partition_rows >= 0]
    variables = []

    for column in remaining_columns:
      try:
        pa_row_shapes = self._ms_factory.instance.row_shapes(column, (partition_rows,))
      except pa.lib.ArrowInvalid as e:
        warnings.warn(
          f"Ignoring secondary column {column} due to {e}",
          ColumnShapeImputationWarning,
        )
        continue

      if isinstance(pa_row_shapes, pa.FixedSizeListArray):
        ndim = pa_row_shapes.type.list_size
        pa_row_shapes = pa_row_shapes.filter(pac.is_valid(pa_row_shapes))
        pa_row_shapes = pac.list_flatten(pa_row_shapes, recursive=True)
        row_shapes = pa_row_shapes.to_numpy().reshape(-1, ndim)
        row_shapes = np.unique(row_shapes, axis=0)

        if row_shapes.shape[0] != 1:
          warnings.warn(
            f"Ignoring secondary column {column} without "
            f"a distinct row shape {row_shapes}",
            ColumnShapeImputationWarning,
          )
          continue

        row_shape = tuple(row_shapes[0].tolist())
        column_dims: Tuple[str, ...] = ()
        sizes = dim_sizes.copy()

        # Identify the obvious candidates
        for d, dim in reversed(list(enumerate(row_shape, 1))):
          if sizes.get("polarization", -1) == dim:
            column_dims = ("polarization",) + column_dims
            del sizes["polarization"]
          elif sizes.get("frequency", -1) == dim:
            column_dims = ("frequency",) + column_dims
            del sizes["frequency"]
          else:
            column_dims += (f"{column}-{d}",)

        array = MainMSv2Array(
          self._ms_factory,
          self._structure_factory,
          self._partition_key,
          column,
          (sizes["time"], sizes["baseline_id"]) + row_shape,
          self._main_column_descs[column].dtype,
          np.nan,
        )

        variables.append(
          (column, Variable(("time", "baseline_id") + column_dims, array))
        )
      elif isinstance(pa_row_shapes, pa.NullArray):
        raise NotImplementedError(f"Secondary scalar column {column}")
      else:
        warnings.warn(
          f"Ignoring scalar secondary column {column}", ColumnShapeImputationWarning
        )
        continue

    return variables

  def get_variables(self) -> Mapping[str, Variable]:
    structure = self._structure_factory.instance
    partition = structure[self._partition_key]
    ant1, ant2 = partition.antenna_pairs
    assert (partition.nbl,) == ant1.shape

    antenna = self._subtable_factories["ANTENNA"].instance
    ant_names = antenna["NAME"].to_numpy().astype(str)
    ant1_names = ant_names[ant1]
    ant2_names = ant_names[ant2]

    spw_id = partition.spw_id
    pol_id = partition.pol_id
    spw = self._subtable_factories["SPECTRAL_WINDOW"].instance
    pol = self._subtable_factories["POLARIZATION"].instance
    field = self._subtable_factories["FIELD"].instance

    spw_table_desc = table_desc(spw)
    chan_freq_coldesc = ColumnDesc.from_descriptor("CHAN_FREQ", spw_table_desc)

    chan_freq = spw["CHAN_FREQ"][spw_id].as_py()
    chan_width = spw["CHAN_WIDTH"][spw_id].as_py()
    spw_name = spw["NAME"][spw_id].as_py()
    spw_freq_group_name = spw["FREQ_GROUP_NAME"][spw_id].as_py()
    spw_ref_freq = spw["REF_FREQUENCY"][spw_id].as_py()
    freq_coder = SpectralCoordCoder(chan_freq_coldesc).with_var_ref_cols(
      lambda c: spw[c].take([spw_id]).to_numpy()
    )

    corr_type = Polarisations.from_values(pol["CORR_TYPE"][pol_id].as_py()).to_str()

    dim_sizes = {
      "time": len(partition.time),
      "baseline_id": partition.nbl,
      "frequency": len(chan_freq),
      "polarization": len(corr_type),
      **FIXED_DIMENSION_SIZES,
    }

    row_map = partition.row_map
    missing_rows = np.count_nonzero(row_map == -1)
    if missing_rows > 0:
      warnings.warn(
        f"{missing_rows} / {row_map.size} ({100.0 * missing_rows / row_map.size:.1f}%) "
        f"rows missing from the full (time, baseline_id) grid "
        f"in partition {self._partition_key}. "
        f"Dataset variables will be padded with nans "
        f"in the case of data variables "
        f"and flags will be set for these cases. "
        f"This situation is benign, especially if auto-corelations "
        f"have been requested on a dataset without them. "
        f"See {PARTITIONING_LINK}",
        IrregularBaselineGridWarning,
      )

    STANDARD_VARIABLES = (
      "TIME_CENTROID",
      "EFFECTIVE_INTEGRATION_TIME",
      "UVW",
      "VISIBILITY",
    )
    processed_columns = {
      MSV4_to_MSV2_COLUMN_SCHEMAS[v].name for v in STANDARD_VARIABLES
    }

    data_vars = [
      (v, self._variable_from_column(v, dim_sizes)) for v in STANDARD_VARIABLES
    ]

    if "FLAG" in self._main_column_descs:
      data_vars.append(("FLAG", self._variable_from_column("FLAG", dim_sizes)))
      processed_columns.add("FLAG")
    else:
      data_vars.append(("FLAG", self._variable_from_column("FLAG_ROW", dim_sizes)))
      processed_columns.add("FLAG_ROW")

    if "WEIGHT_SPECTRUM" in self._main_column_descs:
      data_vars.append(("WEIGHT", self._variable_from_column("WEIGHT", dim_sizes)))
      processed_columns.add("WEIGHT_SPECTRUM")
    else:
      data_vars.append(("WEIGHT", self._variable_from_column("WEIGHT_ROW", dim_sizes)))
      processed_columns.add("WEIGHT_ROW")

    data_vars.extend(self.secondary_variables(processed_columns, dim_sizes))
    field = maybe_impute_field_table(field, partition.field_ids)
    field_names = field.take(partition.field_ids)["NAME"].to_numpy().astype(str)

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

    # Add time coordinate attributes
    # The coder will create most of them from the measures,
    # but we also need to manually add the integration_time
    time_coder = TimeCoder(self._main_column_descs["TIME"])

    time_attrs: Dict[str, Any] = {
      "integration_time": {"attrs": {"type": "quantity", "units": "s"}, "data": 0.0}
    }

    if partition.interval.size == 1:
      # Single unique value
      time_attrs["integration_time"]["data"] = partition.interval.item()
    elif np.allclose(partition.interval[:, None], partition.interval[None, :]):
      # Tolerate some jitter in the unique values
      time_attrs["integration_time"]["data"] = np.mean(partition.interval)
    else:
      # There are multiple unique interval values,
      # a regular grid isn't possible
      warnings.warn(
        f"Missing/Multiple intervals {partition.interval} "
        f"found in partition {self._partition_key}. "
        f"Consider trying different partitioning strategies "
        f"to produce partitions with regular intervals. "
        f"MSv4 cannot strictly represent this case so "
        f"time.attrs['integration_time'] will be set to 'nan' and "
        f"(time, baseline_id) shaped TIME and INTEGRATION_TIME columns "
        f"will be added. "
        f"{'They contain nans in missing rows. ' if missing_rows else ''}"
        f"See {PARTITIONING_LINK}",
        IrregularTimeGridWarning,
      )
      time_attrs["integration_time"]["data"] = np.nan
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
    frequency = freq_coder.decode(Variable("frequency", chan_freq))
    freq_attrs: Dict[str, Any | Dict[str, Any]] = {
      "spectral_window_name": spw_name or "<Unknown>",
      "reference_frequency": {
        "attrs": frequency.attrs.copy(),
        "data": spw_ref_freq,
      },
      "channel_width": {
        "attrs": {"type": "quantity", "units": "Hz"},
      },
      "effective_channel_width": "EFFECTIVE_CHANNEL_WIDTH",
    }

    if spw_freq_group_name:
      freq_attrs["frequency_group_name"] = spw_freq_group_name

    # Ignore the last channel width (data may have been averaged)
    uchan_width = np.unique(chan_width[:-1] if len(chan_width) > 1 else chan_width)

    if uchan_width.size == 1:
      freq_attrs["channel_width"]["data"] = uchan_width.item()
    elif np.allclose(uchan_width[:, None], uchan_width[None, :]):
      freq_attrs["channel_width"]["data"] = np.mean(uchan_width)
    else:
      warnings.warn(
        f"Multiple distinct channel widths {uchan_width} "
        f"found in partition {self._partition_key}. "
        f"MSv4 cannot strictly represent this case and so "
        f"frequency.attrs['channel_width'] will be set to 'nan' and "
        f"a (frequency,) shaped CHANNEL_WIDTH column will be added. "
        f"See {PARTITIONING_LINK}",
        IrregularChannelGridWarning,
      )
      freq_attrs["channel_width"]["data"] = np.nan
      data_vars.append(("CHANNEL_WIDTH", Variable("frequency", chan_width)))

    frequency.attrs.update(freq_attrs)
    coordinates.append(("frequency", frequency))

    return FrozenDict(sorted(data_vars + coordinates))

  def _observation_info(self) -> Dict[str, Any]:
    partition = self._structure_factory.instance[self._partition_key]
    obs = self._subtable_factories["OBSERVATION"].instance
    obs = maybe_impute_observation_table(obs, [partition.obs_id])
    time_coder = TimeCoder(ColumnDesc.from_descriptor("RELEASE_DATE", table_desc(obs)))
    if (release_date_ref := time_coder.measinfo.get("Ref")) != "UTC":
      warnings.warn(
        f"OBSERVATION::RELEASE_DATE Ref {release_date_ref} != UTC "
        f"This error is benign if the accuracy of the above column ",
        FrameConversionWarning,
      )

    decoded_time = time_coder.decode(
      Variable("o", obs["RELEASE_DATE"].take([partition.obs_id]))
    )
    utc_seconds = decoded_time.values[0]

    return dict(
      sorted(
        {
          "observer": [obs["OBSERVER"][partition.obs_id].as_py()],
          "project": obs["PROJECT"][partition.obs_id].as_py(),
          "intents": [partition.obs_mode],
          "release_date": datetime.fromtimestamp(utc_seconds, timezone.utc).isoformat(),
        }.items()
      )
    )

  def _processor_info(self) -> Dict[str, Any]:
    partition = self._structure_factory.instance[self._partition_key]
    proc = self._subtable_factories["PROCESSOR"].instance
    proc = maybe_impute_processor_table(proc, [partition.proc_id])

    return dict(
      sorted(
        {
          "type": proc["TYPE"][partition.proc_id].as_py(),
          "sub_type": proc["SUB_TYPE"][partition.proc_id].as_py(),
        }.items()
      )
    )

  def get_attrs(self) -> Dict[Any, Any]:
    return {
      "observation_info": self._observation_info(),
      "processor_info": self._processor_info(),
    }
