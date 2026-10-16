"""Pure preflight for a fresh MSv2 write; no table is opened or created."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from xarray import DataTree, Variable

from xarray_ms.backend.msv2.measure_encoding import (
  FixedMeasureEncoding,
  check_shared_reference,
  encode_fixed_measure,
)
from xarray_ms.backend.msv2.metadata_plan import (
  MetadataRows,
  PartitionKeys,
  build_metadata,
  extract_metadata,
)
from xarray_ms.errors import (
  FreshMSv2TargetError,
  FreshMSv2ValidationError,
)
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES

CANONICAL_DIMS = ("time", "baseline_id", "frequency", "polarization")
STRUCTURAL_COORDS = {
  "baseline_antenna1_name": "baseline_id",
  "baseline_antenna2_name": "baseline_id",
  "field_name": "time",
  "scan_name": "time",
}
SEMANTIC_ROLES = ("correlated_data", "flag", "weight", "uvw", "field_and_source")
RESERVED_COLUMNS = frozenset(
  {
    "DATA",
    "FLAG",
    "FLAG_ROW",
    "WEIGHT",
    "WEIGHT_SPECTRUM",
    "UVW",
    "TIME",
    "TIME_CENTROID",
    "INTERVAL",
    "EXPOSURE",
    "ANTENNA1",
    "ANTENNA2",
    "ANTENNA3",
    "BASELINE_REF",
    "DATA_DESC_ID",
    "FIELD_ID",
    "OBSERVATION_ID",
    "PROCESSOR_ID",
    "STATE_ID",
    "SCAN_NUMBER",
    "FEED1",
    "FEED2",
    "ARRAY_ID",
    "FLAG_CATEGORY",
    "CORRECTED_WEIGHT_SPECTRUM",
    "FLOAT_DATA",
    "IMAGING_WEIGHT",
    "LAG_DATA",
    "SIGMA",
    "SIGMA_SPECTRUM",
    "SOURCE_ID",
    "SUB_SCAN_NUMBER",
    "PHASE_ID",
    "PULSAR_BIN",
    "PULSAR_GATE_ID",
    "FEED3",
    "TIME_EXTRA_PREC",
    "UVW2",
    "VIDEO_POINT",
  }
)
UNSUPPORTED_OPTIONAL_METADATA = frozenset(
  {
    "field_and_source_ephemeris_xds",
    "gain_curve_xds",
    "phase_calibration_xds",
    "phased_array_xds",
    "pointing_xds",
    "system_calibration_xds",
    "weather_xds",
  }
)


@dataclass(frozen=True)
class FreshMSv2Partition:
  """Resolved references within one correlated dataset."""

  path: str
  correlated_data: str
  flag: str
  weight: str
  uvw: str
  field_and_source: str
  antenna: str
  additional_correlated_data: tuple[tuple[str, str], ...]
  measures: tuple[FixedMeasureEncoding, ...]
  foreign_keys: PartitionKeys


@dataclass(frozen=True)
class FreshMSv2Plan:
  """Immutable input summary for a later fresh-MS writer."""

  target: str
  data_group: str
  partitions: tuple[FreshMSv2Partition, ...]
  visibility_mappings: tuple[tuple[str, str], ...]
  visibility_columns: tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...]
  metadata: MetadataRows


def _target_path(target: str | os.PathLike[str]) -> str:
  try:
    raw = os.fspath(target)
  except TypeError as e:
    raise FreshMSv2TargetError("Target must be a nonempty filesystem path") from e
  if not isinstance(raw, str) or not raw.strip():
    raise FreshMSv2TargetError("Target must be a nonempty filesystem path")
  path = os.path.abspath(os.path.expanduser(raw))
  if os.path.lexists(path):
    raise FreshMSv2TargetError(f"Target already exists: {path}")
  parent = os.path.dirname(path)
  if not os.path.isdir(parent):
    raise FreshMSv2TargetError(f"Target parent is missing or not a directory: {parent}")
  return path


def _field_path(tree: DataTree, node: DataTree, reference: str) -> str:
  if not isinstance(reference, str) or not reference.strip():
    raise FreshMSv2ValidationError(
      f"Partition {node.path}: field_and_source must reference a DataTree node"
    )
  # Absolute paths and root-relative paths emitted by readers are both valid.
  # A simple child name is resolved relative to the correlated node.
  candidates = (
    (reference,)
    if reference.startswith("/")
    else (f"{node.path.rstrip('/')}/{reference}", f"/{reference}")
  )
  for path in candidates:
    try:
      resolved = tree.root[path]
    except (KeyError, ValueError):
      continue
    if resolved.attrs.get("type") != "field_and_source":
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: field_and_source node {resolved.path} has "
        f"invalid type {resolved.attrs.get('type')!r}"
      )
    return resolved.path
  raise FreshMSv2ValidationError(
    f"Partition {node.path}: field_and_source node {reference!r} does not exist"
  )


def _group(node: DataTree, name: str, roles: tuple[str, ...]) -> Mapping:
  groups = node.attrs.get("data_groups")
  if not isinstance(groups, Mapping) or name not in groups:
    raise FreshMSv2ValidationError(
      f"Partition {node.path}: data_groups has no group {name!r}"
    )
  group = groups[name]
  if not isinstance(group, Mapping):
    raise FreshMSv2ValidationError(f"Partition {node.path}: group {name!r} is invalid")
  for role in roles:
    if role not in group:
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: group {name!r} lacks role {role!r}"
      )
  return group


def _variable(node: DataTree, group: Mapping, role: str, name: str):
  reference = group[role]
  if not isinstance(reference, str) or reference not in node.data_vars:
    raise FreshMSv2ValidationError(
      f"Partition {node.path}: group {name!r} role {role!r} "
      f"references missing data variable {reference!r}"
    )
  return node.data_vars[reference]


def plan_fresh_msv2(
  tree: DataTree,
  target: str | os.PathLike[str],
  data_group: str = "base",
  additional_visibility: Mapping[str, str] | None = None,
) -> FreshMSv2Plan:
  """Validate an MSv4 tree for a *new* MSv2 target without writing anything.

  ``additional_visibility`` maps source data-group names to destination MSv2
  visibility columns; the selected ``data_group`` always maps to ``DATA``.
  """
  destination = _target_path(target)
  if not isinstance(tree, DataTree):
    raise FreshMSv2ValidationError("Source must be an xarray DataTree")
  if not isinstance(data_group, str) or not data_group:
    raise FreshMSv2ValidationError("data_group must be a nonempty name")
  if additional_visibility is not None and not isinstance(
    additional_visibility, Mapping
  ):
    raise FreshMSv2ValidationError("additional_visibility must be a mapping")

  supplied_mappings = additional_visibility or {}
  if any(not isinstance(group, str) for group in supplied_mappings):
    raise FreshMSv2ValidationError("Additional data-group names must be strings")
  mappings = []
  destinations = {"DATA"}
  for group, column in sorted(supplied_mappings.items()):
    if not isinstance(group, str) or not group or group == data_group:
      raise FreshMSv2ValidationError(f"Invalid additional data group {group!r}")
    if not isinstance(column, str) or not re.fullmatch(
      r"[A-Za-z][A-Za-z0-9_]*", column
    ):
      raise FreshMSv2ValidationError(f"Invalid destination column {column!r}")
    folded_column = column.upper()
    if folded_column in destinations or folded_column in RESERVED_COLUMNS:
      raise FreshMSv2ValidationError(
        f"Destination column {column!r} is duplicated or writer-controlled"
      )
    destinations.add(folded_column)
    mappings.append((group, column))

  nodes = sorted(
    (n for n in tree.subtree if n.attrs.get("type") in CORRELATED_DATASET_TYPES),
    key=lambda n: n.path,
  )
  if not nodes:
    raise FreshMSv2ValidationError("No correlated datasets found in DataTree")

  partitions = []
  extracted = []
  visibility_shapes: dict[str, set[tuple[int, int]]] = {
    column: set() for _, column in ((data_group, "DATA"), *mappings)
  }
  visibility_dtypes: dict[str, set[str]] = {
    column: set() for column in visibility_shapes
  }
  shared_measures: dict[str, FixedMeasureEncoding] = {}
  for node in nodes:
    ds = node.to_dataset(inherit=True)
    for dim in CANONICAL_DIMS:
      if dim not in ds.coords or ds[dim].dims != (dim,) or ds.sizes.get(dim, 0) == 0:
        raise FreshMSv2ValidationError(
          f"Partition {node.path}: missing canonical coordinate {dim!r}"
        )
    for coord, dim in STRUCTURAL_COORDS.items():
      if coord not in ds.coords or ds[coord].dims != (dim,):
        raise FreshMSv2ValidationError(
          f"Partition {node.path}: coordinate {coord!r} must be 1-D over {dim!r}"
        )
    if (
      "uvw_label" not in ds.coords
      or ds["uvw_label"].dims != ("uvw_label",)
      or ds.sizes.get("uvw_label") != 3
    ):
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: uvw_label must be a 1-D coordinate of size 3"
      )
    for attr in ("observation_info", "processor_info"):
      if not isinstance(node.attrs.get(attr), Mapping):
        raise FreshMSv2ValidationError(
          f"Partition {node.path}: attrs[{attr!r}] must be a mapping"
        )
    antenna_node = node.children.get("antenna_xds")
    if antenna_node is None or antenna_node.attrs.get("type") != "antenna":
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: required antenna_xds child with type 'antenna' "
        "is missing or invalid"
      )
    unsupported_metadata = sorted(
      name
      for name, child in node.children.items()
      if name in UNSUPPORTED_OPTIONAL_METADATA
      or child.attrs.get("type") == "field_and_source_ephemeris"
    )
    if unsupported_metadata:
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: unsupported optional metadata dataset(s) "
        f"{unsupported_metadata}"
      )
    selected_group = _group(node, data_group, SEMANTIC_ROLES)
    base = _variable(node, selected_group, "correlated_data", data_group)
    if base.dims != CANONICAL_DIMS or base.dtype not in (
      np.dtype("complex64"),
      np.dtype("complex128"),
    ):
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: base visibility requires dimensions "
        f"{CANONICAL_DIMS} and complex64 or complex128 dtype; "
        f"got {base.dims}, {base.dtype}"
      )
    visibility_shapes["DATA"].add(base.shape[2:])
    visibility_dtypes["DATA"].add(base.dtype.str)
    flag = _variable(node, selected_group, "flag", data_group)
    if (
      flag.dims != base.dims
      or flag.shape != base.shape
      or flag.dtype not in (np.dtype("bool"), np.dtype("uint8"))
    ):
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: FLAG requires base visibility dimensions/shape "
        "and bool or uint8 dtype"
      )
    weight = _variable(node, selected_group, "weight", data_group)
    if (
      weight.dims != base.dims
      or weight.shape != base.shape
      or weight.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: WEIGHT requires base visibility dimensions/shape "
        "and float32 or float64 dtype"
      )
    uvw = _variable(node, selected_group, "uvw", data_group)
    if (
      uvw.dims != ("time", "baseline_id", "uvw_label")
      or uvw.shape != (base.shape[0], base.shape[1], 3)
      or uvw.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
      raise FreshMSv2ValidationError(
        f"Partition {node.path}: UVW requires (time, baseline_id, uvw_label) "
        "dimensions, matching time/baseline sizes, size-3 uvw_label, "
        "and float32 or float64 dtype"
      )
    field = _field_path(tree, node, selected_group["field_and_source"])
    measures = [
      encode_fixed_measure(ds["time"].variable, node.path, "time", "MAIN::TIME"),
      encode_fixed_measure(
        ds["frequency"].variable, node.path, "frequency", "SPECTRAL_WINDOW::CHAN_FREQ"
      ),
      encode_fixed_measure(uvw.variable, node.path, selected_group["uvw"], "MAIN::UVW"),
    ]
    reference = ds["frequency"].attrs.get("reference_frequency")
    if isinstance(reference, Mapping) and isinstance(reference.get("attrs"), Mapping):
      reference_data = np.asarray(reference.get("data"))
      if reference_data.ndim != 0:
        raise FreshMSv2ValidationError(
          f"Partition {node.path} frequency: reference_frequency data must be scalar"
        )
      reference_measure = encode_fixed_measure(
        Variable((), reference_data.item(), dict(reference["attrs"])),
        node.path,
        "reference_frequency",
        "SPECTRAL_WINDOW::REF_FREQUENCY",
      )
      measures.append(reference_measure)
    else:
      raise FreshMSv2ValidationError(
        f"Partition {node.path} frequency: reference_frequency "
        "requires data and measure attrs"
      )
    for metadata_node, variable_name, column in (
      (antenna_node, "ANTENNA_POSITION", "ANTENNA::POSITION"),
      (tree.root[field], "FIELD_PHASE_CENTER_DIRECTION", "FIELD::PHASE_DIR"),
    ):
      if variable_name not in metadata_node.data_vars:
        raise FreshMSv2ValidationError(
          f"Partition {node.path}: {metadata_node.path} requires "
          f"{variable_name!r} as a data variable"
        )
      measures.append(
        encode_fixed_measure(
          metadata_node.data_vars[variable_name].variable,
          metadata_node.path,
          variable_name,
          column,
        )
      )
    for measure in measures:
      if previous := shared_measures.get(measure.column):
        check_shared_reference(previous, measure)
      else:
        shared_measures[measure.column] = measure
    extracted.append(
      extract_metadata(
        node, antenna_node, tree.root[field], measures[1].frame, reference_measure.frame
      )
    )
    additional = []
    for name, column in mappings:
      extra_group = _group(node, name, ("correlated_data",))
      extra = _variable(node, extra_group, "correlated_data", name)
      if (
        extra.dims != base.dims
        or extra.shape != base.shape
        or extra.dtype != base.dtype
      ):
        raise FreshMSv2ValidationError(
          f"Partition {node.path}: group {name!r} visibility dimensions, "
          "shape and dtype must match the base visibility"
        )
      additional.append((name, extra_group["correlated_data"]))
      visibility_shapes[column].add(extra.shape[2:])
      visibility_dtypes[column].add(extra.dtype.str)
    partitions.append(
      (
        node.path,
        selected_group["correlated_data"],
        selected_group["flag"],
        selected_group["weight"],
        selected_group["uvw"],
        field,
        antenna_node.path,
        tuple(additional),
        tuple(measures),
      )
    )
  metadata, foreign_keys = build_metadata(extracted)
  if any(row.focus_length is not None for row in metadata.feed_rows):
    raise FreshMSv2ValidationError(
      "ANTENNA_FOCUS_LENGTH requires optional FEED::FOCUS_LENGTH column; "
      "fresh creation cannot add subtable columns"
    )
  resolved_partitions = [
    FreshMSv2Partition(*partition, keys)
    for partition, keys in zip(partitions, foreign_keys, strict=True)
  ]
  for column, dtypes in visibility_dtypes.items():
    if len(dtypes) != 1:
      raise FreshMSv2ValidationError(
        f"Visibility column {column!r} has incompatible partition dtypes "
        f"{sorted(dtypes)}"
      )
  return FreshMSv2Plan(
    destination,
    data_group,
    tuple(resolved_partitions),
    ((data_group, "DATA"), *mappings),
    tuple(
      (
        column,
        next(iter(visibility_dtypes[column])),
        tuple(sorted(visibility_shapes[column])),
      )
      for column in visibility_shapes
    ),
    metadata,
  )


def datatree_plan_msv2(
  tree: DataTree,
  target: str | os.PathLike[str],
  data_group: str = "base",
  additional_visibility: Mapping[str, str] | None = None,
) -> FreshMSv2Plan:
  """DataTree-facing spelling of :func:`plan_fresh_msv2`."""
  return plan_fresh_msv2(tree, target, data_group, additional_visibility)
