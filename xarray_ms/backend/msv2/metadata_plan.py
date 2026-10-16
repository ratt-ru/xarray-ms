"""Pure MSv4 metadata extraction and deterministic MSv2 row assignment."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from xarray import Dataset, DataTree

from xarray_ms.casa_types import Polarisations
from xarray_ms.errors import FreshMSv2ValidationError


@dataclass(frozen=True)
class AntennaRow:
  name: str
  station: str
  mount: str
  telescope: str
  position: tuple[float, ...]
  dish_diameter: float


@dataclass(frozen=True)
class FeedRow:
  antenna_id: int
  feed_id: int
  polarization_type: tuple[str, ...]
  receptor_angle: tuple[float, ...]
  focus_length: float | None
  spectral_window_id: int = -1


@dataclass(frozen=True)
class SourceRow:
  name: str


@dataclass(frozen=True)
class FieldRow:
  name: str
  phase_direction: tuple[float, ...]
  source_id: int


@dataclass(frozen=True)
class SpectralWindowRow:
  name: str
  intents: tuple[str, ...]
  channel_frequency: tuple[float, ...]
  frame: str
  reference_frequency: float
  reference_frame: str
  channel_width: tuple[float, ...]
  effective_channel_width: tuple[float, ...]
  resolution: tuple[float, ...]
  frequency_group_name: str | None


@dataclass(frozen=True)
class PolarizationRow:
  corr_type: tuple[int, ...]
  corr_product: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DataDescriptionRow:
  spectral_window_id: int
  polarization_id: int


@dataclass(frozen=True)
class ObservationRow:
  observer: str
  project_uid: str
  release_date: str
  telescope: str


@dataclass(frozen=True)
class ProcessorRow:
  type: str
  sub_type: str


@dataclass(frozen=True)
class StateRow:
  scan_intents: tuple[str, ...]


@dataclass(frozen=True)
class MetadataRows:
  antenna_rows: tuple[AntennaRow, ...]
  feed_rows: tuple[FeedRow, ...]
  field_rows: tuple[FieldRow, ...]
  source_rows: tuple[SourceRow, ...]
  spectral_window_rows: tuple[SpectralWindowRow, ...]
  polarization_rows: tuple[PolarizationRow, ...]
  data_description_rows: tuple[DataDescriptionRow, ...]
  observation_rows: tuple[ObservationRow, ...]
  processor_rows: tuple[ProcessorRow, ...]
  state_rows: tuple[StateRow, ...]


@dataclass(frozen=True)
class PartitionKeys:
  antenna1_ids: tuple[int, ...]
  antenna2_ids: tuple[int, ...]
  feed1_ids: tuple[int, ...]
  feed2_ids: tuple[int, ...]
  field_ids: tuple[int, ...]
  scan_numbers: tuple[int, ...]
  data_desc_id: int
  observation_id: int
  processor_id: int
  state_id: int


@dataclass(frozen=True)
class _Extracted:
  path: str
  antennas: tuple[AntennaRow, ...]
  feeds: tuple[tuple[str, tuple[str, ...], tuple[float, ...], float | None], ...]
  fields: tuple[tuple[str, tuple[float, ...], str], ...]
  spw: SpectralWindowRow
  polarization: PolarizationRow
  observation: ObservationRow
  processor: ProcessorRow
  state: StateRow
  baseline1: tuple[str, ...]
  baseline2: tuple[str, ...]
  field_names: tuple[str, ...]
  scans: tuple[int, ...]


def _fail(context: str, message: str) -> None:
  raise FreshMSv2ValidationError(f"{context}: {message}")


def _strings(values: Any, context: str, *, unique: bool = False) -> tuple[str, ...]:
  result = tuple(np.asarray(values).tolist())
  if any(not isinstance(v, str) or not v for v in result):
    _fail(context, "requires nonempty string values")
  if unique and len(set(result)) != len(result):
    _fail(context, "names must be unique")
  return result


def _numbers(values: Any, context: str) -> tuple[float, ...]:
  data = np.asarray(values)
  if data.dtype.kind not in "iuf" or not np.all(np.isfinite(data)):
    _fail(context, "requires finite numeric values")
  return tuple(float(v) for v in data.flat)


def _variable(ds: Dataset, name: str, dims: tuple[str, ...], context: str):
  if name not in ds or ds[name].dims != dims:
    _fail(context, f"{name} requires dimensions {dims}")
  return ds[name]


def _data_variable(ds: Dataset, name: str, dims: tuple[str, ...], context: str):
  if name not in ds.data_vars:
    _fail(context, f"{name} must be a data variable with dimensions {dims}")
  return _variable(ds, name, dims, context)


def _quantity(variable, unit: str, context: str) -> None:
  if variable.attrs.get("type") != "quantity" or variable.attrs.get("units") != unit:
    _fail(context, f"requires quantity units {unit!r}")


def _string(value: Any, context: str) -> str:
  if not isinstance(value, str):
    _fail(context, "requires a string")
  return value


def _sequence(value: Any, context: str) -> tuple[str, ...]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    _fail(context, "requires a sequence of strings")
  if any(not isinstance(v, str) for v in value):
    _fail(context, "requires a sequence of strings")
  return tuple(value)


def extract_metadata(
  node: DataTree, antenna: DataTree, field: DataTree, frame: str, reference_frame: str
) -> _Extracted:
  """Read only small metadata arrays, never visibility or UVW payloads."""
  path = node.path
  ds = node.to_dataset(inherit=True)
  ad = antenna.to_dataset()
  fd = field.to_dataset()
  ac = f"Partition {path} antenna node {antenna.path}"
  fc = f"Partition {path} field node {field.path}"
  names = _strings(
    _variable(ad, "antenna_name", ("antenna_name",), ac).values,
    f"{ac} antenna_name",
  )
  if not names:
    _fail(ac, "antenna_name must not be empty")
  if len(set(names)) != len(names):
    _fail(
      ac,
      "multiple FEED configurations for one antenna within a partition are "
      "ambiguous because baselines do not identify FEED_ID",
    )
  _strings(
    _variable(ad, "cartesian_pos_label", ("cartesian_pos_label",), ac).values,
    f"{ac} cartesian_pos_label",
  )
  if tuple(ad.cartesian_pos_label.values.tolist()) != ("x", "y", "z"):
    _fail(ac, "cartesian_pos_label must be x, y, z")
  receptors = _strings(
    _variable(ad, "receptor_label", ("receptor_label",), ac).values,
    f"{ac} receptor_label",
    unique=True,
  )
  if not receptors:
    _fail(ac, "receptor_label must not be empty")
  positions = _numbers(
    _data_variable(
      ad, "ANTENNA_POSITION", ("antenna_name", "cartesian_pos_label"), ac
    ).values,
    f"{ac} ANTENNA_POSITION",
  )
  diameter_var = _data_variable(ad, "ANTENNA_DISH_DIAMETER", ("antenna_name",), ac)
  _quantity(diameter_var, "m", f"{ac} ANTENNA_DISH_DIAMETER")
  diameter = _numbers(
    diameter_var.values,
    f"{ac} ANTENNA_DISH_DIAMETER",
  )
  effective_var = _data_variable(
    ad, "ANTENNA_EFFECTIVE_DISH_DIAMETER", ("antenna_name",), ac
  )
  _quantity(effective_var, "m", f"{ac} ANTENNA_EFFECTIVE_DISH_DIAMETER")
  effective = _numbers(
    effective_var.values,
    f"{ac} ANTENNA_EFFECTIVE_DISH_DIAMETER",
  )
  if diameter != effective:
    _fail(ac, "ANTENNA_EFFECTIVE_DISH_DIAMETER differs from ANTENNA_DISH_DIAMETER")
  angle_var = _data_variable(
    ad, "ANTENNA_RECEPTOR_ANGLE", ("antenna_name", "receptor_label"), ac
  )
  _quantity(angle_var, "rad", f"{ac} ANTENNA_RECEPTOR_ANGLE")
  angles = _numbers(
    angle_var.values,
    f"{ac} ANTENNA_RECEPTOR_ANGLE",
  )
  pol_types = _strings(
    np.asarray(
      _variable(ad, "polarization_type", ("antenna_name", "receptor_label"), ac).values
    ).flat,
    f"{ac} polarization_type",
  )
  station = _strings(
    _variable(ad, "station_name", ("antenna_name",), ac).values, f"{ac} station_name"
  )
  mount = _strings(_variable(ad, "mount", ("antenna_name",), ac).values, f"{ac} mount")
  telescope = _strings(
    _variable(ad, "telescope_name", ("antenna_name",), ac).values,
    f"{ac} telescope_name",
  )
  overall = ad.attrs.get("overall_telescope_name")
  if overall is not None and (
    not isinstance(overall, str) or any(t != overall for t in telescope)
  ):
    _fail(ac, "overall_telescope_name conflicts with telescope_name")
  focus = None
  if "ANTENNA_FOCUS_LENGTH" in ad:
    _quantity(ad["ANTENNA_FOCUS_LENGTH"], "m", f"{ac} ANTENNA_FOCUS_LENGTH")
    focus = _numbers(
      _data_variable(ad, "ANTENNA_FOCUS_LENGTH", ("antenna_name",), ac).values,
      f"{ac} ANTENNA_FOCUS_LENGTH",
    )
  antennas = tuple(
    AntennaRow(n, s, m, t, positions[i * 3 : i * 3 + 3], diameter[i])
    for i, (n, s, m, t) in enumerate(zip(names, station, mount, telescope, strict=True))
  )
  nr = len(receptors)
  feeds = tuple(
    (
      name,
      pol_types[i * nr : i * nr + nr],
      angles[i * nr : i * nr + nr],
      focus[i] if focus is not None else None,
    )
    for i, name in enumerate(names)
  )

  field_names = _strings(
    _variable(fd, "field_name", ("field_name",), fc).values,
    f"{fc} field_name",
    unique=True,
  )
  if not field_names:
    _fail(fc, "field_name must not be empty")
  sky = _strings(
    _variable(fd, "sky_dir_label", ("sky_dir_label",), fc).values, f"{fc} sky_dir_label"
  )
  if tuple(sky) != ("ra", "dec"):
    _fail(fc, "sky_dir_label must be ra, dec")
  unsupported_field_data = sorted(
    set(fd.data_vars).difference({"FIELD_PHASE_CENTER_DIRECTION"})
  )
  if unsupported_field_data:
    _fail(fc, f"unsupported FIELD/SOURCE data variable(s) {unsupported_field_data}")
  phase = _numbers(
    _data_variable(
      fd, "FIELD_PHASE_CENTER_DIRECTION", ("field_name", "sky_dir_label"), fc
    ).values,
    f"{fc} FIELD_PHASE_CENTER_DIRECTION",
  )
  sources = _strings(
    _variable(fd, "source_name", ("field_name",), fc).values, f"{fc} source_name"
  )
  fields = tuple(
    (name, phase[i * 2 : i * 2 + 2], sources[i]) for i, name in enumerate(field_names)
  )

  freq = ds["frequency"]
  sc = f"Partition {path} frequency SPECTRAL_WINDOW"
  frequencies = _numbers(freq.values, f"{sc} frequency")
  attrs = freq.attrs
  name = _string(attrs.get("spectral_window_name"), f"{sc} spectral_window_name")
  intents = _sequence(
    attrs.get("spectral_window_intents"), f"{sc} spectral_window_intents"
  )
  reference = attrs.get("reference_frequency")
  if not isinstance(reference, Mapping) or not isinstance(
    reference.get("attrs"), Mapping
  ):
    _fail(sc, "reference_frequency requires data and measure attrs")
  ref_attrs = reference["attrs"]
  if ref_attrs.get("type") != "spectral_coord" or ref_attrs.get("units") != "Hz":
    _fail(sc, "reference_frequency requires a frequency measure in Hz")
  ref = _numbers(reference.get("data"), f"{sc} reference_frequency")
  if len(ref) != 1:
    _fail(sc, "reference_frequency must be scalar")
  width_info = attrs.get("channel_width")
  if (
    not isinstance(width_info, Mapping)
    or not isinstance(width_info.get("attrs"), Mapping)
    or width_info["attrs"].get("type") != "quantity"
    or width_info["attrs"].get("units") != "Hz"
  ):
    _fail(sc, "channel_width requires Hz quantity attrs")
  width_data = np.asarray(width_info.get("data"))
  if width_data.ndim != 0 or width_data.dtype.kind not in "iuf":
    _fail(sc, "channel_width must be numeric scalar")
  if np.isnan(width_data):
    channel_width = _data_variable(ds, "CHANNEL_WIDTH", ("frequency",), sc)
    _quantity(channel_width, "Hz", f"{sc} CHANNEL_WIDTH")
    widths = _numbers(channel_width.values, f"{sc} CHANNEL_WIDTH")
  else:
    widths = (float(width_data),) * len(frequencies)
  if len(widths) != len(frequencies) or not all(np.isfinite(widths)):
    _fail(sc, "channel widths must be finite per frequency")
  effective_name = attrs.get("effective_channel_width")
  if effective_name is not None and (
    not isinstance(effective_name, str) or not effective_name
  ):
    _fail(sc, "effective_channel_width must name a variable")
  if effective_name in ds.data_vars:
    effective_width = _data_variable(ds, effective_name, ("frequency",), sc)
    _quantity(effective_width, "Hz", f"{sc} {effective_name}")
    effective_widths = _numbers(effective_width.values, f"{sc} {effective_name}")
  elif effective_name in ds:
    _fail(sc, f"effective_channel_width {effective_name!r} must be a data variable")
  elif effective_name in (None, "EFFECTIVE_CHANNEL_WIDTH"):
    # The current xarray-ms reader emits this conventional label without a
    # variable. In that one known representation, CHAN_WIDTH is the bounded
    # semantic reconstruction available to the fresh writer.
    effective_widths = widths
  else:
    _fail(sc, f"effective_channel_width references missing variable {effective_name!r}")
  group_name = attrs.get("frequency_group_name")
  if group_name is not None:
    _string(group_name, f"{sc} frequency_group_name")
  spw = SpectralWindowRow(
    name,
    intents,
    frequencies,
    frame,
    ref[0],
    reference_frame,
    widths,
    effective_widths,
    widths,
    group_name,
  )

  try:
    pol = Polarisations.from_values(ds["polarization"].values.tolist())
  except (TypeError, ValueError) as e:
    raise FreshMSv2ValidationError(f"Partition {path} polarization: {e}") from e
  feed_polarizations = {feed[1] for feed in feeds}
  if len(feed_polarizations) != 1:
    _fail(
      f"Partition {path} polarization",
      "all antenna feeds must use the same receptor ordering",
    )
  feed_polarization = next(iter(feed_polarizations))
  if len(set(feed_polarization)) != len(feed_polarization):
    _fail(f"Partition {path} polarization", "feed receptor types must be unique")
  corr_product = []
  for correlation in pol.to_str():
    if len(correlation) != 2:
      _fail(
        f"Partition {path} polarization {correlation!r}",
        "only two-receptor correlations are supported",
      )
    try:
      corr_product.append(tuple(feed_polarization.index(p) for p in correlation))
    except ValueError as e:
      _fail(
        f"Partition {path} polarization {correlation!r}",
        f"is incompatible with feed receptors {feed_polarization}",
      )
      raise AssertionError("unreachable") from e
  polarization = PolarizationRow(tuple(pol.to_ints()), tuple(corr_product))
  observation = node.attrs["observation_info"]
  oc = f"Partition {path} observation_info"
  observer = observation.get("observer")
  if not isinstance(observer, list) or len(observer) != 1:
    _fail(oc, "observer must be a one-element list")
  observer_name = _string(observer[0], f"{oc} observer")
  project = _string(observation.get("project_UID"), f"{oc} project_UID")
  release = _string(observation.get("release_date"), f"{oc} release_date")
  try:
    parsed = datetime.fromisoformat(release)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
      raise ValueError("missing timezone")
  except ValueError as e:
    raise FreshMSv2ValidationError(
      f"{oc} release_date must be timezone-aware and parseable: {e}"
    ) from e
  if len(set(telescope)) != 1:
    _fail(oc, "multiple telescope names in one partition")
  obs = ObservationRow(
    observer_name, project, parsed.astimezone(timezone.utc).isoformat(), telescope[0]
  )
  proc_info = node.attrs["processor_info"]
  proc = ProcessorRow(
    _string(proc_info.get("type"), f"Partition {path} processor_info type"),
    _string(proc_info.get("sub_type"), f"Partition {path} processor_info sub_type"),
  )
  state = StateRow(
    _sequence(
      ds["scan_name"].attrs.get("scan_intents"),
      f"Partition {path} scan_name scan_intents",
    )
  )
  scans = []
  for label in ds["scan_name"].values.tolist():
    try:
      if isinstance(label, bool):
        raise ValueError("boolean is not a scan number")
      number = int(label)
      if (
        isinstance(label, (float, np.floating))
        and (not np.isfinite(label) or label != number)
        or isinstance(label, str)
        and str(number) != label.strip()
      ):
        raise ValueError("not integral")
    except (ValueError, TypeError, OverflowError) as e:
      raise FreshMSv2ValidationError(
        f"Partition {path} scan_name {label!r}: expected integral scan number"
      ) from e
    scans.append(number)
  return _Extracted(
    path,
    antennas,
    feeds,
    fields,
    spw,
    polarization,
    obs,
    proc,
    state,
    _strings(
      ds["baseline_antenna1_name"].values, f"Partition {path} baseline_antenna1_name"
    ),
    _strings(
      ds["baseline_antenna2_name"].values, f"Partition {path} baseline_antenna2_name"
    ),
    _strings(ds["field_name"].values, f"Partition {path} field_name"),
    tuple(scans),
  )


def _ordered(rows):
  return tuple(sorted(set(rows), key=repr))


def build_metadata(
  extracted: Sequence[_Extracted],
) -> tuple[MetadataRows, tuple[PartitionKeys, ...]]:
  antennas_by_name: dict[str, tuple[AntennaRow, str]] = {}
  fields_by_name: dict[str, tuple[tuple[str, tuple[float, ...], str], str]] = {}
  for part in extracted:
    for row in part.antennas:
      if previous := antennas_by_name.get(row.name):
        if previous[0] != row:
          _fail(
            f"Partition {part.path} antenna {row.name!r}",
            f"conflicts with partition {previous[1]}",
          )
      else:
        antennas_by_name[row.name] = (row, part.path)
    for field in part.fields:
      if prior_field := fields_by_name.get(field[0]):
        if prior_field[0] != field:
          _fail(
            f"Partition {part.path} field {field[0]!r}",
            f"conflicts with partition {prior_field[1]}",
          )
      else:
        fields_by_name[field[0]] = (field, part.path)
  antenna_rows = tuple(antennas_by_name[n][0] for n in sorted(antennas_by_name))
  ant_ids = {r.name: i for i, r in enumerate(antenna_rows)}
  source_rows = tuple(
    SourceRow(n)
    for n in sorted({f[2] for f, _ in fields_by_name.values()} - {"UNKNOWN"})
  )
  source_directions: dict[str, tuple[float, ...]] = {}
  for field, _ in fields_by_name.values():
    _, direction, source = field
    if source == "UNKNOWN":
      continue
    if source in source_directions and source_directions[source] != direction:
      _fail(
        f"SOURCE {source!r}",
        "multiple FIELD directions cannot define one fixed SOURCE direction",
      )
    source_directions[source] = direction
  source_ids = {r.name: i for i, r in enumerate(source_rows)}
  field_rows = tuple(
    FieldRow(n, fields_by_name[n][0][1], source_ids.get(fields_by_name[n][0][2], -1))
    for n in sorted(fields_by_name)
  )
  field_ids = {r.name: i for i, r in enumerate(field_rows)}

  # A spectral window's canonical antenna grid is defined by its full FEED
  # cohort. Scope otherwise equal spectral payloads to that cohort so a reader
  # never unions antennas/configurations from unrelated partitions.
  def scoped_spw(part):
    cohort = _ordered(
      (ant_ids[n], pol, angle, focus) for n, pol, angle, focus in part.feeds
    )
    return part.spw, cohort

  scoped_spws = _ordered(scoped_spw(part) for part in extracted)
  spw_ids = {key: i for i, key in enumerate(scoped_spws)}
  spw_rows = tuple(payload for payload, _ in scoped_spws)
  feed_rows = tuple(
    FeedRow(antenna_id, 0, pol, angle, focus, spw_id)
    for spw_id, (_, cohort) in enumerate(scoped_spws)
    for antenna_id, pol, angle, focus in cohort
  )
  pol_rows = _ordered(p.polarization for p in extracted)
  pol_ids = {r: i for i, r in enumerate(pol_rows)}
  dd_rows = _ordered(
    DataDescriptionRow(spw_ids[scoped_spw(p)], pol_ids[p.polarization])
    for p in extracted
  )
  obs_rows = _ordered(p.observation for p in extracted)
  proc_rows = _ordered(p.processor for p in extracted)
  state_rows = _ordered(p.state for p in extracted)
  metadata = MetadataRows(
    antenna_rows,
    feed_rows,
    field_rows,
    source_rows,
    spw_rows,
    pol_rows,
    dd_rows,
    obs_rows,
    proc_rows,
    state_rows,
  )
  for field_row in field_rows:
    if field_row.source_id != -1 and not 0 <= field_row.source_id < len(source_rows):
      _fail(f"FIELD {field_row.name!r}", "SOURCE_ID does not reference a planned row")
  for data_description_row in dd_rows:
    if not 0 <= data_description_row.spectral_window_id < len(spw_rows):
      _fail("DATA_DESCRIPTION", "SPECTRAL_WINDOW_ID does not reference a planned row")
    if not 0 <= data_description_row.polarization_id < len(pol_rows):
      _fail("DATA_DESCRIPTION", "POLARIZATION_ID does not reference a planned row")
  keys = []
  for part in extracted:
    spw_id = spw_ids[scoped_spw(part)]
    local_feeds = {n: 0 for n, _, _, _ in part.feeds}
    local_ants = {r.name for r in part.antennas}
    local_fields = {f[0] for f in part.fields}
    for name in (*part.baseline1, *part.baseline2):
      if name not in local_ants:
        _fail(
          f"Partition {part.path} baseline antenna {name!r}", "missing from antenna_xds"
        )
    for name in part.field_names:
      if name not in local_fields:
        _fail(
          f"Partition {part.path} field {name!r}", "missing from field_and_source node"
        )
    assignment = PartitionKeys(
      tuple(ant_ids[n] for n in part.baseline1),
      tuple(ant_ids[n] for n in part.baseline2),
      tuple(local_feeds[n] for n in part.baseline1),
      tuple(local_feeds[n] for n in part.baseline2),
      tuple(field_ids[n] for n in part.field_names),
      part.scans,
      dd_rows.index(DataDescriptionRow(spw_id, pol_ids[part.polarization])),
      obs_rows.index(part.observation),
      proc_rows.index(part.processor),
      state_rows.index(part.state),
    )
    valid_feeds = {
      (row.antenna_id, row.feed_id, row.spectral_window_id) for row in feed_rows
    }
    if any(
      (*pair, spw_id) not in valid_feeds
      for pair in zip(assignment.antenna1_ids, assignment.feed1_ids, strict=True)
    ) or any(
      (*pair, spw_id) not in valid_feeds
      for pair in zip(assignment.antenna2_ids, assignment.feed2_ids, strict=True)
    ):
      _fail(
        f"Partition {part.path}", "FEED foreign key does not reference a planned row"
      )
    for values, count in (
      (assignment.antenna1_ids + assignment.antenna2_ids, len(antenna_rows)),
      (assignment.field_ids, len(field_rows)),
      ((assignment.data_desc_id,), len(dd_rows)),
      ((assignment.observation_id,), len(obs_rows)),
      ((assignment.processor_id,), len(proc_rows)),
      ((assignment.state_id,), len(state_rows)),
    ):
      if any(value < 0 or value >= count for value in values):
        _fail(f"Partition {part.path}", "foreign key does not reference a planned row")
    keys.append(assignment)
  return metadata, tuple(keys)
