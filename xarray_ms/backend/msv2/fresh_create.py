"""Create a private, zero-MAIN-row MSv2 from a validated fresh-write plan."""

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
from arcae.lib.arrow_tables import Table, ms_descriptor

from xarray_ms.backend.msv2.fresh_plan import FreshMSv2Plan
from xarray_ms.backend.msv2.measures_encoders import EpochCoder
from xarray_ms.casa_types import NUMPY_TO_CASA_MAP, FrequencyMeasures
from xarray_ms.errors import FreshMSv2TargetError

SUBTABLES = (
  "ANTENNA",
  "FEED",
  "FIELD",
  "SOURCE",
  "SPECTRAL_WINDOW",
  "POLARIZATION",
  "DATA_DESCRIPTION",
  "OBSERVATION",
  "PROCESSOR",
  "STATE",
)
ADDROWS_BATCH = 128


def _ragged(values):
  return len({np.shape(value) for value in values}) > 1


def _metadata_columns(plan):
  m = plan.metadata
  source_direction = {
    m.source_rows[r.source_id].name: r.phase_direction
    for r in reversed(m.field_rows)
    if r.source_id >= 0
  }
  return {
    "ANTENNA": (
      m.antenna_rows,
      {
        "NAME": [r.name for r in m.antenna_rows],
        "STATION": [r.station for r in m.antenna_rows],
        "MOUNT": [r.mount for r in m.antenna_rows],
        "POSITION": [r.position for r in m.antenna_rows],
        "DISH_DIAMETER": [r.dish_diameter for r in m.antenna_rows],
        "TYPE": ["GROUND-BASED"] * len(m.antenna_rows),
        "OFFSET": [(0.0, 0.0, 0.0)] * len(m.antenna_rows),
        "FLAG_ROW": [False] * len(m.antenna_rows),
      },
    ),
    "FEED": (
      m.feed_rows,
      {
        "ANTENNA_ID": [r.antenna_id for r in m.feed_rows],
        "FEED_ID": [r.feed_id for r in m.feed_rows],
        "SPECTRAL_WINDOW_ID": [r.spectral_window_id for r in m.feed_rows],
        "NUM_RECEPTORS": [len(r.polarization_type) for r in m.feed_rows],
        "POLARIZATION_TYPE": [r.polarization_type for r in m.feed_rows],
        "RECEPTOR_ANGLE": [r.receptor_angle for r in m.feed_rows],
        "POSITION": [(0.0, 0.0, 0.0)] * len(m.feed_rows),
        "BEAM_OFFSET": [((0.0,) * len(r.polarization_type),) * 2 for r in m.feed_rows],
        "POL_RESPONSE": [
          np.eye(len(r.polarization_type), dtype=np.complex64) for r in m.feed_rows
        ],
        "BEAM_ID": [-1] * len(m.feed_rows),
        "TIME": [0.0] * len(m.feed_rows),
        "INTERVAL": [0.0] * len(m.feed_rows),
      },
    ),
    "FIELD": (
      m.field_rows,
      {
        "NAME": [r.name for r in m.field_rows],
        "SOURCE_ID": [r.source_id for r in m.field_rows],
        "PHASE_DIR": [[r.phase_direction] for r in m.field_rows],
        "DELAY_DIR": [[r.phase_direction] for r in m.field_rows],
        "REFERENCE_DIR": [[r.phase_direction] for r in m.field_rows],
        "NUM_POLY": [0] * len(m.field_rows),
        "TIME": [0.0] * len(m.field_rows),
        "CODE": [""] * len(m.field_rows),
        "FLAG_ROW": [False] * len(m.field_rows),
      },
    ),
    "SOURCE": (
      m.source_rows,
      {
        "NAME": [r.name for r in m.source_rows],
        "SOURCE_ID": list(range(len(m.source_rows))),
        "DIRECTION": [source_direction[r.name] for r in m.source_rows],
        "PROPER_MOTION": [(0.0, 0.0)] * len(m.source_rows),
        "SPECTRAL_WINDOW_ID": [-1] * len(m.source_rows),
        "NUM_LINES": [0] * len(m.source_rows),
        "TIME": [0.0] * len(m.source_rows),
        "INTERVAL": [0.0] * len(m.source_rows),
        "CALIBRATION_GROUP": [0] * len(m.source_rows),
        "CODE": [""] * len(m.source_rows),
      },
    ),
    "SPECTRAL_WINDOW": (
      m.spectral_window_rows,
      {
        "NAME": [r.name for r in m.spectral_window_rows],
        "CHAN_FREQ": [r.channel_frequency for r in m.spectral_window_rows],
        "CHAN_WIDTH": [r.channel_width for r in m.spectral_window_rows],
        "EFFECTIVE_BW": [r.effective_channel_width for r in m.spectral_window_rows],
        "RESOLUTION": [r.resolution for r in m.spectral_window_rows],
        "REF_FREQUENCY": [r.reference_frequency for r in m.spectral_window_rows],
        "MEAS_FREQ_REF": [
          FrequencyMeasures[r.frame].value for r in m.spectral_window_rows
        ],
        "NUM_CHAN": [len(r.channel_frequency) for r in m.spectral_window_rows],
        "TOTAL_BANDWIDTH": [
          sum(abs(v) for v in r.channel_width) for r in m.spectral_window_rows
        ],
        "FREQ_GROUP_NAME": [
          r.frequency_group_name or "" for r in m.spectral_window_rows
        ],
        "FREQ_GROUP": [0] * len(m.spectral_window_rows),
        "IF_CONV_CHAIN": [0] * len(m.spectral_window_rows),
        "NET_SIDEBAND": [1] * len(m.spectral_window_rows),
        "FLAG_ROW": [False] * len(m.spectral_window_rows),
      },
    ),
    "POLARIZATION": (
      m.polarization_rows,
      {
        "CORR_TYPE": [r.corr_type for r in m.polarization_rows],
        "CORR_PRODUCT": [r.corr_product for r in m.polarization_rows],
        "NUM_CORR": [len(r.corr_type) for r in m.polarization_rows],
        "FLAG_ROW": [False] * len(m.polarization_rows),
      },
    ),
    "DATA_DESCRIPTION": (
      m.data_description_rows,
      {
        "SPECTRAL_WINDOW_ID": [r.spectral_window_id for r in m.data_description_rows],
        "POLARIZATION_ID": [r.polarization_id for r in m.data_description_rows],
        "FLAG_ROW": [False] * len(m.data_description_rows),
      },
    ),
    "OBSERVATION": (
      m.observation_rows,
      {
        "OBSERVER": [r.observer for r in m.observation_rows],
        "PROJECT": [r.project_uid for r in m.observation_rows],
        "RELEASE_DATE": [
          datetime.fromisoformat(r.release_date).timestamp()
          + EpochCoder.MJD_OFFSET_SECONDS
          for r in m.observation_rows
        ],
        "TELESCOPE_NAME": [r.telescope for r in m.observation_rows],
        "TIME_RANGE": [(0.0, 0.0)] * len(m.observation_rows),
        "SCHEDULE_TYPE": [""] * len(m.observation_rows),
        "FLAG_ROW": [False] * len(m.observation_rows),
      },
    ),
    "PROCESSOR": (
      m.processor_rows,
      {
        "TYPE": [r.type for r in m.processor_rows],
        "SUB_TYPE": [r.sub_type for r in m.processor_rows],
        "TYPE_ID": [-1] * len(m.processor_rows),
        "MODE_ID": [-1] * len(m.processor_rows),
        "FLAG_ROW": [False] * len(m.processor_rows),
      },
    ),
    "STATE": (
      m.state_rows,
      {
        "OBS_MODE": [",".join(r.scan_intents) for r in m.state_rows],
        "SIG": [True] * len(m.state_rows),
        "REF": [False] * len(m.state_rows),
        "CAL": [0.0] * len(m.state_rows),
        "LOAD": [0.0] * len(m.state_rows),
        "SUB_SCAN": [0] * len(m.state_rows),
        "FLAG_ROW": [False] * len(m.state_rows),
      },
    ),
  }


def _measure_keywords(plan):
  keywords = {
    measure.column: measure.to_column_keywords()
    for part in plan.partitions
    for measure in part.measures
  }
  # Native measure columns require one quantum unit per component.
  for value in keywords.values():
    count = {"uvw": 3, "position": 3, "direction": 2}.get(value["MEASINFO"]["type"], 1)
    value["QuantumUnits"] *= count
  keywords["MAIN::TIME_CENTROID"] = keywords["MAIN::TIME"]
  keywords["OBSERVATION::TIME_RANGE"] = keywords["MAIN::TIME"]
  keywords["FIELD::DELAY_DIR"] = keywords["FIELD::PHASE_DIR"]
  keywords["FIELD::REFERENCE_DIR"] = keywords["FIELD::PHASE_DIR"]
  if plan.metadata.source_rows:
    # SOURCE directions are taken from the corresponding FIELD phase centres.
    keywords["SOURCE::DIRECTION"] = keywords["FIELD::PHASE_DIR"]
  return keywords


def _create(plan, staging):
  from xarray_ms.backend.msv2.writes import DataVariableInfo, synthesise_column_desc

  keywords = _measure_keywords(plan)
  desc = ms_descriptor("MAIN")
  managers = []
  canonical = ms_descriptor("MAIN", complete=True)
  columns = _main_columns(plan)
  for column, dtype, shapes in columns:
    canonical_desc = canonical.get(column)
    if column == "FLAG":
      # casacore accepts BOOL as an alias of canonical "boolean"; avoid a
      # spurious NonCanonicalColumnWarning from the legacy descriptor helper.
      canonical_desc = {**canonical_desc, "valueType": "BOOL"}
    desc[column] = synthesise_column_desc(
      column,
      column,
      DataVariableInfo(0, set(shapes), {np.dtype(dtype)}),
      canonical_desc,
      managers,
    )
    if column == "FLAG":
      desc[column]["valueType"] = canonical["FLAG"]["valueType"]
  for qualified, value in keywords.items():
    table, column = qualified.split("::")
    if table == "MAIN":
      desc[column]["keywords"] = value
  dminfo = {f"*{i + 1}": group for i, group in enumerate(managers)}
  with Table.ms_from_descriptor(staging, table_desc=desc, dminfo=dminfo, ninstances=1):
    pass
  for name, (rows, columns) in _metadata_columns(plan).items():
    # SOURCE is optional in arcae's default MS skeleton; its constructor also
    # registers the SOURCE table keyword in MAIN.
    if name == "SOURCE":
      table = Table.ms_from_descriptor(staging, "SOURCE", ninstances=1)
    else:
      table = Table.from_filename(
        os.path.join(staging, name), readonly=False, ninstances=1
      )
    with table:
      for qualified, value in keywords.items():
        table_name, column = qualified.split("::")
        if table_name == name:
          table.putcolkeywords(column, value)
      if rows:
        for start in range(0, len(rows), ADDROWS_BATCH):
          table.addrows(min(ADDROWS_BATCH, len(rows) - start))
        for column, values in columns.items():
          if _ragged(values):
            for row, value in enumerate(values):
              table.putcol(
                column, np.asarray(value)[None, ...], index=(slice(row, row + 1),)
              )
          else:
            table.putcol(column, np.asarray(values))


def _main_columns(plan):
  """Visibility and writer-controlled arrays share one creation-time schema."""
  data_shapes = next(
    shapes for column, _, shapes in plan.visibility_columns if column == "DATA"
  )
  return (
    *plan.visibility_columns,
    ("FLAG", np.dtype("bool").str, data_shapes),
    ("WEIGHT_SPECTRUM", np.dtype("float32").str, data_shapes),
    ("UVW", np.dtype("float64").str, ((3,),)),
    ("WEIGHT", np.dtype("float32").str, tuple(sorted({(s[1],) for s in data_shapes}))),
    ("SIGMA", np.dtype("float32").str, tuple(sorted({(s[1],) for s in data_shapes}))),
  )


def _verify(plan, staging):
  keywords = _measure_keywords(plan)
  metadata = _metadata_columns(plan)
  with Table.from_filename(staging, readonly=True, ninstances=1) as table:
    if table.nrow() != 0:
      raise ValueError("Fresh MAIN must have zero rows")
    if not {column for column, _, _ in _main_columns(plan)} <= set(table.columns()):
      raise ValueError("Fresh MAIN is missing planned array columns")
    for column, dtype, shapes in _main_columns(plan):
      descriptor = table.getcoldesc(column)
      expected_type = (
        "boolean" if column == "FLAG" else NUMPY_TO_CASA_MAP[np.dtype(dtype).type]
      )
      if descriptor["valueType"].lower() != expected_type.lower():
        raise ValueError(f"Incorrect dtype for MAIN::{column}")
      expected_shape = shapes[0] if descriptor.get("_c_order") else shapes[0][::-1]
      if len(shapes) == 1 and tuple(descriptor.get("shape", ())) != expected_shape:
        raise ValueError(f"Incorrect cell shape for MAIN::{column}")
      if descriptor["dataManagerType"] not in ("TiledColumnStMan", "TiledShapeStMan"):
        raise ValueError(f"Incorrect storage manager for MAIN::{column}")
    missing = set(SUBTABLES) - set(table.getkeywords())
    if missing:
      raise ValueError(f"Fresh MAIN is missing subtable linkage: {sorted(missing)}")
    for qualified, expected in keywords.items():
      name, column = qualified.split("::")
      if name == "MAIN" and table.getcolkeywords(column) != expected:
        raise ValueError(f"Incorrect measure keywords for {qualified}")
  for name in SUBTABLES:
    rows, columns = metadata[name]
    with Table.from_filename(
      os.path.join(staging, name), readonly=True, ninstances=1
    ) as table:
      if table.nrow() != len(rows):
        raise ValueError(f"Incorrect row count in {name}")
      for column, expected in columns.items():
        if _ragged(expected):
          for row, value in enumerate(expected):
            actual = table.getcol(column, index=(slice(row, row + 1),))
            if not np.array_equal(actual, np.asarray(value)[None, ...]):
              raise ValueError(f"Incorrect {name}::{column} row {row}")
        elif not np.array_equal(table.getcol(column), np.asarray(expected)):
          raise ValueError(f"Incorrect {name}::{column}")
      for qualified, expected in keywords.items():
        table_name, column = qualified.split("::")
        if table_name == name and table.getcolkeywords(column) != expected:
          raise ValueError(f"Incorrect measure keywords for {qualified}")


def create_fresh_msv2(plan: FreshMSv2Plan) -> str:
  """Create and verify a fresh MS, publishing only after all handles close."""
  return _transaction(plan)


def _transaction(plan, populate=None):
  """Shared private staging and no-replace publication for fresh outputs."""
  if not isinstance(plan, FreshMSv2Plan):
    raise TypeError("Expected a FreshMSv2Plan")
  target = plan.target
  if os.path.lexists(target):
    raise FreshMSv2TargetError(f"Target already exists: {target}")
  parent = os.path.dirname(target)
  if not os.path.isdir(parent):
    raise FreshMSv2TargetError(f"Target parent is missing: {parent}")
  attempt = tempfile.mkdtemp(
    prefix=f".{os.path.basename(target)}.", suffix=".staging", dir=parent
  )
  staging = os.path.join(attempt, "ms")
  try:
    _create(plan, staging)
    _verify(plan, staging)
    if populate is not None:
      populate(staging)
    if os.path.lexists(target):
      raise FreshMSv2TargetError(f"Target already exists: {target}")
    _publish_no_replace(staging, target)
  finally:
    shutil.rmtree(attempt)
  return target


def _publish_no_replace(staging: str, target: str) -> None:
  """Atomically publish without replacing a concurrent target."""
  if sys.platform == "linux":
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(
      -100, os.fsencode(staging), -100, os.fsencode(target), 1
    )  # AT_FDCWD and RENAME_NOREPLACE
    if result == -1:
      error = ctypes.get_errno()
      if error == errno.EEXIST:
        raise FreshMSv2TargetError(f"Target already exists: {target}")
      raise OSError(error, os.strerror(error), target)
  elif sys.platform == "darwin":
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renamex_np(os.fsencode(staging), os.fsencode(target), 0x00000004)
    if result == -1:
      error = ctypes.get_errno()
      if error == errno.EEXIST:
        raise FreshMSv2TargetError(f"Target already exists: {target}")
      raise OSError(error, os.strerror(error), target)
  elif sys.platform == "win32":
    os.rename(staging, target)
  else:
    raise NotImplementedError(
      f"Atomic no-replace publish unsupported on {sys.platform}"
    )
