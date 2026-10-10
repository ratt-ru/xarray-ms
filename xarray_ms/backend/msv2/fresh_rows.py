"""Admission and bounded row payloads for the complete-grid writer."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from xarray_ms.backend.msv2.fresh_plan import FreshMSv2Partition
from xarray_ms.backend.msv2.measure_encoding import encode_fixed_measure
from xarray_ms.backend.msv2.measures_encoders import EpochCoder
from xarray_ms.errors import FreshMSv2ValidationError


@dataclass(frozen=True)
class RowGrid:
  partition: FreshMSv2Partition
  time: np.ndarray
  order: np.ndarray
  interval: float


def _fail(message):
  raise FreshMSv2ValidationError(message)


def admit_rows(tree, plan, row_grid, rows_per_batch):
  if row_grid != "complete":
    _fail("row_grid must explicitly assert 'complete'")
  if (
    isinstance(rows_per_batch, bool)
    or not isinstance(rows_per_batch, int)
    or rows_per_batch <= 0
  ):
    _fail("rows_per_batch must be a positive integer")
  grids = []
  buckets = {}
  for part in plan.partitions:
    ds = tree[part.path].to_dataset()
    marker = ds.attrs.get("xarray_ms_row_grid")
    if marker is not None:
      if not isinstance(marker, Mapping) or set(marker) != {"status", "missing_rows"}:
        _fail(f"{part.path}: malformed row grid provenance")
      missing = marker["missing_rows"]
      if (
        type(missing) is not int
        or missing < 0
        or marker["status"] not in ("complete", "padded")
      ):
        _fail(f"{part.path}: malformed row grid provenance")
      if marker["status"] != "complete" or missing != 0:
        _fail(f"{part.path}: padded or contradictory row grid provenance")
    if ds.time.dtype.kind not in "iuf":
      _fail(f"{part.path}: time must have a real numeric dtype")
    time = np.asarray(ds.time.values, dtype=np.float64)
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
      _fail(f"{part.path}: time must be finite and strictly increasing")
    encoded = EpochCoder.encode_array(time)
    if not np.all(np.isfinite(encoded)) or not np.all(np.diff(encoded) > 0):
      _fail(f"{part.path}: encoded CASA times must remain distinct and finite")
    if ds.uvw_label.values.tolist() != ["u", "v", "w"]:
      _fail(f"{part.path}: uvw_label must be u, v, w")
    keys = part.foreign_keys
    pairs = list(zip(keys.antenna1_ids, keys.antenna2_ids, strict=True))
    names = tree[part.antenna].antenna_name.values.tolist()
    ants = sorted(
      i for i, a in enumerate(plan.metadata.antenna_rows) if a.name in names
    )
    cross = {(a, b) for a in ants for b in ants if a < b}
    auto = cross | {(a, a) for a in ants}
    if len(set(pairs)) != len(pairs) or set(pairs) not in (cross, auto):
      _fail(
        f"{part.path}: baselines must be exactly canonical crosses or autos and crosses"
      )
    interval = ds.time.attrs.get("integration_time")
    attrs = interval.get("attrs") if isinstance(interval, Mapping) else None
    if (
      not isinstance(attrs, Mapping)
      or attrs.get("type") != "quantity"
      or attrs.get("units") != "s"
    ):
      _fail(f"{part.path}: integration_time requires a seconds quantity")
    value = np.asarray(interval.get("data"))
    if (
      value.ndim != 0
      or value.dtype.kind not in "iuf"
      or not np.isfinite(value)
      or value <= 0
    ):
      _fail(f"{part.path}: integration_time must be finite and positive")
    with np.errstate(over="ignore"):
      edges = encoded[[0, -1]] + np.array([-0.5, 0.5]) * float(value)
    if not np.all(np.isfinite(edges)):
      _fail(f"{part.path}: integration edges must remain finite")
    for name in ("TIME", "INTEGRATION_TIME"):
      if name in ds:
        _fail(f"{part.path}: irregular {name} is unsupported")
    for name in ("TIME_CENTROID", "EFFECTIVE_INTEGRATION_TIME"):
      if name not in ds:
        continue
      var = ds[name]
      if (
        var.dims not in (("time",), ("time", "baseline_id"))
        or var.dtype.kind not in "iuf"
      ):
        _fail(f"{part.path}: invalid {name} dimensions or dtype")
      if name == "TIME_CENTROID":
        measure = encode_fixed_measure(
          var.variable, part.path, name, "MAIN::TIME_CENTROID"
        )
        reference = next(m for m in part.measures if m.column == "MAIN::TIME")
        if measure.msv2_type != reference.msv2_type or measure.frame != reference.frame:
          _fail(f"{part.path}: TIME_CENTROID reference must match TIME")
      elif var.attrs.get("type") != "quantity" or var.attrs.get("units") != "s":
        _fail(f"{part.path}: EFFECTIVE_INTEGRATION_TIME requires seconds")
    grid = RowGrid(
      part,
      time,
      np.asarray(sorted(range(len(pairs)), key=pairs.__getitem__)),
      float(value),
    )
    # Axis-sized sets, never a time x baseline row-key set.
    scalar = (keys.data_desc_id, keys.observation_id, keys.processor_id, keys.state_id)
    # Deduplicate the actual native keys: adding the MJD epoch offset can
    # collapse distinct Unix values from separate partition axes.
    temporal = set(
      zip(encoded.tolist(), keys.field_ids, keys.scan_numbers, strict=True)
    )
    baseline = set(
      zip(
        keys.antenna1_ids,
        keys.antenna2_ids,
        keys.feed1_ids,
        keys.feed2_ids,
        strict=True,
      )
    )
    for old_time, old_baseline in buckets.setdefault(scalar, []):
      if temporal & old_time and baseline & old_baseline:
        _fail(f"{part.path}: overlapping MAIN row keys across partitions")
    buckets[scalar].append((temporal, baseline))
    grids.append(grid)
  return tuple(grids)


def row_batches(tree, grids, bound):
  for grid in grids:
    part = grid.partition
    ds = tree[part.path].to_dataset()
    k = part.foreign_keys
    for t, time in enumerate(grid.time):
      for start in range(0, len(grid.order), bound):
        b = grid.order[start : start + bound]
        count = len(b)
        selection = {"time": slice(t, t + 1), "baseline_id": b}

        def read(name):
          return np.asarray(ds[name].isel(**selection).values).reshape(
            (count,) + ds[name].shape[2:]
          )

        flags = read(part.flag)
        weights = read(part.weight)
        uvw = read(part.uvw).astype(np.float64)
        if not np.all((flags == 0) | (flags == 1)):
          _fail(f"{part.path}: FLAG uint8 values must be 0 or 1")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
          _fail(f"{part.path}: weights must be finite and nonnegative")
        with np.errstate(over="ignore"):
          weights = weights.astype(np.float32)
        if not np.all(np.isfinite(weights)):
          _fail(f"{part.path}: weights overflow float32")
        if not np.all(np.isfinite(uvw)):
          _fail(f"{part.path}: UVW must be finite")

        def temporal(name, default):
          if name not in ds:
            return np.full(count, default, dtype=np.float64)
          var = ds[name]
          selection = {"time": slice(t, t + 1)}
          if "baseline_id" in var.dims:
            selection["baseline_id"] = b
          values = np.asarray(var.isel(**selection).values, dtype=np.float64).ravel()
          if not np.all(np.isfinite(values)):
            _fail(f"{part.path}: {name} must be finite")
          return np.broadcast_to(values, (count,)).copy()

        centroid = temporal("TIME_CENTROID", time)
        exposure = temporal("EFFECTIVE_INTEGRATION_TIME", grid.interval)
        if np.any(exposure < 0) or np.any(exposure > grid.interval):
          _fail(f"{part.path}: exposure must be between zero and interval")
        # Accumulate in double precision to avoid overflowing a float32 sum.
        weight = weights.mean(axis=1, dtype=np.float64).astype(np.float32)
        sigma = np.zeros_like(weight)
        np.divide(1.0, np.sqrt(weight), out=sigma, where=weight > 0)
        rows = {
          "DATA": read(part.correlated_data),
          "FLAG": flags.astype(bool),
          "WEIGHT_SPECTRUM": weights,
          "UVW": uvw,
          "WEIGHT": weight,
          "SIGMA": sigma,
          "FLAG_ROW": flags.astype(bool).all(axis=(1, 2)),
          "TIME": EpochCoder.encode_array(np.full(count, time)),
          "TIME_CENTROID": EpochCoder.encode_array(centroid),
          "INTERVAL": np.full(count, grid.interval),
          "EXPOSURE": exposure,
          "FIELD_ID": np.full(count, k.field_ids[t], dtype=np.int32),
          "SCAN_NUMBER": np.full(count, k.scan_numbers[t], dtype=np.int32),
          "ARRAY_ID": np.zeros(count, dtype=np.int32),
        }
        for col, values in (
          ("ANTENNA1", k.antenna1_ids),
          ("ANTENNA2", k.antenna2_ids),
          ("FEED1", k.feed1_ids),
          ("FEED2", k.feed2_ids),
        ):
          rows[col] = np.asarray(values, dtype=np.int32)[b]
        for col, value in (
          ("DATA_DESC_ID", k.data_desc_id),
          ("OBSERVATION_ID", k.observation_id),
          ("PROCESSOR_ID", k.processor_id),
          ("STATE_ID", k.state_id),
        ):
          rows[col] = np.full(count, value, dtype=np.int32)
        yield rows
