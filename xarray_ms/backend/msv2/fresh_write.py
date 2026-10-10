"""Transactional reduced complete-grid MSv4 to fresh MSv2 writer."""

import os

import numpy as np
from arcae.lib.arrow_tables import Table

from xarray_ms.backend.msv2.fresh_create import SUBTABLES, _transaction
from xarray_ms.backend.msv2.fresh_plan import plan_fresh_msv2
from xarray_ms.backend.msv2.fresh_rows import admit_rows, row_batches
from xarray_ms.backend.msv2.measures_encoders import EpochCoder


def _populate(tree, plan, grids, staging, bound):
  row = 0
  with Table.from_filename(staging, readonly=False, ninstances=1) as table:
    for columns in row_batches(tree, grids, bound):
      count = len(columns["TIME"])
      table.addrows(count)
      for column, values in columns.items():
        table.putcol(column, values, index=(slice(row, row + count),))
      row += count
  ranges = {}
  for grid in grids:
    obs = grid.partition.foreign_keys.observation_id
    lo, hi = (
      EpochCoder.encode_array(grid.time[[0, -1]])
      + np.array([-0.5, 0.5]) * grid.interval
    )
    if obs in ranges:
      old = ranges[obs]
      lo, hi = min(lo, old[0]), max(hi, old[1])
    ranges[obs] = (lo, hi)
  with Table.from_filename(
    os.path.join(staging, "OBSERVATION"), readonly=False, ninstances=1
  ) as table:
    for obs, value in ranges.items():
      table.putcol("TIME_RANGE", np.asarray([value]), index=(slice(obs, obs + 1),))
  _verify_populated(tree, plan, grids, staging, bound, row, ranges)


def _verify_populated(tree, plan, grids, staging, bound, count, ranges):
  with Table.from_filename(staging, readonly=True, ninstances=1) as table:
    if table.nrow() != count or set(SUBTABLES) - set(table.getkeywords()):
      raise ValueError("Incorrect populated MAIN count or subtable closure")
    row = 0
    for columns in row_batches(tree, grids, bound):
      size = len(columns["TIME"])
      for column, expected in columns.items():
        actual = table.getcol(column, index=(slice(row, row + size),))
        if actual.shape != expected.shape or not np.array_equal(
          actual, expected, equal_nan=True
        ):
          raise ValueError(f"Incorrect populated MAIN::{column} at row {row}")
      row += size
  for name in SUBTABLES:
    with Table.from_filename(
      os.path.join(staging, name), readonly=True, ninstances=1
    ) as table:
      if table.nrow() != len(
        getattr(
          plan.metadata,
          {
            "DATA_DESCRIPTION": "data_description_rows",
            "SPECTRAL_WINDOW": "spectral_window_rows",
          }.get(name, name.lower() + "_rows"),
        )
      ):
        raise ValueError(f"Incorrect populated {name} count")
      if name == "OBSERVATION":
        for obs, expected in ranges.items():
          if not np.array_equal(
            table.getcol("TIME_RANGE", index=(slice(obs, obs + 1),)), [expected]
          ):
            raise ValueError("Incorrect OBSERVATION::TIME_RANGE")


def write_fresh_msv2(
  tree, target, *, row_grid, data_group="base", rows_per_batch=128
) -> str:
  """Write the selected group to DATA in a new, complete-grid MSv2.

  ``row_grid='complete'`` is a required caller assertion. Known padded sources
  are rejected. Weights narrow to float32; overflow is rejected. WEIGHT is the
  channel mean and SIGMA is its reciprocal square root, with zero for zero weight.
  All source payload reads and native writes are bounded by ``rows_per_batch``.
  """
  plan = plan_fresh_msv2(tree, target, data_group)
  grids = admit_rows(tree, plan, row_grid, rows_per_batch)
  return _transaction(
    plan, lambda staging: _populate(tree, plan, grids, staging, rows_per_batch)
  )
