import warnings
from typing import Dict, Tuple

import numpy as np
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.antenna_selection import PartitionAntennaSelection
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.factories.subtable import LazySubtableColumns
from xarray_ms.backend.msv2.subtable_index import SubtableIndexFactory
from xarray_ms.errors import ColumnShapeImputationWarning

# MSv4 variables, mapped onto per-receptor MSv2 SYSCAL columns
# and their per-receptor and per-channel variants
SYSCAL_COLUMNS: Dict[str, Tuple[str, str]] = {
  "TCAL": ("TCAL", "TCAL_SPECTRUM"),
  "TRX": ("TRX", "TRX_SPECTRUM"),
  "TSKY": ("TSKY", "TSKY_SPECTRUM"),
  "TSYS": ("TSYS", "TSYS_SPECTRUM"),
  "TANT": ("TANT", "TANT_SPECTRUM"),
  "TANT_SYS": ("TANT_TSYS", "TANT_TSYS_SPECTRUM"),
}


class SystemCalibrationFactory(DatasetFactory):
  """Factory class for generating the system_calibration_xds dataset
  for a partition of the Measurement Set"""

  def get_dataset(
    self,
    selection: PartitionAntennaSelection,
    antenna_xds: Dataset,
    syscal_factory: SubtableIndexFactory,
    interpolate: bool = False,
  ) -> Dataset | None:
    """Generate the system_calibration_xds dataset for a partition.

    SYSCAL data is read lazily.

    Parameters
    ----------
    selection : PartitionAntennaSelection
        The partition-specific antenna selection shared with antenna_xds.
    antenna_xds : Dataset
        The antenna_xds of the partition, providing antenna
        and receptor coordinates.
    syscal_factory : SubtableIndexFactory
        Factory producing the index of the SYSCAL subtable,
        partitioned by SPECTRAL_WINDOW_ID.
    interpolate : bool
        If True, SYSCAL rows are associated with the
        correlated data timesteps, producing a ``time`` dimension.
        Otherwise, the SYSCAL timesteps are retained,
        producing a ``time_system_cal`` dimension.

    Returns
    -------
    Dataset | None
        The system_calibration_xds dataset, or None if the SYSCAL subtable
        is absent, empty or has no rows relevant to this partition.
    """
    partition = self._structure_factory.instance[self._partition_key]
    group = (("SPECTRAL_WINDOW_ID", partition.spw_id),)

    if group not in syscal_factory.instance.groups:
      return None

    columns = LazySubtableColumns.from_partition(
      syscal_factory,
      group,
      selection.antenna_ids,
      partition,
      interpolate,
      antenna_first=True,
    )

    if columns is None:
      return None

    time_dim = "time" if interpolate else "time_system_cal"
    nreceptors = antenna_xds.sizes["receptor_label"]
    nchan: int | None = None
    data_vars = {}

    for name, (column, spectrum_column) in SYSCAL_COLUMNS.items():
      # Prefer the per-channel variant
      if (column_type := columns.column_type(spectrum_column)) is not None:
        (row_shape, dtype), source = column_type, spectrum_column
      elif (column_type := columns.column_type(column)) is not None:
        (row_shape, dtype), source = column_type, column
      else:
        continue

      if row_shape == (nreceptors,):
        var = columns.float_variable(
          source, ("antenna_name", time_dim, "receptor_label"), (nreceptors,), dtype
        )
      elif (
        len(row_shape) == 2
        and row_shape[1] == nreceptors
        and (nchan is None or nchan == row_shape[0])
      ):
        nchan = row_shape[0]
        # (chan, receptor) columns are presented as (receptor, chan)
        var = columns.float_variable(
          source,
          ("antenna_name", time_dim, "receptor_label", "frequency_system_cal"),
          (nreceptors, nchan),
          dtype,
          column_axes=(1, 0),
        )
      else:
        warnings.warn(
          f"Ignoring SYSCAL::{source} with row shape {row_shape}, "
          f"which is inconsistent with {nreceptors} receptors "
          f"or {nchan} channels",
          ColumnShapeImputationWarning,
        )
        continue

      # SYSCAL temperatures are Kelvin, but some columns,
      # such as TANT_TSYS, record no units
      if "units" not in var.attrs:
        var.attrs.update({"type": "quantity", "units": "K"})

      data_vars[name] = var

    if (phase_diff := columns.column_type("PHASE_DIFF")) is not None:
      row_shape, dtype = phase_diff
      if row_shape == ():
        data_vars["PHASE_DIFFERENCE"] = columns.float_variable(
          "PHASE_DIFF", ("antenna_name", time_dim), (), dtype
        )

    if not data_vars:
      return None

    coords = {
      # The schema types this time coordinate as "time"
      time_dim: columns.time_variable(time_dim, "time"),
      "antenna_name": Variable("antenna_name", selection.antenna_names),
      "station_name": antenna_xds.station_name.variable,
      "receptor_label": antenna_xds.receptor_label.variable,
      "polarization_type": antenna_xds.polarization_type.variable,
    }

    if nchan is not None:
      # SYSCAL spectra are generally at a different
      # resolution to the correlated data
      coords["frequency_system_cal"] = Variable(
        "frequency_system_cal", np.arange(nchan)
      )

    return Dataset(
      data_vars=data_vars, coords=coords, attrs={"type": "system_calibration"}
    )
