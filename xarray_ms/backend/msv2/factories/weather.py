from typing import Callable, Dict

import numpy as np
import numpy.typing as npt
from xarray import Dataset, Variable

from xarray_ms.backend.msv2.antenna_selection import PartitionAntennaSelection
from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.factories.subtable import (
  LazySubtableColumns,
  partition_span,
)
from xarray_ms.backend.msv2.subtable_grid import select_rows
from xarray_ms.backend.msv2.subtable_index import SubtableIndexFactory

WEATHER_COLUMNS = (
  "H2O",
  "IONOS_ELECTRON",
  "PRESSURE",
  "REL_HUMIDITY",
  "TEMPERATURE",
  "DEW_POINT",
  "WIND_DIRECTION",
  "WIND_SPEED",
)


def _hpa_to_pa(data: npt.NDArray) -> npt.NDArray:
  return data.astype(np.float64) * 100.0


# Conversion of MSv2 WEATHER units to MSv4 units:
# {msv2_units: (msv4_units, transform)}
UNIT_CONVERSIONS: Dict[str, tuple[str, Callable[[npt.NDArray], npt.NDArray] | None]] = {
  "hPa": ("Pa", _hpa_to_pa),
  "mbar": ("Pa", _hpa_to_pa),
  "m-2": ("/m^2", None),
}


class WeatherFactory(DatasetFactory):
  """Factory class for generating the weather_xds dataset
  for a partition of the Measurement Set"""

  def get_dataset(
    self,
    selection: PartitionAntennaSelection,
    antenna_xds: Dataset,
    weather_factory: SubtableIndexFactory,
    interpolate: bool = False,
  ) -> Dataset | None:
    """Generate the weather_xds dataset for a partition.

    WEATHER rows are identified either by ``NS_WX_STATION_ID``, in which
    case the weather stations are all stations, or by ``ANTENNA_ID``, in which
    case the weather stations are the partition antennas. In both cases,
    only stations with WEATHER rows relevant to the partition are included.
    WEATHER data is read lazily.

    Parameters
    ----------
    selection : PartitionAntennaSelection
        The partition-specific antenna selection shared with antenna_xds.
    antenna_xds : Dataset
        The antenna_xds of the partition, providing
        station names and positions of antenna weather stations.
    weather_factory : SubtableIndexFactory
        Factory producing the index of the WEATHER subtable.
    interpolate : bool
        If True, WEATHER rows are associated with the
        correlated data timesteps, producing a ``time`` dimension.
        Otherwise, the WEATHER timesteps are retained,
        producing a ``time_weather`` dimension.

    Returns
    -------
    Dataset | None
        The weather_xds dataset, or None if the WEATHER subtable
        is absent, empty or has no rows relevant to this partition.
    """
    weather = weather_factory.instance

    if len(weather.groups) == 0:
      return None

    ((group, index),) = weather.groups.items()
    partition = self._structure_factory.instance[self._partition_key]

    # Stations with rows relevant to the partition
    key = index["ANTENNA_ID"]
    candidate_ids = np.unique(key[key >= 0])
    selected = select_rows(
      index["TIME"],
      index["VALID_INTERVAL"],
      key,
      candidate_ids,
      partition_span(partition),
    )
    relevant_ids = np.unique(key[selected])

    if weather.key_column == "NS_WX_STATION_ID":
      station_ids = relevant_ids
      station_names = self._station_names(station_ids)
    else:
      present = np.isin(selection.antenna_ids, relevant_ids)
      station_ids = selection.antenna_ids[present]
      station_names = antenna_xds.station_name.values[present]

    if station_ids.size == 0:
      return None

    columns = LazySubtableColumns.from_partition(
      weather_factory,
      group,
      station_ids,
      partition,
      interpolate,
      antenna_first=True,
    )

    if columns is None:
      return None

    time_dim = "time" if interpolate else "time_weather"
    dims = ("station_name", time_dim)
    data_vars = {}

    for column in WEATHER_COLUMNS:
      if (column_type := columns.column_type(column)) is None:
        continue

      row_shape, dtype = column_type

      if row_shape != ():
        continue

      units = columns.coder_factory.create(column).decode(Variable((), 0.0)).attrs
      units = units.get("units")
      msv4_units, transform = UNIT_CONVERSIONS.get(units, (units, None))
      kwargs = {} if transform is None else {"transform": transform}
      var = columns.float_variable(column, dims, (), dtype, **kwargs)
      if msv4_units is not None:
        var.attrs["units"] = msv4_units
      data_vars[column] = var

    if not data_vars:
      return None

    coords = {
      # The time coordinate type is named after its dimension
      time_dim: columns.time_variable(time_dim, time_dim),
      "station_name": Variable("station_name", station_names),
    }

    if weather.key_column == "NS_WX_STATION_ID":
      first = np.searchsorted(key, station_ids, side="left")
      position = self._station_positions(weather, index, first)
    else:
      position = antenna_xds.ANTENNA_POSITION.values[present]

    if position is not None:
      # Weather station positions are recorded in the same frame as
      # antenna positions, but MSv2 records no measures information for them
      data_vars["STATION_POSITION"] = Variable(
        ("station_name", "cartesian_pos_label"),
        position,
        antenna_xds.ANTENNA_POSITION.attrs,
      )
      coords["cartesian_pos_label"] = Variable("cartesian_pos_label", ["x", "y", "z"])

    return Dataset(data_vars=data_vars, coords=coords, attrs={"type": "weather"})

  def _station_names(self, station_ids: npt.NDArray[np.integer]) -> npt.NDArray:
    """Station names from the ASDM_STATION subtable, if present"""
    asdm_station = self._subtable_factories["ASDM_STATION"].instance

    if "name" in asdm_station.column_names:
      names = asdm_station["name"].to_numpy(zero_copy_only=False).astype(str)
      if np.all((station_ids >= 0) & (station_ids < len(names))):
        return names[station_ids]

    return np.asarray([f"Station_{i}" for i in station_ids])

  @staticmethod
  def _station_positions(weather, index, first) -> npt.NDArray | None:
    """Positions of each weather station, from its first WEATHER row"""
    if weather.column_row_type("NS_WX_STATION_POSITION", int(index["row"][0])) is None:
      return None

    rows = index["row"][first]
    order = np.argsort(rows)
    position = weather.table_factory.instance.getcol(
      "NS_WX_STATION_POSITION", (rows[order],)
    )
    result = np.empty_like(position)
    result[order] = position
    return result
