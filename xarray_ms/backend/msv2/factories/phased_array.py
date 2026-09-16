import numpy as np
from xarray import DataArray, Dataset, Variable

from xarray_ms.backend.msv2.factories.core import DatasetFactory
from xarray_ms.backend.msv2.measures_encoders import MSv2CoderFactory
from xarray_ms.backend.msv2.table_utils import AntennaSelection
from xarray_ms.errors import InvalidMeasurementSet


class PhasedArrayFactory(DatasetFactory):
  """Factory class for generating the phased_array_xds dataset
  for a partition of the Measurement Set"""

  def get_dataset(
    self,
    selection: AntennaSelection,
    receptor_label: DataArray,
    polarization_type: DataArray,
  ) -> Dataset | None:
    """Generate the phased_array_xds dataset for a partition of the Measurement Set.

    Parameters
    ----------
    selection : AntennaSelection
        The partition-specific antenna and feed selection shared with antenna_xds.
    receptor_label : DataArray
        The receptor labels for the phased array, extracted from the ANTENNA table
        subset relevant to the partition.
    polarization_type : DataArray
        The polarization types for the phased array, extracted from the ANTENNA table
        subset relevant to the partition.

    Returns
    -------
    Dataset | None
        An xarray Dataset containing the phased array information for the specified
        partition, or None if the PHASED_ARRAY subtable is not present.
    """
    import pyarrow.compute as pac

    phased_array = self._subtable_factories["PHASED_ARRAY"].instance

    # MS does not contain the optional PHASED_ARRAY subtable,
    # or it is empty -- happens for some OSKAR SKA Mid simulations.
    if phased_array is None or len(phased_array) == 0:
      return None

    # Take the row subset that corresponds to the selected antenna_ids, but
    # match the order of the selection, NOT the order of the PHASED_ARRAY table.
    # The antenna_name coordinate of phased_array_xds must be aligned with
    # that of antenna_xds.
    phased_array_antenna_ids = phased_array["ANTENNA_ID"].to_numpy()

    if not set(selection.antenna_ids).issubset(set(phased_array_antenna_ids)):
      raise InvalidMeasurementSet(
        f"PHASED_ARRAY table does not contain all antenna_ids "
        f"for partition {self._partition_key}. "
        f"antenna_ids in PHASED_ARRAY: {phased_array_antenna_ids.tolist()}, "
        f"antenna_ids in partition: {selection.antenna_ids.tolist()}"
      )

    mapping_antenna_row = {
      antenna_id: row_index
      for row_index, antenna_id in enumerate(phased_array_antenna_ids)
    }
    row_indices = [
      mapping_antenna_row[antenna_id] for antenna_id in selection.antenna_ids
    ]
    phased_array = phased_array.take(row_indices)

    if len(phased_array) == 0:
      raise InvalidMeasurementSet(
        f"No entries were found in PHASED_ARRAY matching "
        f"antenna_ids = {selection.antenna_ids.tolist()}"
      )

    # Coordinate axes
    coordinate_axes = (
      pac.list_flatten(phased_array["COORDINATE_AXES"], recursive=True)
      .to_numpy()
      .reshape(-1, 3, 3)
    )
    coordinate_axes_var = Variable(
      dims=("antenna_name", "cartesian_pos_label_local", "cartesian_pos_label"),
      data=coordinate_axes,
    )
    num_stations = len(coordinate_axes)

    # Element offsets from their respective station positions
    element_offset = (
      pac.list_flatten(phased_array["ELEMENT_OFFSET"], recursive=True)
      .to_numpy()
      .reshape(num_stations, 3, -1)
    )
    element_offset_var = Variable(
      dims=("antenna_name", "cartesian_pos_label_local", "element_id"),
      data=element_offset,
    )
    num_elements = element_offset.shape[-1]

    # Element flags
    num_receptors = len(receptor_label)
    element_flag = (
      pac.list_flatten(phased_array["ELEMENT_FLAG"], recursive=True)
      .to_numpy()
      .astype(bool)  # otherwise we get uint8 which is not schema-compliant
      .reshape(num_stations, num_receptors, num_elements)
    )
    element_flag_var = Variable(
      dims=("antenna_name", "receptor_label", "element_id"),
      data=element_flag,
    )

    coder_factory = MSv2CoderFactory.from_arrow_table(phased_array)

    data_vars = {
      "PHASED_ARRAY_COORDINATE_AXES": coder_factory.create("COORDINATE_AXES").decode(
        coordinate_axes_var
      ),
      "PHASED_ARRAY_ELEMENT_OFFSET": coder_factory.create("ELEMENT_OFFSET").decode(
        element_offset_var
      ),
      "PHASED_ARRAY_ELEMENT_FLAG": coder_factory.create("ELEMENT_FLAG").decode(
        element_flag_var
      ),
    }

    coords = {
      "antenna_name": selection.unique_antenna_names,
      "receptor_label": receptor_label,
      "polarization_type": polarization_type,
      "cartesian_pos_label": ["x", "y", "z"],
      "cartesian_pos_label_local": ["p", "q", "r"],
      "element_id": np.arange(num_elements),
    }

    return Dataset(
      data_vars=data_vars,
      coords=coords,
      attrs={"type": "phased_array"},
    )
