class IrregularGridWarning(UserWarning):
  """Base Warning for irregular grids"""


class IrregularTimeGridWarning(IrregularGridWarning):
  """Warning raised when the intervals associated
  with each timestep are not homogenous"""


class IrregularBaselineGridWarning(IrregularGridWarning):
  """Warning raised when missing baselines are
  present in the Measurement Set"""


class IrregularChannelGridWarning(IrregularGridWarning):
  """Warning raised when an irregular channel grid
  is encountered"""


class MissingMetadataWarning(UserWarning):
  """Warning raised when metadata is missing"""


class ColumnShapeImputationWarning(UserWarning):
  """Warning raised when a colum shape cannot be imputed"""


class ImputedMetadataWarning(MissingMetadataWarning):
  """Warning raised when metadata is imputed
  if the original metadata is missing"""


class FrameConversionWarning(UserWarning):
  """Warning raised if there's no defined conversion from a CASA
  to astropy reference frame"""


class InvalidMeasurementSet(ValueError):
  """Raised when the Measurement Set foreign key indexing is invalid"""


class InvalidPartitionKey(ValueError):
  """Raised when a string representing a partition key is invalid"""


class PartitioningError(ValueError):
  """Raised when a logical error is encountered during Measurement Set partitioning"""


class MissingMeasuresInfo(ValueError):
  """Raised when Measures information is missing from the column"""


class MissingQuantumUnits(ValueError):
  """Raised when QuantumUnits information is missing from the column"""


class MultipleQuantumUnits(ValueError):
  """Raised when there are multiple QuantumUnit value types in the column"""
