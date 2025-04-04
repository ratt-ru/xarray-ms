class IrregularGridWarning(UserWarning):
  """Warning raised when the intervals associated
  with each timestep are not homogenous"""


class MissingMetadataWarning(UserWarning):
  """Warning raised when metadata is missing"""


class ImputedMetadataWarning(MissingMetadataWarning):
  """Warning raised when metadata is imputed
  if the original metadata is missing"""


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
