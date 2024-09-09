class IrregularGridWarning(UserWarning):
  """Warning raised when the intervals associated
  with each timestep are not homogenous"""


class InvalidMeasurementSet(ValueError):
  """Raised when the Measurement Set foreign key indexing is invalid"""


class InvalidPartitionKey(ValueError):
  """Raised when a string representing a partition key is invalid"""


class MissingMeasuresInfo(ValueError):
  """Raised when Measures information is missing from the column"""


class MissingQuantumUnits(ValueError):
  """Raised when QuantumUnits information is missing from the column"""
