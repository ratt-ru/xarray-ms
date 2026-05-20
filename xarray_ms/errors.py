class IgnoredArgument(UserWarning):
  """Issued when keyword arguments are passed that this backend does not support."""


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


class DuplicateAntennaNameWarning(UserWarning):
  """Warning raised when duplicate antenna names are found in the ANTENNA table
  and are made unique by appending a numeric suffix"""


class FrameConversionWarning(UserWarning):
  """Warning raised if there's no defined conversion from a CASA
  to astropy reference frame"""


class InvalidMeasurementSet(ValueError):
  """Raised when the Measurement Set is invalid in some way"""


class ComplexMeasurementSet(ValueError):
  """Raised when complexity in the Measurement Set prevents processing"""


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
