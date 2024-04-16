from datetime import datetime

import numpy as np

# Modified Julian Date Epoch
MJD_EPOCH: datetime = datetime(1858, 11, 17)
# UTC Epoch
UTC_EPOCH: datetime = datetime(1970, 1, 1)
# Difference in seconds between the UTC epoch and MJD epoch
MJD_OFFSET_SECONDS: float = (UTC_EPOCH - MJD_EPOCH).total_seconds()


def mjds_to_utcs(mjds: np.ndarray) -> np.ndarray:
  """Convert Modified Julian Date in seconds to UTC in seconds"""
  return mjds - MJD_OFFSET_SECONDS


def utcs_to_mjds(utcs: np.ndarray) -> np.ndarray:
  """Convert UTC in seconds to Modified Julian Date"""
  return utcs + MJD_OFFSET_SECONDS
