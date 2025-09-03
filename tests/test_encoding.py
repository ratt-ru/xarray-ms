from datetime import datetime, timezone

import numpy as np
import pytest
import xarray.testing as xt
from arcae.lib.arrow_tables import ms_descriptor
from xarray import Variable

from xarray_ms.backend.msv2.encoders import TimeCoder
from xarray_ms.casa_types import ColumnDesc

MAIN_TABLE_DESC = ms_descriptor("MAIN", True)
COLUMN_DESCS = {
  c: ColumnDesc.from_descriptor(c, MAIN_TABLE_DESC)
  for c in MAIN_TABLE_DESC
  if not c.startswith("_")
}
SECONDS_IN_DAY = 24 * 60 * 60


# Worked examples from https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/DateConv/dateconv.pl
@pytest.mark.parametrize(
  "mjd, utc",
  [
    (51544.0, datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
    (59999.874340277776, datetime(2023, 2, 24, 20, 59, 3, tzinfo=timezone.utc)),
    (60592.87866898148, datetime(2024, 10, 9, 21, 5, 17, tzinfo=timezone.utc)),
  ],
)
def test_utc_time_encoder_roundtrip(mjd, utc):
  """Test conversion of Modified Julian Date
  to UTC in seconds, and vice versa"""

  time_coder = TimeCoder(COLUMN_DESCS["TIME"])
  time = Variable(("dummy",), np.array([mjd * SECONDS_IN_DAY]))
  decoded_time = time_coder.decode(time)
  utc_seconds = decoded_time.values[0]
  assert utc == datetime.fromtimestamp(utc_seconds, timezone.utc)
  assert decoded_time.attrs == {
    "type": "time",
    "units": "s",
    "scale": "utc",
    "format": "unix",
  }
  encoded_time = time_coder.encode(decoded_time)
  xt.assert_equal(encoded_time, time)
  assert encoded_time.attrs == {}
