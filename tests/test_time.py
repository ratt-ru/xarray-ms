from datetime import datetime, timezone

import pytest

from xarray_ms.time import mjds_to_utcs, utcs_to_mjds

SECONDS_IN_DAY = 24 * 60 * 60


# Worked examples from https://astroconverter.com/utcmjd.html
@pytest.mark.parametrize(
  "mjd, utc",
  [
    (51544.0, datetime(2000, 1, 1, 0, 0, 0)),
    (59999.874340277776, datetime(2023, 2, 24, 20, 59, 3)),
    (60592.87866898148, datetime(2024, 10, 9, 21, 5, 17)),
  ],
)
def test_mjd_to_utc(mjd, utc):
  """Test conversion of Modified Julian Date
  to UTC in seconds, and vice versa"""
  utc_seconds = mjds_to_utcs(mjd * SECONDS_IN_DAY)
  assert utc == datetime.utcfromtimestamp(utc_seconds)

  assert utc_seconds == utc.replace(tzinfo=timezone.utc).timestamp()
  assert utcs_to_mjds(utc_seconds) == mjd * SECONDS_IN_DAY
