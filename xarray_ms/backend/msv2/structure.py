from dataclasses import dataclass
from numbers import Number
from typing import Any, Iterator, Mapping, Tuple

import arcae
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from xarray_ms.testing.casa_types import Polarisations

PARTITION_COLUMNS = ["FIELD_ID", "PROCESSOR_ID", "DATA_DESC_ID", "FEED1", "FEED2"]
SORTING_COLUMNS = ["TIME", "ANTENNA1", "ANTENNA2"]


@dataclass
class PartitionData:
  time: npt.NDArray[np.float64]
  antenna1: npt.NDArray[np.int32]
  antenna2: npt.NDArray[np.int32]
  tbl_to_row: npt.NDArray[np.int64]
  row: npt.NDArray[np.int64]
  chan_freq: npt.NDArray[np.float64]
  corr_type: npt.NDArray[np.int32]


PartitionKeyT = Tuple[Tuple[str, Number]]


class MSv2Structure(Mapping):
  """Holds structural information about an MSv2 dataset"""

  _ms: str
  _partitions: Mapping[PartitionKeyT, PartitionData]
  _ant: pa.Table
  _ddid: pa.Table
  _feed: pa.Table
  _spw: pa.Table
  _pol: pa.Table

  def __getitem__(self, key: PartitionKeyT) -> PartitionData:
    return self._partitions[key]

  def __iter__(self) -> Iterator[Any]:
    return iter(self._partitions)

  def __len__(self) -> int:
    return len(self._partitions)

  def __init__(self, ms):
    self._ms = ms

    with arcae.table(ms, readonly=True, lockoptions="nolock") as T:
      read_columns = set(PARTITION_COLUMNS) | set(SORTING_COLUMNS)
      index = T.to_arrow(columns=read_columns)

    index = index.append_column("row", pa.array(np.arange(len(index), dtype=np.int64)))
    agg_cmd = [(c, "list") for c in (set(read_columns) - set(PARTITION_COLUMNS))]
    agg_cmd += [("row", "list")]
    partitions = index.group_by(PARTITION_COLUMNS).aggregate(agg_cmd)
    renames = {f"{c}_list": c for c, _ in agg_cmd}
    partitions = partitions.rename_columns(
      renames.get(c, c) for c in partitions.column_names
    )

    with arcae.table(f"{ms}::ANTENNA", lockoptions="nolock") as A:
      self._ant = A.to_arrow()

    with arcae.table(f"{ms}::FEED", lockoptions="nolock") as F:
      self._feed = F.to_arrow()

    with arcae.table(f"{ms}::DATA_DESCRIPTION", lockoptions="nolock") as D:
      self._ddid = D.to_arrow()

    with arcae.table(f"{ms}::SPECTRAL_WINDOW", lockoptions="nolock") as S:
      self._spw = S.to_arrow()

    with arcae.table(f"{ms}::POLARIZATION", lockoptions="nolock") as P:
      self._pol = P.to_arrow()

    self._partitions = {}

    # Full resolution baseline map
    ant1, ant2 = np.triu_indices(len(self._ant), 0)
    ant1, ant2 = (a.astype(np.int32) for a in (ant1, ant2))

    for p in range(len(partitions)):
      key = tuple(sorted((c, partitions[c][p].as_py()) for c in PARTITION_COLUMNS))
      sort_column_names = ["TIME", "ANTENNA1", "ANTENNA2", "row"]
      sort_columns = [partitions[c][p].values.to_numpy() for c in sort_column_names]
      search_columns = {n: c for n, c in zip(sort_column_names, sort_columns)}
      utime = np.unique(search_columns["TIME"])[:, None]
      putime, pant1, pant2 = (
        a.ravel() for a in np.broadcast_arrays(utime, ant1[None, :], ant2[None, :])
      )

      sdtype = np.dtype([("t", putime.dtype), ("a1", pant1.dtype), ("a2", pant2.dtype)])

      full_res = np.zeros(putime.size, sdtype)
      full_res["t"] = putime
      full_res["a1"] = pant1
      full_res["a2"] = pant2

      search = np.zeros(sort_columns[0].size, sdtype)
      search["t"] = search_columns["TIME"]
      search["a1"] = search_columns["ANTENNA1"]
      search["a2"] = search_columns["ANTENNA2"]

      idx = np.searchsorted(full_res, search)
      tbl_to_row = np.full(putime.size, -1, np.int64)
      tbl_to_row[idx] = search_columns["row"]

      index = np.lexsort(tuple(reversed(sort_columns[:-1])))
      partition_data = {
        n.lower(): c[index] for n, c in zip(sort_column_names, sort_columns)
      }

      ddid = partitions["DATA_DESC_ID"][p].as_py()
      spw_id = self._ddid["SPECTRAL_WINDOW_ID"][ddid].as_py()
      pol_id = self._ddid["POLARIZATION_ID"][ddid].as_py()
      corr_type = Polarisations.from_values(self._pol["CORR_TYPE"][pol_id].as_py())

      partition_data.update(
        (
          ("tbl_to_row", tbl_to_row.reshape((utime.shape[0], ant1.shape[0]))),
          ("chan_freq", self._spw["CHAN_FREQ"][spw_id].as_py()),
          ("corr_type", corr_type.to_str()),
        )
      )

      self._partitions[key] = PartitionData(**partition_data)
