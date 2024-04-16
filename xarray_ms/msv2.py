from dataclasses import dataclass
from numbers import Number
from typing import Dict, Tuple

import numpy as np
from arcae.lib.arrow_tables import Table

from xarray_ms.query import groupby_clause, orderby_clause, select_clause, where_clause

PARTITION_COLUMNS = [
  "FIELD_ID",
  "PROCESSOR_ID",
  "DATA_DESC_ID",
  "SCAN_NUMBER",
  "FEED1",
  "FEED2",
]
PARTITION_COLUMNS = ["FIELD_ID", "PROCESSOR_ID", "DATA_DESC_ID", "FEED1", "FEED2"]
SORTING_COLUMNS = ["TIME", "ANTENNA1", "ANTENNA2"]


@dataclass
class PartitionIndex:
  time: np.ndarray[np.float64]
  antenna1: np.ndarray[np.int32]
  antenna2: np.ndarray[np.int32]
  row: np.ndarray[np.int64]


PartitionMapT = Dict[Tuple[Tuple[str, Number], ...], PartitionIndex]


def partition(path: str) -> PartitionMapT:
  """Obtain a partitioning of an MSv2 dataset"""
  sort_group_cols = [f"GAGGR({c}) as GROUP_{c}" for c in SORTING_COLUMNS]
  sort_group_cols.append("GROWID() as __tablerow__")
  sort_group_cols.append("GCOUNT() as __tablerows__")

  groupby = groupby_clause(PARTITION_COLUMNS)
  select = select_clause(PARTITION_COLUMNS + sort_group_cols)
  query = f"{select}\nFROM\n\t{path}\n{groupby}"

  with Table.from_taql(query) as Q:
    partitions = Q.to_arrow()

  partition_map: PartitionMapT = {}

  for p in range(len(partitions)):
    nrows = partitions["__tablerows__"][p].as_py()
    key = tuple(sorted((c, partitions[c][p].as_py()) for c in PARTITION_COLUMNS))
    row = partitions["__tablerow__"][p].values.to_numpy()
    sort_cols = tuple(
      partitions[f"GROUP_{c}"][p].values.to_numpy() for c in SORTING_COLUMNS
    )
    assert all(nrows == len(a) for a in sort_cols)
    index = np.lexsort(tuple(reversed(sort_cols)))
    kw = {c.lower(): v[index] for c, v in zip(SORTING_COLUMNS, sort_cols)}
    partition_map[key] = PartitionIndex(row=row, **kw)

  return partition_map


def xds_from_ms(path: str):
  groupby = groupby_clause(PARTITION_COLUMNS)
  select = select_clause(PARTITION_COLUMNS)
  query = f"{select}\nFROM\n\t{path}\n{groupby}"

  with Table.from_taql(query) as Q:
    partitions = Q.to_arrow()

  for partition in partitions.to_pylist():
    select = select_clause(SORTING_COLUMNS + ["INTERVAL"])
    orderby = orderby_clause(SORTING_COLUMNS)
    where = where_clause(partition.keys(), partition.values())
    query = f"{select}\nFROM\n\t{path}\n{where}\n{orderby}"

    with Table.from_taql(query) as Q:
      time_bl_index = Q.to_arrow()
      uintervals = time_bl_index["INTERVAL"].unique().to_numpy()

      if uintervals.size > 1:
        pstr = ",".join(f"{k}={v}" for k, v in partition.items())
        print(
          f"Partition [{pstr}] of length {len(time_bl_index)} "
          f"doesn't have unique intervals {uintervals}"
        )
