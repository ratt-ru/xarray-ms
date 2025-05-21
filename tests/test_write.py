from typing import Iterable

import numpy as np
import xarray
from xarray import Dataset, DataTree
from xarray.backends.api import dump_to_store

from xarray_ms.backend.msv2.entrypoint import MSv2Store
from xarray_ms.backend.msv2.entrypoint_utils import CommonStoreArgs
from xarray_ms.errors import MissingEncodingError
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES


def test_store(monkeypatch, simmed_ms, tmp_path):
  def datatree_to_msv2(
    dt: DataTree, variables: str | Iterable[str], write_inherited_coords: bool = False
  ):
    assert isinstance(dt, DataTree)

    if isinstance(variables, str):
      list_vars = [variables]
    else:
      list_vars = list(variables)

    if len(list_vars) == 0:
      return

    for node in dt.subtree:
      # Only write visibility data
      if node.attrs.get("type", None) not in CORRELATED_DATASET_TYPES:
        continue

      at_root = node is dt
      ds = node.to_dataset(inherit=write_inherited_coords or at_root)
      ds.to_msv2(list_vars)

  def dataset_to_msv2(ds: Dataset, variables: Iterable[str]):
    assert isinstance(ds, Dataset)

    if isinstance(variables, str):
      list_vars = [variables]
    else:
      list_vars = list(variables)

    if len(list_vars) == 0:
      return

    try:
      common_store_args = ds.encoding["common_store_args"]
      partition_key = ds.encoding["partition_key"]
    except KeyError as e:
      raise MissingEncodingError(
        f"Expected encoding key {e} is not present on "
        f"the dataset of type {ds.attrs.get('type', None)}. "
        f"at path {ds.path} "
        f"Writing back to a Measurement Set "
        f"is not possible without this information"
      ) from e

    # Recover common arguments used to create the original store
    # This will likely re-use existing table and structure factories
    store_args = CommonStoreArgs(**common_store_args)
    msv2_store = MSv2Store.open(
      ms=store_args.ms,
      partition_schema=store_args.partition_schema,
      partition_key=partition_key,
      preferred_chunks=store_args.preferred_chunks,
      auto_corrs=store_args.auto_corrs,
      ninstances=1,
      epoch=store_args.epoch,
      structure_factory=store_args.structure_factory,
    )

    # Strip out coordinates and attributes
    ignored_vars = set(ds.data_vars) - set(list_vars)
    ds = ds.drop_vars(ds.coords).drop_vars(ignored_vars).drop_attrs()
    try:
      dump_to_store(ds, msv2_store)
    finally:
      msv2_store.close()

  monkeypatch.setattr(Dataset, "to_msv2", dataset_to_msv2, raising=False)
  monkeypatch.setattr(DataTree, "to_msv2", datatree_to_msv2, raising=False)

  with xarray.open_datatree(simmed_ms) as xdt:
    # Overwrite UVW coordinates with zeroes
    for ds in xdt.subtree:
      if "UVW" in ds.data_vars:
        assert not np.all(ds.UVW == 0)
        ds.UVW[:] = 0

    xdt.to_msv2("UVW")

  with xarray.open_datatree(simmed_ms) as xdt:
    for ds in xdt.subtree:
      if "UVW" in ds.data_vars:
        assert np.all(ds.UVW == 0)
