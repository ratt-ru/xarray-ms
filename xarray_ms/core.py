import json

import xarray
from xarray.backends.api import open_datatree
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree

try:
  import zarr
except ImportError:
  zarr = None


def encode_attributes(ds: Dataset) -> Dataset:
  """Encode the antenna_xds attribute of a Dataset."""

  # Attempt to encode the the antenna_xds attribute
  ant_xds = ds.attrs.get("antenna_xds", None)
  if ant_xds is None:
    return ds
  elif isinstance(ant_xds, Dataset):
    ant_xds = json.dumps(ant_xds.to_dict())
    return ds.assign_attrs(antenna_xds=ant_xds)
  else:
    raise TypeError(
      f"antenna_xds attribute must be an xarray Dataset "
      f"but a {type(ant_xds)} was present"
    )


def decode_attributes(ds: Dataset) -> Dataset:
  """Decode the antenna_xds attribute of a Dataset."""
  # Attempt to decode the the antenna_xds attribute
  ant_xds = ds.attrs["antenna_xds"]
  if isinstance(ant_xds, str):
    antenna_dict = json.loads(ant_xds)
    ant_ds = Dataset.from_dict(antenna_dict)
    return ds.assign_attrs(antenna_xds=ant_ds)
  elif isinstance(ant_xds, Dataset):
    return ds
  else:
    raise TypeError(
      f"antenna_xds must be an xarray Dataset or a JSON encoded Dataset "
      f"but a {type(ant_xds)} was present"
    )


def xds_from_zarr(*args, **kwargs):
  """Read a Measurement Set-like :class:`~xarray.Dataset` from a Zarr store.

  Thin wrapper around :func:`xarray.open_zarr`."""
  if zarr is None:
    raise ImportError("pip install zarr")

  return decode_attributes(xarray.open_zarr(*args, **kwargs))


def xds_to_zarr(ds: Dataset, *args, **kwargs) -> None:
  """Write a Measurement Set-like :class:`~xarray.Dataset` to a Zarr store.

  Thin wrapper around :meth:`xarray.Dataset.to_zarr`.
  """
  if zarr is None:
    raise ImportError("pip install zarr")

  return encode_attributes(ds).to_zarr(*args, **kwargs)


def xdt_from_zarr(*args, **kwargs):
  """Read a Measurement Set-like :class:`~xarray.core.datatree.DataTree`
  from a Zarr store.

  Thin wrapper around :func:`xarray.backends.api.open_datatree`."""
  if zarr is None:
    raise ImportError("pip install zarr")

  return open_datatree(*args, **kwargs).map_over_subtree(decode_attributes)


def xdt_to_zarr(dt: DataTree, *args, **kwargs) -> None:
  """Read a Measurement Set-like :class:`~xarray.core.datatree.DataTree`
  to a Zarr store

  Thin wrapper around :meth:`xarray.core.datatree.DataTree.to_zarr`.
  """
  if zarr is None:
    raise ImportError("pip install zarr")

  return dt.map_over_subtree(encode_attributes).to_zarr(*args, **kwargs)
