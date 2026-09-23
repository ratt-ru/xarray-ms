import warnings

HAS_WRITE_SUPPORT = False


def _install_write_support():
  """Patch write support methods onto xarray.Dataset and xarray.DataTree"""
  global HAS_WRITE_SUPPORT

  if HAS_WRITE_SUPPORT:
    return

  try:
    from xarray import Dataset, DataTree

    from xarray_ms.backend.msv2.writes import (
      dataset_to_msv2,
      datatree_to_msv2,
      sync_msv2,
    )
  except ImportError as e:
    warnings.warn(f"Engaging write support failed due to {e}", UserWarning)
    HAS_WRITE_SUPPORT = False
  else:
    Dataset.to_msv2 = dataset_to_msv2
    DataTree.to_msv2 = datatree_to_msv2
    DataTree.sync_msv2 = sync_msv2
    HAS_WRITE_SUPPORT = True


def multithreaded_writes() -> bool:
  """Return True if multithreaded write support is enabled"""
  return HAS_WRITE_SUPPORT


_install_write_support()
