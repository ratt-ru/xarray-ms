import warnings
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version as package_version
from typing import Any, Dict, List, Literal, Mapping, Set, Tuple

import numpy as np
import numpy.typing as npt
from arcae.lib.arrow_tables import ms_descriptor
from packaging.version import parse as parse_version
from xarray import Dataset, DataTree
from xarray.backends.common import ArrayWriter

from xarray_ms.backend.msv2.entrypoint import MSv2Store
from xarray_ms.backend.msv2.entrypoint_utils import CommonStoreArgs
from xarray_ms.casa_types import NUMPY_TO_CASA_MAP
from xarray_ms.errors import MissingEncodingError
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES, MAIN_PREFIX_DIMS

# https://github.com/pydata/xarray/pull/10771
if parse_version(package_version("xarray")) >= parse_version("2025.09.01"):
  from xarray.backends.api import _finalize_store
  from xarray.backends.writers import dump_to_store
else:
  from xarray.backends.api import _finalize_store, dump_to_store

ShapeSetType = Set[Tuple[int, ...]]
DTypeSetType = Set[npt.DTypeLike]


@dataclass
class DataVariableInfo:
  counts: int
  shapes: ShapeSetType
  dtypes: DTypeSetType


DataVariableInfoMap = Dict[Tuple[str, str], DataVariableInfo]


MSV4_WRITE_MAP = {
  "UVW": "UVW",
  "VISIBILITY": "DATA",
  "WEIGHT": "WEIGHT_SPECTRUM",
  "FLAG": "FLAG",
}


# MSv4 variables that will not be written back to
# the MSv2 as they are effectively metadata
IGNORED_VARIABLES = [
  # Standard MSv4 variables
  "EFFECTIVE_INTEGRATION_TIME",
  "TIME_CENTROID",
  "TIME_CENTROID_EXTRA_PRECISION",
  "EFFECTIVE_CHANNEL_WIDTH",
  "FREQUENCY_CENTROID",
  # Added by Correlated Factory if grids are irregular
  "TIME",
  "INTEGRATION_TIME",
  "CHANNEL_WIDTH",
]


def validate_column_desc(
  var_name: str, column: str, shapes: ShapeSetType, column_desc: Dict
) -> None:
  # Validate the variable ndim against the column ndim
  if (ndim := column_desc.get("ndim")) is not None:
    # Multi-dimensional CASA column configuraiton
    multidim = ndim == -1
    ndims = {len(s) for s in shapes}
    if not multidim and len(ndims) > 1:
      raise ValueError(
        f"{column} descriptor specifies a fixed ndim ({ndim}) "
        f"but {var_name} has multiple trailing shapes with dimensions {shapes}"
      )
  elif not all(len(shape) == 0 for shape in shapes):
    # ndim implies column only has a row-dimension
    raise ValueError(
      f"{column} descriptor specifies a row only column "
      f"but {var_name} has trailing shape(s) {shapes}"
    )

  # Validate the variable shape if the column is fixed
  if (fixed_shape := column_desc.get("shape")) is not None:
    fixed_shape = tuple(fixed_shape)
    if len(shapes) > 1:
      raise ValueError(
        f"Variable {var_name} has multiple trailing shapes {shapes} "
        f"but {column} specifies a fixed shape {fixed_shape}"
      )

    if (var_shape := next(iter(shapes))) != fixed_shape:
      raise ValueError(
        f"Variable {var_name} has trailing shape {var_shape} "
        f"but {column} has a fixed shape {fixed_shape}"
      )


def fit_tile_shape(shape: Tuple[int, ...], dtype: npt.DTypeLike) -> Dict[str, np.int32]:
  """
  Args:
    shape: FORTRAN ordered tile shape
    dtype: tile data type

  Returns:
    A :code:`{DEFAULTTILESHAPE: tile_shape}` dictionary
  """
  nbytes = np.dtype(dtype).itemsize
  min_tile_dims = [512]
  max_tile_dims = [np.inf]

  for dim in shape:
    min_tile = min(dim, 4)  # Don't tile <=4 elements.
    # For dims which are not exact powers of two, treat them as though
    # they are floored to the nearest power of two.
    max_tile = int(min(2 ** int(np.log2(dim)) / 8, 64))
    max_tile = min_tile if max_tile < min_tile else max_tile
    min_tile_dims.append(min_tile)
    max_tile_dims.append(max_tile)

    tile_shape = min_tile_dims.copy()
    growth_axis = 0

    while np.prod(tile_shape) * nbytes < 1024**2:  # 1MB tiles.
      if tile_shape[growth_axis] < max_tile_dims[growth_axis]:
        tile_shape[growth_axis] *= 2
      growth_axis = (growth_axis + 1) % len(tile_shape)

  # The tile shape is C ordered
  return {"DEFAULTTILESHAPE": list(tile_shape[::-1])}


def generate_column_descriptor(
  table_desc: Dict[str, Any],
  data_var_map: DataVariableInfoMap,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  Synthesises a column descriptor from:

  1. An existing table descriptor.
  2. A complete table descriptor.
  3. The shapes and data types associated with
    xarray Variables on a DataTree.

  Args:
    table_desc: Table descriptor containing existing
      column definitions.
    data_var_map: A mapping of shapes and data types
      associated with each xarray Variable.

  Returns:
    A (descriptor, dminfo) tuple containing any columns
    and data managers that should be created
  """
  canonical_table_desc = ms_descriptor("MAIN", complete=True)
  actual_desc = {}
  dm_groups = []

  for (var_name, msv2_column), data_var_info in data_var_map.items():
    # If there are existing descriptors, either for
    # columns present on the table, or in the canonical definition
    # validate that the variable shape matches the column
    if column_desc := table_desc.get(msv2_column):
      validate_column_desc(var_name, msv2_column, data_var_info.shapes, column_desc)
    elif column_desc := canonical_table_desc.get(msv2_column):
      validate_column_desc(var_name, msv2_column, data_var_info.shapes, column_desc)
    else:
      # Construct a column descriptor and possibly an associated data manager
      # Unify variable numpy types
      dtype = np.result_type(*data_var_info.dtypes)

      if dtype is object:
        raise NotImplementedError(
          f"Types of variable {var_name} ({list(data_var_info.dtypes)}) "
          f"resolves to an object. "
          f"Writing of objects is not supported"
        )

      try:
        casa_type = NUMPY_TO_CASA_MAP[np.dtype(dtype).type]
      except KeyError as e:
        raise ValueError(
          f"No CASA type matched NumPy dtype {dtype}\n{NUMPY_TO_CASA_MAP}"
        ) from e

      column_desc = {"valueType": casa_type, "option": 0}

      if len(data_var_info.shapes) == 1:
        # If the shape is fixed, Tile the column
        # column descriptor shapes are fortran ordered
        fixed_shape = tuple(reversed(next(iter(data_var_info.shapes))))
        row_only = len(fixed_shape) == 0
        if not row_only:
          column_desc["option"] |= 4
          column_desc["shape"] = list(fixed_shape)
          column_desc["ndim"] = len(fixed_shape)
        column_desc["dataManagerGroup"] = dm_group = f"{msv2_column}_GROUP"
        column_desc["dataManagerType"] = dm_type = "TiledColumnStMan"

        dm_groups.append(
          {
            "COLUMNS": [msv2_column],
            "NAME": dm_group,
            "TYPE": dm_type,
            "SPEC": fit_tile_shape(fixed_shape, dtype),
          }
        )
      else:
        # Variably shaped, use a StandardStMan for now
        # but consider a TiledCellStMan in future
        column_desc["option"] = 0
        column_desc["dataManagerGroup"] = "StandardStMan"
        column_desc["dataManagerType"] = "StandardStMan"

      actual_desc[msv2_column] = column_desc

  dminfo = {f"*{i + 1}": g for i, g in enumerate(dm_groups)}
  return actual_desc, dminfo


def msv2_store_from_dataset(ds: Dataset, region="auto") -> MSv2Store:
  try:
    common_store_args = ds.encoding["common_store_args"]
    partition_key = ds.encoding["partition_key"]
  except KeyError as e:
    raise MissingEncodingError(
      f"Expected encoding key {e} is not present on "
      f"a dataset of type {ds.attrs.get('type')}. "
      f"Writing back to a Measurement Set "
      f"is not possible without this information"
    ) from e

  # Recover common arguments used to create the original store
  # This will re-use existing table and structure factories
  store_args = CommonStoreArgs(**common_store_args)
  return MSv2Store.open(
    ms=store_args.ms,
    partition_schema=store_args.partition_schema,
    partition_key=partition_key,
    preferred_chunks=store_args.preferred_chunks,
    auto_corrs=store_args.auto_corrs,
    ninstances=store_args.ninstances,
    epoch=store_args.epoch,
    structure_factory=store_args.structure_factory,
    write_region=region,
  )


WriteMapT = Tuple[str, str] | List[Tuple[str, str]] | Dict[str, str] | None


def promote_write_map(variables: WriteMapT) -> Dict[str, str]:
  """Promotes the supplied write variable mapping into a Dict[str, str]"""
  type_except = TypeError(
    f"{variables} should be one of:\n"
    f"1. a Tuple[str, str] variable -> column mapping\n"
    f"2. An Iterable[Tuple[str, str]].\n"
    f"3. A Dict[str, str] variable -> column mapping"
  )

  def check_tuple(tup: Tuple[str, str]) -> Tuple[str, str]:
    if (
      isinstance(tup, tuple) and len(tup) == 2 and all(isinstance(t, str) for t in tup)
    ):
      return tup

    raise type_except

  if variables is None:
    return {}
  elif isinstance(variables, dict):
    return variables
  elif isinstance(variables, tuple):
    return dict([check_tuple(variables)])
  elif isinstance(variables, (list, set)):
    return dict([check_tuple(v) for v in variables])
  else:
    raise type_except


def sync_msv2(dt: DataTree, write_map: WriteMapT = None):
  assert isinstance(dt, DataTree)
  vis_datasets = [
    n for n in dt.subtree if n.attrs.get("type") in CORRELATED_DATASET_TYPES
  ]

  if len(vis_datasets) == 0:
    warnings.warn("No visibility datasets were found on the DataTree")
    return

  # Get a table factory from the MSv2Store
  table_factory = msv2_store_from_dataset(next(iter(vis_datasets)).ds).table_factory
  table_desc = table_factory.instance.tabledesc()

  write_map = {**MSV4_WRITE_MAP, **promote_write_map(write_map)}
  var_info_map: DataVariableInfoMap = defaultdict(
    lambda: DataVariableInfo(0, set(), set())
  )

  for nodes_visited, node in enumerate(vis_datasets, 1):
    assert isinstance(node, DataTree)
    for name, var in node.data_vars.items():
      if name in IGNORED_VARIABLES:
        continue

      # Don't try to create anything that doesn't translate
      # to a MAIN MSv2 row dimension
      if var.dims[: len(MAIN_PREFIX_DIMS)] != MAIN_PREFIX_DIMS:
        warnings.warn(
          f"Ignoring {name} in {node.path}  "
          f"dimensions {var.dims} do not begin with {MAIN_PREFIX_DIMS}",
          UserWarning,
        )
        continue

      entry = var_info_map[(name, write_map.get(name, name))]
      entry.counts += 1
      entry.shapes.add(var.shape[len(MAIN_PREFIX_DIMS) :])
      entry.dtypes.add(var.dtype)

  # Ensure that the identified variables are universally
  # defined across all DataTree.
  for key, entry in list(var_info_map.items()):
    if entry.counts != nodes_visited:
      warnings.warn(
        f"Ignoring {key[0]} which does not appear in all "
        f"visibility datasets {list(dt.children)}",
        UserWarning,
      )
      var_info_map.pop(key)

  # Generate column descriptors and add the columns
  column_descs, dminfo = generate_column_descriptor(table_desc, var_info_map)
  table_factory.instance.addcols(column_descs, dminfo)
  assert set(column_descs.keys()).issubset(table_factory.instance.columns())


def datatree_to_msv2(
  dt: DataTree,
  write_map: WriteMapT = None,
  compute: Literal[True] = True,
  write_inherited_coords: bool = False,
) -> None:
  assert isinstance(dt, DataTree)

  if (
    len(
      vis_datasets := [
        n for n in dt.subtree if n.attrs.get("type") in CORRELATED_DATASET_TYPES
      ]
    )
    == 0
  ):
    warnings.warn(
      "No visibility datasets were found for write on the DataTree", UserWarning
    )
    return

  for node in vis_datasets:
    at_root = node is dt.root
    ds = node.to_dataset(inherit=write_inherited_coords or at_root)
    dataset_to_msv2(ds, write_map=write_map, compute=compute)


def dataset_to_msv2(
  ds: Dataset,
  write_map: WriteMapT = None,
  compute: Literal[True] = True,
  region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] = "auto",
) -> None:
  assert isinstance(ds, Dataset)

  # Strip out
  # 1. ancilliary variables
  # 2. coordinates
  # 3. attributes
  write_map = {**MSV4_WRITE_MAP, **promote_write_map(write_map)}
  ignored = set(IGNORED_VARIABLES) & set(ds.data_vars)
  write_ds = ds.drop_vars(ignored | set(ds.coords)).drop_attrs()
  write_ds = write_ds.rename_vars(
    {k: v for k, v in write_map.items() if k in write_ds.data_vars}
  )

  msv2_store = msv2_store_from_dataset(write_ds, region)
  msv2_store.set_write_region(write_ds)
  writer = ArrayWriter()
  dump_to_store(write_ds, msv2_store, writer)
  writes = writer.sync(compute=compute)

  if compute:
    _finalize_store(writes, msv2_store)
  else:
    # NOTE: Within xarray to_xxx implementations
    # return delayed_close_after_writes(...)
    # would be typically in this code path.
    # We return None here as this introduces a strict
    # dependency on dask
    return msv2_store.close()
