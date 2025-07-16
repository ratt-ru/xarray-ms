import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, Set, Tuple

import numpy as np
import numpy.typing as npt
from arcae.lib.arrow_tables import ms_descriptor
from xarray import Dataset, DataTree
from xarray.backends.api import dump_to_store

from xarray_ms.backend.msv2.entrypoint import MSv2Store
from xarray_ms.backend.msv2.entrypoint_utils import CommonStoreArgs
from xarray_ms.casa_types import NUMPY_TO_CASA_MAP
from xarray_ms.errors import MissingEncodingError
from xarray_ms.msv4_types import CORRELATED_DATASET_TYPES

ShapeSetType = Set[Tuple[int, ...]]
DTypeSetType = Set[npt.DTypeLike]
ShapeAndDTypeType = Dict[Tuple[str, str], Tuple[ShapeSetType, DTypeSetType]]


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
    shape: tile shape
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

  return {"DEFAULTTILESHAPE": list(tile_shape[::-1])}


def generate_column_descriptor(
  table_desc: Dict[str, Any],
  shapes_and_dtypes: ShapeAndDTypeType,
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
    shapes_and_dtypes: A :code:`{var_name: (set(shapes), set(dtypes))}`
      mapping of shapes and data types associated with each xarray
      Variable.

  Returns:
    A (descriptor, dminfo) tuple containing any columns
    and data managers that should be created
  """
  canonical_table_desc = ms_descriptor("MAIN", complete=True)
  actual_desc = {}
  dm_groups = []

  for (var_name, column), (shapes, dtypes) in shapes_and_dtypes.items():
    # If there are existing descriptors, either for
    # columns present on the table, or in the canonical definition
    # validate that the variable shape matches the column
    if column_desc := table_desc.get(column):
      validate_column_desc(var_name, column, shapes, column_desc)
    elif column_desc := canonical_table_desc.get(column):
      validate_column_desc(var_name, column, shapes, column_desc)
    else:
      # Construct a column descriptor and possibly an
      # associated data manager
      # Unify variable numpy types
      dtype = np.result_type(*dtypes)

      if dtype is object:
        raise NotImplementedError(
          f"Types of variable {var_name} ({list(dtypes)}) "
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

      if len(shapes) == 1:
        # Fixed shape, Tile the column
        fixed_shape = tuple(reversed(next(iter(shapes))))
        row_only = len(fixed_shape) == 0
        if not row_only:
          column_desc["option"] |= 4
          column_desc["shape"] = list(fixed_shape)
          column_desc["ndim"] = len(fixed_shape)
        column_desc["dataManagerGroup"] = dm_group = f"{column}_GROUP"
        column_desc["dataManagerType"] = dm_type = "TiledColumnStMan"

        dm_groups.append(
          {
            "COLUMNS": [column],
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

      actual_desc[column] = column_desc

  dminfo = {f"*{i + 1}": g for i, g in enumerate(dm_groups)}
  return actual_desc, dminfo


def msv2_store_from_dataset(ds: Dataset) -> MSv2Store:
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
  # This will likely re-use existing table and structure factories
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
  )


def datatree_to_msv2(
  dt: DataTree, variables: str | Iterable[str], write_inherited_coords: bool = False
):
  assert isinstance(dt, DataTree)
  list_var_names = [variables] if isinstance(variables, str) else list(variables)
  set_var_names = set(list_var_names)

  if len(set_var_names) == 0:
    warnings.warn("Empty 'variables'")
    return

  vis_datasets = [
    n for n in dt.subtree if n.attrs.get("type") in CORRELATED_DATASET_TYPES
  ]

  if len(vis_datasets) == 0:
    warnings.warn("No visibility datasets were found on the DataTree")
    return

  shapes_and_dtypes: ShapeAndDTypeType = defaultdict(lambda: (set(), set()))

  for node in vis_datasets:
    assert isinstance(node, DataTree)
    if not set_var_names.issubset(node.data_vars.keys()):
      raise ValueError(
        f"{set_var_names} are not present in all visibility DataTree nodes"
      )

    for n in set_var_names:
      var = node.data_vars[n]
      PREFIX_DIMS = ("time", "baseline_id")

      if var.dims[:2] != PREFIX_DIMS:
        raise ValueError(f"{n} dimensions {var.dims} do not start with {PREFIX_DIMS}")

      shapes, dtypes = shapes_and_dtypes[(n, n)]

      shapes.add(var.shape[len(PREFIX_DIMS) :])
      dtypes.add(var.dtype)

  table_factory = msv2_store_from_dataset(next(iter(vis_datasets)).ds)._table_factory
  table_desc = table_factory.instance.tabledesc()
  column_descs, dminfo = generate_column_descriptor(table_desc, shapes_and_dtypes)
  table_factory.instance.addcols(column_descs, dminfo)
  assert set(column_descs.keys()).issubset(table_factory.instance.columns())

  for node in vis_datasets:
    at_root = node is dt.root
    node = node.to_dataset(inherit=write_inherited_coords or at_root)
    node.to_msv2(list_var_names)


def dataset_to_msv2(ds: Dataset, variables: str | Iterable[str]):
  assert isinstance(ds, Dataset)
  list_vars = [variables] if isinstance(variables, str) else list(variables)

  if len(list_vars) == 0:
    return

  # Strip out coordinates and attributes
  msv2_store = msv2_store_from_dataset(ds)
  ignored_vars = set(ds.data_vars) - set(list_vars)
  ds = ds.drop_vars(ds.coords).drop_vars(ignored_vars).drop_attrs()
  try:
    dump_to_store(ds, msv2_store)
  finally:
    msv2_store.close()
