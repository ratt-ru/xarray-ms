import json
from typing import Any, Dict

import pyarrow as pa


def table_desc(table: pa.Table) -> Dict[str, Any]:
  """Extract the CASA table descriptor stored in an Arrow table constructed by arcae"""
  try:
    arcae_metadata = table.schema.metadata[b"__arcae_metadata__"]
  except KeyError:
    raise KeyError("__arcae_metadata__ was not present in the table metadata")

  try:
    return json.loads(arcae_metadata)["__casa_descriptor__"]
  except KeyError:
    raise KeyError(
      f"arcae metadata {arcae_metadata} does not contain a __casa_descriptor__"
    )
