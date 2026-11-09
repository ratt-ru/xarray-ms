from datetime import datetime

import pytest
import xarray
from numpy.testing import assert_array_equal


def _column_desc(name):
  return {
    "_c_order": True,
    "comment": f"{name} column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    "valueType": "COMPLEX",
  }


DATA_GROUP_TABLE_DESC = {
  c: _column_desc(c) for c in ("CORRECTED_DATA", "MODEL_DATA", "RESIDUAL_DATA")
}


def _add_data_columns(chunk_desc, data_dict):
  data_dict = data_dict.copy()
  dims, data = data_dict["DATA"]
  data_dict["CORRECTED_DATA"] = (dims, 2 * data)
  data_dict["MODEL_DATA"] = (dims, 3 * data)
  data_dict["RESIDUAL_DATA"] = (dims, 4 * data)
  return data_dict


DATA_GROUP_MS = {
  "name": "data_groups.ms",
  "transform_data": _add_data_columns,
  "table_desc": DATA_GROUP_TABLE_DESC,
}


def _correlated_nodes(dt):
  return [n for n in dt.children.values() if n.attrs.get("type") == "visibility"]


def _variables(data_group):
  return {k: data_group[k] for k in ("correlated_data", "flag", "weight", "uvw")}


@pytest.mark.parametrize("simmed_ms", [DATA_GROUP_MS], indirect=True)
def test_standard_data_groups(simmed_ms):
  """base, corrected and model data groups are present with their columns"""
  dt = xarray.open_datatree(simmed_ms)
  nodes = _correlated_nodes(dt)
  assert len(nodes) > 0

  for node in nodes:
    data_groups = node.attrs["data_groups"]
    assert list(data_groups.keys()) == ["base", "corrected", "model"]

    for name, correlated_data in [
      ("base", "VISIBILITY"),
      ("corrected", "VISIBILITY_CORRECTED"),
      ("model", "VISIBILITY_MODEL"),
    ]:
      assert _variables(data_groups[name]) == {
        "correlated_data": correlated_data,
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
      }
      path = data_groups[name]["field_and_source"]
      assert path == f"{node.name}/field_and_source_{name}_xds"
      assert dt[path].attrs["type"] == "field_and_source"

    assert "field_and_source_xds" not in node.children
    assert "CORRECTED_DATA" not in node
    assert "MODEL_DATA" not in node
    # RESIDUAL_DATA is still presented as a secondary variable
    assert "RESIDUAL_DATA" in node

    vis = node.VISIBILITY.values
    assert_array_equal(node.VISIBILITY_CORRECTED.values, 2 * vis)
    assert_array_equal(node.VISIBILITY_MODEL.values, 3 * vis)
    assert node.VISIBILITY_MODEL.attrs["units"] == "Jy"


@pytest.mark.parametrize("simmed_ms", [{"name": "base.ms"}], indirect=True)
def test_base_data_group_only(simmed_ms):
  """Only the base data group is present without
  CORRECTED_DATA and MODEL_DATA columns"""
  dt = xarray.open_datatree(simmed_ms)
  for node in _correlated_nodes(dt):
    assert list(node.attrs["data_groups"].keys()) == ["base"]
    assert "VISIBILITY_CORRECTED" not in node
    assert "VISIBILITY_MODEL" not in node


@pytest.mark.parametrize("simmed_ms", [DATA_GROUP_MS], indirect=True)
@pytest.mark.parametrize("data_groups", ["RESIDUAL_DATA", ["RESIDUAL_DATA"]])
def test_bare_custom_data_group(simmed_ms, data_groups):
  """Strings name custom data groups that inherit from the base data group"""
  dt = xarray.open_datatree(simmed_ms, data_groups=data_groups)
  for node in _correlated_nodes(dt):
    data_groups = node.attrs["data_groups"]
    assert list(data_groups.keys()) == ["base", "corrected", "model", "RESIDUAL_DATA"]
    assert _variables(data_groups["RESIDUAL_DATA"]) == {
      "correlated_data": "RESIDUAL_DATA",
      "flag": "FLAG",
      "weight": "WEIGHT",
      "uvw": "UVW",
    }
    path = data_groups["RESIDUAL_DATA"]["field_and_source"]
    assert path == f"{node.name}/field_and_source_RESIDUAL_DATA_xds"
    assert dt[path].attrs["type"] == "field_and_source"


@pytest.mark.parametrize("simmed_ms", [DATA_GROUP_MS], indirect=True)
def test_mapping_data_groups(simmed_ms):
  """Mappings add custom data groups and override standard data groups"""
  data_groups = {
    "residual": {"correlated_data": "RESIDUAL_DATA", "description": "Residuals"},
    "corrected": {"weight": "SIGMA_SPECTRUM"},
  }
  dt = xarray.open_datatree(simmed_ms, data_groups=data_groups)
  for node in _correlated_nodes(dt):
    groups = node.attrs["data_groups"]
    assert list(groups.keys()) == ["base", "corrected", "model", "residual"]
    assert _variables(groups["corrected"]) == {
      "correlated_data": "VISIBILITY_CORRECTED",
      "flag": "FLAG",
      "weight": "SIGMA_SPECTRUM",
      "uvw": "UVW",
    }
    assert groups["residual"]["correlated_data"] == "RESIDUAL_DATA"
    assert groups["residual"]["description"] == "Residuals"
    assert groups["residual"]["weight"] == "WEIGHT"


@pytest.mark.parametrize("simmed_ms", [DATA_GROUP_MS], indirect=True)
def test_open_dataset_data_groups(simmed_ms):
  """open_dataset presents data groups with relative field_and_source names"""
  ds = xarray.open_dataset(simmed_ms, data_groups="RESIDUAL_DATA")
  data_groups = ds.attrs["data_groups"]
  assert list(data_groups.keys()) == ["base", "corrected", "model", "RESIDUAL_DATA"]
  for name, group in data_groups.items():
    assert group["field_and_source"] == f"field_and_source_{name}_xds"
    assert datetime.fromisoformat(group["date"]).tzinfo is not None


@pytest.mark.parametrize("simmed_ms", [{"name": "base.ms"}], indirect=True)
def test_override_absent_standard_data_group(simmed_ms):
  """Standard data groups without their column can be created
  if their correlated data is specified"""
  with pytest.raises(ValueError, match="Unable to override standard data group"):
    xarray.open_dataset(simmed_ms, data_groups={"model": {"weight": "WEIGHT"}})

  ds = xarray.open_dataset(
    simmed_ms, data_groups={"model": {"correlated_data": "VISIBILITY"}}
  )
  assert ds.attrs["data_groups"]["model"]["correlated_data"] == "VISIBILITY"


@pytest.mark.parametrize("simmed_ms", [DATA_GROUP_MS], indirect=True)
@pytest.mark.parametrize(
  "data_groups, exception, match",
  [
    ("MISSING_DATA", ValueError, "which is not a variable"),
    ({"residual": {"flag": "MISSING_FLAG"}}, ValueError, "must specify"),
    (
      {"residual": {"correlated_data": "RESIDUAL_DATA", "flag": "MISSING_FLAG"}},
      ValueError,
      "which is not a variable",
    ),
    ({"residual": {"correlated": "RESIDUAL_DATA"}}, ValueError, "Invalid keys"),
    ("model", ValueError, "standard data groups"),
    (["model", "RESIDUAL_DATA"], ValueError, "standard data groups"),
    ({"residual": "RESIDUAL_DATA"}, TypeError, "must map"),
    ({"residual": {"correlated_data": 1}}, TypeError, "must be strings"),
    (1, TypeError, "must be a string"),
  ],
)
def test_invalid_data_groups(simmed_ms, data_groups, exception, match):
  with pytest.raises(exception, match=match):
    xarray.open_datatree(simmed_ms, data_groups=data_groups)
