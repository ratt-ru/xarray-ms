"""Tests for the ``driver`` / ``driver_kwargs`` arguments and the
deprecation of ``ninstances``."""

import pytest
import xarray

from xarray_ms.backend.msv2.entrypoint import MSv2EntryPoint
from xarray_ms.backend.msv2.entrypoint_utils import (
  DEFAULT_MAIN_NINSTANCES,
  CommonStoreArgs,
  resolve_driver_kwargs,
  table_driver_kwargs,
)

# Positional layout of Table.from_filename after Multiton normalisation:
# (filename, ninstances, readonly, lockoptions, cache_size)
_NINSTANCES = 1
_CACHE_SIZE = 4


def test_resolve_driver_kwargs_default():
  """The default materialises a bounded cache and warns about nothing."""
  assert resolve_driver_kwargs("arcae", None) == {"cache_size": 256}


def test_resolve_driver_kwargs_unsupported_driver():
  with pytest.raises(ValueError, match="Unsupported driver 'bogus'"):
    resolve_driver_kwargs("bogus", None)


@pytest.mark.filterwarnings("ignore:.*?'ninstances' argument is deprecated")
def test_resolve_driver_kwargs_does_not_mutate_input():
  """The returned dictionary is a deep copy safe to mutate downstream."""
  user = {"cache_size": 256, "tables": {"POINTING": {"cache_size": 512}}}
  resolved = resolve_driver_kwargs("arcae", user, ninstances=4)
  resolved["tables"]["POINTING"]["cache_size"] = 0
  assert user == {"cache_size": 256, "tables": {"POINTING": {"cache_size": 512}}}


def test_resolve_driver_kwargs_ninstances_deprecated():
  """A supplied ninstances warns and is folded into the main table scope."""
  with pytest.warns(DeprecationWarning, match="'ninstances' argument is deprecated"):
    resolved = resolve_driver_kwargs("arcae", None, ninstances=4)
  assert resolved["tables"]["MAIN"]["ninstances"] == 4


def test_resolve_driver_kwargs_explicit_main_beats_deprecated():
  """An explicit main-table override wins over the deprecated ninstances."""
  dk = {"tables": {"MAIN": {"ninstances": 2}}}
  with pytest.warns(DeprecationWarning):
    resolved = resolve_driver_kwargs("arcae", dk, ninstances=8)
  assert resolved["tables"]["MAIN"]["ninstances"] == 2


def test_table_driver_kwargs_main_defaults_ninstances():
  assert table_driver_kwargs({"cache_size": 256}, "MAIN") == {
    "cache_size": 256,
    "ninstances": DEFAULT_MAIN_NINSTANCES,
  }


def test_table_driver_kwargs_flat_applies_to_subtable():
  """Flat kwargs apply uniformly; subtables get no implicit ninstances."""
  assert table_driver_kwargs({"cache_size": 256}, "POINTING") == {"cache_size": 256}


def test_table_driver_kwargs_per_table_override():
  dk = {"cache_size": 256, "tables": {"POINTING": {"cache_size": 512}}}
  assert table_driver_kwargs(dk, "POINTING") == {"cache_size": 512}
  assert table_driver_kwargs(dk, "SPECTRAL_WINDOW") == {"cache_size": 256}


def test_common_store_args_default_cache(simmed_ms):
  """Default driver_kwargs bound the main table and every subtable."""
  store_args = CommonStoreArgs(simmed_ms)

  assert store_args.ms_factory._args[_NINSTANCES] == DEFAULT_MAIN_NINSTANCES
  assert store_args.ms_factory._args[_CACHE_SIZE] == 256

  for factory in store_args.subtable_factories.values():
    assert factory._kw == {"cache_size": 256}


def test_common_store_args_pointing_override(simmed_ms):
  """A per-subtable override reaches only that subtable."""
  driver_kwargs = {"cache_size": 256, "tables": {"POINTING": {"cache_size": 512}}}
  store_args = CommonStoreArgs(simmed_ms, driver_kwargs=driver_kwargs)

  assert store_args.subtable_factories["POINTING"]._kw == {"cache_size": 512}
  assert store_args.subtable_factories["SPECTRAL_WINDOW"]._kw == {"cache_size": 256}


@pytest.mark.filterwarnings("ignore:.*?'ninstances' argument is deprecated")
def test_common_store_args_deprecated_ninstances_main_only(simmed_ms):
  """The deprecated ninstances affects the main table, not subtables."""
  driver_kwargs = resolve_driver_kwargs("arcae", None, ninstances=1)
  store_args = CommonStoreArgs(simmed_ms, driver_kwargs=driver_kwargs)

  assert store_args.ms_factory._args[_NINSTANCES] == 1
  # Subtables always default to a single instance
  assert "ninstances" not in store_args.subtable_factories["POINTING"]._kw


def test_open_dataset_ninstances_deprecation(simmed_ms):
  with pytest.warns(DeprecationWarning, match="'ninstances' argument is deprecated"):
    xarray.open_dataset(simmed_ms, ninstances=4)


def test_open_dataset_unsupported_driver(simmed_ms):
  with pytest.raises(ValueError, match="Unsupported driver 'bogus'"):
    MSv2EntryPoint().open_dataset(simmed_ms, driver="bogus")


def test_open_datatree_ninstances_deprecation_warns_once(simmed_ms, recwarn):
  """The deprecation is raised exactly once despite the re-entrant open path."""
  xarray.open_datatree(simmed_ms, ninstances=4)
  deprecations = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
  ninstances_warnings = [
    w for w in deprecations if "'ninstances' argument is deprecated" in str(w.message)
  ]
  assert len(ninstances_warnings) == 1
