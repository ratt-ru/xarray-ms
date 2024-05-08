from cacheout import CacheManager, LRUCache


def on_table_delete(key, value, cause):
  """Close arcae tables on cache eviction"""
  value.close()


CACHESET = CacheManager(
  {
    "tables": {
      "cache_class": LRUCache,
      "maxsize": 100,
      "ttl": 5 * 60,
      "on_delete": on_table_delete,
    },
    "row_maps": {
      "cache_class": LRUCache,
      "maxsize": 100,
      "ttl": 5 * 60,
    },
  }
)

assert isinstance(CACHESET["tables"], LRUCache)
assert isinstance(CACHESET["row_maps"], LRUCache)
