import concurrent.futures as cf
import json
import multiprocessing
import os

import appdirs
import requests

from xarray_ms.testing.corpus import METADATA_URL, dropbox_url

HEADERS = {"User-Agent": "Wget/1.20.3"}


def test_corpus():
  response = requests.get(METADATA_URL, headers=HEADERS)
  payload = json.loads(response.text)

  _ = payload["version"]
  ncpus = multiprocessing.cpu_count()
  user_cache_dir = appdirs.user_cache_dir("xarray-ms")
  os.makedirs(user_cache_dir, exist_ok=True)

  def download_and_save(url: str, name: str):
    path = os.path.sep.join((user_cache_dir, name))
    with open(path, "wb") as f:
      response = requests.get(url, headers=HEADERS, stream=True)
      for chunk in response.iter_content(chunk_size=1024 * 1024):
        f.write(chunk)

  with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:
    futures = []
    names = []
    meta = []

    for name, metadata in payload["metadata"].items():
      if metadata["dtype"] != "CASA MS v2":
        continue

      print(metadata["dtype"])

      url = dropbox_url(metadata["file"], metadata["id"], metadata["rlkey"])
      futures.append(pool.submit(download_and_save, url, metadata["file"]))
      names.append(name)
      meta.append(metadata)

    done, _ = cf.wait(futures, return_when="ALL_COMPLETED")
    [f.result() for f in done]
