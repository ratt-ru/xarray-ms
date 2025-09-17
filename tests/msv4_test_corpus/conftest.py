from __future__ import annotations

import enum
import json
import os.path
import urllib.request
import zipfile
from contextlib import ExitStack
from dataclasses import InitVar, dataclass, field
from hashlib import sha256

import platformdirs
import pytest

BASE_URL = "https://downloadnrao.org"
METADATA_URL = f"{BASE_URL}/file.download.json"
HEADERS = {"user-agent": "Wget/1.16 (linux-gnu)"}
ONE_MB = 1024**2


class DatasetType(enum.Enum):
  """An enumeration over the dataset types
  present in the MSv4 Dataset Test Corpus"""

  MSV2 = 0
  CALTABLE = 1
  NUMPY = 2
  BINARY = 3
  JSON = 4
  NA = 5
  ZARR = 6
  IMAGE = 7
  MSV4 = 8
  MODEL = 9
  CASA_IMAGE = 10
  ZARR_IMAGE = 11
  HASH = 12

  @classmethod
  def from_str(cls, dtype: str) -> DatasetType:
    try:
      return {
        "CASA MS v2": DatasetType.MSV2,
        "CASA Cal Table": DatasetType.CALTABLE,
        "npy": DatasetType.NUMPY,
        "bin": DatasetType.BINARY,
        "json": DatasetType.JSON,
        "na": DatasetType.NA,
        "zarr": DatasetType.ZARR,
        "image": DatasetType.IMAGE,
        "Msv4": DatasetType.MSV4,
        "Model": DatasetType.MODEL,
        "CASA Image": DatasetType.CASA_IMAGE,
        "Zarr Image": DatasetType.ZARR_IMAGE,
        "hash": DatasetType.HASH,
      }[dtype]
    except KeyError:
      raise ValueError(f"Unhandled dataset type {dtype}")


@dataclass
class DatasetMetadata:
  file: InitVar[str]
  dtype: InitVar[str]
  hash: InitVar[str]
  size: InitVar[str]

  filename: str = field(init=False)
  type: DatasetType = field(init=False)
  filesize: int = field(init=False)
  checksum: str = field(init=False)

  path: str
  telescope: str
  mode: str

  def __post_init__(self, file: str, dtype: str, hash: str, size: str):
    self.filename = file
    self.filesize = int(size)
    self.checksum = hash
    self.type = DatasetType.from_str(dtype)


# panel_cutoff_mask url is wrong
IGNORE_DATASETS = {"panel_cutoff_mask"}
DESIRED_DATASET_TYPES = {DatasetType.MSV2}


def download_item(ds_meta: DatasetMetadata, output_file: str, checksum_file: str):
  """Download an item described by a ``ds_meta`` object to ``output_file``,
  storing it's checksum in ``checksum_file``"""
  with ExitStack() as stack:
    url = f"{BASE_URL}/{ds_meta.path}/{ds_meta.filename}"
    request = urllib.request.Request(url, headers=HEADERS)
    response = stack.enter_context(urllib.request.urlopen(request))
    archive = stack.enter_context(open(output_file, "wb"))
    checkfile = stack.enter_context(open(checksum_file, "w"))
    digest = sha256()

    while data := response.read(ONE_MB):
      archive.write(data)
      digest.update(data)

    if (checksum := digest.hexdigest()) != ds_meta.checksum:
      raise ValueError(
        f"Running download checksum {checksum} "
        f"does not match the metadata checksum {ds_meta}"
      )

    checkfile.write(checksum)


@pytest.fixture(scope="session")
def msv4_corpus_metadata():
  request = urllib.request.Request(METADATA_URL, headers=HEADERS)
  with urllib.request.urlopen(request) as response:
    return json.loads(response.read())


@pytest.fixture(scope="session")
def msv4_corpus_dataset(request, msv4_corpus_metadata, tmp_path_factory):
  assert isinstance(request.param, str)
  user_cache_dir = platformdirs.user_cache_dir("xarray-ms", ensure_exists=True)
  test_data_dir = os.path.join(user_cache_dir, "msv4-test-data")

  metadata = msv4_corpus_metadata["metadata"]
  ds_meta = DatasetMetadata(**metadata[request.param])
  ds_cache_dir = os.path.join(test_data_dir, ds_meta.path)
  ds_archive_file = os.path.join(ds_cache_dir, ds_meta.filename)
  ds_checksum_file = f"{ds_archive_file}.sha256sum"
  os.makedirs(ds_cache_dir, exist_ok=True)

  # Do a checksum and file size check if we have these data points
  # Download if these are available
  local_checksum: None | str = None
  local_filesize: None | int = None

  if os.path.isfile(ds_checksum_file):
    with open(ds_checksum_file, "r") as f:
      local_checksum = f.read().strip()

  if os.path.isfile(ds_archive_file):
    local_filesize = os.path.getsize(ds_archive_file)

  if (
    local_checksum is None
    or local_filesize is None
    or local_checksum != ds_meta.checksum
    # metadata file size can be unreliable, ignore and
    # trust in archiving integrity checks
    # or local_filesize != ds_meta.filesize
  ):
    download_item(ds_meta, ds_archive_file, ds_checksum_file)

  root, _ = os.path.splitext(ds_meta.filename)
  dataset_path = tmp_path_factory.mktemp(root)

  with zipfile.ZipFile(ds_archive_file, mode="r") as archive:
    archive.extractall(dataset_path)

  yield request.param, dataset_path
