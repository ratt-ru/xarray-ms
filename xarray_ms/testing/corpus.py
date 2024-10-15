# Constants defining the location of the metadata file
# describing the test corpus
METADATA_FILENAME = "file.download.json"
METADATA_ID = "1m53led1mchpdc4m3pv37"
METADATA_RLKEY = "enkp8m1hv437nu6p020owflrt&st=11psoc6n"


def dropbox_url(filename: str, file_id: str, rlkey: str) -> str:
  """Returns the URL for a dropbox file"""
  return f"https://www.dropbox.com/scl/fi/{file_id}/{filename}?rlkey={rlkey}"


# The URL for the metadata file describing the test corpus
METADATA_URL = dropbox_url(METADATA_FILENAME, METADATA_ID, METADATA_RLKEY)
