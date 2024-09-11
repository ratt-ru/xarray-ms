import dataclasses
import os
import tempfile
import typing
from typing import (
  Any,
  Dict,
  Generator,
  List,
  Tuple,
)

import numpy as np
import numpy.typing as npt
from arcae.lib.arrow_tables import Table

from xarray_ms.casa_types import DataDescArgType, DataDescription, Feed

# First of February 2023
FIRST_FEB_2023_MJDS = 2459976.50000 * 86400

# Default simulation parameters
DEFAULT_SIM_PARAMS = {"ntime": 5, "data_description": [(8, ["XX", "XY", "YX", "YY"])]}

# Additional Columns to add
ADDITIONAL_COLUMNS = {
  "DATA": {
    "_c_order": True,
    "comment": "DATA column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    # 'shape': ...,  # Variably shaped
    "valueType": "COMPLEX",
  },
  "WEIGHT_SPECTRUM": {
    "_c_order": True,
    "comment": "WEIGHT_SPECTRUM column",
    "dataManagerGroup": "StandardStMan",
    "dataManagerType": "StandardStMan",
    "keywords": {},
    "maxlen": 0,
    "ndim": 2,
    "option": 0,
    # 'shape': ...,  # Variably shaped
    "valueType": "FLOAT",
  },
}


@dataclasses.dataclass
class PartitionDescriptor:
  """
  Describes a partition of data to be simulated, mostly containing
  value ranges for the Measurement Set indexing columns.
  These ranges are broadcast against each other to form the
  full Measurement Set index for the partition. So for e.g.
  they could be:

  .. code-block:: python

      desc = PartitionDescriptor(
          DATA_DESC_ID=[0],
          PROCESSOR_ID=[0, 1, 2],
          FIELD_ID=[0, 1],
          ...
      )
  """

  DATA_DESC_ID: npt.NDArray[np.int32]
  PROCESSOR_ID: npt.NDArray[np.int32]
  FIELD_ID: npt.NDArray[np.int32]
  ANTENNA1: npt.NDArray[np.int32]
  ANTENNA2: npt.NDArray[np.int32]
  FEED1: npt.NDArray[np.int32]
  FEED2: npt.NDArray[np.int32]
  TIME: npt.NDArray[np.float64]
  data_description: DataDescription
  feed_map: Dict[Tuple[str], Feed]

  chunk_id: int
  simulate_data: bool = False
  auto_corrs: bool = True
  dump_rate: float = 8.0


DDIDArgType = List[Tuple[npt.NDArray[np.float64], List[str]]]


class MSStructureSimulator:
  """
  Simulates the a Measurement Set with a valid indexing schema for both
  the main table and subtables. Other data is generated with ramp functions.

  This simulator is suitable for generating small test datasets and should
  not be used for large or real data.
  """

  ntime: int
  nantenna: int
  auto_corrs: bool
  dump_rate: float
  time_chunks: int
  time_start: float
  partition_names: List[str]
  partition_indices: npt.NDArray[np.int32]
  simulate_data: bool
  model: Dict[str, Any]
  data_description: DataDescription

  def __init__(
    self,
    ntime: int,
    time_chunks: int = 10,
    dump_rate: float = 8,
    time_start: float = FIRST_FEB_2023_MJDS,
    nproc: int = 1,
    nfield: int = 1,
    nantenna: int = 3,
    data_description: DataDescArgType | None = None,
    partition: Tuple[str, ...] = ("PROCESSOR_ID", "FIELD_ID", "DATA_DESC_ID"),
    auto_corrs: bool = True,
    simulate_data: bool = True,
  ):
    assert ntime >= 1
    assert time_chunks > 0
    assert dump_rate > 0
    assert nproc >= 1
    assert nfield >= 1
    assert nantenna >= 1

    self.data_description = DataDescription.from_spw_corr_pairs(data_description)
    self.feeds = self.data_description.polarisation_set().feed_map()

    # Generate antenna1 and antenna2 from a range of antenna ids
    antenna_id = np.arange(nantenna, dtype=np.int32)
    antenna_square = antenna_id - antenna_id[:, None]
    ant1, ant2 = np.triu_indices_from(antenna_square, 0 if auto_corrs else 1)
    ant1, ant2 = ant1.astype(np.int32), ant2.astype(np.int32)

    # Generate feed1 and feed2 from a range of feed ids
    feed_id = np.arange(self.nfeed, dtype=np.int32)
    feed_square = feed_id - feed_id[:, None]
    feed1, feed2 = np.triu_indices_from(feed_square, 0)
    feed1, feed2 = feed1.astype(np.int32), feed2.astype(np.int32)

    valid_partitions = {
      "PROCESSOR_ID": np.arange(nproc, dtype=np.int32),
      "FIELD_ID": np.arange(nfield, dtype=np.int32),
      "DATA_DESC_ID": np.arange(len(self.data_description), dtype=np.int32),
      "ANTENNA1": ant1,
      "ANTENNA2": ant2,
      "FEED1": feed1,
      "FEED2": feed2,
    }

    try:
      partition_indices = [(p, valid_partitions[p]) for p in partition]
    except KeyError as e:
      raise ValueError(f"{e} is not a valid partitioning index") from e

    bp_names, bp_indices = zip(*partition_indices)
    cbp_names = typing.cast(List[str], bp_names)
    cbp_indices = typing.cast(List[npt.NDArray[np.int32]], bp_indices)
    bcbp_indices = typing.cast(
      npt.NDArray, self.broadcast_partition_indices(cbp_indices)
    )

    self.ntime = ntime
    self.nantenna = nantenna
    self.auto_corrs = auto_corrs
    self.dump_rate = dump_rate
    self.time_chunks = time_chunks
    self.time_start = time_start
    self.simulate_data = simulate_data
    self.partition_names = cbp_names
    self.partition_indices = bcbp_indices
    self.model = {
      "data_description": self.data_description,
      "feed_map": self.feeds,
      "dump_rate": dump_rate,
      **valid_partitions,
    }

  @property
  def nfeed(self) -> int:
    return len(self.feeds)

  def simulate_ms(self, output_ms: str) -> None:
    """Simulate data into the given measurement set name"""
    table_desc = ADDITIONAL_COLUMNS if self.simulate_data else {}
    # Generate descriptors, create simulated data from the descriptors
    # and write simulated data to the main Measurement Set
    with Table.ms_from_descriptor(output_ms, "MAIN", table_desc) as T:
      startrow = 0

      for chunk_desc in self.generate_descriptors():
        data_dict = self.data_factory(chunk_desc)
        (nrow,) = data_dict["TIME"][1].shape
        T.addrows(nrow)

        for column, (_, data) in data_dict.items():
          T.putcol(column, data, index=(slice(startrow, startrow + nrow),))

        startrow += nrow

    kw = {"readonly": False}
    nddid = len(self.data_description)

    with Table.from_filename(f"{output_ms}::FEED", **kw) as T:
      T.addrows(len(self.feeds))
      for r, feed in enumerate(self.feeds.values()):
        pol_types = np.array(feed.polarisation_types())[None, :]
        index = (np.array([r]),)
        T.putcol("NUM_RECEPTORS", np.array([feed.nreceptors]), index=index)
        T.putcol("BEAM_OFFSET", np.zeros((1, 2, feed.nreceptors)), index=index)
        T.putcol("RECEPTOR_ANGLE", np.zeros((1, feed.nreceptors)), index=index)
        T.putcol("POLARIZATION_TYPE", pol_types, index=index)
        T.putcol(
          "POL_RESPONSE",
          np.zeros((1, feed.nreceptors, feed.nreceptors), np.complex64),
          index=index,
        )

    # Populate the main DATA_DESCRIPTION table
    # For simplicity we have a new spw and pol id's in each row
    with Table.from_filename(f"{output_ms}::DATA_DESCRIPTION", **kw) as T:
      T.addrows(nddid)
      T.putcol("SPECTRAL_WINDOW_ID", np.arange(nddid))
      T.putcol("POLARIZATION_ID", np.arange(nddid))

    # Partially populate the SPECTRAL_WINDOW table
    with Table.from_filename(f"{output_ms}::SPECTRAL_WINDOW", **kw) as T:
      T.addrows(nddid)
      for r, (chan_freq, _) in enumerate(self.data_description):
        (nchan,) = chan_freq.shape
        index = (np.array([r]),)
        chan_width = np.full(nchan, (chan_freq[-1] - chan_freq[0]) / nchan)
        T.putcol("NUM_CHAN", np.array([nchan]), index=index)
        T.putcol("CHAN_FREQ", chan_freq[None, :], index=index)
        T.putcol("CHAN_WIDTH", chan_width[None, :], index=index)
        T.putcol("RESOLUTION", chan_freq[None, :], index=index)
        T.putcol("EFFECTIVE_BW", chan_width[None, :], index=index)

    # Partially populate the POLARIZATION table
    with Table.from_filename(f"{output_ms}::POLARIZATION", **kw) as T:
      T.addrows(nddid)
      for r, (_, corrs) in enumerate(self.data_description):
        ncorr = len(corrs)
        index = (np.array([r]),)
        corr_type = corrs.to_numpy()[None, :]
        T.putcol("NUM_CORR", np.array([ncorr]), index=index)
        T.putcol("CORR_TYPE", corr_type, index=index)
        T.putcol(
          "CORR_PRODUCT",
          np.asarray(corrs.corr_product())[None, :],
          index=index,
        )

    # Partially populate the ANTENNA table
    with Table.from_filename(f"{output_ms}::ANTENNA", **kw) as T:
      T.addrows(self.nantenna)
      position = np.arange(self.nantenna * 3, dtype=np.float64).reshape(
        self.nantenna, 3
      )
      T.putcol("POSITION", position)
      T.putcol("OFFSET", position)  # Use a ramp here too
      T.putcol("NAME", np.asarray([f"ANTENNA-{i}" for i in range(self.nantenna)]))
      T.putcol("MOUNT", np.asarray(["ALT-AZ" for _ in range(self.nantenna)]))
      T.putcol("STATION", np.asarray([f"STATION-{i}" for i in range(self.nantenna)]))

  def generate_descriptors(self) -> Generator[PartitionDescriptor, None, None]:
    """Generates a sequence of descriptors, each corresponding to a partition"""
    chunk_id = 0

    for partition_index in self.partition_indices:
      partition = [
        (n, np.array([p], np.int32))
        for n, p in zip(self.partition_names, partition_index)
      ]
      tstart: float = self.time_start

      for t in range(0, self.ntime, self.time_chunks):
        ndumps = min(self.ntime - t, self.time_chunks)
        time = np.arange(ndumps, dtype=np.float64) * self.dump_rate + tstart
        yield PartitionDescriptor(
          **{
            "TIME": time,
            "chunk_id": chunk_id,
            "simulate_data": self.simulate_data,
            "auto_corrs": self.auto_corrs,
            **self.model,
            **dict(partition),
          }
        )
        tstart += ndumps * self.dump_rate
        chunk_id += 1

  @staticmethod
  def broadcast_partition_indices(
    partitions: List[npt.NDArray[np.int32]],
  ) -> npt.NDArray[np.int32]:
    partition_index = list(range(len(partitions)))
    bparts = [
      partition[tuple(slice(None) if p == i else None for i in partition_index)]
      for p, partition in enumerate(partitions)
    ]
    return np.stack([a.ravel() for a in np.broadcast_arrays(*bparts)], axis=1)

  @staticmethod
  def data_factory(
    desc: PartitionDescriptor,
  ) -> Dict[str, Tuple[Tuple[str, ...], npt.NDArray]]:
    """Creates simulated MS data from a partition descriptor"""
    try:
      ddid = desc.DATA_DESC_ID.item()
    except ValueError as e:
      raise ValueError(
        f"Only single DATA_DESC_ID " f"per partition allowed {desc.DATA_DESC_ID}"
      ) from e

    try:
      chan_freq, corrs = desc.data_description[ddid]
    except ValueError:
      raise ValueError(f"Unable to find a descriptor for " f"DATA_DESC_ID={ddid}")

    # Use on FEED for this ddid
    feed = desc.feed_map[tuple(corrs.polarisation_types())]

    array_groups: List[Tuple[List[str], List[npt.NDArray]]] = [
      (["PROCESSOR_ID"], [desc.PROCESSOR_ID]),
      (["FIELD_ID"], [desc.FIELD_ID]),
      (["DATA_DESC_ID"], [desc.DATA_DESC_ID]),
      (["TIME"], [desc.TIME]),
      (["FEED1", "FEED2"], [np.asarray([feed.feed_id])] * 2),
      (["ANTENNA1", "ANTENNA2"], [desc.ANTENNA1, desc.ANTENNA2]),
    ]

    # Broadcast indexing columns against each other
    barrays: List[Tuple[str, npt.NDArray]] = []
    columns: Any
    arrays: Any

    for i, (columns, arrays) in enumerate(array_groups):
      assert len(columns) == len(arrays)
      index = tuple(slice(None) if i == ii else None for ii in range(len(array_groups)))
      barrays.extend((c, a[index]) for c, a in zip(columns, arrays))

    columns, arrays = zip(*barrays)
    arrays = np.broadcast_arrays(*arrays)
    np_arrays: List[Tuple[str, Tuple[str, ...], npt.NDArray]] = [
      (n, ("row",), a.ravel()) for n, a in zip(columns, arrays)
    ]

    # Dimensions
    nrows = len(next(iter(np_arrays))[2])
    (nchan,) = chan_freq.shape
    ncorr = len(corrs)

    # Add the INTERVAL column
    interval = np.full(nrows, desc.dump_rate, dtype=np.float64)
    np_arrays.append(("INTERVAL", ("row",), interval))

    if desc.simulate_data:
      # ramps
      uvw = np.arange(nrows * 3, dtype=np.float64).reshape(nrows, 3)
      data = np.arange(nrows * nchan * ncorr, dtype=np.complex64).reshape(
        nrows, nchan, ncorr
      )
      data.imag = data.real

      # Alternating flags
      flag = np.zeros(nrows * nchan * ncorr, dtype=bool)
      flag[::2] = 1
      flag = flag.reshape(nrows, nchan, ncorr)

      np_arrays.extend(
        [
          ("UVW", ("row", "uvw"), uvw),
          ("DATA", ("row", "chan", "corr"), data),
          ("FLAG", ("row", "chan", "corr"), flag),
          ("WEIGHT_SPECTRUM", ("row", "chan", "corr"), data.real),
        ]
      )

    return {column: (dims, data) for column, dims, data in np_arrays}


def simulate(name=None, **sim_params) -> str:
  """
  Create a Measurement Set in a temporary directory,
  with the given simulation parameters.
  Return the directory
  """
  tmpdir = tempfile.mkdtemp()
  ms_path = os.path.join(tmpdir, name or "simulated.ms")
  simulator = MSStructureSimulator(**{**DEFAULT_SIM_PARAMS, **sim_params})
  simulator.simulate_ms(ms_path)
  return ms_path
