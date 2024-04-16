from __future__ import annotations

from collections.abc import Iterator, MutableSet, Sequence
from enum import IntEnum
from numbers import Integral
from typing import Any, Dict, Iterable, List, Tuple, overload

import numpy as np
import numpy.typing as npt

PRIMARY_KEY_COLUMNS = [
  "FIELD_ID",
  "DATA_DESC_ID",
  "PROCESSOR_ID",
  "FEED1",
  "FEED2",
  "FEED3",
  "ANTENNA1",
  "ANTENNA2",
  "ANTENNA3",
  "TIME",
  "TIME_EXTRA_PREC",
]

INDEXABLE_COLUMNS = PRIMARY_KEY_COLUMNS + [
  "ARRAY_ID",
  "OBSERVATION_ID",
  "PHASE_ID",
  "SCAN_NUMBER",
  "STATE_ID",
]


class Stokes(IntEnum):
  """
  Enumeration of stokes types as defined in
  Measurement Set 2.0 and Stokes.h in casacore:
  https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
  """

  Undefined = 0
  I = 1  # noqa: E741
  Q = 2
  U = 3
  V = 4
  RR = 5
  RL = 6
  LR = 7
  LL = 8
  XX = 9
  XY = 10
  YX = 11
  YY = 12
  RX = 13
  RY = 14
  LX = 15
  LY = 16
  XR = 17
  XL = 18
  YR = 19
  YL = 20
  PP = 21
  PQ = 22
  QP = 23
  QQ = 24
  RCircular = 25
  LCircular = 26
  Linear = 27
  Ptotal = 28
  Plinear = 29
  PFtotal = 30
  PFlinear = 31
  Pangle = 32

  @staticmethod
  def from_value(stokes: int | str | Stokes) -> Stokes:
    if isinstance(stokes, str):
      try:
        return Stokes[stokes]
      except KeyError:
        raise ValueError(f"{stokes} is not a valid Stokes parameter")
    elif isinstance(stokes, Integral):
      return Stokes(stokes)
    elif isinstance(stokes, Stokes):
      return stokes
    else:
      raise TypeError(
        f"Cannot convert {stokes} "
        f"of type {type(stokes)} "
        f"to a Stokes Enumeration"
      )


class Polarisations(Sequence):
  """A sequence of Stokes parameters"""

  _pols: List[Stokes]

  def __init__(self, *polarisations: Stokes):
    self._pols = list(polarisations)

  @overload
  def __getitem__(self, index: int) -> Stokes: ...

  @overload
  def __getitem__(self, index: slice) -> List[Stokes]: ...

  def __getitem__(self, index):
    return self._pols[index]

  def __len__(self) -> int:
    return len(self._pols)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Polarisations):
      return NotImplemented
    return self._pols == other._pols

  def __hash__(self) -> int:
    return hash(tuple(self._pols))

  def __str__(self) -> str:
    return f"[{','.join((c.name for c in self._pols))}]"

  def __repr__(self) -> str:
    return f"Polarisations({','.join((repr(c) for c in self._pols))})"

  @staticmethod
  def from_values(values: Iterable[Any]) -> Polarisations:
    """Creates a Polarisations object from a sequence of values"""
    if all(isinstance(ct, Stokes) for ct in values):
      return Polarisations(*values)

    return Polarisations(*(Stokes.from_value(v) for v in values))

  @property
  def is_mixed_feed(self) -> bool:
    if all(Stokes.XX <= p <= Stokes.YY for p in self._pols):
      return False

    if all(Stokes.RR <= p <= Stokes.LL for p in self._pols):
      return False

    if all(Stokes.PP <= p <= Stokes.QQ for p in self._pols):
      return False

    return True

  def polarisation_types(self) -> List[str]:
    """Derive polarisation types"""
    lookup = set()
    return list(
      p
      for p in "".join(self.to_str())
      if p not in lookup and lookup.add(p) is None  # type: ignore
    )

  def corr_product(self) -> List[Tuple[int, ...]]:
    """Derive correlation product"""
    pol_map = {p: i for i, p in enumerate(self.polarisation_types())}
    return [tuple(pol_map[ct] for ct in pols) for pols in self.to_str()]

  def to_ints(self) -> List[int]:
    return [s.value for s in self._pols]

  def to_numpy(self) -> npt.NDArray[np.int32]:
    return np.asarray(self.to_ints(), np.int8)

  def to_str(self) -> List[str]:
    return [s.name for s in self._pols]


DataDescCanonicalType = Tuple[npt.NDArray[np.float64], Polarisations]
DataDescArgType = Tuple[npt.NDArray[np.float64], Sequence[str]]


class DataDescription(Sequence):
  """Approximates a DATA_DESCRIPTION table

  Internally maintains a list of (CHAN_FREQ, CORR_TYPE) equivalents
  """

  data_description: List[DataDescCanonicalType]

  @overload
  def __init__(self, data_description: List[DataDescCanonicalType]): ...

  @overload
  def __init__(self, data_description: List[DataDescArgType]): ...

  def __init__(self, data_descriptions):
    self.data_description = [
      (
        chan_freq,
        Polarisations.from_values(*corr_type)
        if isinstance(corr_type, list)
        else corr_type,
      )
      for chan_freq, corr_type in data_descriptions
    ]

  def __len__(self):
    return len(self.data_description)

  @overload
  def __getitem__(
    self, index: int
  ) -> Tuple[npt.NDArray[np.float64], Polarisations]: ...

  @overload
  def __getitem__(
    self, index: slice
  ) -> List[Tuple[npt.NDArray[np.float64], Polarisations]]: ...

  def __getitem__(self, index):
    return self.data_description[index]

  def polarisation_set(self) -> PolarisationSet:
    _, corrs = zip(*self.data_description)
    return PolarisationSet(corrs)

  @staticmethod
  def from_spw_corr_pairs(
    pairs: DataDescArgType | DataDescCanonicalType | DataDescription | None = None,
  ) -> "DataDescription":
    if pairs is None:
      # MeerKAT 4 channels, 2 linear correlations
      return DataDescription([(np.linspace(0.856e9, 2 * 0.856e9, 4), ["XX", "YY"])])

    # pass through
    if isinstance(pairs, DataDescription):
      return pairs

    # Must have a sequence at this point
    if not isinstance(pairs, Sequence):
      raise TypeError(
        f"type of pairs {type(pairs)} must be a list " f"or a DataDescription object"
      )

    result = []

    for pair in pairs:
      try:
        chan_freq, corrs = pair
      except ValueError as e:
        raise ValueError(f"{pair} should be a (chan_freq, corrs) tuple") from e

      if isinstance(chan_freq, int):
        # MeerKAT L-bad
        chan_freq = np.linspace(0.856e9, 2 * 0.856e9, chan_freq)
      elif isinstance(chan_freq, (list, tuple)):
        chan_freq = np.asarray(chan_freq)
      elif isinstance(chan_freq, np.ndarray):
        pass
      else:
        raise TypeError(
          f"chan_freq must be an int, tuple, list of ndarray. " f"Got {type(chan_freq)}"
        )

      if isinstance(corrs, (list, tuple, np.ndarray)):
        corrs = Polarisations.from_values(corrs)
      elif isinstance(corrs, Polarisations):
        pass
      else:
        raise TypeError(
          f"corrs must be a list, tuple, ndarray or Polarisations. "
          f"Got {type(corrs)}"
        )

      result.append((chan_freq, corrs))

    return DataDescription(result)


class Feed:
  _receptor_one: str
  _receptor_two: str
  _feed_id: int

  def __init__(self, polarisation_types: List[str], feed_id: int = 0):
    try:
      self._receptor_one, self._receptor_two = polarisation_types
    except ValueError as e:
      raise ValueError(
        f"Only two feed receptors are currently allowed. "
        f"Received {polarisation_types}"
      ) from e

    self._feed_id = feed_id

  def __eq__(self, other) -> bool:
    return (
      isinstance(other, Feed)
      and self._receptor_one == other._receptor_one
      and self._receptor_two == other._receptor_two
      and self.feed_id == other.feed_id
    )

  def __hash__(self) -> int:
    return hash((self._receptor_one, self._receptor_two, self.feed_id))

  @property
  def feed_id(self) -> int:
    return self._feed_id

  @property
  def nreceptors(self) -> int:
    return 2

  def polarisation_types(self) -> List[str]:
    return [self._receptor_one, self._receptor_two]


class PolarisationSet(MutableSet):
  _pols: MutableSet[Polarisations]

  def __init__(self, pols: Iterable[Polarisations]):
    self._pols = set(pols)

  def add(self, pols: Polarisations) -> None:
    self._pols.add(pols)

  def discard(self, pols: Polarisations) -> None:
    return self._pols.discard(pols)

  def __contains__(self, pols: object) -> bool:
    if not isinstance(pols, Polarisations):
      return NotImplemented
    return pols in self._pols

  def __hash__(self) -> int:
    return hash(self._pols)

  def __iter__(self) -> Iterator[Polarisations]:
    return iter(self._pols)

  def __len__(self) -> int:
    return len(self._pols)

  def feed_map(self) -> Dict[Tuple[str, ...], Feed]:
    result: Dict[Tuple[str, ...], Feed] = {}
    i = 0

    for c in self._pols:
      pol_type = c.polarisation_types()
      key = tuple(pol_type)
      try:
        result[key]
      except KeyError:
        result[key] = Feed(pol_type, i)
        i += 1

    return result
