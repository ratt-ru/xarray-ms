# xarray-ms: xarray MSv4 view over a CASA Measurement Set v2

[ratt-ru/xarray-ms](https://github.com/ratt-ru/xarray-ms) presents an xarray
[Measurement Set v4](https://xradio.readthedocs.io/en/stable/measurement_set/schema.html) interface over
CASA [Measurement Set v2](https://casa.nrao.edu/Memos/229.html) data.

### Core API

Open a Measurement Set as a lazy xarray DataTree via the standard xarray backend API:

```python
import xarray
datatree = xarray.open_datatree("/data/data.ms", partition_schema=["FIELD_ID"])
```

Each partition becomes a node in the DataTree, holding a Dataset with dimensions
`(time, baseline_id, frequency, polarization)`.

### Partitioning

MSv2 is tabular: one row per `(TIME, ANTENNA1, ANTENNA2, DATA_DESC_ID, ...)`
composite primary key.
MSv4 requires a regular `(time, baseline_id, frequency, polarization)` grid.
xarray-ms bridges the two by partitioning rows into groups that can each be
viewed as a grid, then building a **row map** that projects grid coordinates back
onto MSv2 row numbers.

#### 1. Partition and sort

`TablePartitioner` in [partition.py](xarray_ms/backend/msv2/partition.py) reads
only the indexing columns of the MAIN table, along with the original MSv2 row
number of each row, and then:

1. sorts all rows by a compound key: *partition columns first, then sort columns,
   then the rest*; the row numbers and remaining columns are permuted along with
   the keys so that they stay row-aligned,
2. locates the partition boundaries by looking for changes in the partition
   column values of the sorted rows,
3. slices the sorted columns at those boundaries into one group per partition key.

Partition columns are `DATA_DESC_ID`, `OBSERVATION_ID`, `PROCESSOR_ID` and
`STATE::OBS_MODE` (derived from `STATE_ID`) by default; `FIELD_ID`,
`SCAN_NUMBER`, `SOURCE_ID`, `SUB_SCAN_NUMBER` and `STATE_ID` can be added via
`partition_schema`. Sort columns are always `TIME`, `ANTENNA1`, `ANTENNA2` — the
MSv2 primary key that induces the `(time, baseline_id)` grid.

The result maps each partition key onto that partition's indexing columns, in
`(TIME, ANTENNA1, ANTENNA2)` order, including the original MSv2 row numbers.

#### 2. Establish the grid axes

`MSv2Structure` in [structure.py](xarray_ms/backend/msv2/structure.py) turns each
group into a `PartitionData`. It is itself a `Mapping` of the form
`{((PCOL_0, PVAL_0), ..., (PCOL_N, PVAL_N)): PartitionData}`.

- **time axis**: the unique `TIME` values form the `time` coordinate, and the
  inverse of that unique operation gives each row its timestep index.
- **baseline axis**: the antenna set is taken from `FEED::ANTENNA_ID` for the
  partition's `SPECTRAL_WINDOW_ID` and feeds — *not* from the antennas actually
  present in the rows — so the antenna and baseline counts are canonical for the
  instrument. `ANTENNA1`/`ANTENNA2` values must be a subset of these, else
  `InvalidMeasurementSet` is raised.

This makes the grid a **superset** of the grid implied by the rows: every
timestep is crossed with every canonical baseline, whether or not a row exists.

#### 3. Build `PartitionData.row_map`

`row_map` is an `(time, baseline_id)` array of MSv2 row numbers — the inverse of the
`row -> (time, baseline_id)` mapping, and the key data structure of the backend.

It is built by scattering a partition's rows, into a single array that is
pre-filled with `-1`. Each row's destination is computed
from its timestep index and its baseline index, the latter derived from its
antenna pair by an upper-triangular baseline numbering; the row's MSv2 row number
is written there.

Before the scatter, rows are normalised so that the baseline index is
well-defined:

- auto-correlations are dropped if the canonical grid excludes them,
- antenna pairs with `ANTENNA1 > ANTENNA2` are swapped to the upper-triangle
  convention,
- antenna IDs are remapped to a contiguous `0..antenna-1` range when
  `FEED::ANTENNA_ID` is not already contiguous.

Two rows landing on the same `(time, baseline_id)` cell is an error: it means the
partition cannot be expressed as a grid, and `PartitioningError` is raised. The
fix is generally a finer `partition_schema`.

#### 4. `-1` means "missing row"

Entries left at `-1` are grid cells for which MSv2 has no row: a baseline that
was flagged away by CASA `split --keepflags=False`, an antenna that dropped out
for part of the observation, or auto-correlations requested on a dataset without
them. `-1` is never a valid MSv2 row number, so it is unambiguous.

`-1` propagates straight through to the reads. When xarray requests a block of a
data variable, `MSv2Array` ([array.py](xarray_ms/backend/msv2/array.py)) indexes
`row_map` with the `(time, baseline_id)` part of the selection, flattens the
result into a list of rows — possibly containing `-1`s — and hands it to `arcae`
together with an output buffer pre-filled with that variable's default value.
`arcae` skips negative indices, leaving those positions of the buffer untouched,
so imputation is simply the buffer's fill value: `nan` for data variables,
flagged for flags. No separate gather, scatter or masking pass is needed.

Other consumers filter the `-1`s out instead: the non-negative entries identify
the rows that actually exist, e.g. when probing column shapes, and their count
drives the `IrregularBaselineGridWarning`.

#### 5. Subtable propagation

MSv2 specifies a composite primary key for the MAIN table: it contains
foreign keys that reference rows in sub-tables. e.g. the `FIELD_ID` column
references rows in the `FIELD` sub-table.
MSv4 has no such indirection: metadata is either attached to the correlated
dataset itself, or lives in a child dataset of the partition's DataTree node.
It follows that the columns in the in the `partition_schema` direct how
subtable data is propagated to each partition.

#### Irregularity warnings

Signal when the grid cannot be made perfectly regular:

- `IrregularTimeGridWarning` — non-uniform `INTERVAL`; `integration_time=nan`
  is set and per-row `TIME` / `INTEGRATION_TIME` columns are added.
- `IrregularChannelGridWarning` — non-uniform `CHAN_WIDTH`; `channel_width=nan`
  is set and a per-channel `CHANNEL_WIDTH` column is added.
- `IrregularBaselineGridWarning` — `row_map` contains `-1`s, i.e. baselines
  missing for some timesteps; imputed with defaults (benign in most cases).

### Measures

MSv2 and MSv4 record measures (units, measure type and reference frame) in
different places:

- **MSv2** stores them once per column, in the column descriptor keywords:
  `MEASINFO["type"]` (e.g. `position`), `MEASINFO["Ref"]` for a fixed frame
  (e.g. `ITRF`) *or* `MEASINFO["VarRefCol"]` naming a column of per-row frame
  codes, and `QuantumUnits` (e.g. `["m", "m", "m"]`).
- **MSv4** stores them as `attrs` on each variable: `type`, `units` and a
  frame-like key (`frame`, `scale`/`format` for time, `observer` for frequency),
  plus schema-only attributes such as `coordinate_system`, `origin` and
  `origin_object_name` that have no MSv2 equivalent.

#### 1. Adapt, then decode

Measures are translated column by column, in
[measures_adapters.py](xarray_ms/backend/msv2/measures_adapters.py) and
[measures_encoders.py](xarray_ms/backend/msv2/measures_encoders.py):

```python
coder_factory = MSv2CoderFactory.from_arrow_table(subtable)  # or .from_table_desc(desc)
var = coder_factory.create("COLUMN").decode(var)
```

- The **measures adapter** reads the column descriptor. `ArrowTableMeasuresAdapter`
  (used by `from_arrow_table`) also resolves `VarRefCol` against the table's
  rows. Because the subtable has already been cut down to the partition's rows,
  it must find exactly one frame, else `PartitioningError` is raised. The fix is
  a finer `partition_schema`. `ColumnDescMeasuresAdapter` (used by
  `from_table_desc`) supports a fixed `Ref` only.
- `MSv2CoderFactory.create` picks a **coder** from `MEASINFO["type"]`:

  | MSv2 type   | Coder             | MSv4 `type`      | Frame key / mapping                     |
  |-------------|-------------------|------------------|-----------------------------------------|
  | `epoch`     | `EpochCoder`      | `time`           | `scale`; also converts MJD → unix data |
  | `frequency` | `FrequencyCoder`  | `spectral_coord` | `observer`, e.g. `LSRK` → `lsrk`        |
  | `direction` | `DirectionCoder`  | `sky_coord`      | `frame`, e.g. `J2000` → `fk5`           |
  | `position`  | `PositionCoder`   | `location`       | `frame`, `ITRF` → `ITRS`                |
  | `uvw`       | `UvwCoder`        | `uvw`            | `frame`, e.g. `J2000` → `fk5`           |

#### 2. How to construct a measures variable

Split the attributes by where they come from:

1. put the MSv4-only attributes, i.e. those the MSv4 schema requires but MSv2
   cannot express, on the `Variable` when you create it,
2. pass the variable through `decode`, which adds the attributes derived from
   the MSv2 measures (`type`, `units`, frame).

`ANTENNA_POSITION` in [antenna.py](xarray_ms/backend/msv2/factories/antenna.py)
is the reference example:

```python
antenna_position = Variable(
  ("antenna_name", "cartesian_pos_label"),
  position,
  {"coordinate_system": "geocentric", "origin_object_name": "earth"},
)
ant_coder_factory.create("POSITION").decode(antenna_position)
# attrs: coordinate_system=geocentric, origin_object_name=earth,
#        type=location, units=m, frame=ITRS
```

**Do not** assign `var.attrs = {...}` after decoding. This replaces the derived
attributes and silently drops any not listed, almost always the `frame`.
`PHASED_ARRAY_ELEMENT_OFFSET` used to do this (see
[#176](https://github.com/ratt-ru/xarray-ms/pull/176#discussion_r4083467399));
it now follows the two-step pattern in
[phased_array.py](xarray_ms/backend/msv2/factories/phased_array.py).
Its `frame` is the MSv2 `Ref` (`ITRF` → `ITRS`) as recorded, even though the
offsets are topocentric.

#### 3. When hardcoding is legitimate

Only when MSv2 holds no measures information for the column:

- `PHASED_ARRAY_COORDINATE_AXES` is a rotation matrix with no `MEASINFO`;
  its `units: dimensionless` and `type: rotation_matrix` are set directly.
- `DATA` carries no units in MSv2, so `MSV4_to_MSV2_COLUMN_SCHEMAS` in
  [correlated.py](xarray_ms/backend/msv2/factories/correlated.py) supplies
  `VisibilityCoder` (`units: Jy`) instead of the default coder.

### Backend

Uses `arcae` (not python-casacore) for high-performance CASA Table access.

#### Write support

There are two versions of arcae available. They can be distinguished
by the `arcae.safe_multithreaded_writes()` function returning `True` if writes
are supported when multiple (i.e. > 1) instances of a casacore Table
are opened by arcae.

* `arcae >= 0.5.0, < 0.6.0` supports writing if only a single casacore Table
  instance is requested (i.e. `ninstances=1`). This functionality is
  currently available in arcae's `main` branch.
  `xarray-ms >= 0.5.0, < 0.6.0` and xarray-ms' `main` branch mirror this
  functionality and versioning setup.
* `arcae >= 0.4.0, < 0.5.0` supports writing when multiple instances
  of a casacore Table are requested (i.e. `ninstances > 1`) by supporting
  a Multiple-Reader Single-Writer pattern.
  This functionality is currently available via `alpha` releases in
  arcae's `0.4.0-dev` branch. `xarray-ms >= 0.4.0, < 0.5.0` and xarray-ms'
  `write-support`  branch mirror this functionality and versioning setup.
  The intention is to merge both these branches into each repository's
  `main` branch after undergoing sufficient testing.

## When relevant
- [Documentation](doc/source) → Documentation markdown source
- Adding or changing a variable carrying units or frames → read [Measures](#measures)

## Tooling

This package is managed by the `uv` tool.

- Install: `uv sync`
- Test: `uv run --group test py.test --msv4_test_corpus tests/`.
  Remove the `--mvs4_test_corpus` flag for quick tests that don't require real data.
- Docs: Sphinx; source in `doc/source/`, hosted on ReadTheDocs.
- Pre-commit hooks gate commits; run them before proposing a change as done: `uv run --dev pre-commit run -a`
- [Changelog](doc/source/changelog.rst): Provide changelog entries when creating a PR.

### Test Data Generation

[simulator.py](xarray_ms/testing/simulator.py) contains a `simulate` function
that constructs a `MSStructureSimulator` class used for simulating the structure of
a Measurement Set for test fixtures. It creates:

1. A MAIN MSv2 table containing:

  a. The columns forming the composite primary key.
  b. Ramp data for visibilities, weights and UVW coordinates in the MAIN table.

2. Sub-tables whose rows are referenced through the composite primary key.
  Sub-tables are populated with valid, but not fully physically realistic data.
