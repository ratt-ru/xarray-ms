.. _partitioning-guide:

Partitioning Guide
==================

`Measurement Set v4.0 <msv4-spec_>`_ specifies a series of datasets with
``time``, ``baseline_id`` and ``frequency`` coordinates where
``time`` and ``frequency`` have associated ``integration_time`` and
``channel_width`` attributes.
In the best case, this represents monotonic, equidistant values along
``time`` and ``frequency`` and the standard quadratic relation between
antennas in the case of ``baseline_id``.
Observational data recorded directly off an interferometer and stored
for archival purposes will commonly follow a
``(time, baseline_id, frequency)`` ordering.

The usefulness of this representation and ordering is that it is
simple and easy for software to reason about.
This is desirable as it simplifies our software.

The challenge in converting from MSv2 to MSv4 is formulating a
partitioning strategy to handle irregularity in an MSv2 dataset.

Measurement Set v2.0 irregularity
---------------------------------

By contrast the `Measurement Set v2.0 <msv2-spec_>`_ is a tabular format that
does not enforce any notion of regularity (although much software assumes it).
The ``TIME`` and ``INTERVAL`` columns in the MAIN MSv2 table
describe the midpoint in time at which a sample was measured
and the amount of time (integration time) taken to measure the sample,
while the ``ANTENNA1`` and ``ANTENNA2`` columns define the baseline along
which the sample was measured.
``TIME``, ``ANTENNA1`` and ``ANTENNA2`` are *keys* in the tabular MAIN table
and there is no requirement that the measurements they index are ordered,
or even form a regular ``(time, baseline_id)`` grid.
Additionally, the ``DATA_DESC_ID`` column establishes a relation to the
``SPECTRAL_WINDOW::CHAN_FREQ`` and ``SPECTRAL_WINDOW::CHAN_WIDTH`` columns
representing the frequency centroid and bandwidth of the sample, respectively.

The challenge that MSv2 poses to radio astronomy software in the worst case
is that it can represent overlapped or disjoint measurements in time and frequency
for one or more baselines.
However, most observational data is well-behaved:
Measurements are commonly ordered by ``TIME, ANTENNA1, ANTENNA2``
and ``CHAN_FREQ`` commonly increases monotically with
equidistant values (i.e. ``CHAN_WIDTH`` values are uniform) but this cannot
always be assumed.
Any regularity in an MSv2 MS is achieved through convention rather
than enforcement.


Choosing a partitioning strategy
--------------------------------

By default, MSv2 measurements are partitioned by ``DATA_DESC_ID``,
``OBSERVATION_ID``, ``PROCESSOR_ID`` and the
``STATE::OBS_MODE`` (via ``STATE_ID``) columns.

.. autodata:: xarray_ms.backend.msv2.structure.DEFAULT_PARTITION_COLUMNS

For example, it follows from the previous section that,
in order to achieve regularity in frequency, *partition*
MSv2 measurements by the ``DATA_DESC_ID`` column.

Partitioning always uses these columns, but additional columns can be
selected if finer grained partitioning is required:

.. autodata:: xarray_ms.backend.msv2.structure.VALID_PARTITION_COLUMNS

Note that ``OBS_MODE`` and ``SUB_SCAN_NUMBER`` are columns in the ``STATE``
subtable, while ``SOURCE_ID`` is a column of the ``FIELD`` subtable.
Partitioning on these columns is achieved by joining on the ``STATE_ID``
and ``FIELD_ID`` columns, respectively.


Within these partitions, measurements are sorted by
``TIME``, ``ANTENNA1`` and ``ANTENNA2``
to form a grid.

.. _time-partitioning:

Partitioning in time
++++++++++++++++++++

Compared to frequency, achieving regularity in time requires more thought
as it depends on identifying partitions of MSv2 where data:

1. contains monotically increasing ``TIME`` (after ordering).
2. is dumped with a uniform ``INTERVAL``.
3. ideally contains no gaps: i.e. ``(TIME - INTERVAL)[1:] == (TIME + INTERVAL)[:-1]``.

For example, ``OBS_MODE`` specifying ``STATE::OBS_MODE`` via ``STATE_ID``
is a good default partitioner, as it represents a shift in the
interferometer's mode of operation: It identifies when
the interferometer is e.g. slewing/observing a calibrator/observing a target.

Other valid partitioning columns are:

- ``FIELD_ID``: Observing a field for a period of time.
- ``SOURCE_ID``: Observing a source within a field for a period of time.
- ``SCAN_NUMBER``: A coarse, logical number (i.e. scan) associated with the data.
- ``SUB_SCAN_NUMBER``: A finer, logical number (i.e. scan) associated with the data.
  This specifies ``STATE::SUB_SCAN_NUMBER`` (via ``STATE_ID``).
- ``STATE_ID``: The state of an interferometer.

as these columns frequently identify measurement groupings where
the interferometer is consistently dumping.

.. code-block:: python

    import xarray_ms
    import xarray

    # Also partition by SCAN_NUMBER and FIELD_ID
    dt = xarray.open_datatree(ms, partition_schema=["SCAN_NUMBER", "FIELD_ID"])

.. _missing-baselines:

Missing Baselines
-----------------

Baselines can be missing for distinct ``TIME`` values.
This can occur when Measurement Sets are passed through the
CASA ``split`` task with ``keepflags=False`` set, for instance.

Having all baselines present can be useful
for simplifying calibration algorithms and cases where
auto-correlations are requested, but none are present in the data.

``xarray-ms`` will impute these missing data points with default values
(``nan`` in the case of data, ``1`` in the case of flags).

Irregular Grid Warnings
-----------------------

Given the specified partitioning schema, ``xarray-ms`` will partition
the MSv2 by the supplied columns and attempt to establish a regular
``(time, baseline_id, frequency)`` grid.
If this is not possible, three classes of warning can be issued,
related to each of the three dimensions.

:class:`~xarray_ms.errors.IrregularTimeGridWarning`
+++++++++++++++++++++++++++++++++++++++++++++++++++

This warning is raised when it is impossible
to identify a unique ``INTERVAL`` value for a partition.
This is required to assign a single ``integration_time``
attribute to the ``time`` coordinate.

The above check is relaxed slightly by excluding the last time
in the partition (to handle averaged data) and by allowing
a degree of jitter in the ``INTERVAL`` column.

Generally, this happens if the requested partitioning schema
does not satisfy the criteria described in :ref:`time-partitioning`.
The solution is to experiment with other partitioning columns.

Should the user wish to continue with this case,
``xarray-ms`` sets ``integration_time=nan``
and adds ``(time, baseline_id)``-shaped,
``TIME`` and ``INTEGRATION_TIME`` columns.
Downstream applications should account for this.

:class:`~xarray_ms.errors.IrregularChannelGridWarning`
++++++++++++++++++++++++++++++++++++++++++++++++++++++

This warning is raised when it is impossible to identify a unique
``CHAN_WIDTH`` value for the partition.
This is required to assign a single ``channel_width``
attribute to the ``frequency`` coordinate.

Should the user wish to continue with this
case ``xarray-ms`` sets ``channel_width=nan``
and adds ``(frequency,)``-shaped ``CHANNEL_WIDTH`` columns.
Downstream application should account for this.

:class:`~xarray_ms.errors.IrregularBaselineGridWarning`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

This warning is raised when baselines were missing for a
particular timestep.
This is a relatively benign warning as ``xarray-ms`` will
impute missing values (See :ref:`missing-baselines`).
