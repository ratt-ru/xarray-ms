Tutorial
========

The `Measurement Set v2.0 <msv2-spec_>`_ is a tabular format that
includes notions of regularity or, the shape of the data, in the MAIN table.
This is accomplished through the ``DATA_DESC_ID`` column which defines the
Spectral Window and Polarisation Configuration associated with each row:
the shape of the visibility in each row of the ``DATA`` column can
vary per-row.

By contrast `Measurement Set v4.0 <msv4-spec_>`_ specifies a
collection of Datasets of ndarrays on a regular grid.
To move data between the two formats, it is necessary to partition
or group MSv2 rows by the same shape and configuration.

In xarray-ms, this is accomplished by specifying ``partition_columns``
when opening a Measurement Set.
Different columns may be used to define the partition, but
:code:`[DATA_DESC_ID, FIELD_ID, OBSERVATION_ID]` is a reasonable choice.

Opening a Measurement Set
-------------------------

As xarray-ms implements an `xarray backend <xarray_backend_>`_,
it is possible to use the standard :func:`xarray.open_dataset`
to open up a single partition of a Measurement Set.

.. ipython:: python
  :okwarning:

  import xarray_ms
  from xarray_ms.testing.simulator import simulate
  import xarray

  # Simulate a Measurement Set with 3
  # channel and polarisation configurations
  ms = simulate("test.ms", data_description=[
    (8, ("XX", "XY", "YX", "YY")),
    (4, ("RR", "LL")),
    (16, ("RR", "RL", "LR", "LL"))])

  ds = xarray.open_dataset(ms,
    partition_columns=["DATA_DESC_ID", "FIELD_ID", "OBSERVATION_ID"])

  ds

Opening a specific partition
++++++++++++++++++++++++++++++

Because we've simulated multiple Data Description values in
our Measurement Set, xarray-ms has automatically opened the first partition
containing 8 frequencies and 4 linear polarisations.
To open the second partition a ``partition_key`` can be also be
passed to :func:`xarray.open_dataset`.

.. ipython:: python

  ds = xarray.open_dataset(ms,
    partition_columns=["DATA_DESC_ID", "FIELD_ID", "OBSERVATION_ID"],
    partition_key=(("DATA_DESC_ID", 1), ("FIELD_ID", 0), ("OBSERVATION_ID", 0)))

  ds

and it can be seen that the dataset refers to the second partition
containing 4 frequencies and 2 circular polarisations.

Selecting a subset of the data
++++++++++++++++++++++++++++++

By default, :func:`xarray.open_dataset` will return a dataset
with a lazy view over the data.
xarray has extensive functionality for
`indexing and selecting data <xarray_indexing_and_selecting_>`_.

For example, one could select select some specific dimensions out:

.. ipython:: python

  ds = xarray.open_dataset(ms,
    partition_columns=["DATA_DESC_ID", "FIELD_ID", "OBSERVATION_ID"],
    partition_key=(("DATA_DESC_ID", 1), ("FIELD_ID", 0), ("OBSERVATION_ID", 0)))

  subds = ds.isel(time=slice(1, 3), baseline=[1, 3, 5], frequency=slice(2, 4))
  subds

At this point, the dataset is still lazy -- no Data variables have been loaded
into memory.

Loading in a lazy dataset
+++++++++++++++++++++++++

By calling load on the lazy dataset, all the Data Variables are loaded onto the
dataset as numpy arrays.

.. ipython:: python

  subds.load()
