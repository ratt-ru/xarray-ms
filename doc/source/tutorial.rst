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
it is possible to use the standard :func:`xarray.open_datatree`
to open multiple partitions of a Measurement Set.

.. ipython:: python
  :okwarning:

  import xarray_ms
  from xarray_ms.testing.simulator import simulate
  from xarray.backends.api import open_datatree

  # Simulate a Measurement Set with 2 channel and polarisation configurations
  ms = simulate("test.ms", data_description=[
    (8, ("XX", "XY", "YX", "YY")),
    (4, ("RR", "LL"))])

  dt = open_datatree(ms, partition_columns=[
      "DATA_DESC_ID",
      "FIELD_ID",
      "OBSERVATION_ID"])

  dt

Selecting a subset of the data
++++++++++++++++++++++++++++++

By default, :func:`~xarray.backends.api.open_datatree` will return a dataset
with a lazy view over the data.
xarray has extensive functionality for
`indexing and selecting data <xarray_indexing_and_selecting_>`_.

For example, one could select select some specific dimensions out:

.. ipython:: python

  dt = open_datatree(ms,
    partition_columns=["DATA_DESC_ID", "FIELD_ID", "OBSERVATION_ID"])

  subdt = dt.isel(time=slice(1, 3), baseline=[1, 3, 5], frequency=slice(2, 4))
  subdt

At this point, the DataTree is still lazy -- no Data variables have been loaded
into memory.

Loading a DataTree
++++++++++++++++++

By calling load on the lazy datatree, all the Data Variables are loaded onto the
dataset as numpy arrays.

.. ipython:: python

  subdt.load()
