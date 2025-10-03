Tutorial
========

The `Measurement Set v2.0 <msv2-spec_>`_ is a tabular format that
includes notions of regularity or, the shape of the data, in the MAIN table.
This is achieved through the ``DATA_DESC_ID`` column which defines the
Spectral Window and Polarisation Configuration associated with each row:
the shape of the visibility in each row of the ``DATA`` column can
vary per-row.

By contrast `Measurement Set v4.0 <msv4-spec_>`_ specifies a
collection of Datasets of ndarrays on a regular grid.
To move data between the two formats, it is necessary to partition
or group MSv2 rows by the same shape and configuration.

In xarray-ms, this is accomplished by specifying a ``partition_schema``
when opening a Measurement Set.
Different columns may be used to define the partition.
See :ref:`partitioning-guide` for more information.

Opening a Measurement Set
-------------------------

As xarray-ms implements an `xarray backend <xarray_backend_>`_,
it is possible to use the :func:`xarray.open_datatree` function
to open multiple partitions of a Measurement Set.

.. ipython:: python
  :okwarning:

  import xarray_ms
  import xarray
  import xarray.testing
  from xarray_ms.testing.simulator import simulate

  # Simulate a Measurement Set with 2 channel and polarisation configurations
  ms = simulate("test.ms", data_description=[
    (8, ("XX", "XY", "YX", "YY")),
    (4, ("RR", "LL"))])

  dt = xarray.open_datatree(ms, partition_schema=["FIELD_ID"])

  dt

.. warning::

  The MSv4 spec is still under development and the arrangement and naming
  of the DataTree branches is likely to change.

Selecting a subset of the data
++++++++++++++++++++++++++++++

By default, :func:`~xarray.open_datatree` will return a datatree
with a lazy view over the data.
xarray has extensive functionality for
`indexing and selecting data <xarray_indexing_and_selecting_>`_.

For example, one could select select some specific dimensions out:

.. ipython:: python

  dt = xarray.open_datatree(ms, partition_schema=["FIELD_ID"])

  subdt = dt.isel(time=slice(1, 3), baseline_id=[0, 2], frequency=slice(2, 4))
  subdt

At this point, the ``subdt`` DataTree is still lazy -- no Data variables have been loaded
into memory.

Loading a DataTree
++++++++++++++++++

By calling load on the lazy datatree, all the Data Variables are loaded onto the
dataset as numpy arrays.

.. ipython:: python

  subdt.load()

Opening a Measurement Set with dask_
------------------------------------

Generally speaking, observational data will be too large to fit in memory.
Either portions of the dataset must be selected and loaded, or it must be
processed in chunks.

Data processing using a chunked storage engine such as dask_
can be enabled by specifying the ``chunks`` parameter:

.. ipython:: python

  dt = xarray.open_datatree(ms, partition_schema=["FIELD_ID"],
    chunks={"time": 2, "frequency": 2}, auto_corrs=True)

  dt

Per-partition chunking
++++++++++++++++++++++

Different chunking may be desired, especially when applied to
different channelisation and polarisation configurations.
In these cases, the ``preferred_chunks`` argument can be used
to specify different chunking setups for each partition.

.. ipython:: python

  dt = xarray.open_datatree(ms, partition_schema=["FIELD_ID"],
    chunks={},
    preferred_chunks={
      (("DATA_DESC_ID", 0),): {"time": 2, "frequency": 4},
      (("DATA_DESC_ID", 1),): {"time": 3, "frequency": 2}})

See the ``preferred_chunks`` argument of
:meth:`~xarray_ms.backend.msv2.entrypoint.MSv2EntryPoint.open_datatree`
for more information.

.. ipython:: python

  dt


Writing a DataTree to Zarr
--------------------------

zarr_ is a chunked storage format designed for use with distributed file systems.
Once a DataTree view of the data has been established, it is trivial to export
this to a zarr_ store.

.. ipython:: python
  :okwarning:

  import os.path
  import tempfile

  dt = xarray.open_datatree(ms, partition_schema=["FIELD_ID"],
    chunks={},
    preferred_chunks={
      (("DATA_DESC_ID", 0),): {"time": 2, "frequency": 4},
      (("DATA_DESC_ID", 1),): {"time": 3, "frequency": 2}})

  zarr_path = f"{tempfile.mkdtemp()}{os.path.sep}test.zarr"
  dt.to_zarr(zarr_path, consolidated=True, compute=True)

It is then trivial to open this using ``open_datatree``:

.. ipython:: python

  dt2 = xarray.open_datatree(zarr_path)
  xarray.testing.assert_identical(dt, dt2)


Writing a DataTree to Cloud Storage
-----------------------------------

xarray incorporates standard functionality for writing xarray datasets to cloud storage.
Here we will use the ``s3fs`` package to write to an S3 bucket.

.. code-block:: python

  import s3fs

  # custom-profile in .aws/credentials
  s3 = s3fs.S3FileSystem(profile="custom-profile",
                         client_kwargs={"region_name": "af-south-1"})
  # A path in a bucket
  store = s3fs.mapping.S3Map("bucket/scratch/test.zarr", s3=s3,
                             check=True, create=False)
  dt.to_zarr(store=store, mode="w", compute=True, consolidated=True)

See the xarray documentation on
`Cloud Storage Buckets <https://docs.xarray.dev/en/stable/user-guide/io.html#cloud-storage-buckets_>`_
for information on interfacing with other cloud providers.
