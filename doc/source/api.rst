API
===

Opening Measurement Sets
------------------------

The standard :func:`xarray.backends.api.open_dataset` and
:func:`xarray.backends.api.open_datatree` methods should
be used to open either a :class:`~xarray.Dataset` or a
:class:`~xarray.DataTree`.

.. code-block:: python

    >>> dataset = xarray.open_dataset(
                    "/data/data.ms",
                    partition_columns=["DATA_DESC_ID", "FIELD_ID"])
    >>> datatree = xarray.backends.api.open_datatree(
                    "/data/data.ms",
                    partition_columns=["DATA_DESC_ID", "FIELD_ID"])

These methods defer to the relevant methods on the
`Entrypoint Class <entrypoint-class_>`_.
Consult the method signatures for information on extra
arguments that can be passed.


.. _entrypoint-class:

Entrypoint Class
----------------

Entrypoint class for the MSv2 backend.

.. autoclass:: xarray_ms.backend.msv2.entrypoint.MSv2PartitionEntryPoint
    :members: open_dataset, open_datatree


Reading from Zarr
-----------------

Thin wrappers around :func:`xarray.Dataset.open_zarr`
and :func:`xarray.DataTree.open_zarr` that encode
:class:`~xarray.Dataset` attributes as JSON.

.. autofunction:: xarray_ms.xds_from_zarr
.. autofunction:: xarray_ms.xdt_from_zarr

Writing to Zarr
---------------

Thin wrappers around :func:`xarray.Dataset.to_zarr`
and :func:`xarray.DataTree.to_zarr` that encode
:class:`~xarray.Dataset` attributes as JSON.

.. autofunction:: xarray_ms.xds_to_zarr
.. autofunction:: xarray_ms.xdt_to_zarr
