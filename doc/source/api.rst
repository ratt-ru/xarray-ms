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
                    partition_schema=["DATA_DESC_ID", "FIELD_ID"])
    >>> datatree = xarray.backends.api.open_datatree(
                    "/data/data.ms",
                    partition_schema=["DATA_DESC_ID", "FIELD_ID"])

These methods defer to the relevant methods on the
`Entrypoint Class <entrypoint-class_>`_.
Consult the method signatures for information on extra
arguments that can be passed.


.. _entrypoint-class:

Entrypoint Class
----------------

Entrypoint class for the MSv2 backend.

.. autoclass:: xarray_ms.backend.msv2.entrypoint.MSv2EntryPoint
    :members: open_dataset, open_datatree
