API
===

Opening Measurement Sets
------------------------

The standard :func:`xarray.open_datatree` method should
be used to open a :class:`~xarray.DataTree` interface
to the underlying Measurement Set data.

.. code-block:: python

    >>> datatree = xarray.open_datatree("/data/data.ms", partition_schema=["FIELD_ID"])

These methods defer to the relevant methods on the
`Entrypoint Class <entrypoint-class_>`_.
Consult the method signatures for information on extra
arguments that can be passed.


.. _entrypoint-class:

Entrypoint Class
----------------

Entrypoint class for the MSv2 backend.

.. autoclass:: xarray_ms.backend.msv2.entrypoint.MSv2EntryPoint
    :members: open_datatree, open_dataset

.. _partitioning-schema:
