xarray-ms
=========

xarray-ms presents a Measurement Set v4 view (MSv4) over
`CASA Measurement Sets <https://casa.nrao.edu/Memos/229.html>`_ (MSv2).
It provides access to MSv2 data via the xarray API, allowing MSv4 compliant applications
to be developed on well-understood MSv2 data.

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

  ms

  dt = xarray.open_datatree(ms)

  dt

Measurement Set v4
------------------

NRAO_/SKAO_ are developing a new xarray-based `Measurement Set v4 specification <msv4-spec_>`_.
While there are many changes some of the major highlights are:

* xarray_ is used to define the specification.
* MSv4 data consists of Datasets of ndarrays on a regular time-channel grid.
  MSv2 data is tabular and, while in many instances the time-channel grid is regular,
  this is not guaranteed, especially after MSv2 datasets have been transformed by various tasks.


xarray_ Datasets are self-describing and they are therefore easier to reason about and work with.
Additionally, the regularity of data will make writing MSv4-based software less complex.

xradio
------

`casangi/xradio <xradio_>`_ provides a reference implementation that converts
CASA v2 Measurement Sets to Zarr v4 Measurement Sets using the python-casacore_
package.

Why xarray-ms?
--------------

* By developing against an MSv4 xarray view over MSv2 data,
  developers can develop applications on well-understood data,
  and then seamlessly transition to newer formats.
  Data can also be exported to newer formats (principally zarr_) via xarray's
  native I/O routines.
  However, the xarray view of either format looks the same to the software developer.

* xarray-ms builds on xarray's
  `backend API <https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html>`_:
  Implementing a formal CASA MSv2 backend has a number of benefits:

  * xarray's internal I/O routines such as ``open_dataset`` and ``open_datatree``
    can dispatch to the backend to load data.
  * Similarly xarray's `lazy loading mechanism <xarray_lazy_>`_ dispatches
    through the backend.
  * Automatic access to any `chunked array types <xarray_chunked_arrays_>`_
    supported by xarray including, but not limited to dask_.
  * Arbitrary chunking along any xarray dimension.

* xarray-ms uses arcae_, a high-performance backend to CASA Tables implementing
  a subset of python-casacore_'s interface.
* Some limited support for irregular MSv2 data via padding.
* Refer to the :ref:`MSv4 compliance and roadmap <compliance-and-roadmap>`
  section for information on adherence to the specification.
