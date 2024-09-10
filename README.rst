=========
xarray-ms
=========

.. image:: https://img.shields.io/pypi/v/xarray-ms.svg
        :target: https://pypi.python.org/pypi/xarray-ms

.. image:: https://github.com/ratt-ru/xarray-ms/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/ratt-ru/xarray-ms/actions/workflows/ci.yml

.. image:: https://readthedocs.org/projects/xarray-ms/badge/?version=latest
        :target: https://xarray-ms.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

====

xarray-ms presents a Measurement Set v4 view (MSv4) over
`CASA Measurement Sets <https://casa.nrao.edu/Memos/229.html>`_ (MSv2).
It provides access to MSv2 data via the xarray API, allowing MSv4 compliant applications
to be developed on well-understood MSv2 data.

.. code-block:: python

  >>> import xarray_ms
  >>> import xarray
  >>> ds = xarray.open_dataset("/data/L795830_SB001_uv.MS/",
                               chunks={"time": 2000, "baseline": 1000})
  >>> ds
    <xarray.Dataset> Size: 70GB
    Dimensions:                     (time: 28760, baseline: 2775, frequency: 16,
                                    polarization: 4, uvw_label: 3)
    Coordinates:
        antenna1_name               (baseline) object 22kB dask.array<chunksize=(1000,), meta=np.ndarray>
        antenna2_name               (baseline) object 22kB dask.array<chunksize=(1000,), meta=np.ndarray>
        baseline_id                 (baseline) int64 22kB dask.array<chunksize=(1000,), meta=np.ndarray>
      * frequency                   (frequency) float64 128B 1.202e+08 ... 1.204e+08
      * polarization                (polarization) <U2 32B 'XX' 'XY' 'YX' 'YY'
      * time                        (time) float64 230kB 1.601e+09 ... 1.601e+09
    Dimensions without coordinates: baseline, uvw_label
    Data variables:
        EFFECTIVE_INTEGRATION_TIME  (time, baseline) float64 638MB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        FLAG                        (time, baseline, frequency, polarization) uint8 5GB dask.array<chunksize=(2000, 1000, 16, 4), meta=np.ndarray>
        TIME_CENTROID               (time, baseline) float64 638MB dask.array<chunksize=(2000, 1000), meta=np.ndarray>
        UVW                         (time, baseline, uvw_label) float64 2GB dask.array<chunksize=(2000, 1000, 3), meta=np.ndarray>
        VISIBILITY                  (time, baseline, frequency, polarization) complex64 41GB dask.array<chunksize=(2000, 1000, 16, 4), meta=np.ndarray>
        WEIGHT                      (time, baseline, frequency, polarization) float32 20GB dask.array<chunksize=(2000, 1000, 16, 4), meta=np.ndarray>
    Attributes:
        antenna_xds:          <xarray.Dataset> Size: 4kB\nDimensions:           (...
        version:              0.0.1
        creation_date:        2024-09-10T14:29:22.587984+00:00
        data_description_id:  0

Measurement Set v4
------------------

NRAO_/SKAO_ are developing a new xarray-based `Measurement Set v4 specification <msv4-spec_>`_.
While there are many changes some of the major highlights are:

* xarray_ is used to define the specification.
* MSv4 data consists of Datasets of ndarrays on a regular time-channel grid.
  MSv2 data is tabular and, while in many instances the time-channel grid is regular,
  this was not guaranteed, especially after MSv2 datasets had been transformed by various tasks.


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

Work in Progress
----------------

The Measurement Set v4 specification is currently under active development.
xarray-ms is currently under active development and does not yet
have feature parity with xradio_.

Most measures information and many secondary sub-tables are currently missing.
However, the most important parts of the ``MAIN`` tables,
as well as the ``ANTENNA``, ``POLARIZATON`` and ``SPECTRAL_WINDOW``
sub-tables are implemented and should be sufficient
for basic algorithm development.

.. _SKAO: https://www.skao.int/
.. _NRAO: https://public.nrao.edu/
.. _msv4-spec: https://docs.google.com/spreadsheets/d/14a6qMap9M5r_vjpLnaBKxsR9TF4azN5LVdOxLacOX-s/
.. _xradio: https://github.com/casangi/xradio
.. _dask-ms: https://github.com/ratt-ru/dask-ms
.. _arcae: https://github.com/ratt-ru/arcae
.. _dask: https://www.dask.org/
.. _python-casacore: https://github.com/casacore/python-casacore/
.. _xarray: https://xarray.dev/
.. _xarray_backend: https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html
.. _xarray_lazy: https://docs.xarray.dev/en/latest/internals/internal-design.html#lazy-indexing-classes
.. _xarray_chunked_arrays: https://docs.xarray.dev/en/latest/internals/chunked-arrays.html
.. _zarr: https://zarr.dev/
