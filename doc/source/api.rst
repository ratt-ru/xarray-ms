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

Creating a fresh Measurement Set
--------------------------------

Call :func:`xarray_ms.plan_fresh_msv2` to validate an MSv4 DataTree and
resolve its visibility destinations and metadata without creating tables.
Pass the resulting plan to :func:`xarray_ms.create_fresh_msv2` to create
and verify a new MSv2 skeleton with zero MAIN rows. Creation does not
replace an existing target, including one created concurrently; invalid
input raises :class:`~xarray_ms.errors.FreshMSv2ValidationError`, an
existing target raises :class:`~xarray_ms.errors.FreshMSv2TargetError`,
and verification failure does not publish the staging table.

.. autofunction:: xarray_ms.plan_fresh_msv2

.. autofunction:: xarray_ms.create_fresh_msv2

Writing a complete fresh Measurement Set
-------------------------------------------

Use :func:`xarray_ms.write_fresh_msv2` to materialize one selected data group
into ``DATA``, together with flags, weights, UVW and the required structural
columns. The writer reuses the fresh schema and metadata planner. It constructs
and checks the schema through fresh table handles, writes and verifies the
private output, closes all handles, then publishes without replacing an
existing destination. A failed attempt removes its private staging directory.

.. code-block:: python

    import xarray as xr
    from xarray_ms import write_fresh_msv2

    with xr.open_datatree("complete.zarr", engine="zarr") as tree:
        write_fresh_msv2(
            tree, "new.ms", row_grid="complete", data_group="base",
            rows_per_batch=128,
        )

``row_grid="complete"`` is a required assertion that every time/baseline cell
represents a real row. Each partition must contain the full cross-correlation
baseline set for its antennas, optionally including all auto-correlations.
Reversed antenna pairs, duplicate rows, incomplete baseline sets and known
padded grids are refused. The reader's ``xarray_ms_row_grid`` attribute records
known missing rows and survives Zarr serialization. For older or independently
constructed trees without that provenance, the caller must establish occupancy
before making the assertion. Flags and visibility values do not establish row
presence; a fully flagged row can still be a real row.

Rows are ordered by partition path, increasing time and antenna pair.
The supported profile uses fixed measure references and a finite positive
scalar ``time.attrs["integration_time"]`` quantity in seconds. Time centroids
and effective integration times may be supplied with their supported metadata;
otherwise they default to time and integration interval respectively. Irregular
time representations are outside this initial materializer's scope.

``WEIGHT_SPECTRUM`` uses float32, including conversion of float64 inputs.
Weights must remain finite and nonnegative after conversion. ``WEIGHT`` is
the channel mean of spectral weights; ``SIGMA`` is their reciprocal square root,
or zero for zero row weight. ``FLAG_ROW`` is true when all
channel/polarization flags are true.
These are semantic reconstruction policies, not native-column preservation.

``rows_per_batch`` bounds writer row batches, each containing complete
channel/polarization cells. Native mutations use one table instance and occur
serially. An input Dask task can allocate more memory than the writer's batch
size; choose source chunks accordingly. Existing ``DataTree.to_msv2`` update
behavior is unchanged.

Additional visibility destinations remain supported by the planning and
zero-row skeleton APIs, but are deferred in this materializer. Sparse-grid
reconstruction, unrestricted optional metadata and variable measure references
are also deferred. This semantic writer does not replace the exact-logical
native preservation sidecar used by the MSv2 workflow project.

.. autofunction:: xarray_ms.write_fresh_msv2

Generated interoperability demonstrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xarray_ms.testing.fresh.make_complete_tree()`` constructs the named
``complete-crosscorr-v1`` profile: three antennas, four timesteps, twelve
cross-correlation rows, eight channels near 1.4 GHz and XX/YY polarizations.
It contains synthetic visibilities, flags, weights and nonzero UVW. This is
structural/interoperability evidence, not a physically validated observation
or qualification of the full Paper I demonstrator.

.. code-block:: python

    import xarray as xr
    from xarray_ms import write_fresh_msv2
    from xarray_ms.testing.fresh import make_complete_tree

    source = make_complete_tree()
    source.to_zarr("complete.zarr", consolidated=True)
    source.close()
    del source
    with xr.open_datatree("complete.zarr", engine="zarr") as tree:
        write_fresh_msv2(tree, "complete.ms", row_grid="complete",
                        rows_per_batch=2)

The test group pairs Zarr 2 with xarray before 2026.4. The generated profile
was checked with Python 3.13, xarray 2026.2.0, Zarr 2.18.7 and arcae 0.4.0a14.
Tests compare the reopened arrays and a flag-aware weighted visibility average,
and separately remove an original MS before reconstructing from its Zarr export.
Native writes always use ``ninstances=1``; the newer arcae 0.5 branch is not
qualified by this run.

A WSClean 3.6 smoke run consumed the generated MS and produced finite
64-by-64 Stokes-I dirty and restored images using:

.. code-block:: console

    wsclean -j 1 -size 64 64 -scale 60asec -niter 0 -pol I \
        -no-update-model-required -name smoke complete.ms

The tested runtime was the local container
``ghcr.io/shinobi-dosho/wsclean:3.6-idg-d0.1.0``, image ID
``sha256:b88aefe62ea18e9ea4f06a83a11c157156734e14ce61a67668ee04c065e8f5f8``.
This smoke test establishes downstream readability for this generated profile;
it does not establish imaging accuracy, calibration quality, real-observation
coverage or exact native restoration.


.. _entrypoint-class:

Entrypoint Class
----------------

Entrypoint class for the MSv2 backend.

.. autoclass:: xarray_ms.backend.msv2.entrypoint.MSv2EntryPoint
    :members: open_datatree, open_dataset

.. _partitioning-schema:
