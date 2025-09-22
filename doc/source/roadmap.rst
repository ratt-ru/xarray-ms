.. _compliance-and-roadmap:

Measurement Set v4 Compliance
=============================

xarray-ms fully implements the loading of correlated data from
MSv2 datasets into the Measurement Set v4.0 specification.
This covers the MSv2 ``MAIN`` table, as well as the
``DATA_DESCRIPTION``, ``SPECTRAL_WINDOW``, ``POLARIZATION``,
``FEED``, ``OBSERVATION``, ``STATE`` and ``PROCESSOR`` subtables
whose synthesis is presented in correlated data datasets
within an xarray DataTree.

Care has been taken to convert measures information from MSv2 into
MSv4 metadata attributes, where appropriate.

In particular, it loads the MSv2 dataset present in the
`Measurement Set v4 test suite <msv4-test-suite_>`_ except for:

- ALMA Measurement Sets which sometimes do not correctly link
  the ANTENNNA and MAIN table via the FEED table.
  This will need to be addressed heuristically.
- Single-dish Measurement Sets.
  This is not difficult as it involves loading in
  ``MAIN::FLOAT_DATA`` into the ``SPECTRUM`` variable and
  renaming ``FIELD_PHASE_CENTER_DIRECTION`` to
  ``FIELD_REFERENCE_CENTER_DIRECTION`` in the
  ``field_and_source_xds`` dataset.

MSv4 specifies a set of optional datasets, of which the following are implemented:

- antenna_xds
- field_and_source_xds (required components)

The following optional datasets are not yet implemented:

- field_and_source_ephemeris_xds
- pointing_xds
- system_calibration_xds
- gain_curve_xds
- phase_calibration_xds
- weather_xds
- phased_array_xds

Roadmap
-------

The existing coverage of the specification arguably represents a Pareto distribution of the required data for writing Radio Astronomy software in an MSv4 paradigm, but we aim to address the remaining cases as follows in order of priority:

- phased_array_xds
- pointing_xds

as this will more fully support LOFAR and SKA-LOW. The following datasets are probably required for VLBI:

- system_calibration_xds
- weather_xds

while the following are arguably required for calibration and other software that will need to be developed
for the MSv4 paradigm:

- gain_curve_xds
- phase_calibration_xds
- field_and_source_ephemeris_xds
- single dish systems

This is a rough strategy and doesn't need to be set in stone.
Please reach out or contribute PR's if you have specific requirements.

.. _msv4-test-suite: https://github.com/ratt-ru/xarray-ms/blob/main/tests/msv4_test_corpus/test_msv_corpus.py
