:tocdepth: 1

Changelog
=========

0.3.7 (03-10-2025)
------------------
* Documentation updates (:pr:`134`)
* Temporarily restrict xarray to \< 2025.9.1 (:pr:`136`) until
  `xarray#10808 <https://github.com/pydata/xarray/issues/10808_>`_
  is resolved.
* Restrict arcae to \< 0.4.0 to prevent
  API-breaking write support changes (:pr:`136`)
* Provide a physically realistic SPECTRAL_WINDOW::REF_FREQUENCY in simulated data (:pr:`133`)


0.3.6 (22-09-2025)
------------------
* Document Partitioning Strategies and Irregular Grid Handling (:pr:`132`)
* Document MSv4 compliance and roadmap (:pr:`131`)

0.3.5 (17-09-2025)
------------------
* Remove deploy to test-pypi (:pr:`130`)

0.3.4 (17-09-2025)
------------------
* Add user-agent headers to urlib.request.urlopen (:pr:`129`)

0.3.3 (17-09-2025)
------------------
* Support non-standard columns (:pr:`126`)
* Test on macos-15 (:pr:`127`)
* Correct MSv4 schema issues (:pr:`125`)
* Support TAI Measure encoding in time data (:pr:`124`)
* Distinguish between time, baseline and channel irregularity (:pr:`123`)
* Test xarray-ms against the full MSv4 Test Corpus (:pr:`120`)
* Handle negative SOURCE_IDs in the FIELD subtable (:pr:`122`)
* Cleanup Dataset Factories (:pr:`118`)
* Add field_and_source_xds (:pr:`114`, :pr:`119`, :pr:`122`)
* Fix integer selection (:pr:`117`)
* Correct clamp of the slice stop to be min(shape[d], s.stop) (:pr:`115`)
* Rename deprecated ``repository_url`` to ``repository-url`` (:pr:`112`)
* Use trusted publishing when publishing to pypi (:pr:`111`)
* Update pre-commit hooks (:pr:`110`)
* Implement PEP 621 in pyproject.toml (:pr:`109`)

0.3.2 (21-06-2025)
------------------
* Upgrade to arcae 0.3.0 (:pr:`108`)
* Call MSv2Array.__getitem__ rather than MSv2Array._getitem which is not guaranteed to be present (:pr:`107`)

0.3.1 (11-06-2025)
------------------
* Fix low-resolution broadcasting (:pr:`106`)

0.3.0 (10-06-2025)
------------------
* Upgrade to arcae 0.2.9 to elide selection checks on ignored rows (:pr:`105`)

0.2.9 (02-06-2025)
------------------
* Handle negative foreign keys during imputation of subtables (:pr:`102`)
* Fix documentation typo (:pr:`99`)
* Update Work in Progress documentation (:pr:`98`)
* Remove stray test case print (:pr:`97`)

0.2.8 (01-04-2025)
------------------
* Update copyright to reflect NRF and RATT in
  both the BSD3 license and documentation (:pr:`96`)

0.2.7 (01-04-2025)
------------------
* Fix changelog formatting (:pr:`95`)
* Add ``PROCESSOR_ID`` to the default partitioning columns (:pr:`94`)
* Support ``processor_info`` on the correlated dataset (:pr:`94`)

0.2.6 (31-03-2025)
------------------
* Allow some jitter in the ``INTERVAL`` column when setting ``time.integration_time`` (:pr:`93`)
* Impute missing ``FIELD``, ``STATE`` and ``OBSERVATION`` subtable data (:pr:`92`)
* Increase MSv2Structure cache timeout from 1 to 5 minutes (:pr:`91`)
* Check for ``TIME`` and ``INTEGRATION_TIME`` in the case of multiple ``INTERVAL`` values (:pr:`90`)

0.2.5 (24-03-2025)
------------------
* Support ``field_name``, ``scan_number`` and ``sub_scan_number`` coordinates
  on the Correlated Dataset  (:pr:`88`)
* Support fallback to ``WEIGHT`` if ``WEIGHT_SPECTRUM`` is not present (:pr:`87`)

0.2.4 (19-03-2025)
------------------
* Fix no-autocorrelation case when constructing partition row maps (:pr:`85`)
* Default auto correlations to `False` (:pr:`85`)
* Refactor dataset factories into `factories` subpackage (:pr:`83`, :pr:`86`)
* Use a ``CommonStoreArgs`` class to default initialise common store arguments (:pr:`83`)
* Release resources when datasets or datatrees are closed (:pr:`81`)
* Use creator attribute to record xarray-ms version (:pr:`80`)
* Generalise the TableFactory class into a Multiton class (:pr:`79`)
* Refactor partitioning logic to be more robust (:pr:`78`)
* The set of antennas related to a partition in the ``FEED`` table is
  used to create the antenna dataset for that partition (:pr:`78`)
* Metadata extraction moved to dataset factories (:pr:`78`)
* Extend the antenna dataset implementation (:pr:`77`)
* Fix MSv2Store._partition_key typing (:pr:`76`)
* Add observation_info attribute (:pr:`74`)
* Add ``ANTENNA_DISH_DIAMETER`` variable to antenna dataset (:pr:`73`)
* Add cartesian_pos_label labels to antenna dataset (:pr:`72`)
* Allow fallback to string values in partition keys (:pr:`71`)
* Report irregular channel widths with an IrregularGridWarning (:pr:`70`)
* Tighten ``SOURCE_ID`` partitioning checks (:pr:`69`)
* Check that each partition has a unique feed index pair (:pr:`68`)
* Remove unused and commented out test cases (:pr:`67`)


0.2.3 (28-02-2025)
------------------
* Remove superfluous hollow DataTree node containing the Measurement Set name.
  Visibility partition structure changes to ``msname_partition_000``. (:pr:`66`)

0.2.2 (27-02-2025)
------------------
* Add u, v and w labels to the uvw_label coordinate (:pr:`65`)
* Remove ellipsoid_pos_label from ANTENNA_POSITION component coordinate (:pr:`64`)
* Move README content into the Documentation (:pr:`62`)
* Allow varying intervals in the last timestep of a partition (:pr:`61`)
* Rename ANTENNA dataset to antenna_xds (:pr:`60`)
* Depend on arcae ^0.2.7 (:pr:`59`)
* Fix test cases that succeeded after attributes changed (:pr:`57`)
* Make MSv2Array transform a property (:pr:`56`)
* Further partitioning improvement and alignment with MSv4 (:pr:`55`)
* Use epoch to distinguish multiple instances of the same dataset (:pr:`54`)
* Use np.logical_or.reduce for generating diffs over more than 2 partitioning arrays (:pr:`53`)
* Improve Missing Column error (:pr:`52`)
* Fix ``open_datatree`` instructions in the README (:pr:`51`)
* Skip test case that segfaults on numpy 2.2.2 (:pr:`50`)
* Upgrade to xarray 2025.1.1 (:pr:`49`)
* Add documentation link to MSv2EntryPoint class (:pr:`47`)
* Change visibility partition structure to ``msname/partition-001`` (:pr:`46`)
* Rename ``baseline`` dimension to ``baseline_id`` (:pr:`44`)
* Loosen xarray version requirement to \>= 2024.9.0 (:pr:`44`)
* Change ``partition_chunks`` to ``preferred_chunks`` (:pr:`44`)
* Allow arcae to vary in the 0.2.x range (:pr:`42`)
* Pin xarray to 2024.9.0 (:pr:`42`)
* Add test case for irregular grids (:pr:`39`, :pr:`40`, :pr:`41`)
* Rename MSv2PartitionEntryPoint to MSv2EntryPoint (:pr:`38`)
* Move ``chunks`` kwarg functionality in MSv2PartitionEntryPoint.open_datatree
  to ``partition_chunks`` (:pr:`37`)
* Set MSv4 version to 4.0.0 (:pr:`34`)
* Fix changelog highlighting in install instructions (:pr:`33`)
* Add basic read tests (:pr:`32`)
* Fix Dataset and DataTree equivalence checks in test cases (:pr:`31`)

0.2.1 (04-10-2024)
------------------
* Parallelise row partitioning (:pr:`28`, :pr:`30`)
* Upgrade to arcae 0.2.5 (:pr:`29`)
* Rename antenna{1,2}_name to baseline_antenna{1,2}_name (:pr:`26`)
* Update Cloud Storage write documentation (:pr:`25`, :pr:`27`)
* Use datatree as the primary representation (:pr:`24`)
* Remove unnecessary coordinate attributes (:pr:`23`)
* Disable navigation sidebars (:pr:`19`)
* Add Github Issue and PR templates (:pr:`17`)
* Improve key resolution (:pr:`15`)
* Add a basic tutorial (:pr:`13`)

0.2.0 (11-09-2024)
------------------

* Initial release
