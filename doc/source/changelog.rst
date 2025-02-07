:tocdepth: 1

Changelog
=========

X.Y.Z (DD-MM-YYYY)
------------------
* Fix test cases that succeeded after attributes changed (:pr:`57`)
* Make MSv2Array transform a property (:pr:`56`)
* Further partitioning improvement and alignment with MSv4 (:pr:`55`)
* Use epoch to dintinguish multiple instances of the same dataset (:pr:`54`)
* Use np.logical_or.reduce for generating diffs over more than 2 partitioning arrays (:pr:`53`)
* Improve Missing Column error (:pr:`52`)
* Fix `open_datatree` instructions in the README (:pr:`51`)
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
