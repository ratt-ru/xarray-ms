:tocdepth: 1

Changelog
=========

X.Y.Z (DD-MM-YYYY)
------------------
* Add test case for irregular grids (:pr:`39`)
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
