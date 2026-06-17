Installation
============

.. code-block:: bash

    $ pip install xarray-ms

xarray-ms has a minimal set of dependencies.
If cubed_, dask_ or zarr_ support is required for use with xarray,
they must be installed separately.

.. code-block:: bash

  pip install cubed dask[array] distributed zarr

Development
===========

Install with the dev, doc and testing dependencies using uv_:

.. code-block:: bash

  $ uv sync --group dev --group test --group doc

The pre-commit hooks can be manually executed as follows:

.. code-block:: bash

  $ uv run --dev pre-commit run -a

Test Suite
----------

After installing the dependencies above, run the following command
within the xarray-ms source code directory to execute the test suite:

.. code-block:: bash

  $ uv run --group test py.test tests/


Documentation
-------------

Run the following command within the doc sub-directory to
build the Sphinx documentation

.. code-block:: bash

  $ cd doc
  $ make html

Release Process
---------------

For a new version number, say ``0.2.0``, perform the following operations
on the ``main`` branch:

1. Edit ``doc/source/changelog.rst`` to reflect the new version.
2. Run

   .. code-block:: bash

      $ tbump --dry-run 0.2.0

3. If 2. succeeds, run

   .. code-block:: bash

      $ tbump 0.2.0

.. _cubed: https://cubed-dev.github.io/cubed/
.. _dask: https://www.dask.org/
.. _zarr: https://zarr.dev/
