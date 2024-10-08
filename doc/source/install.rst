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

Firstly, install Python `Poetry <poetry_>`_.

.. _poetry: https://python-poetry.org/

Then, the following commands will install the required dependencies,
optional testing dependencies, documentation and development dependencies
in a suitable virtual environment:

.. code-block:: bash

  $ cd /code/arcae
  $ poetry env use 3.11
  $ poetry install -E testing --with doc --with dev
  $ poetry run pre-commit install
  $ poetry shell

The pre-commit hooks can be manually executed as follows:

.. code-block:: bash

  $ poetry run pre-commit run -a


Test Suite
----------

Run the following command within the arcae source code directory to
execute the test suite

.. code-block:: bash

  $ cd /code/arcae
  $ poetry install -E testing --with dev
  $ poetry run py.test -s -vvv tests/


Documentation
-------------

Run the following command within the doc sub-directory to
build the Sphinx documentation

.. code-block:: bash

  $ cd /code/arcae
  $ poetry install --with doc
  $ poetry shell
  $ cd doc
  $ make html

.. _cubed: https://cubed-dev.github.io/cubed/
.. _dask: https://www.dask.org/
.. _zarr: https://zarr.dev/

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
