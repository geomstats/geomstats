Geomstats
=========

*Geomstats provides code for computations and statistics on manifolds with geometric structures.*

The package is organized into two main modules:
`geometry` and `learning`.

The module `geometry` implements concepts in Riemannian geometry,
such as manifolds and Riemannian metrics, with an object-oriented approach.

The module `learning` implements statistics and learning algorithms for data
on manifolds. The code is object-oriented and classes inherit from
scikit-learn's base classes and mixins.

To learn how to use `geomstats`, visit the page :ref:`first_steps`.
To contribute to `geomstats` visit the page :ref:`contributing`.

**QUICK INSTALL**

.. code-block:: bash

    pip3 install -r requirements.txt
    pip3 install geomstats

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   first-steps.rst
   api-reference.rst
   contributing.rst
