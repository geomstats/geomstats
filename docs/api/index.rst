=============
API Reference
=============

The API reference gives an overview of Geomstats implementation.

The module `geometry` implements concepts in differential geometry, to
perform computations on manifolds and Riemannian metrics, with associated exponential
and logarithmic maps, geodesics, and parallel transport.

The module `learning` implements statistics and learning algorithms for data
on manifolds, such as estimation, clustering and dimension reduction.
The code is object-oriented and classes inherit from
scikit-learn's base classes and mixins.

In both modules, the operations are vectorized for batch computation and provide
support for different execution backends---namely NumPy, Autograd and PyTorch.
The module `backend` implements the operations needed to use Geomstats seamlessly with any backend.

.. toctree::
   :maxdepth: 3
   :caption: Packages & Modules

   modules
