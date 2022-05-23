Geomstats
=========

Geomstats is an open-source Python package for computations and statistics on
nonlinear manifolds. The mathematical definition of
`manifold <https://en.wikipedia.org/wiki/Manifold>`_ is beyond the scope of this documentation.
However, in order to use Geomstats, you can visualize it as a smooth subset of the
Euclidean space. Simple examples of manifolds include the sphere or the space of 3D rotations.

Data from many application fields are elements of manifolds. For instance,
the manifold of 3D rotations SO(3), or the manifold of 3D rotations and translations SE(3),
appear naturally when performing statistical learning on articulated objects like the human
spine or robotics arms. Other examples of data that belong to manifolds
are introduced in our `paper <https://arxiv.org/abs/2004.04667>`_.

Computations on manifolds require special tools of
`differential geometry <https://en.wikipedia.org/wiki/Differential_geometry>`_. Computing
the mean of two rotation matrices :math:`R_1, R_2` as :math:`\frac{R_1 + R_2}{2}` does not
generally give a rotation matrix. Statistics for data on manifolds need to be extended to
"geometric statistics" to perform consistent operations.

In this context, Geomstats provides code to fulfill four objectives:

- provide educational support to learn "hands-on" differential geometry and geometric statistics, through its examples and visualizations.
- foster research in differential geometry and geometric statistics by providing operations on manifolds to gain intuition on results of a research paper;
- democratize the use of geometric statistics by implementing user-friendly geometric learning algorithms using Scikit-Learn API; and
- provide a platform to share learning algorithms on manifolds.

The `source code <https://github.com/geomstats/geomstats>`_ is freely available on GitHub.

The package is organized into two main modules:
`geometry` and `learning`.

The module `geometry` implements concepts in differential geometry,
such as manifolds and Riemannian metrics, with associated exponential
and logarithmic maps, geodesics, and parallel transport.

The module `learning` implements statistics and learning algorithms for data
on manifolds, such as estimation, clustering and dimension reduction.
The code is object-oriented and classes inherit from
scikit-learn's base classes and mixins.

In both modules, the operations are vectorized for batch computation and provide
support for different execution backends---namely NumPy, PyTorch, and TensorFlow.

To learn how to use `geomstats`, visit :ref:`first_steps`. To contribute to `geomstats` visit :ref:`contributing`. To learn more about differential geometry and manifolds, visit :ref:`explanation`. To find more advanced examples, visit :ref:`tutorials`.



.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/index
   explanation/index
   tutorials/index
   contributing/index
   api/index
   gsod <gsod>

