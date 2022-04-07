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

To learn how to use `geomstats`, visit the page :ref:`first_steps`.
To contribute to `geomstats` visit the page :ref:`contributing`.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   first-steps.rst
   examples.rst
   api-reference.rst
   contributing.rst
   gsod.rst

.. nbgallery::
   :maxdepth: 1
   :caption: Tutorials


   notebooks/00_foundations__introduction_to_geomstats.ipynb
   notebooks/01_foundations__manifolds.ipynb
   notebooks/03_practical_methods__data_on_manifolds.ipynb
   notebooks/04_practical_methods__from_vector_spaces_to_manifolds.ipynb
   notebooks/05_practical_methods__simple_machine_learning_on_tangent_spaces.ipynb
   notebooks/06_practical_methods__riemannian_frechet_mean_and_tangent_pca.ipynb
   notebooks/07_practical_methods__riemannian_kmeans.ipynb
   notebooks/08_practical_methods__information_geometry.ipynb
   notebooks/09_practical_methods__implement_your_own_riemannian_geometry.ipynb
   notebooks/10_practical_methods__shape_analysis.ipynb
   notebooks/11_real_world_applications__cell_shapes_analysis.ipynb
   notebooks/12_real_world_applications__emg_sign_classification_in_spd_manifold.ipynb
   notebooks/13_real_world_applications__graph_embedding_and_clustering_in_hyperbolic_space.ipynb
   notebooks/14_real_world_applications__hand_poses_analysis_in_kendall_shape_space.ipynb
   notebooks/15_real_world_applications__optic_nerve_heads_analysis_in_kendall_shape_space.ipynb
   notebooks/16_real_world_applications__visualizations_in_kendall_shape_spaces.ipynb