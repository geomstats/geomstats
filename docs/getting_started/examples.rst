.. _examples:

=============
Code Examples
=============

To learn how to use geomstats, you can look at the many examples in the repository of
`code examples <https://github.com/geomstats/geomstats/tree/main/examples>`_.

If you have installed geomstats via Git, you can run them from the command line as follows.

.. code-block:: bash

    python3 examples/plot_grid_h2.py

These examples allow getting intuition on manifolds and concepts from differential geometry, as well as running learning algorithms.

Learn differential geometry
===========================

Assume that your data naturally belongs to the
`hyperbolic plane H2 <https://en.wikipedia.org/wiki/Hyperbolic_geometry#Models_of_the_hyperbolic_plane>`_
and you want to get intuition on the geometry of this space.
The space H2 has a negative curvature. The geodesics - i.e.
the curves of shortest length - on H2 are not straight lines.
How do they look? To answer this question, you can run
the example that
`plots geodesics on H2 <https://github.com/geomstats/geomstats/blob/main/examples/plot_geodesics_h2.py>`_.

Next, you might be interested in the shapes of "squares" on the negatively curved manifold H2. To visualize squares on H2, you can run the examples that plot squares using the
`Poincare disk <https://github.com/geomstats/geomstats/blob/main/examples/plot_square_h2_poincare_disk.py>`_,
the `Klein disk <https://github.com/geomstats/geomstats/blob/main/examples/plot_square_h2_klein_disk.py>`_ or the `Poincare half-plane <https://github.com/geomstats/geomstats/blob/main/examples/plot_square_h2_poincare_half_plane.py>`_ representations, which are the three main visualizations of H2.

Interested in other geometries? Just adapt the corresponding codes to the manifold of interest. Note that only low-dimensional manifolds, such as 2D and 3D, come with visualizations.

Run learning algorithms
=======================

Assume that you are interested in performing a clustering of data on the hyperbolic plane. `This example <https://github.com/geomstats/geomstats/blob/main/examples/plot_kmeans_manifolds.py>`_ shows how to run K-means on synthetic data on H2.


Interested in clustering data belonging to other manifolds? Check out this example for clustering `on the circle <https://github.com/geomstats/geomstats/blob/main/examples/plot_online_kmeans_s1.py>`_ or `on the sphere <https://github.com/geomstats/geomstats/blob/main/examples/plot_online_kmeans_s2.py>`_.
