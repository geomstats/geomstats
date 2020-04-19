.. _first_steps:

===========
First steps
===========

The purpose of this guide is to illustrate the possible uses of geomstats.

**INSTALL GEOMSTATS**

From a terminal (OS X & Linux), you can install geomstats and its requirements with ``pip3`` as follows::

    pip3 install -r requirements.txt
    pip3 install geomstats

**CHOOSE THE BACKEND**

You can choose your backend by setting the environment variable ``GEOMSTATS_BACKEND`` to ``numpy``, ``tensorflow`` or ``pytorch``. By default, the numpy backend is used. You should only use the numpy backend for examples with visualizations.

.. code-block:: bash

    export GEOMSTATS_BACKEND=pytorch

**FIRST EXAMPLES**

To use `geomstats` for learning
algorithms to Riemannian manifolds, you need to follow three steps:
- instantiate the manifold of interest,
- instantiate the learning algorithm of interest,
- run the algorithm.

As an example, the following code snippet illustrates the use of K-means on the hypersphere.

.. code-block:: python

    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.learning.online_kmeans import OnlineKMeans

    sphere = Hypersphere(dimension=5)
    clustering = OnlineKMeans(metric=sphere.metric, n_clusters=4)
    clustering = clustering.fit(data)

The following code snippet shows the use of tangent Principal Component Analysis on the
3D rotations.

.. code-block:: python

    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    from geomstats.learning.pca import TangentPCA

    so3 = SpecialOrthogonal(n=3)
    metric = so3.bi_invariant_metric

    tpca = TangentPCA(metric=metric, n_components=2)
    tpca = tpca.fit(data, base_point=metric.mean(data))
    tangent_projected_data = tpca.transform(data)

All geometric computations are performed behind the scenes.
The user only needs a high-level understanding of Riemannian geometry.
Each algorithm can be used with any of the manifolds and metric
implemented in the package.


**NEXT STEPS**

You can find more examples in the repository
`examples <https://github.com/geomstats/geomstats/tree/master/examples>`_ of geomstats.
You can run them from the command line as follows.

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
How do they look like? To answer this question, you can run
the example that
`plots geodesics on H2 <https://github.com/geomstats/geomstats/blob/master/examples/plot_geodesics_h2.py>`_.

Next, you might be interested in the shapes of "squares" on the negatively curved manifold H2. To visualize squares on H2, you can run the examples that plot squares using the
`Poincare disk <https://github.com/geomstats/geomstats/blob/master/examples/plot_square_h2_poincare_disk.py>`_,
the `Klein disk <https://github.com/geomstats/geomstats/blob/master/examples/plot_square_h2_klein_disk.py>`_ or the `Poincare half-plane <https://github.com/geomstats/geomstats/blob/master/examples/plot_square_h2_poincare_half_plane.py>`_ representations, which are the three main visualizations of H2.

Interested in other geometries? Just adapt the corresponding codes to the manifold of interest. Note that only low-dimensional manifolds, such as 2D and 3D, come with visualizations.

Run learning algorithms
=======================

Assume that you are interested in performing a clustering of data on the hyperbolic plane. `This example <https://github.com/geomstats/geomstats/blob/master/examples/plot_kmeans_manifolds.py>`_ shows how to run K-means on synthetic data on H2.

Interested in clustering data belonging to other manifolds? Check out this example for clustering `on the circle <https://github.com/geomstats/geomstats/blob/master/examples/plot_online_kmeans_s1.py>`_ or `on the sphere <https://github.com/geomstats/geomstats/blob/master/examples/plot_online_kmeans_s2.py>`_.
