.. _first_steps:

===========
First steps
===========

The purpose of this guide is to illustrate the possible uses of geomstats.

**Install geomstats.**

From a terminal (OS X & Linux), you can install geomstats and its requirements with ``pip3`` as follows::

    pip3 install -r requirements.txt
    pip3 install geomstats

**Choose the backend.**

You can choose your backend, by setting the environment variable GEOMSTATS_BACKEND to numpy, tensorflow or pytorch. By default, numpy is used. You should only use the numpy backend for examples with visualizations.

.. code-block:: bash

    export GEOMSTATS_BACKEND=numpy

**First examples.**

We illustrate here the use of `geomstats` to generalize learning
algorithms to Riemannian manifolds.

The following code snippet illustrates the use of K-means on the hypersphere.

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


**Next steps.**

You can find more examples in the repository "examples" of geomstats. You can run them from the command line as follows.

.. code-block:: bash

    python3 examples/plot_grid_h2.py
