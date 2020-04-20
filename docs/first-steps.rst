.. _first_steps:

===========
First steps
===========

The purpose of this guide is to illustrate the possible uses of geomstats.

**INSTALL GEOMSTATS WITH PIP3**

From a terminal (OS X & Linux), you can install geomstats and its requirements with ``pip3`` as follows::

    pip3 install -r requirements.txt
    pip3 install geomstats

This installs the latest version uploaded on PyPi.

**INSTALL GEOMSTATS FROM GITHUB**

From a terminal (OS X & Linux), you can install geomstats and its requirements via Git as follows::

    pip3 install -r requirements
    git clone https://github.com/geomstats/geomstats.git

This installs the latest GitHub version, useful for developers.

**CHOOSE THE BACKEND**

You can choose your backend by setting the environment variable ``GEOMSTATS_BACKEND`` to ``numpy``, ``tensorflow`` or ``pytorch``. By default, the numpy backend is used. You should only use the numpy backend for examples with visualizations.

.. code-block:: bash

    export GEOMSTATS_BACKEND=pytorch

**FIRST EXAMPLES**

To use `geomstats` for learning
algorithms on Riemannian manifolds, you need to follow three steps:
- instantiate the manifold of interest,
- instantiate the learning algorithm of interest,
- run the algorithm.
The data should be represented by the structure ``gs.array``, which represents numpy arrays, tensorflow or pytorch tensors, depending on the choice of backend.

As an example, the following code snippet illustrates the use of K-means
on the 5-dimensional hypersphere, assuming ``data`` belongs to this space.

.. code-block:: python

    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.learning.online_kmeans import OnlineKMeans

    sphere = Hypersphere(dimension=5)
    clustering = OnlineKMeans(metric=sphere.metric, n_clusters=4)
    clustering = clustering.fit(data)

The following code snippet shows the use of tangent Principal Component Analysis on the
space of 3D rotations, assuming ``data`` belongs to this space.

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

To see additional examples, visit the page :ref:`examples`.
