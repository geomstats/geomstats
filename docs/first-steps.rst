.. _first_steps:

===========
First steps
===========

The purpose of this guide is to illustrate the possible uses of geomstats.

**INSTALL GEOMSTATS WITH PIP3**

From a terminal (OS X & Linux), you can install geomstats and its requirements with ``pip3`` as follows::

    pip3 install geomstats

This method installs the latest version of geomstats that is uploaded on PyPi.

**INSTALL GEOMSTATS FROM GITHUB**

From a terminal (OS X & Linux), you can install geomstats and its requirements via Git as follows::

    git clone https://github.com/geomstats/geomstats.git
    pip3 install -r requirements.txt

This methods installs the `latest GitHub version <https://github.com/geomstats/geomstats>`_. Developers should install this version, together with the development requirements and the optional requirements to enable ``tensorflow`` and ``pytorch`` backends::

    pip3 install -r dev-requirements.txt -r opt-requirements.txt

**CHOOSE THE BACKEND**

Geomstats can run seemlessly with ``numpy``, ``tensorflow`` or ``pytorch``. Note that ``pytorch`` and ``tensorflow`` requirements are optional, as geomstats can be used with ``numpy`` only. By default, the ``numpy`` backend is used. The visualizations are only available with this backend.

To get the ``tensorflow`` and ``pytorch`` versions compatible with geomstats, install the `optional requirements <https://github.com/geomstats/geomstats/blob/master/opt-requirements.txt>`_::

    pip3 install -r opt-requirements.txt

You can choose your backend by setting the environment variable ``GEOMSTATS_BACKEND`` to ``numpy``, ``tensorflow`` or ``pytorch``, and importing the ``backend`` module. From the command line:

.. code-block:: bash

    export GEOMSTATS_BACKEND=pytorch

and in the Python3 code:

.. code-block:: python

    import geomstats.backend as gs


**FIRST EXAMPLES**

To use `geomstats` for learning
algorithms on Riemannian manifolds, you need to follow three steps:
- instantiate the manifold of interest,
- instantiate the learning algorithm of interest,
- run the algorithm.
The data should be represented by the structure ``gs.array``, which represents numpy arrays, tensorflow or pytorch tensors, depending on the choice of backend.

As an example, the following code snippet illustrates the use of K-means
on simulated data on the 5-dimensional hypersphere.

.. code-block:: python

    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.learning.online_kmeans import OnlineKMeans

    sphere = Hypersphere(dim=5)

    data = sphere.random_uniform(n_samples=10)

    clustering = OnlineKMeans(metric=sphere.metric, n_clusters=4)
    clustering = clustering.fit(data)

The following code snippet shows the use of tangent Principal Component Analysis on simulated data on the
space of 3D rotations.

.. code-block:: python

    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    from geomstats.learning.pca import TangentPCA

    so3 = SpecialOrthogonal(n=3, point_type='vector')
    metric = so3.bi_invariant_metric

    data = so3.random_uniform(n_samples=10)

    tpca = TangentPCA(metric=metric, n_components=2)
    tpca = tpca.fit(data)
    tangent_projected_data = tpca.transform(data)

All geometric computations are performed behind the scenes.
The user only needs a high-level understanding of Riemannian geometry.
Each algorithm can be used with any of the manifolds and metric
implemented in the package.

To see additional examples, visit the page :ref:`examples`.
