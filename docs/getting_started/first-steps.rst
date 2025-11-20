.. _first_steps:

===========
First steps
===========

The purpose of this guide is to guide through the installation of geomstats and illustrate the possible uses of geomstats.
Install is possible via ``pip3``, ``conda`` or by cloning the Github repository.


**INSTALL GEOMSTATS WITH PIP3**

From a terminal (OS X & Linux), you can install geomstats and its requirements with ``pip3`` as follows::

    pip3 install geomstats

This method installs the latest version of geomstats that is uploaded on PyPi.


**INSTALL GEOMSTATS WITH CONDA**

From a terminal (OS X & Linux) or an Anaconda prompt (Windows), you can install geomstats and its
requirements with ``conda`` as follows:

::

    conda install -c conda-forge geomstats

This method installs the latest version of geomstats that is uploaded on
conda-forge. Note that geomstats is only available with Python3.


**INSTALL GEOMSTATS FROM GITHUB**

From a terminal (OS X & Linux), you can install geomstats and its requirements via Git as follows::

    git clone https://github.com/geomstats/geomstats.git
    cd geomstats
    pip3 install .

This methods installs the `latest GitHub version <https://github.com/geomstats/geomstats>`_. Developers should install this version, together with the development requirements and the optional requirements to enable the ``autograd`` and ``pytorch`` backends::

    pip3 install .[dev,opt]

If you use the flag ``-e``, geomstats will be installed in editable mode, i.e. local changes are immediately reflected in your installation.


**CHOOSE THE BACKEND**

Geomstats can run seamlessly with ``numpy`` or ``pytorch``. Note that ``pytorch`` requirement is optional, as geomstats can be used with ``numpy`` only. By default, the ``numpy`` backend is used. The visualizations are only available with this backend.

To get the ``autograd`` and ``pytorch`` versions compatible with geomstats, install the optional requirements::

    pip3 install geomstats[opt]

To install only the requirements for a given backend do::

    pip3 install geomstats[<backend_name>]

You can choose your backend by setting the environment variable ``GEOMSTATS_BACKEND`` to ``numpy``, ``autograd``  or ``pytorch``, and importing the ``backend`` module. From the command line:

.. code-block:: bash

    export GEOMSTATS_BACKEND=<backend_name>

and in the Python3 code:

.. code-block:: python

    import geomstats.backend as gs


**FIRST EXAMPLES**

To use `geomstats` for learning
algorithms on Riemannian manifolds, you need to follow three steps:
- instantiate the manifold of interest,
- instantiate the learning algorithm of interest,
- run the algorithm.
The data should be represented by the structure ``gs.array``, which represents numpy arrays or pytorch tensors, depending on the choice of backend.

As an example, the following code snippet illustrates the use of K-means
on simulated data on the 5-dimensional hypersphere.

.. code-block:: python

    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.learning.online_kmeans import OnlineKMeans

    sphere = Hypersphere(dim=5)

    data = sphere.random_uniform(n_samples=10)

    clustering = OnlineKMeans(sphere, n_clusters=4)
    clustering = clustering.fit(data)

The following code snippet shows the use of tangent Principal Component Analysis on simulated data on the
space of 3D rotations.

.. code-block:: python

    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    from geomstats.learning.pca import TangentPCA

    so3 = SpecialOrthogonal(n=3, point_type="vector")

    data = so3.random_uniform(n_samples=10)

    tpca = TangentPCA(so3, n_components=2)
    tpca = tpca.fit(data)
    tangent_projected_data = tpca.transform(data)

All geometric computations are performed behind the scenes.
The user only needs a high-level understanding of Riemannian geometry.
Each algorithm can be used with any of the manifolds and metric
implemented in the package.

To see additional examples, visit the page :ref:`examples`.
