Geomstats
=========

.. list-table::
   :header-rows: 0

   * - **Code**
     - |PyPI version|\ |Zenodo|\
   * - **Continuous Integration**
     - |Build Status|\ |python|\
   * - **Code coverage (numpy, tensorflow, pytorch)**
     - |Coverage Status np|\ |Coverage Status tf|\ |Coverage Status torch|
   * - **Documentation**
     - |doc|\ |binder|\ |tutorial|\
   * - **Community**
     - |contributions|\ |Slack|\ |Twitter|\

**NEWS**: Interested in pushing forward the fields of computational
differential geometry and topology? The `ICLR 2021 challenge of geometry
and topology <https://gt-rl.github.io/challenge>`__ is open for
submissions. Deadline: May 2nd, 2021. $3500 in prizes.

Geomstats is an open-source Python package for computations and
statistics on manifolds. The package is organized into two main modules:
``geometry`` and ``learning``.

The module ``geometry`` implements concepts in differential geometry,
and the module ``learning`` implements statistics and learning
algorithms for data on manifolds.

.. raw:: html

    <img src="https://raw.githubusercontent.com/ninamiolane/geomstats/master/examples/imgs/h2_grid.png" height="120px" width="120px" align="left">


-  To get an overview of ``geomstats``, see our `introductory
   video <https://www.youtube.com/watch?v=Ju-Wsd84uG0&list=PLYx7XA2nY5GejOB1lsvriFeMytD1-VS1B&index=3>`__.
-  To get started with ``geomstats``, see the
   `examples <https://github.com/geomstats/geomstats/tree/master/examples>`__
   and
   `notebooks <https://github.com/geomstats/geomstats/tree/master/notebooks>`__
   directories.
-  The documentation of ``geomstats`` can be found on the `documentation
   website <https://geomstats.github.io/>`__.
-  To follow the scientific literature on geometric statistics, follow
   our twitter-bot `@geomstats-papers <https://twitter.com/geomstats>`__!

If you find ``geomstats`` useful, please kindly cite our
`paper <https://jmlr.org/papers/v21/19-027.html>`__:

::

    @article{JMLR:v21:19-027,
      author  = {Nina Miolane and Nicolas Guigui and Alice Le Brigant and Johan Mathe and Benjamin Hou and Yann Thanwerdas and Stefan Heyder and Olivier Peltre and Niklas Koep and Hadi Zaatiti and Hatem Hajri and Yann Cabanes and Thomas Gerald and Paul Chauchat and Christian Shewmake and Daniel Brooks and Bernhard Kainz and Claire Donnat and Susan Holmes and Xavier Pennec},
      title   = {Geomstats:  A Python Package for Riemannian Geometry in Machine Learning},
      journal = {Journal of Machine Learning Research},
      year    = {2020},
      volume  = {21},
      number  = {223},
      pages   = {1-9},
      url     = {http://jmlr.org/papers/v21/19-027.html}
    }

Install geomstats via pip3
--------------------------

From a terminal (OS X & Linux), you can install geomstats and its
requirements with ``pip3`` as follows:

::

    pip3 install geomstats

This method installs the latest version of geomstats that is uploaded on
PyPi. Note that geomstats is only available with Python3.

Install geomstats via Git
-------------------------

From a terminal (OS X & Linux), you can install geomstats and its
requirements via ``git`` as follows:

::

    git clone https://github.com/geomstats/geomstats.git
    pip3 install -r requirements.txt

This method installs the latest GitHub version of geomstats. Developers
should install this version, together with the development requirements
and the optional requirements to enable ``tensorflow`` and ``pytorch``
backends:

::

    pip3 install -r dev-requirements.txt -r opt-requirements.txt

Choose the backend
------------------

Geomstats can run seamlessly with ``numpy``, ``tensorflow`` or
``pytorch``. Note that ``pytorch`` and ``tensorflow`` requirements are
optional, as geomstats can be used with ``numpy`` only. By default, the
``numpy`` backend is used. The visualizations are only available with
this backend.

To get the ``tensorflow`` and ``pytorch`` versions compatible with
geomstats, install the `optional
requirements <https://github.com/geomstats/geomstats/blob/master/opt-requirements.txt>`__:

::

    pip3 install -r opt-requirements.txt

You can choose your backend by setting the environment variable
``GEOMSTATS_BACKEND`` to ``numpy``, ``tensorflow`` or ``pytorch``, and
importing the ``backend`` module. From the command line:

::

    export GEOMSTATS_BACKEND=pytorch

and in the Python3 code:

::

    import geomstats.backend as gs

Getting started
---------------

To use ``geomstats`` for learning algorithms on Riemannian manifolds,
you need to follow three steps: - instantiate the manifold of interest,
- instantiate the learning algorithm of interest, - run the algorithm.

The data should be represented by a ``gs.array``. This structure
represents numpy arrays, or tensorflow/pytorch tensors, depending on the
choice of backend.

The following code snippet shows the use of tangent Principal Component
Analysis on simulated ``data`` on the space of 3D rotations.

.. code:: python

    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    from geomstats.learning.pca import TangentPCA

    so3 = SpecialOrthogonal(n=3, point_type='vector')
    metric = so3.bi_invariant_metric

    data = so3.random_uniform(n_samples=10)

    tpca = TangentPCA(metric=metric, n_components=2)
    tpca = tpca.fit(data)
    tangent_projected_data = tpca.transform(data)

All geometric computations are performed behind the scenes. The user
only needs a high-level understanding of Riemannian geometry. Each
algorithm can be used with any of the manifolds and metric implemented
in the package.

To see additional examples, go to the
`examples <https://github.com/geomstats/geomstats/tree/master/examples>`__
or
`notebooks <https://github.com/geomstats/geomstats/tree/master/notebooks>`__
directories.

Contributing
------------

See our
`contributing <https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst>`__
guidelines!

Acknowledgements
----------------

This work is supported by:

-  the Inria-Stanford associated team `GeomStats <http://www-sop.inria.fr/asclepios/projects/GeomStats/>`__,
-  the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement `G-Statistics <https://team.inria.fr/epione/en/research/erc-g-statistics/>`__ No. 786854),
-  the French society for applied and industrial mathematics (`SMAI <http://smai.emath.fr/>`__),
-  the National Science Foundation (grant NSF DMS RTG 1501767).

.. |Twitter| image:: https://img.shields.io/twitter/follow/geomstats?label=Follow%20%40geomstats-papers%20%20%20%20&style=social
   :target: https://twitter.com/geomstats
.. |PyPI version| image:: https://badge.fury.io/py/geomstats.svg
   :target: https://badge.fury.io/py/geomstats
.. |Build Status| image:: https://github.com/geomstats/geomstats/actions/workflows/build.yml/badge.svg
   :target: https://github.com/geomstats/geomstats/actions/workflows/build.yml
.. |Slack| image:: https://img.shields.io/badge/Slack-Join-yellow
   :target: https://geomstats.slack.com/
.. |Coverage Status np| image:: https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=numpy
   :target: https://codecov.io/gh/geomstats/geomstats
.. |Coverage Status tf| image:: https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=tensorflow
   :target: https://codecov.io/gh/geomstats/geomstats
.. |Coverage Status torch| image:: https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=pytorch
   :target: https://codecov.io/gh/geomstats/geomstats
.. |Zenodo| image:: https://zenodo.org/badge/108200238.svg
   :target: https://zenodo.org/badge/latestdoi/108200238
.. |python| image:: https://img.shields.io/badge/python-3.6+-blue?logo=python
   :target: https://www.python.org/
.. |tutorial| image:: https://img.shields.io/youtube/views/Ju-Wsd84uG0?label=watch&style=social
   :target: https://www.youtube.com/watch?v=Ju-Wsd84uG0
.. |doc| image:: https://img.shields.io/badge/docs-website-brightgreen?style=flat
   :target: https://geomstats.github.io/?badge=latest
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/geomstats/geomstats/master?filepath=notebooks
.. |contributions| image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
   :target: https://geomstats.github.io/contributing.html
