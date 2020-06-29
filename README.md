# Geomstats
[![PyPI version](https://badge.fury.io/py/geomstats.svg)](https://badge.fury.io/py/geomstats)[![Build Status](https://travis-ci.org/geomstats/geomstats.svg?branch=master)](https://travis-ci.org/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=numpy)](https://codecov.io/gh/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=tensorflow)](https://codecov.io/gh/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=pytorch)](https://codecov.io/gh/geomstats/geomstats) (Coverages for: numpy, tensorflow, pytorch)


Geomstats is an open-source Python package for computations and statistics on manifolds. The package is organized into two main modules:
``geometry`` and ``learning``.

The module `geometry` implements concepts in differential geometry, and the module `learning` implements statistics and learning algorithms for data on manifolds.

<img align="left" src="https://raw.githubusercontent.com/ninamiolane/geomstats/master/examples/imgs/h2_grid.png" width=90 height=90>


- To get started with ```geomstats```, see the [examples](https://github.com/geomstats/geomstats/tree/master/examples) and [notebooks](https://github.com/geomstats/geomstats/tree/master/notebooks) directories.
- The documentation of ```geomstats``` can be found on the [documentation website](https://geomstats.github.io/).
- If you find ``geomstats`` useful, please kindly cite our [paper](https://arxiv.org/abs/2004.04667).

## Install geomstats via pip3

From a terminal (OS X & Linux), you can install geomstats and its requirements with ``pip3`` as follows:

```
pip3 install geomstats
```

This method installs the latest version of geomstats that is uploaded on PyPi. Note that geomstats is only available with Python3.

## Install geomstats via Git

From a terminal (OS X & Linux), you can install geomstats and its requirements via ``git`` as follows:

```
git clone https://github.com/geomstats/geomstats.git
pip3 install -r requirements.txt
```

This method installs the latest GitHub version of geomstats. Developers should install this version, together with the development requirements and the optional requirements to enable ``tensorflow`` and ``pytorch`` backends:

```
pip3 install -r dev-requirements.txt -r opt-requirements.txt
```

## Choose the backend

Geomstats can run seemlessly with ``numpy``, ``tensorflow`` or ``pytorch``. Note that ``pytorch`` and ``tensorflow`` requirements are optional, as geomstats can be used with ``numpy`` only. By default, the ``numpy`` backend is used. The visualizations are only available with this backend.

To get the ``tensorflow`` and ``pytorch`` versions compatible with geomstats, install the [optional requirements](https://github.com/geomstats/geomstats/blob/master/opt-requirements.txt):

```
pip3 install -r opt-requirements.txt
```

You can choose your backend by setting the environment variable ``GEOMSTATS_BACKEND`` to ``numpy``, ``tensorflow`` or ``pytorch``, and importing the ``backend`` module. From the command line:

```
export GEOMSTATS_BACKEND=pytorch
```

and in the Python3 code:

```
import geomstats.backend as gs
```

## Getting started

To use ``geomstats`` for learning
algorithms on Riemannian manifolds, you need to follow three steps:
- instantiate the manifold of interest,
- instantiate the learning algorithm of interest,
- run the algorithm.

The data should be represented by a ``gs.array``. This structure represents numpy arrays, or tensorflow/pytorch tensors, depending on the choice of backend.

The following code snippet shows the use of tangent Principal Component Analysis on simulated ``data`` on the
space of 3D rotations.

```python
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.pca import TangentPCA

so3 = SpecialOrthogonal(n=3, point_type='vector')
metric = so3.bi_invariant_metric

data = so3.random_uniform(n_samples=10)

tpca = TangentPCA(metric=metric, n_components=2)
tpca = tpca.fit(data)
tangent_projected_data = tpca.transform(data)
```

All geometric computations are performed behind the scenes.
The user only needs a high-level understanding of Riemannian geometry.
Each algorithm can be used with any of the manifolds and metric
implemented in the package.

To see additional examples, go to the [examples](https://github.com/geomstats/geomstats/tree/master/examples) or [notebooks](https://github.com/geomstats/geomstats/tree/master/notebooks) directories.

## Contributing

See our [contributing](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst) guidelines!

## Acknowledgements

This work is supported by:
- the Inria-Stanford associated team [GeomStats](http://www-sop.inria.fr/asclepios/projects/GeomStats/),
- the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement [G-Statistics](https://team.inria.fr/epione/en/research/erc-g-statistics/) No. 786854),
- the French society for applied and industrial mathematics ([SMAI](http://smai.emath.fr/)),
- the National Science Foundation (grant NSF DMS RTG 1501767).
