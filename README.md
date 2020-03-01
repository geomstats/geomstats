# Geomstats
[![Build Status](https://travis-ci.org/geomstats/geomstats.svg?branch=master)](https://travis-ci.org/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=numpy)](https://codecov.io/gh/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=tensorflow)](https://codecov.io/gh/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=pytorch)](https://codecov.io/gh/geomstats/geomstats) (Coverages for: numpy, tensorflow, pytorch)


Computations and statistics on manifolds with geometric structures.

<img align="left" src="https://raw.githubusercontent.com/ninamiolane/geomstats/master/examples/imgs/h2_grid.png" width=110 height=110>

- To get started with ```geomstats```, see the [examples directory](https://github.com/geomstats/geomstats/tree/master/examples).
- For more in-depth applications of ``geomstats``, see the [applications repository](https://github.com/geomstats/applications/).
- The documentation of ```geomstats``` can be found on the [documentation website](https://geomstats.github.io/).
- If you find ``geomstats`` useful, please kindly cite our [paper](https://arxiv.org/abs/1805.08308).


## Installation

OS X & Linux:

```
pip3 install -r requirements.txt
pip3 install geomstats
```

Pytorch and tensorflow requirements are optional, as geomstats can be used with numpy only.

To change backend:
```
export GEOMSTATS_BACKEND=pytorch
```

## Getting started

Run example scripts, for example:

```
python3 examples/plot_grid_h2.py
```

## Contributing

Developers can install the dev-requirements:

```
pip3 install -r dev-requirements.txt
```

And run unit tests from this folder:
```
nose2 tests
```

See our [contributing](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst) guidelines!

## Acknowledgements

This work is supported by:
- the Inria-Stanford associated team [GeomStats](http://www-sop.inria.fr/asclepios/projects/GeomStats/),
- the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement [G-Statistics](https://team.inria.fr/epione/en/research/erc-g-statistics/) No. 786854),
- the French society for applied and industrial mathematics ([SMAI](http://smai.emath.fr/)),
- the National Science Foundation (grant NSF DMS RTG 1501767).
