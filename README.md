# Geomstats
[![Build Status](https://travis-ci.org/geomstats/geomstats.svg?branch=master)](https://travis-ci.org/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=numpy)](https://codecov.io/gh/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=tensorflow)](https://codecov.io/gh/geomstats/geomstats)[![Coverage Status](https://codecov.io/gh/geomstats/geomstats/branch/master/graph/badge.svg?flag=pytorch)](https://codecov.io/gh/geomstats/geomstats) (Coverages for: numpy, tensorflow, pytorch)


Computations and statistics on manifolds with geometric structures.

- To get started with ```geomstats```, see the [examples directory](https://github.com/geomstats/geomstats/examples).
- For more in-depth applications of ``geomstats``, see the [applications repository](https://github.com/geomstats/applications/).
- The documentation of ```geomstats``` can be found [here](https://geomstats.github.io/).
- If you use ``geomstats``, please kindly cite our [paper](https://arxiv.org/abs/1805.08308).

<img src="https://raw.githubusercontent.com/ninamiolane/geomstats/master/examples/imgs/gradient_descent.gif" width=350 height=350><img src="https://raw.githubusercontent.com/ninamiolane/geomstats/master/examples/imgs/h2_grid.png" width=300 height=300>


## Installation

OS X & Linux:

```
pip3 install geomstats
```

## Running tests

```
pip3 install nose2
nose2
```

## Getting started

Define your backend by setting the environment variable ```GEOMSTATS_BACKEND``` to ```numpy```, ```tensorflow```, or ```pytorch```:

```
export GEOMSTATS_BACKEND=numpy
```

Then, run example scripts:

```
python3 examples/plot_grid_h2.py
```

## Contributing

See our [CONTRIBUTING.md][link_contributing] file!

## Authors & Contributors

* Alice Le Brigant
* Claire Donnat
* Oleg Kachan
* Benjamin Hou
* Johan Mathe
* Nina Miolane
* Xavier Pennec

## Acknowledgements

This work is partially supported by the National Science Foundation, grant NSF DMS RTG 1501767.

[link_contributing]: https://github.com/geomstats/geomstats/CONTRIBUTING.md
