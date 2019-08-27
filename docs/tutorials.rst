Tutorials
=========

**Choosing the backend.**

You need to set the environment variable GEOMSTATS_BACKEND to numpy, tensorflow or pytorch. Only use the numpy backend for examples with visualizations.

.. code-block:: bash

    export GEOMSTATS_BACKEND=numpy

**A first python example.**

This example shows how to compute a geodesic on the Lie group SE(3), which is the group of rotations and translations in 3D.

.. code-block:: python

    """
    Plot a geodesic of SE(3) equipped
    with its left-invariant canonical metric.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    import geomstats.visualization as visualization

    from geomstats.special_euclidean_group import SpecialEuclideanGroup

    SE3_GROUP = SpecialEuclideanGroup(n=3)
    METRIC = SE3_GROUP.left_canonical_metric

    initial_point = SE3_GROUP.identity
    initial_tangent_vec = [1.8, 0.2, 0.3, 3., 3., 1.]
    geodesic = METRIC.geodesic(initial_point=initial_point,
                               initial_tangent_vec=initial_tangent_vec)

    n_steps = 40
    t = np.linspace(-3, 3, n_steps)

    points = geodesic(t)

    visualization.plot(points, space='SE3_GROUP')
    plt.show()

**More examples.**

You can find more examples in the repository "examples" of geomstats. You can run them from the command line as follows.

.. code-block:: bash

    python3 examples/plot_grid_h2.py
