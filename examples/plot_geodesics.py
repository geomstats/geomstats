"""
Plot a geodesic of SE(3) equipped
with its left-invariant canonical metric.
"""

import numpy as np
import matplotlib.pyplot as plt

from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.visualization as visualization


se3_group = SpecialEuclideanGroup(n=3)
metric = se3_group.left_canonical_metric

initial_point = se3_group.identity
initial_tangent_vec = np.array([1.8, 0.2, 0.3, 3., 3., 1.])
geodesic = metric.geodesic(initial_point=initial_point,
                           initial_tangent_vec=initial_tangent_vec)

points = np.vstack([geodesic(t)
                    for t in np.linspace(-0.7, 0.7, 20)])

visualization.plot_points(points)
plt.show()
