"""
Plot a geodesic of the following Riemannian manifolds:
    - SE(3) with its left-invariant canonical metric: a Lie group
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geomstats.special_euclidean_group import SpecialEuclideanGroup


se3_group = SpecialEuclideanGroup(n=3)
metric = se3_group.left_canonical_metric

initial_point = se3_group.identity
initial_tangent_vec = np.array([1.2, 0.2, 0.3, 6., 0., 0])
geodesic = metric.geodesic(initial_point=initial_point,
                           initial_tangent_vec=initial_tangent_vec)

visualization.plot
