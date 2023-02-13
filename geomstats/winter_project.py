

import os
import sys
import warnings

sys.path.append(os.path.dirname(os.getcwd()))
warnings.filterwarnings("ignore")

import geomstats._backend as gs

### This is causing an error because random isn't found in geomstats -> _backend
gs.random.seed(2020)

import matplotlib
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import geomstats.visualization as visualization


### Might just be using this for practice .. take out later
visualization.tutorial_matplotlib()


from geomstats.geometry.euclidean import Euclidean

dim = 2
n_samples = 2

euclidean = Euclidean(dim=dim)
points_in_linear_space = euclidean.random_point(n_samples=n_samples)
print("Points in linear space:\n", points_in_linear_space)

linear_mean = gs.sum(points_in_linear_space, axis=0) / n_samples
print("Mean of points:\n", linear_mean)


