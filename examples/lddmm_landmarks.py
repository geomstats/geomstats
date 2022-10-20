"""
Visualize geodesics in landmarks space with kernel metric
"""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.landmarks import Landmarks, L2LandmarksMetric, KernelLandmarksMetric
import numpy as np
import matplotlib.pyplot as plt
import time

# Setting : 2 configurations of 2 points in 2D
n_points = 2
landmark_set_a = gs.array([[0., 0.],[1., .1]])
landmark_set_b = gs.array([[1., 0.],[0., .1]])

# set the space and metrics
r2 = Euclidean(dim=2)
gaussian_2 = lambda d: gs.exp(-d/.25**2)
metrics = { 
    "Kernel metric (default gaussian kernel, sigma=1)" : 
            KernelLandmarksMetric(ambient_dimension=2, k_landmarks=n_points),
    "Kernel metric (gaussian kernel, sigma=.25)" : 
            KernelLandmarksMetric(ambient_dimension=2, k_landmarks=n_points, kernel=gaussian_2)
}

# set parameters for geodesics computation and plotting
n_sampling_geod = 100
times = gs.linspace(0., 1., n_sampling_geod)
atol = 1e-6
gs.random.seed(1234)

# main loop
for key in metrics:
    metric = metrics[key]
    space_landmarks_in_euclidean_2d = Landmarks(
        ambient_manifold=r2, k_landmarks=n_points, metric=metric)
    landmarks_a = landmark_set_a
    landmarks_b = landmark_set_b

    # testing exp
    initial_tangent_vec = gs.array([[1., 0.],[-1., 0.]])
    start = time.time()
    landmarks_ab = metric.geodesic(landmarks_a, initial_tangent_vec=initial_tangent_vec)
    end = time.time()
    print("elapsed=", end-start)
    geod = landmarks_ab(times)
    plt.figure()
    plt.plot(landmark_set_a[:,0], landmark_set_a[:,1], 'o')
    plt.quiver(landmark_set_a[:,0], landmark_set_a[:,1], 
                initial_tangent_vec[:,0], initial_tangent_vec[:,1])
    plt.plot(geod[:,:,0], geod[:,:,1])
    plt.title(f"{key},\n geodesic (exp from tangent vector)")

    # testing log
    start = time.time()
    landmarks_ab = metric.geodesic(landmarks_a, landmarks_b)
    end = time.time()
    print("elapsed=", end-start)
    geod = landmarks_ab(times)
    plt.figure()
    plt.plot(landmark_set_a[:,0], landmark_set_a[:,1], 'o')
    plt.plot(landmark_set_b[:,0], landmark_set_b[:,1], 'x')
    plt.plot(geod[:,:,0], geod[:,:,1])
    plt.title(f"{key},\n geodesic (computing log)")

plt.show()