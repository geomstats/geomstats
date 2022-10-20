

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.landmarks import Landmarks, L2LandmarksMetric, KernelLandmarksMetric
import numpy as np
import matplotlib.pyplot as plt
import time

r2 = Euclidean(dim=2)

n_points = 2
landmark_set_a = gs.array([[0., 0.],[1., .1]])
landmark_set_b = gs.array([[1., 0.],[0., .1]])

n_sampling_geod = 100
times = gs.linspace(0., 1., n_sampling_geod)
atol = 1e-6
gs.random.seed(1234)

metrics = [ 
    L2LandmarksMetric(ambient_metric=r2.metric, k_landmarks=n_points),
    KernelLandmarksMetric(ambient_dimension=2, k_landmarks=n_points)
]

for metric in metrics:
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
    plt.plot(geod[:,:,0], geod[:,:,1])

    # testing log
    start = time.time()
    landmarks_ab = metric.geodesic(landmarks_a, landmarks_b)
    end = time.time()
    print("elapsed=", end-start)
    geod = landmarks_ab(times)
    plt.figure()
    plt.plot(geod[:,:,0], geod[:,:,1])

plt.show()