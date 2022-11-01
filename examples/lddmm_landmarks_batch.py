"""
Visualize geodesics in landmarks space with kernel metric
"""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.landmarks import Landmarks, L2LandmarksMetric, KernelLandmarksMetric
import numpy as np
import matplotlib.pyplot as plt
import time

gs.random.seed(1234)

# optional : using keops for kernel convolutions (to be used with pytorch backend)
use_keops = True
# N.B. Usage of keops has no interest in the setting of this example.
# It is usefull only for landmarks sets > 5000 points approx, and using a GPU.
# To install keops, on a linux machine use "pip install pykeops"
# Note : On MacOs, as of Oct 2022, a compiler warning causes the current version 
# of keops on pip to fail when calling formulas. 
# You need to install latest release of keops from git using both commands 
# pip install git+https://github.com/getkeops/keops.git@main#subdirectory=keopscore
# pip install git+https://github.com/getkeops/keops.git@main#subdirectory=pykeops

step = "euler" # step rule for numerical integration
n_steps = 10 # number of integration steps

# Setting : 2 configurations of 2 points in 2D
n_points = 2
landmark_set_a = gs.array([[[0., 0.],[1., .1]],[[0., 0.],[1., .2]]])
landmark_set_b = gs.array([[[1., 0.],[0., .1]],[[1., 0.],[0., .2]]])
nbatch = landmark_set_a.shape[0]

# set the space and metrics
r2 = Euclidean(dim=2)
if use_keops:
    gaussian_1 = lambda d: (-d).exp()
    gaussian_2 = lambda d: (-d/.25**2).exp()
else:
    gaussian_1 = lambda d: gs.exp(-d)
    gaussian_2 = lambda d: gs.exp(-d/.25**2)

metrics = { 
    "Kernel metric (gaussian kernel, sigma=1)" : 
            KernelLandmarksMetric(ambient_dimension=2, k_landmarks=n_points, kernel=gaussian_1, use_keops=use_keops),
    "Kernel metric (gaussian kernel, sigma=.25)" : 
            KernelLandmarksMetric(ambient_dimension=2, k_landmarks=n_points, kernel=gaussian_2, use_keops=use_keops)
}

# set parameters for geodesics computation and plotting
n_sampling_geod = 100
times = gs.linspace(0., 1., n_sampling_geod)
atol = 1e-6

# main loop
for key in metrics:
    metric = metrics[key]
    space_landmarks_in_euclidean_2d = Landmarks(
        ambient_manifold=r2, k_landmarks=n_points, metric=metric)
    landmarks_a = landmark_set_a
    landmarks_b = landmark_set_b

    # testing exp
    initial_cotangent_vec = gs.array([[[1., 0.],[-1., 0.]],[[1., 1.],[-1., 0.]]])
    start = time.time()
    landmarks_ab = metric.geodesic(landmarks_a, initial_cotangent_vec=initial_cotangent_vec, step=step, n_steps=n_steps)
    geod = gs.to_numpy(landmarks_ab(times))
    end = time.time()
    print("elapsed=", end-start)
    print("kernel_matrix=",metric.kernel_matrix(landmarks_a))
    print("cometric_matrix=",metric.cometric_matrix(landmarks_a))
    print("metric_matrix=",metric.metric_matrix(landmarks_a))
    for k in range(nbatch):
        landmark_set_a_k = landmark_set_a[k]
        initial_cotangent_vec_k = initial_cotangent_vec[k]
        geod_k = geod[k]
        plt.figure()
        plt.plot(landmark_set_a_k[:,0], landmark_set_a_k[:,1], 'o')
        plt.quiver(landmark_set_a_k[:,0], landmark_set_a_k[:,1], 
                initial_cotangent_vec_k[:,0], initial_cotangent_vec_k[:,1])
        plt.plot(geod_k[:,:,0], geod_k[:,:,1])
        plt.title(f"set {k}, {key},\n geodesic (exp from cotangent vector)")
        plt.axis("equal")

    # testing log
    start = time.time()
    landmarks_ab = metric.geodesic(landmarks_a, landmarks_b, step=step, n_steps=n_steps)
    geod = gs.to_numpy(landmarks_ab(times))
    end = time.time()
    print("elapsed=", end-start)
    for k in range(nbatch):
        landmark_set_a_k = landmark_set_a[k]
        landmark_set_b_k = landmark_set_b[k]
        geod_k = geod[k]
        plt.figure()
        plt.plot(landmark_set_a_k[:,0], landmark_set_a_k[:,1], 'o')
        plt.plot(landmark_set_b_k[:,0], landmark_set_b_k[:,1], 'x')
        plt.plot(geod_k[:,:,0], geod_k[:,:,1])
        plt.title(f"set {k}, {key},\n geodesic (computing log)")
        plt.axis("equal")

plt.show()

