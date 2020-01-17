"""
Uncertainty of the empirical Fréchet mean on the sphere

The variance of the Fréchet mean FM_n of a sample of n IID
random variables of variance Var is decreasing more slowly
in a sphere than in a Euclidean space.
This example computes the  modulation factor
     alpha = Var( FM_n) / ( n * Var)
for isotropic distributions on hyperspheres or radius 0<r<Pi
in the sphere s3.
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
# import geomstats.visualization as visualization

from geomstats.geometry.hypersphere import Hypersphere

# n = number of samples for computing the mean
# NN = number of trials for the stochastic integral of expectation
# Ntheta = number of points on the curve sig_est(theta)


def EmpiricalVar(n, theta, dim, NN=5000):
    """
    Compute the variance of the empirical Fréchet mean of n samples
    drawn from an isotropic distributions on hyper sub-spheres of radius
    0 < theta < Pi around the north pole of the sphere of dimension dim.
    This is repeated NN times to make a stochastic approximation of the
    expectation.
    """
    assert dim > 2, "Dim > 2 needed to draw a uniform sample on subsphere"
    var = []
    sphere = Hypersphere(dimension=dim)
    subsphere = Hypersphere(dimension=dim-1)

    # Define north pole
    north_pole = np.zeros(dim+1)
    north_pole[dim] = 1.0
    for j in range(NN):
        # Sample n points from the uniform distrib on a subsphere
        # of radius theta (i.e cos(theta) in ambiant space)
        data = gz.zeros((n, dim+1), dtype=float)
        # For sampling on a subsphere, use RandomUniform(dim-1)
        for i in range(n):
            directions = subsphere.random_uniform(n)
            for j in range(dim):
                data[i,j] = gs.sin(theta) * directions[j]
            data[i,dim] = gs.cos(theta)
        ## Compute empirical Frechet mean of the n-sample
        current_mean = sphere.metric.adaptive_gradientdescent_mean(data,init_points=[north_pole])
        var.append(sphere.metric.squared_dist(north_pole,current_mean)
    return (np.mean(var), 2* np.std(var) / gs.sqrt( NN ) )


def main():



    fig = plt.figure(figsize=(15, 5))

    sphere = Hypersphere(dimension=2)

    data = sphere.random_von_mises_fisher(kappa=15, n_samples=140)
    mean = sphere.metric.mean(data)

if __name__ == "__main__":
    main()