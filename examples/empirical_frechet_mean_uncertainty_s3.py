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


import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
# import geomstats.visualization as visualization

from geomstats.geometry.hypersphere import Hypersphere

# n = number of samples for computing the mean
# NN = number of trials for the stochastic integral of expectation
# Ntheta = number of points on the curve sig_est(theta)

def my_random_uniform_sphere(n):
    #locals(x,xn)
    # Random coordinates between -1 and 1
    x = (np.random.random_sample(n+1)*2.0 -1.0)
    nx = np.linalg.norm(x)
    while nx > 1 or nx < 1.e-10:
       x = (np.random.random_sample(n+1)*2.0 -1.0)
       nx = np.linalg.norm(x)
    return(x/nx)

def my_random_sample(N, dim):
    X = np.zeros( (N, dim+1), dtype=gs.float64)
    for i in range(N):
        X[i,:] = my_random_uniform_sphere(dim)
    return X



def empirical_var(n, theta, dim, NN=1000):
    """
    Compute the variance of the empirical Fréchet mean of n samples
    drawn from an isotropic distributions on hyper sub-spheres of radius
    0 < theta < Pi around the north pole of the sphere of dimension dim.
    This is repeated NN times to make a stochastic approximation of the
    expectation.
    """
    assert dim > 1, "Dim > 1 needed to draw a uniform sample on subsphere"
    var = []
    sphere = Hypersphere(dimension=dim)
    subsphere = Hypersphere(dimension=dim-1)

    # Define north pole
    north_pole = np.zeros(dim+1)
    north_pole[dim] = 1.0
    for j in range(NN):
        # Sample n points from the uniform distrib on a subsphere
        # of radius theta (i.e cos(theta) in ambiant space)
        data = gs.zeros((n, dim+1), dtype=gs.float64)
        # For sampling on a subsphere, use RandomUniform(dim-1)
        directions = subsphere.random_uniform(n)
        # directions = my_random_sample(n, dim-1)
        for i in range(n):
            for j in range(dim):
                data[i,j] = gs.sin(theta) * directions[i,j]
            data[i,dim] = gs.cos(theta)
        ## Compute empirical Frechet mean of the n-sample
        current_mean = sphere.metric.adaptive_gradientdescent_mean(data, n_max_iterations=64, init_points=[north_pole])
        var.append(sphere.metric.squared_dist(north_pole, current_mean))
    return (np.mean(var), 2* np.std(var) / np.sqrt( NN ) )


def modulation_factor(n, theta, dim, NN=1000):
    """
    Compute the modulation factor as the ratio between the variance of
    the empirical Fréchet mean of n samples drawn from an isotropic
    distributions on hyper sub-spheres of radius 0 < theta < Pi around
    the north pole of the sphere of dimension dim.
    This is repeated NN times to make a stochastic approximation of the
    expectation and stddev on the modulation factor is returned as second parameter
    """
    (var, stdvar) = empirical_var(n, theta, dim, NN=NN)
    return var * n / theta ** 2, stdvar * n / theta ** 2



def asymptotic_modulation(dim, theta):
    gamma = 1.0 / dim + (1.0 - 1.0 / dim) * theta / np.tan(theta)
    return (1.0 / gamma)**2


def plot_modulation_factor(n, dim, NN=1000, Ntheta=20):
    # modulation factor for n points on the sphere S_dim embedded in R^{dim+1}
    theta = np.linspace(0.000001, np.pi / 2.0 - 0.000001, Ntheta)
    measured_modulation_factor = []
    error = []
    small_var_modulation_factor = []
    asymptotic_modulation_factor = []
    for thetai in theta:
        (var, stdvar) = modulation_factor(n, thetai, dim, NN=NN)
        measured_modulation_factor.append(var)
        error.append(stdvar)
        print(n, thetai, var, stdvar, '\n')
        small_var_modulation_factor.append(1.0 + 2.0 / 3.0 * thetai ** 2 * (1.0 - 1.0 / dim) * (1.0 - 1.0 / (n)))
        asymptotic_modulation_factor.append(asymptotic_modulation(dim, thetai))
    plt.figure()
    plt.errorbar(theta, measured_modulation_factor, yerr=error, color='r', label='Measured')  # plotting t,a separately
    plt.plot(theta, small_var_modulation_factor, 'g', label='Small variance prediction')  # plotting t,a separately
    plt.plot(theta, asymptotic_modulation_factor, 'grey', label='Asymptotic prediction')  # plotting t,a separately
    plt.xlabel(r'Standard deviation $\theta$')
    plt.ylabel(r'Modulation factor $\alpha$')
    plt.title("Convergence rate modulation factor, sphere dim={1}, n={0}".format(n, dim))
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.01)
    ## plt.savefig("Figures/SphVarModulation_N{0}_d{1}.svg".format(n, dim))
    plt.savefig("Figures/SphVarModulation_N{0}_d{1}.pdf".format(n, dim))
    return plt



def multiplot_modulation_factor(dim, NN=1000, Ntheta=20):
    ## Implementation for the sphere S_dim in R^{dim+1}
    theta = np.linspace(0.000001, np.pi / 2.0 - 0.000001, Ntheta)
    small_var_modulation_factor = []
    asymptotic_modulation_actor = []
    plt.figure()
    for thetai in theta:
        small_var_modulation_factor.append(1.0 + 2.0 / 3.0 * thetai ** 2 * (1.0 - 1.0 / dim) * (1.0))
        asymptotic_modulation_actor.append(asymptotic_modulation(dim, thetai))
    plt.plot(theta, small_var_modulation_factor, 'g', label='Small variance prediction')
    plt.plot(theta, asymptotic_modulation_actor, 'grey', label='Asymptotic prediction')
    color = {10: 'red', 20: 'orange', 50: 'olive', 100: 'blue'}
    for n in [10, 20, 50, 100]:
        measured_modulation_factor = []
        for thetai in theta:
            (var, stdvar) = modulation_factor(n, thetai, dim, NN=NN)
            measured_modulation_factor.append(var)
            print(n, thetai, var, stdvar, '\n')
        plt.plot(theta, measured_modulation_factor, color=color[n], label="N={0}".format(n))  # plotting t,a separately
    plt.xlabel(r'Standard deviation $\theta$')
    plt.ylabel(r'Modulation factor $\alpha$')
    plt.legend(loc='best')
    plt.title("Convergence rate modulation factor, sphere, dim={0}, N > 5".format(dim))
    plt.draw()
    plt.pause(0.01)
    ## plt.savefig("Figures/SphVarModulation_N10p_d{0}.svg".format(dim))
    plt.savefig("Figures/SphVarModulation_N10p_d{0}.pdf".format(dim))
    return plt


def main():
    NN = 10;

    print("Var of empirical mean for 1 sample, theta=0.1 in S2", empirical_var(1, 0.1, 2, NN=NN), "\n")
    print("Var of empirical mean for 1 sample, theta=0.1 in S3", empirical_var(1, 0.1, 3, NN=NN), "\n")

    print("Modulation factor for 1 sample theta=0.1 in S2 (should be close to 1):",
          modulation_factor(1, 0.1, 2, NN=NN), "\n")

    print("Modulation factor for 500 sample theta close to Pi/2 in S5 (should be around 25):",
          modulation_factor(500, gs.pi / 2 - 0.001, 5, NN=NN), "\n")

    plot_modulation_factor(2, 2, NN=NN)
    # plot_modulation_factor(3, 2, NN=NN)
    plot_modulation_factor(4, 2, NN=NN)
    # plot_modulation_factor(5, 2, NN=NN)
    plot_modulation_factor(10, 2, NN=NN)

    # plot_modulation_factor(1, 3, NN=NN)
    plot_modulation_factor(2, 3, NN=NN)
    # plot_modulation_factor(3, 3, NN=NN)
    plot_modulation_factor(4, 3, NN=NN)
    # plot_modulation_factor(5, 3, NN=NN)


    multiplot_modulation_factor(3, NN=NN)

if __name__ == "__main__":
    main()