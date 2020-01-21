"""This very simple example demonstrates how geometries from geomstats can be
used in pymanopt to perform optimization on manifolds. It uses the Riemannian
steepest descent solver.

The example currently requires installing the pymanopt HEAD from git:

    pip install git+ssh://git@github.com/pymanopt/pymanopt.git
"""

import functools
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # NOQA

import numpy as np
from numpy import linalg as la, random as rnd
from pymanopt import Problem
from pymanopt.manifolds.manifold import Manifold
from pymanopt.solvers import SteepestDescent

from geomstats.geometry.hypersphere import Hypersphere


def squeeze_ndarray(f):
    """Decorator to remove singleton dimensions from a numpy.ndarray returned
    by the decorated function.
    """
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return np.squeeze(f(*args, **kwargs))
    return inner


class GeomstatsSphere(Manifold):
    """A simple adapter class which proxies calls by pymanopt's solvers to
    manifolds to the underlying geomstats Hypersphere class.
    """

    def __init__(self, n):
        self._sphere = Hypersphere(n-1)

    def norm(self, x, g):
        return self._sphere.metric.norm(g, base_point=x)

    @squeeze_ndarray
    def proj(self, x, g):
        return self._sphere.projection_to_tangent_space(g, base_point=x)

    @squeeze_ndarray
    def retr(self, x, g):
        # geomstats's hypersphere implementation doesn't provide a retraction
        # so use the exponential map instead.
        return self._sphere.metric.exp(g, base_point=x)

    def inner(self, x, u, v):
        return self._sphere.metric.inner_product(u, v, base_point=x)

    @squeeze_ndarray
    def rand(self):
        return self._sphere.random_uniform()


@squeeze_ndarray
def dominant_eigenvector(A):
    """Returns the dominant eigenvector of the symmetric matrix A by minimizing
    the Rayleigh quotient -x' * A * x / (x' * x).
    """
    m, n = A.shape
    assert m == n, "matrix must be square"
    assert np.allclose(np.sum(A - A.T), 0), "matrix must be symmetric"

    def cost(x):
        return -np.inner(x, A @ x)

    def egrad(x):
        return -2 * A @ x

    sphere = GeomstatsSphere(n)
    problem = Problem(manifold=sphere, cost=cost, egrad=egrad)
    solver = SteepestDescent()
    xopt = solver.solve(problem)
    return xopt


if __name__ == "__main__":
    if os.environ.get("GEOMSTATS_BACKEND") != "numpy":
        raise SystemExit(
            "This example currently only supports the numpy backend")

    # Generate random problem data.
    n = 128
    A = rnd.randn(n, n)
    A = 0.5 * (A + A.T)

    # Calculate the actual solution by a conventional eigenvalue decomposition.
    w, v = la.eig(A)
    x = v[:, np.argmax(w)]

    # Solve the problem again with the wrapped geomstats manifold
    # implementation.
    xopt = dominant_eigenvector(A)

    # Make sure both vectors have the same direction. Both are valid
    # eigenvectors, of course, but for comparison we need to get rid of the
    # ambiguity.
    if np.sign(x[0]) != np.sign(xopt[0]):
        xopt = -xopt

    # Print information about the solution.
    print("l2-norm of x:", la.norm(x))
    print("l2-norm of xopt:", la.norm(xopt))
    error_norm = la.norm(x - xopt)
    print("l2-error:", error_norm)
    print("solution found: %s" % np.isclose(error_norm, 0.0, atol=1e-3))
