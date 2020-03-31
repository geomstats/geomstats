"""This very simple example demonstrates how geometries from geomstats can be
used in pymanopt to perform optimization on manifolds. It uses the Riemannian
steepest descent solver.

The example currently requires installing the pymanopt HEAD from git:

    pip install git+ssh://git@github.com/pymanopt/pymanopt.git
"""

import functools
import logging
import os

from pymanopt import Problem
from pymanopt.manifolds.manifold import Manifold
from pymanopt.solvers import SteepestDescent

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere


def squeeze_output(function):
    """Decorator to remove singleton dimensions from a numpy.ndarray returned
    by the decorated function.
    """
    @functools.wraps(function)
    def inner(*args, **kwargs):
        return gs.squeeze(function(*args, **kwargs))
    return inner


class GeomstatsSphere(Manifold):
    """A simple adapter class which proxies calls by pymanopt's solvers to
    `Manifold` subclasses to the underlying geomstats `Hypersphere` class.
    """

    def __init__(self, ambient_dimension):
        self._sphere = Hypersphere(ambient_dimension - 1)

    def norm(self, base_vector, tangent_vector):
        return self._sphere.metric.norm(tangent_vector, base_point=base_vector)

    def inner(self, base_vector, tangent_vector_a, tangent_vector_b):
        return self._sphere.metric.inner_product(
            tangent_vector_a, tangent_vector_b, base_point=base_vector)

    @squeeze_output
    def proj(self, base_vector, tangent_vector):
        return self._sphere.projection_to_tangent_space(
            tangent_vector, base_point=base_vector)

    @squeeze_output
    def retr(self, base_vector, tangent_vector):
        """The retraction operator, which maps a tangent vector in the tangent
        space at a specific point back to the manifold by approximating moving
        along a geodesic. Since geomstats's `Hypersphere` class doesn't provide
        a retraction we use the exponential map instead (see also
        https://hal.archives-ouvertes.fr/hal-00651608/document).
        """
        return self._sphere.metric.exp(tangent_vector, base_point=base_vector)

    @squeeze_output
    def rand(self):
        return self._sphere.random_uniform()


def estimate_dominant_eigenvector(matrix):
    """Returns the dominant eigenvector of the symmetric matrix A by minimizing
    the Rayleigh quotient -x' * A * x / (x' * x).
    """
    num_rows, num_columns = gs.shape(matrix)
    assert num_rows == num_columns, 'matrix must be square'
    assert gs.allclose(gs.sum(matrix - gs.transpose(matrix)), 0.0), \
        'matrix must be symmetric'

    def cost(vector):
        return -gs.dot(vector, gs.dot(matrix, vector))

    def egrad(vector):
        return -2 * gs.dot(matrix, vector)

    sphere = GeomstatsSphere(num_columns)
    problem = Problem(manifold=sphere, cost=cost, egrad=egrad)
    solver = SteepestDescent()
    return solver.solve(problem)


if __name__ == '__main__':
    if os.environ.get('GEOMSTATS_BACKEND') != 'numpy':
        raise SystemExit(
            'This example currently only supports the numpy backend')

    ambient_dimension = 128
    matrix = gs.random.normal(size=(ambient_dimension, ambient_dimension))
    matrix = 0.5 * (matrix + matrix.T)

    eigenvalues, eigenvectors = gs.linalg.eig(matrix)
    dominant_eigenvector = eigenvectors[:, gs.argmax(eigenvalues)]

    dominant_eigenvector_estimate = estimate_dominant_eigenvector(matrix)
    if (gs.sign(dominant_eigenvector[0]) !=
            gs.sign(dominant_eigenvector_estimate[0])):
        dominant_eigenvector_estimate = -dominant_eigenvector_estimate

    logging.info('l2-norm of dominant eigenvector: {}'.format(
        gs.linalg.norm(dominant_eigenvector)))
    logging.info('l2-norm of dominant eigenvector estimate: {}'.format(
        gs.linalg.norm(dominant_eigenvector_estimate)))
    error_norm = gs.linalg.norm(
        dominant_eigenvector - dominant_eigenvector_estimate)
    logging.info('l2-norm of difference vector: {}'.format(error_norm))
    logging.info('solution found: %s' % gs.isclose(error_norm, 0.0, atol=1e-3))
