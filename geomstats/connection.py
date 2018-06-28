"""
Affine connections.
"""

import geomstats.backend as gs


class Connection(object):

    def __init__(self, dimension):
        self.dimension = dimension

    def christoffel_symbol(self, base_point):
        """
        Christoffel symbols associated with the connection.
        """
        raise NotImplementedError(
                'The Christoffel symbols are not implemented.')

    def parallel_transport(self, tangent_vector_a, tangent_vector_b):
        """
        Parallel transport associated with the connection.
        """
        raise NotImplementedError(
                'Parallel transport is not implemented.')

    def riemannian_curvature(self, base_point):
        """
        Riemannian curvature tensor associated with the connection.
        """
        raise NotImplementedError(
                'The Riemannian curvature tensor is not implemented.')

    def geodesic_equation(self):
        """
        The geodesic ordinary differential equation associated
        with the connection.
        """
        raise NotImplementedError(
                'The geodesic equation tensor is not implemented.')

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None, point_ndim=1):
        """
        Geodesic associated with the connection.
        """
        # TODO(nina): integrate the ODE
        raise NotImplementedError(
                'Geodesics are not implemented.')

    def torsion(self, base_point):
        """
        Torsion tensor associated with the connection.
        """
        raise NotImplementedError(
                'The torsion tensor is not implemented.')


class LeviCivitaConnection(Connection):
    """
    Levi-Civita connection associated with a Riemannian metric.
    """
    def __init__(self, metric):
        self.metric = metric
        self.dimension = metric.dimension

    def christoffel_symbols(self, base_point):
        """
        Christoffel symbols associated with the connection.
        """
        # TODO(nina): implement with automatic differentiation.
        raise NotImplementedError(
                'The Christoffel symbols are not implemented.')

    def torsion(self, base_point):
        """
        Torsion tensor associated with the Levi-Civita connection is zero.
        """
        # TODO(nina)
        return gs.zeros((self.dimension,) * 3)
