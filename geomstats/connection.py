"""
Affine connections.
"""

import autograd

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

    def connection(self, tangent_vector_a, tangent_vector_b, base_point):
        """
        Connection applied to tangent_vector_b in the direction of
        tangent_vector_a, both tangent at base_point.
        """
        raise NotImplementedError(
                'connection is not implemented.')

    def exp(self, tangent_vector_a, base_point):
        """
        Connection applied to tangent_vector_b in the direction of
        tangent_vector_a, both tangent at base_point.
        """
        raise NotImplementedError(
                'The affine connection exponential is not implemented.')

    def log(self, point, base_point):
        """
        Connection applied to tangent_vector_b in the direction of
        tangent_vector_a, both tangent at base_point.
        """
        raise NotImplementedError(
                'The affine connection logarithm is not implemented.')

    def pole_ladder_transport(
            self, tangent_vector_a, tangent_vector_b, base_point):
        """
        One step of pole ladder (parallel transport associated with the
        symmetric part of the connection using transvections).
        """
        half_tangent_vector_b = 1. / 2. * tangent_vector_b
        mid_point = self.exp(
                base_point=base_point,
                tangent_vector=half_tangent_vector_b)

        mid_tangent_vector = - self.log(
                base_point=mid_point,
                point=base_point)
        end_point = self.exp(
                base_point=mid_point,
                tangent_vector=mid_tangent_vector)

        base_shoot = self.exp(
                base_point=base_point,
                tangent_vector=tangent_vector_a)
        mid_tangent_vector_to_shoot = - self.log(
                base_point=mid_point,
                end_point=base_shoot)
        end_shoot = self.exp(
                base_point=mid_point,
                tangent_vector=mid_tangent_vector_to_shoot)

        tangent_vector = - self.log(base_point=end_point, point=end_shoot)
        return tangent_vector

    def parallel_transport(
            self, tangent_vector_a, tangent_vector_b, base_point, n_points=1):
        """
        Parallel transport of tangent vector a integrating the connection
        along the (affine connection) geodesic starting at the initial point
        base_point with initial tangent vector the tangent vector b.

        Returns a tangent vector at the point
        exp_(base_point)(tangent_vector_b).
        """
        current_point = gs.copy(base_point)
        geodesic_tangent_vector = 1. / n_points * tangent_vector_b
        transported_tangent_vector = gs.copy(tangent_vector_a)
        for i_point in range(1, n_points):
            transported_tangent_vector = self.pole_ladder_transport(
                tangent_vector_a=transported_tangent_vector,
                tangent_vector_b=geodesic_tangent_vector,
                base_point=current_point)
            current_point = self.exp(
                base_point=current_point,
                tangent_vector=geodesic_tangent_vector)

            frac_tangent_vector_b = (i_point + 1) / n_points * tangent_vector_b
            next_point = self.exp(
                base_point=base_point,
                tangent_vector=frac_tangent_vector_b)
            geodesic_tangent_vector = self.log(
                base_point=current_point,
                point=next_point)

        return transported_tangent_vector

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

    def metric_matrix(self, base_point):
        metric_matrix = self.metric.inner_product_matrix(base_point)
        return metric_matrix

    def cometric_matrix(self, base_point):
        """
        The cometric is the inverse of the metric.
        """
        metric_matrix = self.metric_matrix(base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix)
        return cometric_matrix

    def metric_derivative(self, base_point):
        # TODO(nina): same operation without autograd package?
        metric_derivative = autograd.jacobian(self.metric_matrix)
        return metric_derivative(base_point)

    def christoffel_symbols(self, base_point):
        """
        Christoffel symbols associated with the connection.
        """
        term_1 = gs.einsum('nim,nmkl->nikl',
                           self.cometric_matrix(base_point),
                           self.metric_derivative(base_point))
        term_2 = gs.einsum('nim,nmlk->nilk',
                           self.cometric_matrix(base_point),
                           self.metric_derivative(base_point))
        term_3 = - gs.einsum('nim,nklm->nikl',
                             self.cometric_matrix(base_point),
                             self.metric_derivative(base_point))

        christoffel_symbols = 0.5 * (term_1 + term_2 + term_3)
        return christoffel_symbols

    def torsion(self, base_point):
        """
        Torsion tensor associated with the Levi-Civita connection is zero.
        """
        # TODO(nina)
        return gs.zeros((self.dimension,) * 3)
