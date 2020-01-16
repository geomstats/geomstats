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

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension]
        """
        raise NotImplementedError(
                'The Christoffel symbols are not implemented.')

    def connection(self, tangent_vector_a, tangent_vector_b, base_point):
        """
        Connection applied to tangent_vector_b in the direction of
        tangent_vector_a, both tangent at base_point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        tangent_vec_b: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
                'connection is not implemented.')

    def exp(self, tangent_vec, base_point):
        """
        Exponential map associated to the affine connection.

        Parameters
        ----------
        tangent_vec: array-like, shape=[n_samples, dimension]
                                 or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
                'The affine connection exponential is not implemented.')

    def log(self, point, base_point):
        """
        Logarithm map associated to the affine connection.

        Parameters
        ----------
        point: array-like, shape=[n_samples, dimension]
                           or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        raise NotImplementedError(
                'The affine connection logarithm is not implemented.')

    def pole_ladder_step(self, base_point, next_point, base_shoot):
        """
        One step of pole ladder (parallel transport associated with the
        symmetric part of the connection using transvections).

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        next_point: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        base_shoot: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        transported_tangent_vector: array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]

        end_point: array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]
        """
        mid_tangent_vector_to_shoot = 1. / 2. * self.log(
                base_point=base_point,
                point=next_point)

        mid_point = self.exp(
                base_point=base_point,
                tangent_vec=mid_tangent_vector_to_shoot)

        tangent_vector_to_shoot = - self.log(
                base_point=mid_point,
                point=base_shoot)

        end_shoot = self.exp(
                base_point=mid_point,
                tangent_vec=tangent_vector_to_shoot)

        transported_tangent_vector = - self.log(
            base_point=next_point, point=end_shoot)

        end_point = self.exp(
                base_point=next_point,
                tangent_vec=transported_tangent_vector)

        return transported_tangent_vector, end_point

    def pole_ladder_parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, n_steps=1):
        """
        Approximation of Parallel transport using the pole ladder scheme
        of tangent vector a along the geodesic starting at the initial point
        base_point with initial tangent vector the tangent vector b.

        Returns a tangent vector at the point
        exp_(base_point)(tangent_vector_b).

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        tangent_vec_b: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        n_steps: int, the number of pole ladder steps

        Returns
        -------
        transported_tangent_vector: array-like, shape=[n_samples, dimension]
                                                or shape=[1, dimension]
        """

        current_point = gs.copy(base_point)
        transported_tangent_vector = gs.copy(tangent_vec_a)
        base_shoot = self.exp(base_point=current_point,
                              tangent_vec=transported_tangent_vector)
        for i_point in range(0, n_steps):
            frac_tangent_vector_b = (i_point + 1) / n_steps * tangent_vec_b
            next_point = self.exp(
                base_point=base_point,
                tangent_vec=frac_tangent_vector_b)
            transported_tangent_vector, base_shoot = self.pole_ladder_step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot)
            current_point = next_point

        return transported_tangent_vector

    def riemannian_curvature(self, base_point):
        """
        Riemannian curvature tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
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
        raise NotImplementedError(
                'Geodesics are not implemented.')

    def torsion(self, base_point):
        """
        Torsion tensor associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
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
        """
        Metric matrix defining the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        metric_matrix: array-like, shape=[n_samples, dimension, dimension]
                                   or shape=[1, dimension, dimension]
        """
        metric_matrix = self.metric.inner_product_matrix(base_point)
        return metric_matrix

    def cometric_matrix(self, base_point):
        """
        The cometric is the inverse of the metric.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        cometric_matrix: array-like, shape=[n_samples, dimension, dimension]
                                     or shape=[1, dimension, dimension]
        """
        metric_matrix = self.metric_matrix(base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix)
        return cometric_matrix

    def metric_derivative(self, base_point):
        """

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        metric_derivative = autograd.jacobian(self.metric_matrix)
        return metric_derivative(base_point)

    def christoffel_symbols(self, base_point):
        """
        Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        christoffel_symbols: array-like,
                             shape=[n_samples, dimension, dimension, dimension]
                             or shape=[1, dimension, dimension, dimension]
        """
        cometric_mat_at_point = self.cometric_matrix(base_point)
        metric_derivative_at_point = self.metric_derivative(base_point)
        term_1 = gs.einsum('nim,nmkl->nikl',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_2 = gs.einsum('nim,nmlk->nilk',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_3 = - gs.einsum('nim,nklm->nikl',
                             cometric_mat_at_point,
                             metric_derivative_at_point)

        christoffel_symbols = 0.5 * (term_1 + term_2 + term_3)
        return christoffel_symbols

    def torsion(self, base_point):
        """
        Torsion tensor associated with the Levi-Civita connection is zero.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        torsion: array-like, shape=[dimension, dimension, dimension]
        """
        torsion = gs.zeros((self.dimension,) * 3)
        return torsion

    def exp(self, tangent_vec, base_point):
        """
        Exponential map associated to the metric.

        Parameters
        ----------
        tangent_vec: array-like, shape=[n_samples, dimension]
                                 or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        return self.metric.exp(tangent_vec, base_point)

    def log(self, point, base_point):
        """
        Logarithm map associated to the metric.

        Parameters
        ----------
        point: array-like, shape=[n_samples, dimension]
                           or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        return self.metric.log(point, base_point)
