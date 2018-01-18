"""
Base classes for all abstract manifolds.
The OOP structure is inspired by Andrea Censi
in his module geometry.
"""


class Manifold(object):
    """Base class for differentiable manifolds."""

    def __init__(self, dimension):
        self.dimension = dimension

    def belongs(self, point):
        """Check if the point belongs to the manifold."""
        raise NotImplementedError('belongs is not implemented.')

    def regularize(self, point):
        """
        Regularizes the point's coordinates to the canonical representation
        for this manifold.
        """
        raise NotImplementedError('regularize is not implemented.')


class RiemannianManifold(Manifold):
    """ Base class for (pseudo-/sub-) Riemannian manifolds."""

    def riemannian_inner_product(self, ref_point,
                                 tangent_vec_a, tangent_vec_b):
        """
        Compute the inner product at point ref_point
        between tangent vectors tangent_vec_a and tangent_vec_b.
        """
        raise NotImplementedError(
                'The Riemannian inner product is not implemented.')

    def riemannian_exp(self, ref_point, tangent_vec):
        """
        Compute the Riemannian exponential at point ref_point
        of tangent vector tangent_vec wrt the metric.
        """
        raise NotImplementedError(
                'The Riemannian exponential is not implemented.')

    def riemannian_log(self, ref_point, point):
        """
        Compute the Riemannian logarithm at point ref_point
        of tangent vector tangent_vec wrt the metric.
        """
        raise NotImplementedError(
                'The Riemannian logarithm is not implemented.')

    def riemannian_dist(point_a, point_b):
        """
        Compute the Riemannian distance between points
        point_a and point_b.
        """
        raise NotImplementedError(
                'The Riemannian distance is not implemented.')

    def random_uniform(self):
        """
        Samples a random point in this manifold according to
        the Riemannian measure.
        """
        raise NotImplementedError('random_uniform is not implemented.')

    def riemannian_variance(self, ref_point, points, weights):
        """
        Compute the variance of the points
        in the tangent space at the ref_point.
        """
        raise NotImplementedError(
                'The Riemannian variance is not implemented.')

    def riemannian_mean(self, points, weights):
        """
        Compute the weighted mean of the
        points.

        The geodesic distances are obtained with the
        Riemannian distance.
        """
        raise NotImplementedError(
                'The Riemannian mean is not implemented.')


class LieGroup(Manifold):
    """ Base class for Lie groups."""

    def __init__(self, dimension):
        Manifold.__init__(dimension)
        self.identity = None

    def compose(self, point_a, point_b):
        """
        Composition of the Lie group.
        """
        raise NotImplementedError('The composition is not implemented.')

    def inverse(self, point):
        """
        Inverse law of the Lie group.
        """
        raise NotImplementedError('The inverse is not implemented.')

    def jacobian_translation(point, left_or_right='left'):
        """
        Compute the jacobian matrix of the differential
    of the left translation by the point.
        """
        raise NotImplementedError(
               'The jacobian of the Lie group translation is not implemented.')

    def group_exp(self, point, tangent_vec):
        """
        Compute the group exponential at point ref_point
        of tangent vector tangent_vec.
        """
        raise NotImplementedError(
                'The group exponential is not implemented.')

    def group_log(self, ref_point, point):
        """
        Compute the group logarithm at point ref_point
        of the point point.
        """
        raise NotImplementedError(
                'The group logarithm is not implemented.')
