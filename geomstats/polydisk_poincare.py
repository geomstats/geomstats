"""
The Poincare polydisk
"""


from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric
import geomstats.backend as gs


class PoincarePolydisk(Manifold):
    """Class for the Poincare polydisk."""
    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        self.dimension = dimension
        self.metric = PoincarePolydisk(dimension)

    def belongs(self, point):
        """
        Evaluate if a point belongs to the Poincare polydisk.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        belongs = gs.all(gs.less_equal(gs.abs(point), 1.), axis=1)
        return belongs


class PoincarePolydiskMetric(RiemannianMetric):
    """
    Class for the Riemannian Poincare metric.
    """
    def __init__(self, dimension):
        super(PoincarePolydiskMetric, self).__init__(
                                          dimension=dimension,
                                          signature=(dimension, 0, 0))

    def inner_product_matrix(self, base_point):
        """
        Inner product matrix, independent of the base point.
        """
        diag = (self.dimension - gs.arange(self.dimension)) / (1 - gs.abs(base_point)**2)**2
        inner_prod_mat = gs.diag(diag)
        return inner_prod_mat

    def squared_dist(self, point_a, point_b, epsilon=1e-16):
        """
        Squared geodesic distance between two points.
        Parameters
        ----------
        point_a: array-like, shape=[n_samples, dimension]
                             or shape=[1, dimension]
        point_b: array-like, shape=[n_samples, dimension]
                             or shape=[1, dimension]
        epsilon: security value not to divide by zero
        """
        # log = self.log(point=point_b, base_point=point_a)
        # sq_dist = self.squared_norm(vector=log, base_point=point_a)

        coeff_nbr = point_a.shape[-1]
        sweep_nbr = coeff_nbr + 1
        sq_dist = 0
        for i in range(coeff_nbr):
            delta = gs.linalg.norm((point_a[i] - point_b[i])) / max(gs.linalg.norm(1 - point_a[i] * gs.conjugate(point_b[i])), epsilon)
            delta = min(delta, 1 - epsilon)
            sq_dist += (sweep_nbr - (i + 1)) * (1 / 2 * gs.log((1 + delta) / (1 - delta))) ** 2
        return sq_dist

    def exp(self, tangent_vec, base_point, epsilon=1e-16):
        """
        The Riemannian exponential in the Poincare polydisk.
        """
        # tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        # base_point = gs.to_ndarray(base_point, to_ndim=2)
        theta = gs.angle(tangent_vec, deg=False)
        s = 2 * gs.abs(tangent_vec) / gs.maximum((1 - gs.abs(base_point) ** 2), epsilon)
        return (base_point + gs.exp(1j * theta) + (base_point - gs.exp(1j * theta)) * gs.exp(-s)) / (
                    1 + gs.conjugate(base_point) * gs.exp(1j * theta) + (1 - gs.conjugate(base_point) * gs.exp(1j * theta)) * gs.exp(-s))

    def tau(self, z1, z2, epsilon=1e-16):
        delta = gs.abs((z2 - z1)) / max(gs.abs(1 - z1.conj() * z2), epsilon)
        delta = min(delta, 1 - epsilon)
        return (1 / 2) * gs.log((1 + delta) / (1 - delta))

    def log(self, point, base_point):
        """
        The Riemannian logarithm in the Poincare polydisk.
        """
        # point = gs.to_ndarray(point, to_ndim=2)
        # base_point = gs.to_ndarray(base_point, to_ndim=2)
        dimension = self.dimension
        tangent_vector = gs.zeros([dimension], dtype=complex)
        for j in range(dimension):
            complex_angle = gs.angle(point[j] - base_point[j]) - gs.angle(1 - base_point[j].conj() * point[j])
            complex_norm = tau(base_point[j], point[j]) * (1 - gs.abs(base_point[j]) ** 2)
            tangent_vector[j] = complex_norm * gs.exp(1j * complex_angle)
        return tangent_vector

    def atanh(self, z):
        return (1 / 2) * gs.log((1 + z) / (1 - z))

    def mean(self, points, weights=None, nbr_rec=100, epsilon=1e-16):
        """
        The Frechet mean of (weighted) points is the weighted average of
        the points in the Poincare polydisk.
        """
        if isinstance(points, list):
            points = gs.vstack(points)
        points = gs.to_ndarray(points, to_ndim=2)
        n_points = gs.shape(points)[0]

        if isinstance(weights, list):
            weights = gs.vstack(weights)
        elif weights is None:
            weights = gs.ones((n_points,))

        # weighted_points = gs.einsum('n,nj->nj', weights, points)
        # mean = (gs.sum(weighted_points, axis=0)
        #         / gs.sum(weights))
        # mean = gs.to_ndarray(mean, to_ndim=2)

        points_nbr = points.shape[0]
        if points_nbr == 0:
            raise ValueError("A class is empty.")
        dimension = self.dimension
        mean = gs.mean(points, axis=0)

        sum_distances_power_2 = 0
        for k in range(points_nbr):
            if not (mean == points[k, :]).all():
                sum_distances_power_2 += distance(mean, data[k, :, :]) ** 2
        step_size = sum_distances_power_2 ** (1 / 2) / 4

        for i in range(dimension):
            for j in range(nbr_rec):
                vk = 0
                for l in range(points_nbr):
                    cik = (points[l, i] - mean[i]) / (1 - gs.conjugate(mean[i]) * points[l, i])
                    gradient_vector += weights[l] * gs.sign(cik) * atanh(gs.abs(cik))
                gradient_vector *= (1 - gs.abs(mean[i]) ** 2)
                gradient_vector /= max(distance(mean, exp(mean, gradient_vector)), epsilon)
                mean[i] = exp(mean[i], step_size * vk)

            new_sum_distances_power_2 = 0
            for k in range(points_nbr):
                if not (mean == points[k, :]).all():
                    new_sum_distances_power_2 += distance(mean, points[k, :]) ** 2
            if new_sum_distances_power_2 >= sum_distances_power_2:
                step_size /= 2
            sum_distances_power_2 = new_sum_distances_power_2

        return mean