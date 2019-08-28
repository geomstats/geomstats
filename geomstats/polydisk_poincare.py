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



class PoincarePolydisk(RiemannianMetric):
    """
    Class for the Riemannian Poincare metric.
    """
    def __init__(self, dimension):
        super(PoincareMetric, self).__init__(
                                          dimension=dimension,
                                          signature=(dimension, 0, 0))

    def inner_product_matrix(self, base_point):
        """
        Inner product matrix, independent of the base point.
        """
        inner_prod_mat = gs.eye(self.dimension-1, self.dimension-1)
        first_row = gs.array([0.] * (self.dimension - 1))
        first_row = gs.to_ndarray(first_row, to_ndim=2, axis=1)
        inner_prod_mat = gs.vstack([gs.transpose(first_row),
                                    inner_prod_mat])

        first_column = gs.array([-1.] + [0.] * (self.dimension - 1))
        first_column = gs.to_ndarray(first_column, to_ndim=2, axis=1)
        inner_prod_mat = gs.hstack([first_column,
                                    inner_prod_mat])

        diag = (self.dimension - gs.arange(self.dimension)) / (1 - gs.abs(base_point)**2)**2
        inner_prod_mat = gs.diag(diag)

        return inner_prod_mat

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential in the Poincare polydisk.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        return base_point + tangent_vec

    def exp(self, tangent_vec, base_point, epsilon=1e-16):
        theta = gs.angle(tangent_vec, deg=False)
        s = 2 * gs.abs(tangent_vec) / gs.maximum((1 - gs.abs(base_point) ** 2), epsilon)
        return (base_point + gs.exp(1j * theta) + (base_point - gs.exp(1j * theta)) * gs.exp(-s)) / (
                    1 + gs.conjugate(base_point) * gs.exp(1j * theta) + (1 - gs.conjugate(base_point) * gs.exp(1j * theta)) * gs.exp(-s))

    def log(self, point, base_point):
        """
        The Riemannian logarithm in the Poincare polydisk.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        return point - base_point

    def tau(self, z1, z2, epsilon=1e-16):
        delta = gs.abs((z2 - z1)) / max(gs.abs(1 - z1.conj() * z2), epsilon)
        delta = min(delta, 1 - epsilon)
        return (1 / 2) * gs.log((1 + delta) / (1 - delta))

    def log_map_kahler_mu(self, point, base_point):
        dimension = base_point.shape[0]
        v = gs.zeros([dimension], dtype=complex)
        for j in range(dimension):
            v[j] = tau(base_point[j], point[j]) * gs.exp(1j * (gs.angle(point[j] - base_point[j]) - gs.angle(1 - base_point[j].conj() * point[j]))) * (
                        1 - gs.abs(base_point[j]) ** 2)
        return v


    def mean(self, points, weights=None):
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

        weighted_points = gs.einsum('n,nj->nj', weights, points)
        mean = (gs.sum(weighted_points, axis=0)
                / gs.sum(weights))
        mean = gs.to_ndarray(mean, to_ndim=2)
        return mean

    def atanh(self, z):
        return (1 / 2) * gs.log((1 + z) / (1 - z))

    def kahler_mean_mu(data, nbr_rec=100):
        points_nbr = data.shape[0]
        if points_nbr == 0:
            raise ValueError("A class is empty.")
        space_dimension = data.shape[1]
        kahler_mean_vector = gs.mean(data, axis=0)
        for i in range(space_dimension):
            for j in range(nbr_rec):
                vk = 0
                for l in range(points_nbr):
                    cik = (data[l, i] - kahler_mean_vector[i]) / (1 - gs.conjugate(kahler_mean_vector[i]) * data[l, i])
                    vk += gs.sign(cik) * atanh(gs.abs(cik))
                vk *= (1 - gs.abs(kahler_mean_vector[i]) ** 2)
                kahler_mean_vector[i] = exp(kahler_mean_vector[i], 1 / (j + 1000) * vk)
        return kahler_mean_vector
