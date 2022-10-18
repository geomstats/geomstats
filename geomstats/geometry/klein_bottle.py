import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class KleinBottleMetric(RiemannianMetric):
    def __init__(self, **kwargs):
        super().__init__(dim=2, shape=(2,), default_coords_type="intrinsic")


class KleinBottle(Manifold):
    def __init__(self, dim=2, shape=2):
        super().__init__(dim=2, shape=(2,), metric=None)

    def random_point(self, n_samples=1, bound=None):
        """Uniformly sample points on the manifold.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : unused

        Returns
        -------
        samples : array-like, shape=[n_samples, 2]
            Points sampled on the manifold.
        """
        samples = gs.random.uniform(size=(n_samples, 2))
        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)
        return samples

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the manifold. In this case returns the vector itself.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : unused

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        if gs.all(self.is_tangent(vector)):
            return vector
        raise ValueError("Vector has the wrong shape.")

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Evaluate if the point belongs to :math:`\mathbb{R}^2`.

        This method checks the shape and of the input point.

        Parameters
        ----------
        vector : array-like, shape=[...]
            Point to test.
        base_point: unused here
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the tangent space.
        """
        # copied from VectorSpace
        vector = gs.array(vector)
        minimal_ndim = len(self.shape)
        if self.shape[0] == 1 and len(vector.shape) <= 1:
            vector = gs.transpose(gs.to_ndarray(gs.to_ndarray(vector, 1), 2))
        belongs = vector.shape[-minimal_ndim:] == self.shape
        if vector.ndim <= minimal_ndim:
            return belongs
        return gs.tile(gs.array([belongs]), [vector.shape[0]])

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the set [0,1]^2.

        This method checks the shape and values of the input point.

        Parameters
        ----------
        point : array-like, shape=[...]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        point = gs.array(point)
        minimal_ndim = len(self.shape)
        correct_shape = point.shape[-minimal_ndim:] == self.shape
        if correct_shape:
            correct_values = gs.logical_and(point >= 0, point <= 1)
            return correct_values
        if point.ndim > minimal_ndim:
            return gs.tile(gs.array([correct_shape]), [point.shape[0]])

    @staticmethod
    def equivalent(point1, point2, atol=gs.atol):
        """Evaluate whether two points in :math:`\mathbb R^2` represent equivalent points on the Klein Bottle.

        This method uses the equivalence :math:`(x_1,y_1) \sim (x_2, y_2) \Leftrightarrow
        y_1-y_2 \in \mathbb Z \text{ and } x_1-x_2 \in 2\mathbb Z \text{ or }
        y_1 + y_2 \in \mathbb Z \text{ and } x_1-x_2 \in \mathbb Z\setminus 2\mathbb Z`

        This means points are equivalent if one walks an even number of squares in x-direction and any number of squares
        in y-direction or an uneven number of squares in x-direction and any number of squares in y-direction while
        "mirroring" the y coordinate.

        Parameters
        ----------
        point1: array-like, shape=[..., 2]
            Representation of point on Klein Bottle
        point2: array-like, shape=[..., 2]
            Representation of point on Klein Bottle
        atol: Absolute tolerance to test for belonging to \mathbb Z.

        Returns
        -------
        is_equivalent : array-like, shape=[..., 2]
            Boolean evaluating if points are equivalent
        """
        x_diff = point1[..., 0] - point2[..., 0]
        y_diff = point1[..., 1] - point2[..., 1]
        y_sum = point1[..., 1] + point2[..., 1]
        un_mirrored = gs.logical_and(
            is_close_mod(gs.mod(x_diff, 2.0), 2.0, atol=atol),
            is_close_mod(gs.mod(y_diff, 1.0), 1.0, atol=atol),
        )
        mirrored = gs.logical_and(
            is_close_mod(gs.mod(x_diff - 1, 2.0), 2.0, atol=atol),
            is_close_mod(gs.mod(y_sum, 1.0), 1.0, atol=atol),
        )
        return gs.logical_or(un_mirrored, mirrored)

    def regularize(self, point):
        """Regularize any point in :math:`\mathbb R^2` to its canonical equivalent representation in the square
        :math:`[0,1]^2``

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., 2]
            Regularized point.
        """
        # determine number of steps to take in x direction
        num_steps = gs.cast(gs.abs(gs.floor(point[..., 0])), gs.int64)
        x_canonical = gs.remainder(point[..., 0], 1)
        y_canonical_even = gs.remainder(point[..., 1], 1)
        y_canonical_odd = 1 - y_canonical_even
        point_even = gs.stack([x_canonical, y_canonical_even], axis=-1)
        point_odd = gs.stack([x_canonical, y_canonical_odd], axis=-1)
        return gs.where(gs.remainder(num_steps, 2) == 0, point_even, point_odd)


def is_close_mod(array, divisor, atol):
    return gs.logical_or(
        gs.isclose(array, 0.0, atol), gs.isclose(gs.abs(array), divisor, atol)
    )
