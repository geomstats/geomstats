import geomstats.backend as gs
from geomstats.geometry.base import ComplexMatrixVectorSpace
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.vectorization import get_n_points


def get_random_quaternion(n_points=1):
    # https://stackoverflow.com/a/44031492/11011913
    size = (3, n_points) if n_points > 1 else 3
    u, v, w = gs.random.uniform(size=size)

    return gs.transpose(
        gs.array(
            [
                gs.sqrt(1 - u) * gs.sin(2 * gs.pi * v),
                gs.sqrt(1 - u) * gs.cos(2 * gs.pi * v),
                gs.sqrt(u) * gs.sin(2 * gs.pi * w),
                gs.sqrt(u) * gs.cos(2 * gs.pi * w),
            ]
        )
    )


def get_random_times(n_times, sort=True):
    if n_times == 1:
        return gs.random.rand(1)[0]

    time = gs.random.rand(n_times)
    if sort:
        return gs.sort(time)

    return time


class RandomDataGenerator:
    def __init__(self, space, amplitude=2.0):
        self.space = space
        self.amplitude = amplitude

    def random_point(self, n_points=1):
        return self.space.random_point(n_points)

    def random_tangent_vec(self, base_point):
        tangent_vec = self.space.random_tangent_vec(base_point)
        return tangent_vec / self.amplitude


class VectorSpaceRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return self.random_point(n_points)


class MatrixVectorSpaceRandomDataGenerator(VectorSpaceRandomDataGenerator):
    def random_basis_representation(self, n_points=1):
        dtype = (
            gs.get_default_cdtype()
            if isinstance(self, ComplexMatrixVectorSpace)
            else None
        )

        if n_points == 1:
            return gs.random.rand(self.space.dim, dtype=dtype)

        return gs.reshape(
            gs.random.rand(n_points * self.space.dim),
            (n_points, -1),
            dtype=dtype,
        )


class EmbeddedSpaceRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return self.space.embedding_space.random_point(n_points)


class NFoldManifoldRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        base = self.space.base_manifold
        if not hasattr(base, "embedding_space"):
            raise NotImplementedError("Can't get point to project.")

        n_copies = self.space.n_copies
        point = base.embedding_space.random_point(n_points * n_copies)

        shape = (n_points, n_copies) if n_points > 1 else (n_copies,)
        return gs.reshape(point, shape + base.shape)


class HypersphereIntrinsicRandomDataGenerator(RandomDataGenerator):
    def random_tangent_vec(self, base_point):
        n_points = get_n_points(self.space.point_ndim, base_point)
        batch_shape = (n_points,) if n_points > 1 else ()
        return gs.random.uniform(size=batch_shape + (self.space.dim,)) / self.amplitude


class RankKPSDMatricesRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, amplitude=1.0):
        super().__init__(space, amplitude=amplitude)
        self._square = SquareMatrices(self.space.n)

    def point_to_project(self, n_points=1):
        return self._square.random_point(n_points)


class LieGroupVectorRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, amplitude=1.0):
        super().__init__(space, amplitude=amplitude)
        self._euclidean = Euclidean(self.space.dim, equip=False)

    def point_to_project(self, n_points=1):
        return self._euclidean.random_point(n_points)


class KendalShapeRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    def random_horizontal_vec(self, base_point):
        tangent_vec = self.random_tangent_vec(base_point)
        fiber_bundle = self.space.metric.fiber_bundle
        return (
            fiber_bundle.horizontal_projection(tangent_vec, base_point) / self.amplitude
        )


class GammaRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    def random_point_standard(self, n_points=1):
        return self.space.natural_to_standard(self.random_point(n_points=n_points))

    def random_tangent_vec_standard(self, base_point):
        base_point_natural = self.space.standard_to_natural(base_point)
        tangent_vec_natural = self.random_tangent_vec(base_point_natural)
        return (
            self.space.tangent_natural_to_standard(
                tangent_vec_natural, base_point_natural
            )
            / self.amplitude
        )


class HeisenbergVectorsRandomDataGenerator(VectorSpaceRandomDataGenerator):
    def random_upper_triangular_matrix(self, n_points=1):
        if n_points == 1:
            size = 3
            expected_shape = (3, 3)
            indices = [(0, 1), (0, 2), (1, 2)]
        else:
            size = n_points * 3
            expected_shape = (n_points, 3, 3)
            indices = []
            for k in range(n_points):
                indices.extend([(k, 0, 1), (k, 0, 2), (k, 1, 2)])

        vec = gs.random.uniform(size=size)
        return gs.array_from_sparse(indices, vec, expected_shape) + gs.eye(3)


class ShapeBundleRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, n_discretized_curves=5):
        super().__init__(space)
        self.n_discretized_curves = n_discretized_curves

    def random_tangent_vec(self, base_point):
        n_points = base_point.shape[0] if base_point.ndim > 2 else 1
        point = self.random_point(n_points=n_points)

        geo = self.space.metric.geodesic(initial_point=base_point, end_point=point)

        times = gs.linspace(0.0, 1.0, self.n_discretized_curves)

        geod = geo(times)

        return (
            self.n_discretized_curves * (geod[..., 1, :, :] - geod[..., 0, :, :])
        ) / self.amplitude
