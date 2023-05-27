import geomstats.backend as gs
from geomstats.geometry.base import ComplexVectorSpace, VectorSpace
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.geometry.hypersphere import _Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.vectorization import get_n_points


def _get_random_tangent_vec(space, base_point):
    n_points = get_n_points(space, base_point)
    batch_shape = (n_points,) if n_points > 1 else ()
    vec = gs.random.uniform(size=batch_shape + space.shape)
    return space.to_tangent(vec, base_point)


def _get_random_tangent_vec_hypersphere_intrinsic(space, base_point):
    n_points = get_n_points(space, base_point)
    batch_shape = (n_points,) if n_points > 1 else ()
    return gs.random.uniform(size=batch_shape + (space.dim,))


def _get_random_tangent_vec_vector_space(space, base_point):
    n_points = get_n_points(space, base_point)
    vec = space.random_point(n_points)
    return space.to_tangent(vec, base_point)


def _get_random_tangent_vec_from_embedding_space(space, base_point):
    n_points = get_n_points(space, base_point)
    vec = space.embedding_space.random_point(n_points)
    return space.to_tangent(vec, base_point)


def get_random_tangent_vec(space, base_point):
    if isinstance(space, _Hypersphere) and space.default_coords_type == "intrinsic":
        return _get_random_tangent_vec_hypersphere_intrinsic(space, base_point)

    if isinstance(space, SPDMatrices):
        return space.random_tangent_vec(base_point)

    if isinstance(space, (VectorSpace, ComplexVectorSpace)):
        return _get_random_tangent_vec_vector_space(space, base_point)

    if hasattr(space, "embedding_space"):
        return _get_random_tangent_vec_from_embedding_space(space, base_point)

    return _get_random_tangent_vec(space, base_point)


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
    def __init__(self, space, amplitude=1.0):
        self.space = space
        self.amplitude = amplitude

    def random_point(self, n_points=1):
        return self.space.random_point(n_points)

    def random_tangent_vec(self, base_point):
        return get_random_tangent_vec(self.space, base_point) / self.amplitude


class VectorSpaceRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return self.random_point(n_points)


class EmbeddedSpaceRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return self.space.embedding_space.random_point(n_points)


class DiscreteCurvesRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return Matrices(
            self.space.k_sampling_points, self.space.ambient_manifold.dim
        ).random_point(n_points)


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


class FiberBundleRandomDataGenerator(RandomDataGenerator):
    def __init__(self, total_space, base):
        super().__init__(space=total_space)
        self.base = base

    def base_random_point(self, n_points=1):
        return self.base.random_point(n_points)

    def base_random_tangent_vec(self, base_point):
        return get_random_tangent_vec(self.base, base_point)


class GammaRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    def random_point_standard(self, n_points=1):
        return self.space.natural_to_standard(self.random_point(n_points=n_points))

    def random_tangent_vec_standard(self, base_point):
        base_point_natural = self.space.standard_to_natural(base_point)
        tangent_vec_natural = self.random_tangent_vec(base_point_natural)
        return self.space.tangent_natural_to_standard(
            tangent_vec_natural, base_point_natural
        )
