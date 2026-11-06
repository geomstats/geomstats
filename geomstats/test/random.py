from functools import singledispatch

import geomstats.backend as gs
from geomstats.geometry.base import (
    ComplexMatrixVectorSpace,
    DiffeomorphicManifold,
    DiffeomorphicMatrixVectorSpace,
    DiffeomorphicVectorSpace,
    LevelSet,
    MatrixVectorSpace,
    OpenSet,
    VectorSpace,
)
from geomstats.geometry.diffeo import ReversedDiffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.geometry.manifold import Manifold
from geomstats.vectorization import repeat_point


def get_random_quaternion(n_points=1):
    # https://stackoverflow.com/a/44031492/11011913
    size = (3, n_points) if n_points > 1 else 3
    u, v, w = gs.random.uniform(size=size)

    return gs.stack(
        [
            gs.sqrt(1 - u) * gs.sin(2 * gs.pi * v),
            gs.sqrt(1 - u) * gs.cos(2 * gs.pi * v),
            gs.sqrt(u) * gs.sin(2 * gs.pi * w),
            gs.sqrt(u) * gs.cos(2 * gs.pi * w),
        ],
        axis=-1,
    )


def get_random_times(n_times, sort=True):
    if n_times == 1:
        return gs.random.rand(1)[0]

    time = gs.random.rand(n_times)
    if sort:
        return gs.sort(time)

    return time


class RandomDataGenerator:
    """Random data generator for a space.

    Parameters
    ----------
    space : Manifold
        Space to sample from.
    amplitude : float
        Random tangent vectors are divided by it.
    near_scale : float
        Scale of the tangent vector used by ``random_point_near``.
    """

    def __init__(self, space, amplitude=2.0, near_scale=0.1):
        self.space = space
        self.amplitude = amplitude
        self.near_scale = near_scale

    def random_point(self, n_points=1):
        return self.space.random_point(n_points)

    def random_tangent_vec(self, base_point):
        tangent_vec = self.space.random_tangent_vec(base_point)
        return tangent_vec / self.amplitude

    def random_point_near(self, base_point):
        """Sample a point near ``base_point`` using the metric's exponential.

        The tangent vector used to move away from ``base_point`` is scaled
        by ``self.near_scale``.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point around which to sample.

        Returns
        -------
        point : array-like, shape=[..., *space.shape]
            Point near ``base_point``.
        """
        tangent_vec = self.near_scale * self.random_tangent_vec(base_point)
        return self.space.metric.exp(tangent_vec, base_point)

    def random_tangent_vec_at_identity(self, n_points=1):
        """Sample tangent vectors at the identity of a group.

        Parameters
        ----------
        n_points : int
            Number of tangent vectors.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vectors at the identity.
        """
        identity = repeat_point(self.space.identity, n_points)
        return self.random_tangent_vec(identity)


class RandomDatasetGenerator:
    def __init__(self, space, image_space=None):
        self.space = space
        self.image_space = image_space

    def random_point(self, n_points=1):
        return self.space.random_point(n_points)

    def random_image_point(self, n_points=1):
        return self.image_space.random_point(n_points)


class VectorSpaceRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return self.random_point(n_points)


class MatrixVectorSpaceRandomDataGenerator(VectorSpaceRandomDataGenerator):
    def random_basis_representation(self, n_points=1):
        dtype = (
            gs.get_default_cdtype()
            if isinstance(self.space, ComplexMatrixVectorSpace)
            else None
        )

        if n_points == 1:
            return gs.random.rand(self.space.dim, dtype=dtype)

        return gs.reshape(
            gs.random.rand(n_points * self.space.dim, dtype=dtype),
            (n_points, -1),
        )


@singledispatch
def get_data_generator(space, **kwargs):
    """Get the random data generator registered for a space.

    Parameters
    ----------
    space : Manifold
        Space to sample from.
    **kwargs
        Keyword arguments passed to the data generator constructor,
        e.g. ``amplitude`` or ``near_scale``.

    Returns
    -------
    data_generator : RandomDataGenerator
        Data generator for ``space``.
    """
    raise NotImplementedError(
        f"No data generator registered for {type(space).__name__}."
    )


@get_data_generator.register(Manifold)
def _(space, **kwargs):
    return RandomDataGenerator(space, **kwargs)


@get_data_generator.register(VectorSpace)
def _(space, **kwargs):
    return VectorSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(MatrixVectorSpace)
def _(space, **kwargs):
    return MatrixVectorSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(LevelSet)
@get_data_generator.register(OpenSet)
def _(space, **kwargs):
    return EmbeddedSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(DiffeomorphicManifold)
@get_data_generator.register(DiffeomorphicVectorSpace)
@get_data_generator.register(DiffeomorphicMatrixVectorSpace)
def _(space, **kwargs):
    return DiffeoBasedRandomDataGenerator(
        space.image_space, diffeo=ReversedDiffeo(space.diffeo), **kwargs
    )


class EmbeddedSpaceRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        return self.space.embedding_space.random_point(n_points)

    def random_point_near(self, base_point):
        """Sample a point near ``base_point`` by projection.

        Works without a metric: moves along a tangent vector in the embedding
        space and projects back onto the space.

        The tangent vector used to move away from ``base_point`` is scaled
        by ``self.near_scale``.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point around which to sample.

        Returns
        -------
        point : array-like, shape=[..., *space.shape]
            Point near ``base_point``.
        """
        return self.space.projection(
            base_point + self.near_scale * self.random_tangent_vec(base_point)
        )


class NFoldManifoldRandomDataGenerator(RandomDataGenerator):
    def point_to_project(self, n_points=1):
        base = self.space.base_manifold
        if not hasattr(base, "embedding_space"):
            raise NotImplementedError("Can't get point to project.")

        n_copies = self.space.n_copies
        point = base.embedding_space.random_point(n_points * n_copies)

        shape = (n_points, n_copies) if n_points > 1 else (n_copies,)
        return gs.reshape(point, shape + base.shape)


class RankKPSDMatricesRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, amplitude=1.0, near_scale=0.1):
        super().__init__(space, amplitude=amplitude, near_scale=near_scale)
        self._square = SquareMatrices(self.space.n)

    def point_to_project(self, n_points=1):
        return self._square.random_point(n_points)


class LieGroupVectorRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, amplitude=1.0, near_scale=0.1):
        super().__init__(space, amplitude=amplitude, near_scale=near_scale)
        self._euclidean = Euclidean(self.space.dim, equip=False)

    def point_to_project(self, n_points=1):
        return self._euclidean.random_point(n_points)


class KendalShapeRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    def random_horizontal_vec(self, base_point):
        tangent_vec = self.random_tangent_vec(base_point)
        return (
            self.space.fiber_bundle.horizontal_projection(tangent_vec, base_point)
            / self.amplitude
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


class ShapeBundleRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, n_discretized_curves=5, near_scale=0.1):
        super().__init__(space, near_scale=near_scale)
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


class DiffeoBasedRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, diffeo, amplitude=2.0, near_scale=0.1):
        super().__init__(space, amplitude, near_scale=near_scale)
        self.diffeo = diffeo

    def random_point(self, n_points=1):
        return self.diffeo(self.space.random_point(n_points))

    def random_tangent_vec(self, base_point):
        image_point = base_point
        base_point = self.diffeo.inverse(image_point)

        tangent_vec = self.space.random_tangent_vec(base_point)
        tangent_vec = tangent_vec / self.amplitude
        return self.diffeo.tangent(
            tangent_vec, base_point=base_point, image_point=image_point
        )
