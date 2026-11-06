import random
from functools import singledispatch

import geomstats.backend as gs
from geomstats.geometry.base import (
    ComplexMatrixVectorSpace,
    ComplexVectorSpace,
    ComplexVectorSpaceOpenSet,
    DiffeomorphicManifold,
    DiffeomorphicMatrixVectorSpace,
    DiffeomorphicVectorSpace,
    LevelSet,
    MatrixVectorSpace,
    OpenSet,
    VectorSpace,
)
from geomstats.geometry.complex_manifold import ComplexManifold
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.geometry.diffeo import ReversedDiffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatrices,
)
from geomstats.geometry.general_linear import SquareMatrices
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.geometry.hpd_matrices import HPDMatrices
from geomstats.geometry.lower_triangular_matrices import (
    LowerTriangularMatrices,
    StrictlyLowerTriangularMatrices,
)
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.nfold_manifold import NFoldManifold
from geomstats.geometry.open_hemisphere import (
    OpenHemisphere,
    OpenHemispheresProduct,
)
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.geometry.siegel import Siegel
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import (
    NullRowSumsSymmetricMatrices,
    SymmetricHollowMatrices,
    SymmetricMatrices,
)
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
@get_data_generator.register(ComplexManifold)
def _(space, **kwargs):
    return RandomDataGenerator(space, **kwargs)


@get_data_generator.register(VectorSpace)
@get_data_generator.register(ComplexVectorSpace)
def _(space, **kwargs):
    return VectorSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(MatrixVectorSpace)
@get_data_generator.register(ComplexMatrixVectorSpace)
def _(space, **kwargs):
    return MatrixVectorSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(LevelSet)
@get_data_generator.register(OpenSet)
@get_data_generator.register(ComplexVectorSpaceOpenSet)
def _(space, **kwargs):
    return EmbeddedSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(Siegel)
def _(space, **kwargs):
    return SiegelRandomDataGenerator(space, **kwargs)


@get_data_generator.register(PositiveReals)
def _(space, **kwargs):
    return PositiveRealsRandomDataGenerator(space, **kwargs)


@get_data_generator.register(NullRowSumsSymmetricMatrices)
@get_data_generator.register(StrictlyLowerTriangularMatrices)
@get_data_generator.register(SymmetricHollowMatrices)
def _(space, **kwargs):
    return LevelSetMatrixVectorSpaceRandomDataGenerator(space, **kwargs)


@get_data_generator.register(FullRankCorrelationMatrices)
def _(space, **kwargs):
    return FullRankCorrelationMatricesRandomDataGenerator(space, **kwargs)


@get_data_generator.register(NFoldManifold)
def _(space, **kwargs):
    return NFoldManifoldRandomDataGenerator(space, **kwargs)


@get_data_generator.register(OpenHemisphere)
def _(space, **kwargs):
    return OpenHemisphereRandomDataGenerator(space, **kwargs)


@get_data_generator.register(OpenHemispheresProduct)
def _(space, **kwargs):
    return FactorwiseRandomDataGenerator(space, **kwargs)


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


class SiegelRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    def random_tangent_vec_at_zero(self, n_points=1):
        """Sample tangent vectors at the zero matrix.

        Parameters
        ----------
        n_points : int
            Number of tangent vectors.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vectors at zero.
        """
        zero = gs.zeros(self.space.shape, dtype=gs.get_default_cdtype())
        return self.random_tangent_vec(repeat_point(zero, n_points))


class PositiveRealsRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    """Random data generator for the positive reals.

    Tangent vectors are scaled by their base point, so that their norm for
    the positive reals metric does not depend on the base point.
    """

    def random_tangent_vec(self, base_point):
        return base_point * super().random_tangent_vec(base_point)


class LevelSetMatrixVectorSpaceRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    """Random data generator for matrix vector spaces defined as level sets.

    Points are sampled as for level sets, and basis representations as for
    matrix vector spaces.
    """

    random_basis_representation = (
        MatrixVectorSpaceRandomDataGenerator.random_basis_representation
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


class FactorwiseRandomDataGenerator(RandomDataGenerator):
    """Random data generator sampling each factor of a product manifold.

    Points and tangent vectors are sampled with the data generator of each
    factor and embedded in the product. Useful when a factor needs its own
    data generator; otherwise, a product manifold is sampled as any other
    manifold.

    Parameters
    ----------
    space : ProductManifold
        Space to sample from.
    amplitude : float
        Passed to the data generator of each factor.
    near_scale : float
        Scale of the tangent vector used by ``random_point_near``.
    """

    def __init__(self, space, amplitude=2.0, near_scale=0.1):
        super().__init__(space, amplitude=amplitude, near_scale=near_scale)
        self.factor_data_generators = [
            get_data_generator(factor, amplitude=amplitude, near_scale=near_scale)
            for factor in space.factors
        ]

    def random_point(self, n_points=1):
        return self.space.embed_to_product(
            [
                data_generator.random_point(n_points)
                for data_generator in self.factor_data_generators
            ]
        )

    def random_tangent_vec(self, base_point):
        factor_base_points = self.space.project_from_product(base_point)
        return self.space.embed_to_product(
            [
                data_generator.random_tangent_vec(factor_base_point)
                for data_generator, factor_base_point in zip(
                    self.factor_data_generators, factor_base_points
                )
            ]
        )


class OpenHemisphereRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    """Random data generator for the open hemisphere.

    Tangent vectors sampled in the embedding have unbounded length for the
    pullback metric near the equator (about ``|v| / x_0 ** 2``). If the space
    is equipped, they are rescaled to a bounded metric norm, so geodesics
    stay short enough for ``exp`` to be accurate.

    Parameters
    ----------
    space : OpenHemisphere
        Space to sample from.
    amplitude : float
        Random tangent vectors have metric norm at most ``1 / amplitude``.
    near_scale : float
        Scale of the tangent vector used by ``random_point_near``.
    """

    def random_tangent_vec(self, base_point):
        tangent_vec = self.space.random_tangent_vec(base_point)
        if not hasattr(self.space, "metric"):
            return tangent_vec / self.amplitude

        norm = self.space.metric.norm(tangent_vec, base_point)
        radius = gs.random.uniform(size=norm.shape) / self.amplitude
        return gs.einsum("...,...i->...i", radius / norm, tangent_vec)


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


class FullRankCorrelationMatricesRandomDataGenerator(EmbeddedSpaceRandomDataGenerator):
    def random_diagonal_vec(self, n_points=1):
        return Matrices.diagonal(self.space.embedding_space.random_point(n_points))


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


class MatrixOperationsRandomDataGenerator:
    """Random data generator for matrix operations.

    Matrix operations are not tied to a space, so each sampler draws a new
    random shape unless one is given. Samplers ending in ``_like`` instead
    match the shape and batch size of a given matrix.

    Parameters
    ----------
    m_bounds : tuple[int]
        Bounds of the number of rows.
    n_bounds : tuple[int]
        Bounds of the number of columns.
    """

    def __init__(self, m_bounds=(2, 4), n_bounds=(2, 4)):
        self.m_bounds = m_bounds
        self.n_bounds = n_bounds

        self._n_shape_trials = 20

    def _random_shape(self):
        return (random.randint(*self.m_bounds), random.randint(*self.n_bounds))

    def _random_square_shape(self):
        return (random.randint(*self.m_bounds),)

    def _random_non_square_shape(self):
        m = random.randint(*self.m_bounds)
        for _ in range(self._n_shape_trials):
            n = random.randint(*self.n_bounds)
            if n != m:
                break

        return (m, n)

    @staticmethod
    def _n_points_like(mat):
        return mat.shape[0] if mat.ndim > 2 else 1

    def random_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_shape()

        return Matrices(*shape, equip=False).random_point(n_points)

    def random_mat_like(self, mat):
        return self.random_mat(self._n_points_like(mat), shape=mat.shape[-2:])

    def random_transposed_mat_like(self, mat):
        return self.random_mat(self._n_points_like(mat), shape=mat.shape[-2:][::-1])

    def random_square_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SquareMatrices(*shape).random_point(n_points)

    def random_non_square_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_non_square_shape()

        return self.random_mat(n_points, shape)

    def random_symmetric_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SymmetricMatrices(*shape, equip=False).random_point(n_points)

    def random_diagonal_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        n = shape[0]
        identity = gs.eye(n)
        batch_shape = (n_points,) if n_points > 1 else ()
        coeffs = gs.random.uniform(size=batch_shape + (n,))
        return gs.einsum("...,ij->...ij", coeffs, identity)

    def random_lower_triangular_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return LowerTriangularMatrices(*shape).random_point(n_points)

    def random_upper_triangular_mat(self, n_points=1, shape=None):
        return gs.transpose(self.random_lower_triangular_mat(n_points, shape))

    def random_spd_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SPDMatrices(*shape).random_point(n_points)

    def random_skew_symmetric_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return SkewSymmetricMatrices(*shape).random_point(n_points)

    @staticmethod
    def random_coefficients(n_points=1):
        """Sample two scalar coefficients, independent of `n_points`."""
        return gs.random.uniform(size=(2,))


class ComplexMatrixOperationsRandomDataGenerator(MatrixOperationsRandomDataGenerator):
    """Random data generator for complex matrix operations.

    Same as `MatrixOperationsRandomDataGenerator`, but samples complex
    matrices.
    """

    def random_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_shape()

        return ComplexMatrices(*shape, equip=False).random_point(n_points)

    def random_square_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return self.random_mat(n_points, shape=shape * 2)

    def random_hermitian_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return HermitianMatrices(*shape, equip=False).random_point(n_points)

    def random_hpd_mat(self, n_points=1, shape=None):
        if shape is None:
            shape = self._random_square_shape()

        return HPDMatrices(*shape, equip=False).random_point(n_points)

    def random_skew_hermitian_mat(self, n_points=1, shape=None):
        return ComplexMatrices.to_skew_hermitian(
            self.random_square_mat(n_points, shape=shape)
        )
