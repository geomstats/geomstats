"""Unit tests for Tangent PCA."""

import pytest

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.pca import TangentPCA
from geomstats.vectorization import repeat_out


@tests.conftest.np_and_autograd_only
class TestTangentPCA(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        self.so3 = SpecialOrthogonal(n=3, point_type="vector")
        self.spd = SPDMatrices(3)

        self.n_samples = 10

        self.X = self.so3.random_uniform(n_samples=self.n_samples)
        self.n_components = 2

    def test_tangent_pca_error(self):
        X = self.X
        tpca = TangentPCA(self.so3, n_components=self.n_components)
        tpca.fit(X)
        X_diff_size = gs.ones((self.n_samples, gs.shape(X)[1] + 1))
        with pytest.raises(ValueError):
            tpca.transform(X_diff_size)

    def test_tangent_pca(self):
        X = self.X
        tpca = TangentPCA(self.so3, n_components=gs.shape(X)[1])
        tpca.fit(X)
        self.assertEqual(tpca.n_features_, gs.shape(X)[1])

    def test_fit_mle(self):
        X = self.X
        tpca = TangentPCA(self.so3, n_components="mle")
        tpca.fit(X)
        self.assertEqual(tpca.n_features_, gs.shape(X)[1])

    def test_fit_to_target_explained_variance(self):
        X = self.spd.random_point(n_samples=5)
        target = 0.90
        tpca = TangentPCA(self.spd, n_components=target)
        tpca.fit(X)
        result = gs.cumsum(tpca.explained_variance_ratio_)[-1] > target
        expected = True
        self.assertAllClose(result, expected)

    def test_fit_matrix(self):
        expected = 2
        X = self.spd.random_point(n_samples=5)
        tpca = TangentPCA(space=self.spd, n_components=expected)
        tpca.fit(X)
        result = tpca.n_components_
        self.assertAllClose(result, expected)

    def test_fit_transform_matrix(self):
        expected = 2
        X = self.spd.random_point(n_samples=5)
        tpca = TangentPCA(space=self.spd, n_components=expected)
        tangent_projected_data = tpca.fit_transform(X)
        result = tangent_projected_data.shape[-1]
        self.assertAllClose(result, expected)

    def test_fit_inverse_transform_matrix(self):
        X = self.spd.random_point(n_samples=5)
        tpca = TangentPCA(space=self.spd)
        tangent_projected_data = tpca.fit_transform(X)
        result = tpca.inverse_transform(tangent_projected_data)
        expected = X
        self.assertAllClose(result, expected, atol=1e-6)

    def test_fit_transform_vector(self):
        expected = 2
        tpca = TangentPCA(space=self.so3, n_components=expected)
        tangent_projected_data = tpca.fit_transform(self.X)
        result = tangent_projected_data.shape[-1]
        self.assertAllClose(result, expected)

    def test_fit_inverse_transform_vector(self):
        tpca = TangentPCA(space=self.so3)
        tangent_projected_data = tpca.fit_transform(self.X)
        result = tpca.inverse_transform(tangent_projected_data)
        expected = self.X
        self.assertAllClose(result, expected)

    def test_fit_fit_transform_matrix(self):
        X = self.spd.random_point(n_samples=5)
        tpca = TangentPCA(space=self.spd)
        expected = tpca.fit_transform(X)
        result = tpca.fit(X).transform(X)
        self.assertAllClose(result, expected)

    def test_fit_matrix_se(self):
        se_mat = SpecialEuclidean(n=3, equip=False)

        X = se_mat.random_point(self.n_samples)

        tpca = TangentPCA(space=se_mat)
        tpca.mean_estimator = ExponentialBarycenter(se_mat)
        tangent_projected_data = tpca.fit_transform(X)
        result = tpca.inverse_transform(tangent_projected_data)
        expected = X
        self.assertAllClose(result, expected)

    def test_principal_directions_selection(self):
        class AlmostEuclideanDimTwo(VectorSpace):
            """Class for the almost Euclidean space of dimension 2.

            This manifold almost corresponds to the Euclidean space
            of dimension 2: the Euclidean metric is modified to give
            more weight to the second axis.

            Parameters
            ----------
            dim : int
                Dimension of the Euclidean space.
            """

            def __init__(self, equip=True):
                super().__init__(
                    dim=2,
                    shape=(2,),
                    equip=equip,
                )

            @staticmethod
            def default_metric():
                """Metric to equip the space with if equip is True."""
                return AlmostEuclideanDimTwoMetric

            @property
            def identity(self):
                """Identity of the group.

                Returns
                -------
                identity : array-like, shape=[n]
                """
                return gs.zeros(2)

            def _create_basis(self):
                """Create the canonical basis."""
                return gs.eye(2)

        class AlmostEuclideanDimTwoMetric(RiemannianMetric):
            """Class for the almost Euclidean metric in dimension 2.

            It almost corresponds to the Euclidean space of dimension 2:
            the Euclidean metric is modified to give more weight to the
            second axis. The infinitesimal metric element has the
            following expression: ds^2 = dx1^2 + 10000 dx2^2
            """

            def metric_matrix(self, base_point=None):
                """Compute the inner-product matrix, independent of the base point.

                Parameters
                ----------
                base_point : array-like, shape=[..., dim]
                    Base point.
                    Optional, default: None.

                Returns
                -------
                inner_prod_mat : array-like, shape=[..., dim, dim]
                    Inner-product matrix.
                """
                mat = gs.zeros((2, 2))
                mat[0, 0] = 1
                mat[1, 1] = 10000
                return repeat_out(self._space, mat, base_point, out_shape=(2, 2))

            @staticmethod
            def exp(tangent_vec, base_point, **kwargs):
                """Compute exp map of a base point in tangent vector direction.

                The Riemannian exponential is vector addition in the Euclidean space.

                Parameters
                ----------
                tangent_vec : array-like, shape=[..., dim]
                    Tangent vector at base point.
                base_point : array-like, shape=[..., dim]
                    Base point.

                Returns
                -------
                exp : array-like, shape=[..., dim]
                    Riemannian exponential.
                """
                return base_point + tangent_vec

            @staticmethod
            def log(point, base_point, **kwargs):
                """Compute log map using a base point and other point.

                The Riemannian logarithm is the subtraction in the Euclidean space.

                Parameters
                ----------
                point: array-like, shape=[..., dim]
                    Point.
                base_point: array-like, shape=[..., dim]
                    Base point.

                Returns
                -------
                log: array-like, shape=[..., dim]
                    Riemannian logarithm.
                """
                return point - base_point

        almost_euclidean_dim_two = AlmostEuclideanDimTwo()
        tpca = TangentPCA(space=almost_euclidean_dim_two, n_components=1)
        point_a = gs.array([10, 0], dtype=float)
        point_b = gs.array([0, 1], dtype=float)
        X = gs.stack([point_a, -point_a, point_b, -point_b], axis=0)
        mean = gs.array([0, 0], dtype=float)
        tpca.fit(X=X, base_point=mean)
        variance_axis_1 = almost_euclidean_dim_two.metric.squared_dist(mean, point_a)
        variance_axis_2 = almost_euclidean_dim_two.metric.squared_dist(mean, point_b)
        assert variance_axis_2 >= variance_axis_1
        normed_riemannian_basis_vector_2 = gs.array([0, 0.01])
        assert gs.all(
            tpca.components_in_riemannian_basis == normed_riemannian_basis_vector_2
        )
