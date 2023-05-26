import pytest

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.connection import ConnectionTestCase
from geomstats.vectorization import get_batch_shape


class RiemannianMetricTestCase(ConnectionTestCase):
    def test_metric_matrix(self, base_point, expected, atol):
        res = self.space.metric.metric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_metric_matrix_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.metric_matrix(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_metric_matrix_is_spd(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        metric_matrix = self.space.metric.metric_matrix(base_point)

        res = SPDMatrices(n=self.space.dim).belongs(metric_matrix)
        expected_shape = get_batch_shape(self.space, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    def test_cometric_matrix(self, base_point, expected, atol):
        res = self.space.metric.cometric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_cometric_matrix_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.cometric_matrix(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inner_product_derivative_matrix(self, base_point, expected, atol):
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_product_derivative_matrix_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.inner_product_derivative_matrix(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        # TODO: test inner_product with itself?
        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_product_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_inner_product_is_symmetric(self, n_points, atol):
        """Check inner product is symmetric.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        inner_product_ab = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_product_ba = self.space.metric.inner_product(
            tangent_vec_b, tangent_vec_a, base_point
        )

        self.assertAllClose(inner_product_ab, inner_product_ba, atol=atol)

    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_coproduct_vec(self, n_reps, atol):
        # TODO: check if cotangent generation makes sense
        base_point = self.data_generator.random_point()
        cotangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        cotangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    cotangent_vec_a=cotangent_vec_a,
                    cotangent_vec_b=cotangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["cotangent_vec_a", "cotangent_vec_b", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_squared_norm(self, vector, base_point, expected, atol):
        res = self.space.metric.squared_norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_squared_norm_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vector = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.squared_norm(vector, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(vector=vector, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_norm(self, vector, base_point, expected, atol):
        res = self.space.metric.norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_norm_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vector = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.norm(vector, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(vector=vector, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_norm_is_positive(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        vector = self.data_generator.random_tangent_vec(base_point)

        norm_ = self.space.metric.norm(vector, base_point)

        res = gs.all(norm_ > -atol)
        self.assertTrue(res)

    def test_normalize(self, vector, base_point, expected, atol):
        res = self.space.metric.normalize(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_normalize_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vector = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.normalize(vector, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(vector=vector, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_squared_dist(self, point_a, point_b, expected, atol):
        res = self.space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_squared_dist_vec(self, n_reps, atol):
        point_a, point_b = self.data_generator.random_point(2)

        expected = self.space.metric.squared_dist(point_a, point_b)

        vec_data = generate_vectorization_data(
            data=[dict(point_a=point_a, point_b=point_b, expected=expected, atol=atol)],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_squared_dist_is_symmetric(self, n_points, atol):
        """Check squared distance is symmetric.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        squared_dist_ab = self.space.metric.squared_dist(point_a, point_b)
        squared_dist_ba = self.space.metric.squared_dist(point_b, point_a)

        self.assertAllClose(squared_dist_ab, squared_dist_ba, atol=atol)

    @pytest.mark.random
    def test_squared_dist_is_positive(self, n_points, atol):
        """Check squared distance is positive.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        squared_dist_ = self.space.metric.squared_dist(point_a, point_b)
        res = gs.all(squared_dist_ > -atol)
        self.assertTrue(res)

    def test_dist(self, point_a, point_b, expected, atol):
        # TODO: dist properties mixins? (thinking about stratified)
        res = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_dist_vec(self, n_reps, atol):
        point_a, point_b = self.data_generator.random_point(2)

        expected = self.space.metric.dist(point_a, point_b)

        vec_data = generate_vectorization_data(
            data=[dict(point_a=point_a, point_b=point_b, expected=expected, atol=atol)],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_dist_is_symmetric(self, n_points, atol):
        """Check distance is symmetric.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        dist_ab = self.space.metric.dist(point_a, point_b)
        dist_ba = self.space.metric.dist(point_b, point_a)

        self.assertAllClose(dist_ab, dist_ba, atol=atol)

    @pytest.mark.random
    def test_dist_is_positive(self, n_points, atol):
        """Check distance is positive.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        dist_ = self.space.metric.dist(point_a, point_b)
        res = gs.all(dist_ > -atol)
        self.assertTrue(res)

    @pytest.mark.random
    def test_dist_is_log_norm(self, n_points, atol):
        """Check distance is norm of log.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        log_norm = self.space.metric.norm(
            self.space.metric.log(point_b, point_a), point_a
        )
        dist_ = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(dist_, log_norm, atol=atol)

    @pytest.mark.random
    def test_dist_point_to_itself_is_zero(self, n_points, atol):
        """Check distance of a point to itself is zero.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point = self.data_generator.random_point(n_points)

        dist_ = self.space.metric.dist(point, point)

        expected_shape = get_batch_shape(self.space, point)
        expected = gs.zeros(expected_shape)
        self.assertAllClose(dist_, expected, atol=atol)

    @pytest.mark.random
    def test_dist_triangle_inequality(self, n_points, atol):
        """Check distance satifies triangle inequality.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)
        point_c = self.data_generator.random_point(n_points)

        dist_ab = self.space.metric.dist(point_a, point_b)
        dist_bc = self.space.metric.dist(point_b, point_c)
        rhs = dist_ac = self.space.metric.dist(point_a, point_c)

        lhs = dist_ab + dist_bc
        res = gs.all(lhs + atol >= rhs)
        self.assertTrue(res, f"lhs: {lhs}, rhs: {dist_ac}, diff: {lhs-rhs}")

    def test_diameter(self, points, expected, atol):
        res = self.space.metric.diameter(points)
        self.assertAllClose(res, expected, atol=atol)

    def test_normal_basis(self, basis, base_point, expected, atol):
        res = self.space.metric.normal_basis(basis, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_covariant_riemann_tensor(self, base_point, expected, atol):
        res = self.space.metric.covariant_riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_covariant_riemann_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.covariant_riemann_tensor(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_covariant_riemann_tensor_is_skew_symmetric_1(self, n_points, atol):
        """Check covariant riemannian tensor verifies first skew symmetry.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        # TODO: add definition of first skew symmetry in docstrings
        base_point = self.data_generator.random_point(n_points)

        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_1 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-2, -1], [-1, -2]
        )

        res = gs.all(gs.abs(skew_symmetry_1) < gs.atol)
        self.assertTrue(res)

    @pytest.mark.random
    def test_covariant_riemann_tensor_is_skew_symmetric_2(self, n_points, atol):
        """Check covariant riemannian tensor verifies second skew symmetry.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        # TODO: add definition of second skew symmetry in docstrings
        base_point = self.data_generator.random_point(n_points)

        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_2 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-4, -3], [-3, -4]
        )

        res = gs.all(gs.abs(skew_symmetry_2) < gs.atol)
        self.assertTrue(res)

    @pytest.mark.random
    def test_covariant_riemann_tensor_bianchi_identity(self, n_points, atol):
        """Check covariant riemannian tensor verifies Bianchi identity.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        # TODO: add Bianchi identity in docstrings
        base_point = self.data_generator.random_point(n_points)

        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        bianchi_identity = (
            covariant_metric_tensor
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-2, -1, -3])
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-1, -3, -2])
        )

        res = gs.all(gs.abs(bianchi_identity) < gs.atol)
        self.assertTrue(res)

    @pytest.mark.random
    def test_covariant_riemann_tensor_is_interchange_symmetric(self, n_points, atol):
        """Check covariant riemannian tensor verifies interchange symmetry.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        # TODO: add definition of interchange symmetry in docstrings
        base_point = self.data_generator.random_point(n_points)

        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        interchange_symmetry = covariant_metric_tensor - gs.moveaxis(
            covariant_metric_tensor, [-4, -3, -2, -1], [-2, -1, -4, -3]
        )

        res = gs.all(gs.abs(interchange_symmetry) < gs.atol)
        self.assertTrue(res)

    def test_sectional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_sectional_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            vectorization_type="repeat-0-1",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_scalar_curvature(self, base_point, expected, atol):
        res = self.space.metric.scalar_curvature(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_scalar_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.scalar_curvature(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_parallel_transport_ivp_norm(self, n_points, atol):
        """Check parallel transported norm is preserved.

        This is for parallel transport defined by initial value problem (ivp).

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        direction = self.data_generator.random_tangent_vec(base_point)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        self.assertAllClose(
            self.space.metric.norm(transported, end_point),
            self.space.metric.norm(tangent_vec, base_point),
            atol=atol,
        )

    @pytest.mark.random
    def test_parallel_transport_bvp_norm(self, n_points, atol):
        """Check parallel transported norm is preserved.

        This is for parallel transport defined by boundary value problem (bvp).

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        self.assertAllClose(
            self.space.metric.norm(transported, end_point),
            self.space.metric.norm(tangent_vec, base_point),
            atol=atol,
        )
