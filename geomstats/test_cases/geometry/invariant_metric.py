import pytest

from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.vectorization import repeat_point

# TODO: more random tests (e.g. inner product)

# TODO: add lie algebra related tests


class _InvariantMetricTestCaseMixins(RiemannianMetricTestCase):
    def test_inner_product_at_identity(
        self, tangent_vec_a, tangent_vec_b, expected, atol
    ):
        res = self.space.metric.inner_product_at_identity(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inner_product_at_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.inner_product_at_identity(
            tangent_vec_a, tangent_vec_b
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_invariance(self, n_points, atol):
        left = self.space.metric.left

        group_elem_1 = self.data_generator.random_point(n_points)
        group_elem_2 = self.data_generator.random_point(n_points)

        tangent_vec_11 = self.data_generator.random_tangent_vec(group_elem_1)
        tangent_vec_12 = self.data_generator.random_tangent_vec(group_elem_1)

        dtrans_2 = self.space.tangent_translation_map(
            group_elem_2, left=left, inverse=False
        )

        if left:
            group_elem_3 = self.space.compose(group_elem_2, group_elem_1)
        else:
            group_elem_3 = self.space.compose(group_elem_1, group_elem_2)

        tangent_vec_31 = dtrans_2(tangent_vec_11)
        tangent_vec_32 = dtrans_2(tangent_vec_12)

        inner_prod = self.space.metric.inner_product(
            tangent_vec_11, tangent_vec_12, group_elem_1
        )
        inner_prod_ = self.space.metric.inner_product(
            tangent_vec_31, tangent_vec_32, group_elem_3
        )
        self.assertAllClose(inner_prod, inner_prod_, atol=atol)


class InvariantMetricMatrixTestCase(_InvariantMetricTestCaseMixins):
    def test_structure_constant(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected, atol
    ):
        res = self.space.metric.structure_constant(
            tangent_vec_a, tangent_vec_b, tangent_vec_c
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_structure_constant_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.structure_constant(
            tangent_vec_a, tangent_vec_b, tangent_vec_c
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "tangent_vec_c"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_dual_adjoint(self, tangent_vec_a, tangent_vec_b, expected, atol):
        res = self.space.metric.dual_adjoint(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_dual_adjoint_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.dual_adjoint(tangent_vec_a, tangent_vec_b)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_dual_adjoint_structure_constant(self, n_points, atol):
        base_point = repeat_point(self.space.identity, n_reps=n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        result = self.space.metric.inner_product_at_identity(
            self.space.metric.dual_adjoint(tangent_vec_a, tangent_vec_b), tangent_vec_c
        )
        expected = self.space.metric.structure_constant(
            tangent_vec_a, tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(result, expected, atol=atol)

    def test_connection_at_identity(self, tangent_vec_a, tangent_vec_b, expected, atol):
        res = self.space.metric.connection_at_identity(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_connection_at_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        expected = self.space.metric.connection_at_identity(
            tangent_vec_a, tangent_vec_b
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_connection(self, tangent_vec_a, tangent_vec_b, base_point, expected, atol):
        res = self.space.metric.connection(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_curvature_at_identity(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected, atol
    ):
        res = self.space.metric.curvature_at_identity(
            tangent_vec_a, tangent_vec_b, tangent_vec_c
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_curvature_at_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.curvature_at_identity(
            tangent_vec_a, tangent_vec_b, tangent_vec_c
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "tangent_vec_c"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_sectional_curvature_at_identity(
        self, tangent_vec_a, tangent_vec_b, expected, atol
    ):
        res = self.space.metric.sectional_curvature_at_identity(
            tangent_vec_a, tangent_vec_b
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_sectional_curvature_at_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.sectional_curvature_at_identity(
            tangent_vec_a, tangent_vec_b
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_curvature_derivative_at_identity(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, expected, atol
    ):
        res = self.space.metric.curvature_derivative_at_identity(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_curvature_derivative_at_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_d = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.curvature_derivative_at_identity(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    tangent_vec_d=tangent_vec_d,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=[
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
                "tangent_vec_d",
            ],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.vec
    def test_exp_at_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.exp(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=None,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_exp")

    @pytest.mark.random
    def test_log_after_exp_at_identity(self, n_points, atol):
        base_point = repeat_point(self.space.identity, n_reps=n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        point = self.space.exp(tangent_vec)
        tangent_vec_ = self.space.log(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_exp_after_log_at_identity(self, n_points, atol):
        base_point = repeat_point(self.space.identity, n_reps=n_points)
        end_point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log(end_point, base_point)
        end_point_ = self.space.metric.exp(tangent_vec)

        self.assertAllClose(end_point_, end_point, atol=atol)


class InvariantMetricVectorTestCase(_InvariantMetricTestCaseMixins):
    def test_left_exp_from_identity(self, tangent_vec, expected, atol):
        res = self.space.metric.left_exp_from_identity(tangent_vec)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_left_exp_from_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.left_exp_from_identity(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[dict(tangent_vec=tangent_vec, expected=expected, atol=atol)],
            arg_names=["tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_left_log_from_identity(self, point, expected, atol):
        res = self.space.metric.left_log_from_identity(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_left_exp_from_identity_after_left_log_from_identity(self, n_points, atol):
        end_point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.left_log_from_identity(end_point)
        end_point_ = self.space.metric.left_exp_from_identity(tangent_vec)

        self.assertAllClose(end_point_, end_point, atol=atol)

    @pytest.mark.random
    def test_left_log_from_identity_after_left_exp_from_identity(self, n_points, atol):
        base_point = repeat_point(self.space.identity, n_reps=n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        end_point = self.space.metric.left_exp_from_identity(tangent_vec)
        tangent_vec_ = self.space.metric.left_log_from_identity(end_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_exp_from_identity(self, tangent_vec, expected, atol):
        res = self.space.metric.exp_from_identity(tangent_vec)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_from_identity_vec(self, n_reps, atol):
        base_point = self.space.identity
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.exp_from_identity(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[dict(tangent_vec=tangent_vec, expected=expected, atol=atol)],
            arg_names=["tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_log_from_identity(self, point, expected, atol):
        res = self.space.metric.log_from_identity(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_exp_from_identity_after_log_from_identity(self, n_points, atol):
        end_point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log_from_identity(end_point)
        end_point_ = self.space.metric.exp_from_identity(tangent_vec)

        self.assertAllClose(end_point_, end_point, atol=atol)

    @pytest.mark.random
    def test_log_from_identity_after_exp_from_identity(self, n_points, atol):
        base_point = repeat_point(self.space.identity, n_reps=n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        end_point = self.space.metric.exp_from_identity(tangent_vec)
        tangent_vec_ = self.space.metric.log_from_identity(end_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)


class BiInvariantMetricTestCase(RiemannianMetricTestCase):
    @pytest.mark.random
    def test_invariance(self, n_points, atol):
        group_elem_1 = self.data_generator.random_point(n_points)
        group_elem_2 = self.data_generator.random_point(n_points)

        tangent_vec_11 = self.data_generator.random_tangent_vec(group_elem_1)
        tangent_vec_12 = self.data_generator.random_tangent_vec(group_elem_1)

        dleft_2 = self.space.tangent_translation_map(
            group_elem_2, left=True, inverse=False
        )
        dright_2 = self.space.tangent_translation_map(
            group_elem_2, left=False, inverse=False
        )

        group_elem_3 = self.space.compose(group_elem_2, group_elem_1)
        group_elem_4 = self.space.compose(group_elem_1, group_elem_2)

        tangent_vec_31 = dleft_2(tangent_vec_11)
        tangent_vec_32 = dleft_2(tangent_vec_12)

        tangent_vec_41 = dright_2(tangent_vec_11)
        tangent_vec_42 = dright_2(tangent_vec_12)

        inner_prod = self.space.metric.inner_product(
            tangent_vec_11, tangent_vec_12, group_elem_1
        )
        inner_prod_left = self.space.metric.inner_product(
            tangent_vec_31, tangent_vec_32, group_elem_3
        )
        inner_prod_right = self.space.metric.inner_product(
            tangent_vec_41, tangent_vec_42, group_elem_4
        )
        self.assertAllClose(inner_prod, inner_prod_left, atol=atol)
        self.assertAllClose(inner_prod, inner_prod_right, atol=atol)
