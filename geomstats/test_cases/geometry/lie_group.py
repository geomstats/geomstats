import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import GroupExpTestCaseMixins
from geomstats.vectorization import repeat_point


class _LieGroupTestCaseMixins(GroupExpTestCaseMixins):
    def test_compose(self, point_a, point_b, expected, atol):
        composed = self.space.compose(point_a, point_b)
        self.assertAllClose(composed, expected, atol=atol)

    @pytest.mark.vec
    def test_compose_vec(self, n_reps, atol):
        point_a, point_b = self.data_generator.random_point(2)

        expected = self.space.compose(point_a, point_b)

        vec_data = generate_vectorization_data(
            data=[dict(point_a=point_a, point_b=point_b, expected=expected, atol=atol)],
            arg_names=["point_a", "point_b"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inverse(self, point, expected, atol):
        inverse = self.space.inverse(point)
        self.assertAllClose(inverse, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_vec(self, n_reps, atol):
        point = self.data_generator.random_point()

        expected = self.space.inverse(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_compose_with_inverse_is_identity(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        inverse = self.space.inverse(point)

        identity = self.space.identity
        if n_points > 1:
            identity = gs.broadcast_to(identity, (n_points, *identity.shape))

        identity_ = self.space.compose(point, inverse)
        self.assertAllClose(identity_, identity, atol=atol)

        identity_ = self.space.compose(inverse, point)
        self.assertAllClose(identity_, identity, atol=atol)

    @pytest.mark.random
    def test_compose_with_identity_is_point(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        point_ = self.space.compose(point, self.space.identity)
        self.assertAllClose(point_, point, atol=atol)

        point_ = self.space.compose(self.space.identity, point)
        self.assertAllClose(point_, point, atol=atol)

    def test_log(self, point, base_point, expected, atol):
        vec = self.space.log(point, base_point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_log_vec(self, n_reps, atol):
        point, base_point = self.data_generator.random_point(2)

        expected = self.space.log(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_exp_after_log(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        vec = self.space.log(point, base_point)
        point_ = self.space.exp(vec, base_point)

        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_log_after_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        point = self.space.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.log(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_to_tangent_at_identity_belongs_to_lie_algebra(self, n_points, atol):
        tangent_vec = self.data_generator.random_tangent_vec(
            repeat_point(self.space.identity, n_points)
        )

        res = self.space.lie_algebra.belongs(tangent_vec, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_tangent_translation_map(
        self, point, left, inverse, tangent_vec, expected, atol
    ):
        res = self.space.tangent_translation_map(point, left=left, inverse=inverse)(
            tangent_vec
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_translation_map_vec(self, n_reps, left, inverse, atol):
        point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(point)

        expected = self.space.tangent_translation_map(
            point, left=left, inverse=inverse
        )(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    tangent_vec=tangent_vec,
                    left=left,
                    inverse=inverse,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point", "tangent_vec"],
            expected_name="expected",
            vectorization_type="repeat-1",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_lie_bracket(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        # TODO: any random test for validation here?
        bracket = self.space.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(bracket, expected, atol=atol)

    @pytest.mark.vec
    def test_lie_bracket_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a, tangent_vec_b = self.data_generator.random_tangent_vec(
            repeat_point(base_point, 2)
        )

        expected = self.space.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)

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

    def test_identity(self, expected, atol):
        self.assertAllClose(self.space.identity, expected, atol=atol)


class MatrixLieGroupTestCase(_LieGroupTestCaseMixins, ManifoldTestCase):
    pass


class LieGroupTestCase(_LieGroupTestCaseMixins, ManifoldTestCase):
    # TODO: exp and log not from identity: are they enough tested with log and exp?

    def test_jacobian_translation(self, point, left, expected, atol):
        res = self.space.jacobian_translation(point, left=left)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_jacobian_translation_vec(self, n_reps, left, atol):
        point = self.data_generator.random_point()
        expected = self.space.jacobian_translation(point, left=left)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, left=left, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_exp_from_identity(self, tangent_vec, expected, atol):
        res = self.space.exp_from_identity(tangent_vec)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_from_identity_vec(self, n_reps, atol):
        tangent_vec = self.data_generator.random_tangent_vec(self.space.identity)
        expected = self.space.exp_from_identity(tangent_vec)

        vec_data = generate_vectorization_data(
            data=[dict(tangent_vec=tangent_vec, expected=expected, atol=atol)],
            arg_names=["tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_log_from_identity(self, point, expected, atol):
        vec = self.space.log_from_identity(point)
        self.assertAllClose(vec, expected, atol=atol)

    @pytest.mark.vec
    def test_log_from_identity_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        expected = self.space.log_from_identity(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_exp_from_identity_after_log_from_identity(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.space.log_from_identity(point)

        point_ = self.space.exp_from_identity(tangent_vec)
        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_log_from_identity_after_exp_from_identity(self, n_points, atol):
        tangent_vec = self.data_generator.random_tangent_vec(
            repeat_point(self.space.identity, n_points)
        )

        point = self.space.exp_from_identity(tangent_vec)
        tangent_vec_ = self.space.log_from_identity(point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)
