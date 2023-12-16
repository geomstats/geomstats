import pytest

import geomstats.backend as gs
from geomstats.geometry.diffeo import AutodiffDiffeo
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class CircleSO2Diffeo(AutodiffDiffeo):
    """Diffeomorphism between S1 and SO2."""

    def __init__(self):
        super().__init__(space_shape=(2,), image_space_shape=(2, 2))

    def diffeomorphism(self, base_point):
        second_column = gs.stack([-base_point[..., 1], base_point[..., 0]], axis=-1)
        return gs.stack([base_point, second_column], axis=-1)

    def inverse_diffeomorphism(self, image_point):
        return image_point[..., 0]


class DiffeoTestCase(TestCase):
    tangent_to_multiple = False

    def setup_method(self):
        if not hasattr(self, "data_generator") and hasattr(self, "space"):
            self.data_generator = RandomDataGenerator(self.space)

        if not hasattr(self, "image_data_generator") and hasattr(self, "image_space"):
            self.image_data_generator = RandomDataGenerator(self.image_space)

    def test_diffeomorphism(self, base_point, expected, atol):
        res = self.diffeo.diffeomorphism(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_diffeomorphism_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.diffeo.diffeomorphism(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_diffeomorphism_belongs(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        image_point = self.diffeo.diffeomorphism(base_point)

        belongs = self.image_space.belongs(image_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(belongs, expected)

    def test_inverse_diffeomorphism(self, image_point, expected, atol):
        res = self.diffeo.inverse_diffeomorphism(image_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_diffeomorphism_vec(self, n_reps, atol):
        image_point = self.image_data_generator.random_point()

        expected = self.diffeo.inverse_diffeomorphism(image_point)

        vec_data = generate_vectorization_data(
            data=[dict(image_point=image_point, expected=expected, atol=atol)],
            arg_names=["image_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_inverse_diffeomorphism_belongs(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)

        point = self.diffeo.inverse_diffeomorphism(image_point)

        belongs = self.space.belongs(point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(belongs, expected)

    @pytest.mark.random
    def test_inverse_diffeomorphism_after_diffeomorphism(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        image_point = self.diffeo.diffeomorphism(base_point)
        base_point_ = self.diffeo.inverse_diffeomorphism(image_point)

        self.assertAllClose(base_point_, base_point, atol=atol)

    @pytest.mark.random
    def test_diffeomorphism_after_inverse_diffeomorphism(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)

        point = self.diffeo.inverse_diffeomorphism(image_point)
        image_point_ = self.diffeo.diffeomorphism(point)

        self.assertAllClose(image_point_, image_point, atol=atol)

    def test_tangent_diffeomorphism(
        self, tangent_vec, expected, atol, base_point=None, image_point=None
    ):
        res = self.diffeo.tangent_diffeomorphism(
            tangent_vec, base_point=base_point, image_point=image_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_diffeomorphism_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.diffeo.tangent_diffeomorphism(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_diffeomorphism_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        image_tangent_vec = self.diffeo.tangent_diffeomorphism(tangent_vec, base_point)
        image_point = self.diffeo.diffeomorphism(base_point)

        is_tangent = self.image_space.is_tangent(
            image_tangent_vec, image_point, atol=atol
        )
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    @pytest.mark.random
    def test_tangent_diffeomorphism_with_image_point(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        image_point = self.diffeo.diffeomorphism(base_point)

        image_tangent_vec = self.diffeo.tangent_diffeomorphism(tangent_vec, base_point)
        image_tangent_vec_ = self.diffeo.tangent_diffeomorphism(
            tangent_vec, image_point=image_point
        )

        self.assertAllClose(image_tangent_vec, image_tangent_vec_, atol=atol)

    def test_inverse_tangent_diffeomorphism(
        self,
        image_tangent_vec,
        expected,
        atol,
        image_point=None,
        base_point=None,
    ):
        res = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec,
            image_point=image_point,
            base_point=base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_tangent_diffeomorphism_vec(self, n_reps, atol):
        image_point = self.image_data_generator.random_point()
        image_tangent_vec = self.image_data_generator.random_tangent_vec(image_point)

        expected = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    image_tangent_vec=image_tangent_vec,
                    image_point=image_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["image_tangent_vec", "image_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_inverse_tangent_diffeomorphism_is_tangent(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)
        image_tangent_vec = self.image_data_generator.random_tangent_vec(image_point)

        tangent_vec = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        base_point = self.diffeo.inverse_diffeomorphism(image_point)

        is_tangent = self.space.is_tangent(tangent_vec, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    @pytest.mark.random
    def test_inverse_tangent_diffeomorphism_with_base_point(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)
        image_tangent_vec = self.image_data_generator.random_tangent_vec(image_point)

        base_point = self.diffeo.inverse_diffeomorphism(image_point)

        tangent_vec = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        tangent_vec_ = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, base_point=base_point
        )
        self.assertAllClose(tangent_vec, tangent_vec_, atol=atol)

    @pytest.mark.random
    def test_inverse_tangent_diffeomorphism_after_tangent_diffeomorphism(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        image_tangent_vec = self.diffeo.tangent_diffeomorphism(tangent_vec, base_point)
        image_point = self.diffeo.diffeomorphism(base_point)

        tangent_vec_ = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_tangent_diffeomorphism_after_inverse_tangent_diffeomorphism(
        self, n_points, atol
    ):
        image_point = self.image_data_generator.random_point(n_points)
        image_tangent_vec = self.image_data_generator.random_tangent_vec(image_point)

        tangent_vec = self.diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        base_point = self.diffeo.inverse_diffeomorphism(image_point)

        image_tangent_vec_ = self.diffeo.tangent_diffeomorphism(tangent_vec, base_point)
        self.assertAllClose(image_tangent_vec_, image_tangent_vec, atol=atol)


class AutodiffDiffeoTestCase(DiffeoTestCase):
    def test_jacobian_diffeomorphism(self, base_point, expected, atol):
        res = self.diffeo.jacobian_diffeomorphism(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_jacobian_diffeomorphism_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.diffeo.jacobian_diffeomorphism(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inverse_jacobian_diffeomorphism(self, image_point, expected, atol):
        res = self.diffeo.inverse_jacobian_diffeomorphism(image_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_jacobian_diffeomorphism_vec(self, n_reps, atol):
        image_point = self.image_data_generator.random_point()

        expected = self.diffeo.inverse_jacobian_diffeomorphism(image_point)

        vec_data = generate_vectorization_data(
            data=[dict(image_point=image_point, expected=expected, atol=atol)],
            arg_names=["image_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class DiffeoComparisonTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator") and hasattr(self, "space"):
            self.data_generator = RandomDataGenerator(self.space)

        if not hasattr(self, "image_data_generator") and hasattr(self, "image_space"):
            self.image_data_generator = RandomDataGenerator(self.image_space)

    @pytest.mark.random
    def test_diffeomorphism(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.diffeo.diffeomorphism(base_point)
        res_ = self.other_diffeo.diffeomorphism(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inverse_diffeomorphism(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)

        res = self.diffeo.inverse_diffeomorphism(image_point)
        res_ = self.other_diffeo.inverse_diffeomorphism(image_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_tangent_diffeomorphism(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.diffeo.tangent_diffeomorphism(tangent_vec, base_point)
        res_ = self.other_diffeo.tangent_diffeomorphism(tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inverse_tangent_diffeomorphism(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)
        image_tangent_vec = self.image_data_generator.random_tangent_vec(image_point)

        res = self.diffeo.inverse_tangent_diffeomorphism(image_tangent_vec, image_point)
        res_ = self.other_diffeo.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        self.assertAllClose(res, res_, atol=atol)
