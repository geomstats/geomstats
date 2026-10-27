import pytest

import geomstats.backend as gs
from geomstats.geometry.diffeo import AutodiffDiffeo
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase


class CircleSO2Diffeo(AutodiffDiffeo):
    """Diffeomorphism between S1 and SO2."""

    def __init__(self):
        super().__init__(space_shape=(2,), image_space_shape=(2, 2))

    def __call__(self, base_point):
        second_column = gs.stack([-base_point[..., 1], base_point[..., 0]], axis=-1)
        return gs.stack([base_point, second_column], axis=-1)

    def inverse(self, image_point):
        return image_point[..., 0]


class DiffeoTestCase(TestCase):
    def test_diffeomorphism(self, base_point, expected, atol=gs.atol):
        """Check diffeomorphism.

        Parameters
        ----------
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        expected : array-like, shape=[..., *image_space.shape]
            Expected point on the image space.
        atol : float
            Absolute tolerance.
        """
        res = self.diffeo(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse(self, image_point, expected, atol=gs.atol):
        """Check inverse diffeomorphism.

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        expected : array-like, shape=[..., *domain_space.shape]
            Expected point on the domain space.
        atol : float
            Absolute tolerance.
        """
        res = self.diffeo.inverse(image_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent(
        self, tangent_vec, expected, atol=gs.atol, base_point=None, image_point=None
    ):
        """Check tangent diffeomorphism.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *domain_space.shape]
            Tangent vector to the domain space at `base_point`.
        expected : array-like, shape=[..., *image_space.shape]
            Expected tangent vector to the image space.
        atol : float
            Absolute tolerance.
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
            Optional, default: None.
        image_point : array-like, shape=[..., *image_space.shape]
            Image of `base_point`.
            Optional, default: None.
        """
        res = self.diffeo.tangent(
            tangent_vec, base_point=base_point, image_point=image_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_tangent(
        self,
        image_tangent_vec,
        expected,
        atol=gs.atol,
        image_point=None,
        base_point=None,
    ):
        """Check inverse tangent diffeomorphism.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_space.shape]
            Tangent vector to the image space at `image_point`.
        expected : array-like, shape=[..., *domain_space.shape]
            Expected tangent vector to the domain space.
        atol : float
            Absolute tolerance.
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
            Optional, default: None.
        base_point : array-like, shape=[..., *domain_space.shape]
            Inverse image of `image_point`.
            Optional, default: None.
        """
        res = self.diffeo.inverse_tangent(
            image_tangent_vec,
            image_point=image_point,
            base_point=base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_diffeomorphism_belongs(self, base_point, atol=gs.atol):
        """Check diffeomorphism maps to the image space.

        Parameters
        ----------
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        atol : float
            Absolute tolerance.
        """
        image_point = self.diffeo(base_point)

        assert_belongs(self.image_space, image_point, atol=atol)

    def test_inverse_belongs(self, image_point, atol=gs.atol):
        """Check inverse diffeomorphism maps to the domain space.

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        atol : float
            Absolute tolerance.
        """
        point = self.diffeo.inverse(image_point)

        assert_belongs(self.domain_space, point, atol=atol)

    def test_inverse_after_diffeomorphism(self, base_point, atol=gs.atol):
        """Check inverse is a left inverse of the diffeomorphism.

        Parameters
        ----------
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        atol : float
            Absolute tolerance.
        """
        image_point = self.diffeo(base_point)
        base_point_ = self.diffeo.inverse(image_point)

        self.assertAllClose(base_point_, base_point, atol=atol)

    def test_diffeomorphism_after_inverse(self, image_point, atol=gs.atol):
        """Check inverse is a right inverse of the diffeomorphism.

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        atol : float
            Absolute tolerance.
        """
        point = self.diffeo.inverse(image_point)
        image_point_ = self.diffeo(point)

        self.assertAllClose(image_point_, image_point, atol=atol)

    def test_tangent_is_tangent(self, tangent_vec, base_point, atol=gs.atol):
        """Check tangent diffeomorphism maps to tangent vectors.

        The image of a tangent vector at `base_point` must be tangent to
        the image space at the image of `base_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *domain_space.shape]
            Tangent vector to the domain space at `base_point`.
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        atol : float
            Absolute tolerance.
        """
        image_tangent_vec = self.diffeo.tangent(tangent_vec, base_point)
        image_point = self.diffeo(base_point)

        assert_is_tangent(self.image_space, image_tangent_vec, image_point, atol=atol)

    def test_inverse_tangent_is_tangent(
        self, image_tangent_vec, image_point, atol=gs.atol
    ):
        """Check inverse tangent diffeomorphism maps to tangent vectors.

        The image of a tangent vector at `image_point` must be tangent to
        the domain space at the inverse image of `image_point`.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_space.shape]
            Tangent vector to the image space at `image_point`.
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        base_point = self.diffeo.inverse(image_point)

        assert_is_tangent(self.domain_space, tangent_vec, base_point, atol=atol)

    def test_inverse_tangent_after_tangent(self, tangent_vec, base_point, atol=gs.atol):
        """Check inverse tangent is a left inverse of the tangent.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *domain_space.shape]
            Tangent vector to the domain space at `base_point`.
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        atol : float
            Absolute tolerance.
        """
        image_tangent_vec = self.diffeo.tangent(tangent_vec, base_point)
        image_point = self.diffeo(base_point)

        tangent_vec_ = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_tangent_after_inverse_tangent(
        self, image_tangent_vec, image_point, atol=gs.atol
    ):
        """Check inverse tangent is a right inverse of the tangent.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_space.shape]
            Tangent vector to the image space at `image_point`.
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        base_point = self.diffeo.inverse(image_point)

        image_tangent_vec_ = self.diffeo.tangent(tangent_vec, base_point)
        self.assertAllClose(image_tangent_vec_, image_tangent_vec, atol=atol)

    @pytest.mark.api
    def test_tangent_with_image_point(self, tangent_vec, base_point, atol=gs.atol):
        """Check tangent diffeomorphism with image point.

        Passing `image_point` is the same as passing `base_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *domain_space.shape]
            Tangent vector to the domain space at `base_point`.
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        atol : float
            Absolute tolerance.
        """
        image_point = self.diffeo(base_point)

        image_tangent_vec = self.diffeo.tangent(tangent_vec, base_point)
        image_tangent_vec_ = self.diffeo.tangent(tangent_vec, image_point=image_point)

        self.assertAllClose(image_tangent_vec, image_tangent_vec_, atol=atol)

    @pytest.mark.api
    def test_inverse_tangent_with_base_point(
        self, image_tangent_vec, image_point, atol=gs.atol
    ):
        """Check inverse tangent diffeomorphism with base point.

        Passing `base_point` is the same as passing `image_point`.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_space.shape]
            Tangent vector to the image space at `image_point`.
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        atol : float
            Absolute tolerance.
        """
        base_point = self.diffeo.inverse(image_point)

        tangent_vec = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        tangent_vec_ = self.diffeo.inverse_tangent(
            image_tangent_vec, base_point=base_point
        )
        self.assertAllClose(tangent_vec, tangent_vec_, atol=atol)


class AutodiffDiffeoTestCase(DiffeoTestCase):
    def test_jacobian(self, base_point, expected, atol=gs.atol):
        """Check Jacobian of the diffeomorphism.

        Parameters
        ----------
        base_point : array-like, shape=[..., *domain_space.shape]
            Point on the domain space.
        expected : array-like, shape=[..., *image_space.shape, *domain_space.shape]
            Expected Jacobian.
        atol : float
            Absolute tolerance.
        """
        res = self.diffeo.jacobian(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_jacobian(self, image_point, expected, atol=gs.atol):
        """Check Jacobian of the inverse diffeomorphism.

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_space.shape]
            Point on the image space.
        expected : array-like, shape=[..., *domain_space.shape, *image_space.shape]
            Expected Jacobian.
        atol : float
            Absolute tolerance.
        """
        res = self.diffeo.inverse_jacobian(image_point)
        self.assertAllClose(res, expected, atol=atol)


class DiffeoComparisonTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator") and hasattr(self, "space"):
            self.data_generator = RandomDataGenerator(self.space)

        if not hasattr(self, "image_data_generator") and hasattr(self, "image_space"):
            self.image_data_generator = RandomDataGenerator(self.image_space)

    @pytest.mark.random
    def test_diffeomorphism(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.diffeo(base_point)
        res_ = self.other_diffeo(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inverse(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)

        res = self.diffeo.inverse(image_point)
        res_ = self.other_diffeo.inverse(image_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.diffeo.tangent(tangent_vec, base_point)
        res_ = self.other_diffeo.tangent(tangent_vec, base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_inverse_tangent(self, n_points, atol):
        image_point = self.image_data_generator.random_point(n_points)
        image_tangent_vec = self.image_data_generator.random_tangent_vec(image_point)

        res = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        res_ = self.other_diffeo.inverse_tangent(image_tangent_vec, image_point)
        self.assertAllClose(res, res_, atol=atol)
