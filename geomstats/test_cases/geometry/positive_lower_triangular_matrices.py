import geomstats.backend as gs
from geomstats.geometry.diffeo import AutodiffDiffeo
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class LowerMatrixLog(AutodiffDiffeo):
    """Matrix logarithm diffeomorphism.

    An alternative implementation for testing
    purposes.
    """

    @staticmethod
    def __call__(base_point):
        """Compute the matrix log.

        Parameters
        ----------
        base_point : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        log : array_like, shape=[..., n, n]
            Matrix logarithm of base_point.
        """
        return gs.linalg.logm(base_point)

    @staticmethod
    def inverse(image_point):
        """Compute the matrix exponential.

        Parameters
        ----------
        image_point : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Exponential of image_point.
        """
        return gs.linalg.expm(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.
        """
        raise NotImplementedError("Not implemented.")


class CholeskyMetricTestCase(RiemannianMetricTestCase):
    def test_diag_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.diag_inner_product(
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_strictly_lower_inner_product(
        self, tangent_vec_a, tangent_vec_b, expected, atol
    ):
        res = self.space.metric.strictly_lower_inner_product(
            tangent_vec_a,
            tangent_vec_b,
        )
        self.assertAllClose(res, expected, atol=atol)
