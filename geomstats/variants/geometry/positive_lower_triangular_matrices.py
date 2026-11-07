"""Matrix logarithm on unit-diagonal lower triangular matrices with logm and expm."""

import geomstats.backend as gs
from geomstats.geometry.diffeo import Diffeo


def _block_triangular(mat, perturbation):
    """Build the block upper triangular matrix [[mat, perturbation], [0, mat]].

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Diagonal blocks.
    perturbation : array-like, shape=[..., n, n]
        Upper right block.

    Returns
    -------
    block_mat : array-like, shape=[..., 2 * n, 2 * n]
        Block matrix.
    """
    zeros = gs.zeros_like(mat + perturbation)
    mat = mat + zeros
    perturbation = perturbation + zeros
    top = gs.concatenate([mat, perturbation], axis=-1)
    bottom = gs.concatenate([zeros, mat], axis=-1)
    return gs.concatenate([top, bottom], axis=-2)


class LowerMatrixLog(Diffeo):
    r"""Matrix logarithm diffeomorphism.

    Variant of
    :class:`geomstats.geometry.positive_lower_triangular_matrices.LowerMatrixLog`
    computed with the general matrix logarithm and exponential.

    A diffeomorphism between the lower triangular matrices
    with unit diagonal and strictly lower triangular matrices.

    Notes
    -----
    The differentials use that, for a matrix function :math:`f`,

    .. math::

        f\left(\begin{bmatrix} X & E \\ 0 & X \end{bmatrix}\right)
        = \begin{bmatrix} f(X) & d_X f(E) \\ 0 & f(X) \end{bmatrix}
    """

    @staticmethod
    def __call__(base_point):
        """Compute the matrix log.

        Parameters
        ----------
        base_point : array_like, shape=[..., n, n]
            Lower triangular matrix with unit diagonal.

        Returns
        -------
        log : array_like, shape=[..., n, n]
            Strictly lower triangular matrix.
        """
        return gs.linalg.logm(base_point)

    @staticmethod
    def inverse(image_point):
        """Compute the matrix exponential.

        Parameters
        ----------
        image_point : array_like, shape=[..., n, n]
            Strictly lower triangular matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Lower triangular matrix with unit diagonal.
        """
        return gs.linalg.expm(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
        image_point : array-like, shape=[..., n, n]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., n, n]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        n = base_point.shape[-1]
        block_mat = gs.linalg.logm(_block_triangular(base_point, tangent_vec))
        return block_mat[..., :n, n:]

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., n, n]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., n, n]
            Image point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        if image_point is None:
            image_point = self(base_point)

        n = image_point.shape[-1]
        block_mat = gs.linalg.expm(_block_triangular(image_point, image_tangent_vec))
        return block_mat[..., :n, n:]
