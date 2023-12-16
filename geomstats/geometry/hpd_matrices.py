"""The manifold of Hermitian positive definite (HPD) matrices.

Lead author: Yann Cabanes.


References
----------
.. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
.. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

import math

import geomstats.backend as gs
from geomstats.geometry.base import ComplexVectorSpaceOpenSet
from geomstats.geometry.complex_matrices import ComplexMatrices, ComplexMatricesMetric
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import HermitianMatrices, expmh, powermh
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.spd_matrices import SymMatrixLog, logmh
from geomstats.integrator import integrate
from geomstats.vectorization import repeat_out


class HPDMatrices(ComplexVectorSpaceOpenSet):
    """Class for the manifold of Hermitian positive definite (HPD) matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        super().__init__(
            dim=n**2, shape=(n, n), embedding_space=HermitianMatrices(n), equip=equip
        )
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return HPDAffineMetric

    @staticmethod
    def belongs(point, atol=gs.atol):
        """Check if a matrix is Hermitian with positive eigenvalues.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an HPD matrix.
        """
        return ComplexMatrices.is_hpd(point, atol)

    def projection(self, point, atol=gs.atol):
        """Project a matrix to the space of HPD matrices.

        First the Hermitian part of point is computed, then the eigenvalues
        are floored to gs.atol.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            HPD matrix.
        """
        herm = ComplexMatrices.to_hermitian(point)
        eigvals, eigvecs = gs.linalg.eigh(herm)
        regularized = gs.where(eigvals < atol, atol, eigvals)
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)
        return Matrices.mul(reconstruction, ComplexMatrices.transconjugate(eigvecs))

    def random_point(self, n_samples=1, bound=0.1):
        """Sample in HPD(n) from the log-uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in HPD(n).
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)
        eye = gs.eye(n, dtype=gs.get_default_cdtype())
        samples = gs.stack([eye for i_sample in range(n_samples)], axis=0)
        samples = gs.reshape(samples, size)
        samples += bound * gs.random.rand(*size, dtype=gs.get_default_cdtype())
        samples = self.projection(samples)
        return samples

    def random_tangent_vec(self, base_point, n_samples=1):
        """Sample on the tangent space of HPD(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in the tangent space at base_point.
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)

        sqrt_base_point = gs.cast(
            gs.linalg.sqrtm(base_point),
            base_point.dtype,
        )

        tangent_vec_at_id_aux = gs.random.rand(*size, dtype=gs.get_default_cdtype())
        tangent_vec_at_id_aux *= 2
        tangent_vec_at_id_aux -= 1 + 1j
        tangent_vec_at_id = tangent_vec_at_id_aux + ComplexMatrices.transconjugate(
            tangent_vec_at_id_aux
        )

        return Matrices.mul(sqrt_base_point, tangent_vec_at_id, sqrt_base_point)

    from_vector = HermitianMatrices.__dict__["from_vector"]
    to_vector = HermitianMatrices.__dict__["to_vector"]


class HPDAffineMetric(ComplexRiemannianMetric):
    """Class for the affine-invariant metric on the HPD manifold."""

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the affine-invariant inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[..., n, n]
            Inner-product.
        """
        inv_base_point = GeneralLinear.inverse(base_point)
        aux_a = Matrices.mul(inv_base_point, tangent_vec_a)
        aux_b = Matrices.mul(inv_base_point, tangent_vec_b)

        return Matrices.trace_product(aux_a, aux_b)

    def exp(self, tangent_vec, base_point):
        """Compute the affine-invariant exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric defined in inner_product.
        This gives a Hermitian positive definite matrix.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        sqrt_base_point, inv_sqrt_base_point = powermh(base_point, [1.0 / 2, -1.0 / 2])

        tangent_vec_at_id = Matrices.mul(
            inv_sqrt_base_point, tangent_vec, inv_sqrt_base_point
        )

        tangent_vec_at_id = ComplexMatrices.to_hermitian(tangent_vec_at_id)
        exp_from_id = expmh(tangent_vec_at_id)

        return Matrices.mul(sqrt_base_point, exp_from_id, sqrt_base_point)

    def log(self, point, base_point):
        """Compute the affine-invariant logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in inner_product.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm of point at base_point.
        """
        sqrt_base_point, inv_sqrt_base_point = powermh(base_point, [1.0 / 2, -1.0 / 2])
        point_near_id = Matrices.mul(inv_sqrt_base_point, point, inv_sqrt_base_point)

        # TODO: only this differs
        point_near_id = ComplexMatrices.to_hermitian(point_near_id)

        log_at_id = logmh(point_near_id)
        return Matrices.mul(sqrt_base_point, log_at_id, sqrt_base_point)

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.
        Denoting `tangent_vec_a` by `S`, `base_point` by `A`, and `end_point`
        by `B` or `B = Exp_A(tangent_vec_b)` and :math:`E = (BA^{- 1})^{( 1
        / 2)}`. Then the parallel transport to `B` is:

        .. math::
            S' = ESE^T

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold of HPD matrices. Point to transport from
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed. Unused if `end_point` is given.
            Optional, default: None.
        end_point : array-like, shape=[..., n, n]
            Point on the manifold of HPD matrices. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        if end_point is None:
            end_point = self.exp(direction, base_point)
        sqrt_bp, inv_sqrt_bp = powermh(base_point, [1.0 / 2, -1.0 / 2])
        pdt = powermh(Matrices.mul(inv_sqrt_bp, end_point, inv_sqrt_bp), 1.0 / 2)
        congruence_mat = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        return ComplexMatrices.congruent(tangent_vec, congruence_mat)

    def injectivity_radius(self, base_point):
        """Radius of the largest ball where the exponential is injective.

        Because of the negative curvature of this space, the injectivity radius
        is infinite everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class HPDBuresWassersteinMetric(ComplexRiemannianMetric):
    """Class for the Bures-Wasserstein metric on the HPD manifold."""

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the Bures-Wasserstein inner-product.

        Compute the inner-product of tangent_vec_a :math:`A` and tangent_vec_b
        :math:`B` at point base_point :math:`S=PDP^\top` using the
        Bures-Wasserstein Riemannian metric:

        .. math::
            \frac{1}{2}\sum_{i,j}\frac{[P^\top AP]_{ij}[P^\top BP]_{ij}}{d_i+d_j}

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        eigvals, eigvecs = gs.linalg.eigh(base_point)
        transconj_eigvecs = ComplexMatrices.transconjugate(eigvecs)
        rotated_tangent_vec_a = Matrices.mul(transconj_eigvecs, tangent_vec_a, eigvecs)
        rotated_tangent_vec_b = Matrices.mul(transconj_eigvecs, tangent_vec_b, eigvecs)

        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        result = (
            ComplexMatrices.frobenius_product(
                gs.cast(coefficients, dtype=rotated_tangent_vec_a.dtype)
                * rotated_tangent_vec_a,
                rotated_tangent_vec_b,
            )
            / 2
        )

        return result

    def exp(self, tangent_vec, base_point):
        """Compute the Bures-Wasserstein exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[...,]
            Riemannian exponential.
        """
        eigvals, eigvecs = gs.linalg.eigh(base_point)
        transconj_eigvecs = ComplexMatrices.transconjugate(eigvecs)
        rotated_tangent_vec = Matrices.mul(transconj_eigvecs, tangent_vec, eigvecs)
        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        rotated_sylvester = rotated_tangent_vec * gs.cast(
            coefficients, dtype=rotated_tangent_vec.dtype
        )
        rotated_hessian = gs.einsum("...ij,...j->...ij", rotated_sylvester, eigvals)
        rotated_hessian = Matrices.mul(rotated_hessian, rotated_sylvester)
        hessian = Matrices.mul(eigvecs, rotated_hessian, transconj_eigvecs)

        return base_point + tangent_vec + hessian

    def log(self, point, base_point):
        """Compute the Bures-Wasserstein logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Bures-Wasserstein metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm.
        """
        sqrt_bp, inv_sqrt_bp = powermh(base_point, [0.5, -0.5])
        pdt = powermh(Matrices.mul(sqrt_bp, point, sqrt_bp), 0.5)
        sqrt_product = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        transconj_sqrt_product = ComplexMatrices.transconjugate(sqrt_product)
        return sqrt_product + transconj_sqrt_product - 2 * base_point

    def squared_dist(self, point_a, point_b):
        """Compute the Bures-Wasserstein squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Point.
        point_b : array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            Riemannian squared distance.
        """
        product = gs.matmul(point_a, point_b)
        sqrt_product = gs.linalg.sqrtm(product)
        trace_a = gs.trace(point_a)
        trace_b = gs.trace(point_b)
        trace_prod = gs.trace(sqrt_product)

        squared_dist = gs.real(trace_a + trace_b - 2.0 * trace_prod)

        return gs.where(squared_dist < 0.0, 0.0, squared_dist)

    def parallel_transport(
        self,
        tangent_vec,
        base_point,
        direction=None,
        end_point=None,
        n_steps=10,
        step="rk4",
    ):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Approximation of the solution of the parallel transport of a tangent
        vector a along the geodesic defined by :math:`t \mapsto exp_{(
        base\_point)}(t* tangent\_vec\_b)`. The parallel transport equation is
        formulated in this case in [TP2021]_.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., n, n]
            Initial point of the geodesic.
        direction : array-like, shape=[..., n, n]
            Tangent vector at `base_point`, initial velocity of the geodesic to
            transport along.
        end_point : array-like, shape=[..., n, n]
            Point to transport to.
            Optional, default: None.
        n_steps : int
            Number of steps to use to approximate the solution of the
            ordinary differential equation.
            Optional, default: 100
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use in the integration scheme.
            Optional, default: 'rk4'.

        Returns
        -------
        transported :  array-like, shape=[..., n, n]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.

        See Also
        --------
        Integration module: geomstats.integrator
        """
        if end_point is None:
            end_point = self.exp(direction, base_point)

        horizontal_lift_a = gs.linalg.solve_sylvester(
            base_point, base_point, tangent_vec
        )

        square_root_bp, inverse_square_root_bp = powermh(base_point, [0.5, -0.5])
        end_point_lift = Matrices.mul(square_root_bp, end_point, square_root_bp)
        square_root_lift = powermh(end_point_lift, 0.5)

        horizontal_velocity = gs.matmul(inverse_square_root_bp, square_root_lift)
        partial_horizontal_velocity = Matrices.mul(horizontal_velocity, square_root_bp)
        partial_horizontal_velocity += ComplexMatrices.transconjugate(
            partial_horizontal_velocity
        )

        def force(state, time):
            horizontal_geodesic_t = (
                1 - time
            ) * square_root_bp + time * horizontal_velocity
            geodesic_t = (
                (1 - time) ** 2 * base_point
                + time * (1 - time) * partial_horizontal_velocity
                + time**2 * end_point
            )

            align = Matrices.mul(
                horizontal_geodesic_t,
                ComplexMatrices.transconjugate(horizontal_velocity - square_root_bp),
                state,
            )
            right = align + ComplexMatrices.transconjugate(align)
            return gs.linalg.solve_sylvester(geodesic_t, geodesic_t, -right)

        flow = integrate(force, horizontal_lift_a, n_steps=n_steps, step=step)
        final_align = Matrices.mul(end_point, flow[-1])
        return final_align + ComplexMatrices.transconjugate(final_align)

    def injectivity_radius(self, base_point):
        """Compute the upper bound of the injectivity domain.

        This is the smallest eigen value of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        eigen_values = gs.linalg.eigvalsh(base_point)
        return eigen_values[..., 0] ** 0.5


class HPDEuclideanMetric(ComplexMatricesMetric):
    """Class for the Euclidean metric on the HPD manifold."""

    @staticmethod
    def exp_domain(tangent_vec, base_point):
        """Compute the domain of the Euclidean exponential map.

        Compute the real interval of time where the Euclidean geodesic starting
        at point `base_point` in direction `tangent_vec` is defined.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp_domain : array-like, shape=[..., 2]
            Interval of time where the geodesic is defined.
        """
        invsqrt_base_point = powermh(base_point, -0.5)
        reduced_vec = gs.matmul(invsqrt_base_point, tangent_vec)
        reduced_vec = gs.matmul(reduced_vec, invsqrt_base_point)
        eigvals = gs.linalg.eigvalsh(reduced_vec)
        min_eig = gs.amin(eigvals, axis=-1)
        max_eig = gs.amax(eigvals, axis=-1)
        inf_value = gs.where(max_eig <= 0.0, gs.array(-math.inf), -1.0 / max_eig)
        sup_value = gs.where(min_eig >= 0.0, gs.array(-math.inf), -1.0 / min_eig)
        return gs.stack((inf_value, sup_value), axis=-1)

    def injectivity_radius(self, base_point):
        """Compute the upper bound of the injectivity domain.

        This is the smallest eigen value of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        eigen_values = gs.linalg.eigvalsh(base_point)
        return eigen_values[..., 0]


class HPDLogEuclideanMetric(PullbackDiffeoMetric):
    """Class for the Log-Euclidean metric on the HPD manifold."""

    def __init__(self, space, image_space=None):
        if image_space is None:
            image_space = HermitianMatrices(n=space.n)
        diffeo = SymMatrixLog()
        super().__init__(space, diffeo, image_space)
