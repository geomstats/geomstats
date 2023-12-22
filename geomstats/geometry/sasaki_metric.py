"""Class for the Sasaki metric.

A class implementing the Sasaki metric: The natural metric on the tangent
bundle TM of a Riemannian manifold M.

Lead authors: E. Nava-Yazdani, F. Ambellan, M. Hanik and C. von Tycowicz.
"""
from joblib import Parallel, delayed

import geomstats.backend as gs
from geomstats.geometry.base import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import check_is_batch


class GradientDescent:
    """Gradient descent algorithm."""

    def __init__(self, lrate=0.1, max_iter=100, tol=1e-6):
        self.lrate = lrate
        self.max_iter = max_iter
        self.tol = tol

    def minimize(self, x_ini, i_pt, e_pt, grad, exp):
        """Apply a gradient descent until max_iter or a given tolerance is reached."""
        x = x_ini
        for _ in range(self.max_iter):
            grad_x = grad(x, i_pt, e_pt)
            grad_norm = gs.linalg.norm(grad_x)
            if grad_norm < self.tol:
                break
            grad_x = -self.lrate * grad_x
            x = exp(grad_x, x)
        return x


class TangentBundle(Manifold):
    """Tangent bundle of a space."""

    def __init__(self, space, equip=True):
        self.space = space
        super().__init__(dim=2 * space.dim, shape=(2,) + space.shape, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SasakiMetric

    def _unstack(self, point):
        return (
            point[..., 0, -self.space.point_ndim + 1 :],
            point[..., 1, -self.space.point_ndim + 1 :],
        )

    def _stack(self, space_point, space_tangent_vec):
        return gs.stack([space_point, space_tangent_vec], axis=-self.point_ndim)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        space_point, space_tangent_vec = self._unstack(point)
        return gs.logical_and(
            self.space.belongs(space_point),
            self.space.is_tangent(space_tangent_vec, space_point),
        )

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold according to some distribution.

        If the manifold is compact, preferably a uniform distribution will be used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., *point_shape]
            Points sampled on the manifold.
        """
        space_point = self.space.random_point(n_samples, bound)
        space_tangent_vec = self.space.random_tangent_vec(space_point)
        return self._stack(space_point, space_tangent_vec)

    @staticmethod
    def projection(point):
        """Project a point to the vector space.

        This method is for compatibility and returns `point`. `point` should
        have the right shape,

        Parameters
        ----------
        point: array-like, shape[..., *point_shape]
            Point.

        Returns
        -------
        point: array-like, shape[..., *point_shape]
            Point.
        """
        return gs.copy(point)

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : array-like, shape=[...,]
            Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError("`is_tangent` is not implemented")

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space.

        This method is for compatibility and returns vector.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point in the vector space

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        raise NotImplementedError("`to_tangent` is not implemented")

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec."""
        raise NotImplementedError("`random_tangent_vec` is not implemented")


class SasakiMetric(RiemannianMetric):
    """Implements of the Sasaki metric on the tangent bundle TM of a Riem. manifold M.

    The Sasaki metric is characterized by the following three properties:

    * the canonical projection of TM becomes a Riemannian submersion,
    * parallel vector fields along curves are orthogonal to their fibres, and
    * its restriction to any tangent space is Euclidean.

    Geodesic computations are realized via a discrete formulation of the
    geodesic equation on TM that involve geodesics, parallel translation, and
    the curvature tensor on the base manifold M (see [1]_ for details).
    However, as the implemented energy in the discrete-geodesics-optimization
    as well as the approximations of its gradient slightly differ from those
    proposed in [1]_, we also refer to [2]_ for additional details.

    Parameters
    ----------
    space : Manifold
        Tangent bundle.
    n_jobs: int
        Number of jobs for parallel computing.
        Optional, default: 1.
    n_steps : int
        Number of discrete time steps.
        Optional, default: 3.

    References
    ----------
    .. [1] Muralidharan, P., & Fletcher, P. T. "Sasaki metrics for analysis of
        longitudinal data on manifolds", IEEE CVPR 2012, pp. 1027-1034
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4270017/
    .. [2] Nava-Yazdani, E., Hanik, M., Ambellan, F., & von Tycowicz, C.
        "On Gradient Formulas in an Algorithm for the Logarithm of the Sasaki
        Metric", Technical Report Zuse-Institut Berlin, 2022
        https://nbn-resolving.org/urn/resolver.pl?urn:nbn:de:0297-zib-87174
    """

    def __init__(self, space, n_jobs=1, n_steps=3):
        super().__init__(space=space)
        self.n_jobs = n_jobs
        self.n_steps = n_steps
        self._gradient_descent = GradientDescent()

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a point.

        Exponential map at base_point of tangent_vec computed by
        shooting a Sasaki geodesic using an Euler integration on TM.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2, M.dim]
            Tangent vector in TTM at the base point in TM.
        base_point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.


        Returns
        -------
        exp : array-like, shape=[..., 2, M.dim]
            Point on the tangent bundle TM.
        """
        par_trans = self._space.space.metric.parallel_transport
        eps = 1 / self.n_steps

        v0, w0 = self._space._unstack(tangent_vec)
        p0, u0 = self._space._unstack(base_point)

        for _ in range(self.n_steps):
            p = self._space.space.metric.exp(eps * v0, p0)
            u = par_trans(u0 + eps * w0, p0, end_point=p)
            v = par_trans(
                v0 - eps * (self._space.space.metric.curvature(u0, w0, v0, p0)),
                p0,
                end_point=p,
            )
            w = par_trans(w0, p0, end_point=p)
            p0, u0 = p, u
            v0, w0 = v, w

        return self._space._stack(p, u)

    def log(self, point, base_point):
        """Compute the Riemannian logarithm of a point.

        Logarithmic map at base_point of point computed by iteratively relaxing
        a discretized geodesic between base_point and point.

        Parameters
        ----------
        point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.
        base_point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.

        Returns
        -------
        log : array-like, shape=[..., 2, M.dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        par_trans = self._space.space.metric.parallel_transport

        pu = self.geodesic_discrete(base_point, point)
        pu1 = gs.take(pu, 1, axis=-(self._space.point_ndim + 1))

        p1, u1 = self._space._unstack(pu1)
        p0, u0 = self._space._unstack(base_point)

        w = par_trans(u1, p1, end_point=p0) - u0
        v = self._space.space.metric.log(point=p1, base_point=p0)
        return self.n_steps * self._space._stack(v, w)

    def geodesic_discrete(self, initial_point, end_point):
        """Compute Sakai geodesic employing a variational time discretization.

        Parameters
        ----------
        end_points : array-like, shape=[..., 2, M.shape]
            Points in the tangent bundle TM of manifold M.
        initial_points : array-like, shape=[..., 2, M.shape]
            Points in the tangent bundle TM of manifold M.

        Returns
        -------
        geodesic : array-like, shape=[..., n_steps + 1, 2, M.shape]
            Discrete geodesics of form x(s)=(p(s), u(s)) in Sasaki metric
            connecting initial_point = x(0) and end_point = x(1).
        """
        metric = self._space.space.metric
        par_trans = metric.parallel_transport

        def _grad(pu, i_pt, e_pt):
            """Gradient of discrete geodesic energy."""
            pu = gs.vstack(
                [
                    gs.expand_dims(i_pt, axis=0),
                    pu,
                    gs.expand_dims(e_pt, axis=0),
                ]
            )
            p, u = self._space._unstack(pu)

            p1, p2, p3 = p[:-2], p[1:-1], p[2:]
            u1, u2, u3 = u[:-2], u[1:-1], u[2:]

            eps = 1 / self.n_steps

            v2 = metric.log(p3, p2) / eps
            w2 = (par_trans(u3, p3, end_point=p2) - u2) / eps

            gp = (metric.log(p3, p2) + metric.log(p1, p2)) / (
                2 * eps**2
            ) - metric.curvature(u2, w2, v2, p2)

            gu = (
                par_trans(u3, p3, end_point=p2)
                - 2 * u2
                + par_trans(u1, p1, end_point=p2)
            ) / eps**2

            return -self._space._stack(gp, gu) * eps

        def _geodesic_discrete_single(initial_point, end_point):
            """Calculate the discrete geodesic."""
            ijk = "ijk"[: self._space.space.point_ndim]

            def _scalarmul(scalar, point):
                return gs.einsum(f"...,...{ijk}->...{ijk}", scalar, point)

            p0, u0 = initial_point[0], initial_point[1]
            pL, uL = end_point[0], end_point[1]

            v = metric.log(pL, p0)
            s = gs.linspace(0.0, 1.0, self.n_steps + 1)[1:-1]

            p_ini = metric.exp(gs.einsum(f"p, {ijk}->p{ijk}", s, v), p0)
            u_ini = _scalarmul(
                (1.0 - s), par_trans(u0, p0, end_point=p_ini)
            ) + _scalarmul(s, par_trans(uL, pL, end_point=p_ini))
            pu_ini = self._space._stack(p_ini, u_ini)

            x = self._gradient_descent.minimize(
                pu_ini, initial_point, end_point, _grad, self.exp
            )

            return gs.vstack(
                [
                    gs.expand_dims(initial_point, axis=0),
                    x,
                    gs.expand_dims(end_point, axis=0),
                ]
            )

        is_batch = check_is_batch(self._space.point_ndim, initial_point, end_point)
        if not is_batch:
            return _geodesic_discrete_single(initial_point, end_point)

        initial_point, end_point = gs.broadcast_arrays(initial_point, end_point)

        with Parallel(n_jobs=min(self.n_jobs, len(end_point)), verbose=0) as parallel:
            rslt = parallel(
                delayed(_geodesic_discrete_single)(i_pt, e_pt)
                for i_pt, e_pt in zip(initial_point, end_point)
            )
        return gs.array(rslt)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 2, M.dim]
            Tangent vector in TTM of the tangent bundle TM.
        tangent_vec_b : array-like, shape=[..., 2, M.dim]
            Tangent vector in TTM of the tangent bundle TM.
        base_point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.

        Returns
        -------
        inner_product : array-like, shape=[..., 1]
            Inner-product.
        """
        vec_a_0, vec_a_1 = self._space._unstack(tangent_vec_a)
        vec_b_0, vec_b_1 = self._space._unstack(tangent_vec_b)
        pt, _ = self._space._unstack(base_point)

        inner = self._space.space.metric.inner_product
        return inner(vec_a_0, vec_b_0, pt) + inner(vec_a_1, vec_b_1, pt)
