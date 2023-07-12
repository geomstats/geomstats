"""Class for the Sasaki metric.

A class implementing the Sasaki metric: The natural metric on the tangent
bundle TM of a Riemannian manifold M.

Lead authors: E. Nava-Yazdani, F. Ambellan, M. Hanik and C. von Tycowicz.
"""
from joblib import Parallel, delayed

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 3


def _gradient_descent(x_ini, i_pt, e_pt, grad, exp, lrate=0.1, max_iter=100, tol=1e-6):
    """Apply a gradient descent until max_iter or a given tolerance is reached."""
    x = x_ini
    for _ in range(max_iter):
        grad_x = grad(x, i_pt, e_pt)
        grad_norm = gs.linalg.norm(grad_x)
        if grad_norm < tol:
            break
        grad_x = -lrate * grad_x
        x = exp(grad_x, x)
    return x


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
        Base manifold of the tangent bundle.
    n_jobs: int
        Number of jobs for parallel computing.
        Optional, default: 1.

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

    def __init__(self, space, n_jobs=1):
        shape = (2, gs.prod(gs.array(space.shape)))

        self.n_jobs = n_jobs
        self.dim = 2 * space.dim
        self.shape = shape

        super().__init__(space=space)

    def exp(self, tangent_vec, base_point, n_steps=N_STEPS, **kwargs):
        """Compute the Riemannian exponential of a point.

        Exponential map at base_point of tangent_vec computed by
        shooting a Sasaki geodesic using an Euler integration on TM.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2, M.dim]
            Tangent vector in TTM at the base point in TM.
        base_point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.
        n_steps : int
            Number of discrete time steps.
            Optional, default: N_STEPS.

        Returns
        -------
        exp : array-like, shape=[..., 2, M.dim]
            Point on the tangent bundle TM.
        """
        bs_pts = gs.reshape(base_point, (-1, 2) + self._space.shape)
        tngs = gs.reshape(tangent_vec, bs_pts.shape)

        par_trans = self._space.metric.parallel_transport
        eps = 1 / n_steps

        v0, w0 = gs.take(tngs, 0, axis=1), gs.take(tngs, 1, axis=1)
        p0, u0 = gs.take(bs_pts, 0, axis=1), gs.take(bs_pts, 1, axis=1)
        for _ in range(n_steps):
            p = self._space.metric.exp(eps * v0, p0)
            u = par_trans(u0 + eps * w0, p0, end_point=p)
            v = par_trans(
                v0 - eps * (self._space.metric.curvature(u0, w0, v0, p0)),
                p0,
                end_point=p,
            )
            w = par_trans(w0, p0, end_point=p)
            p0, u0 = p, u
            v0, w0 = v, w

        return gs.reshape(gs.stack([p, u], axis=1), base_point.shape)

    def log(self, point, base_point, n_steps=N_STEPS, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Logarithmic map at base_point of point computed by iteratively relaxing
        a discretized geodesic between base_point and point.

        Parameters
        ----------
        point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.
        base_point : array-like, shape=[..., 2, M.dim]
            Point in the tangent bundle TM of manifold M.
        n_steps : int
            Number of discrete time steps.
            Optional, default: N_STEPS.

        Returns
        -------
        log : array-like, shape=[..., 2, M.dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point, base_point = gs.broadcast_arrays(point, base_point)

        pts = gs.reshape(point, (-1, 2) + self._space.shape)
        bs_pts = gs.reshape(base_point, (-1, 2) + self._space.shape)

        par_trans = self._space.metric.parallel_transport

        pu = self.geodesic_discrete(bs_pts, pts, n_steps)
        if len(pts) == 1:
            pu = gs.expand_dims(pu, axis=0)

        pu1 = gs.take(pu, 1, axis=1)
        p1, u1 = gs.take(pu1, 0, axis=1), gs.take(pu1, 1, axis=1)
        p0, u0 = gs.take(bs_pts, 0, axis=1), gs.take(bs_pts, 1, axis=1)
        w = par_trans(u1, p1, end_point=p0) - u0
        v = self._space.metric.log(point=p1, base_point=p0)
        rslt = n_steps * gs.stack([v, w], axis=1)

        return gs.reshape(gs.array(rslt), point.shape)

    def geodesic_discrete(self, initial_points, end_points, n_steps=N_STEPS, **kwargs):
        """Compute Sakai geodesic employing a variational time discretization.

        Parameters
        ----------
        end_points : array-like, shape=[..., 2, M.shape]
            Points in the tangent bundle TM of manifold M.
        initial_points : array-like, shape=[..., 2, M.shape]
            Points in the tangent bundle TM of manifold M.
        n_steps : int
            n_steps - 1 is the number of intermediate points in the
            discretization of the geodesic from initial_point to end_point
            Optional, default: N_STEPS.

        Returns
        -------
        geodesic : array-like, shape=[..., n_steps + 1, 2, M.shape]
            Discrete geodesics of form x(s)=(p(s), u(s)) in Sasaki metric
            connecting initial_point = x(0) and end_point = x(1).
        """
        metric = self._space.metric
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

            p = gs.take(pu, 0, axis=1)
            u = gs.take(pu, 1, axis=1)

            p1, p2, p3 = p[:-2], p[1:-1], p[2:]
            u1, u2, u3 = u[:-2], u[1:-1], u[2:]

            eps = 1 / n_steps

            v2 = metric.log(p3, p2) / eps
            w2 = (par_trans(u3, p3, end_point=p2) - u2) / eps

            gp = (metric.log(p3, p2) + metric.log(p1, p2)) / (
                2 * eps**2
            ) + metric.curvature(u2, w2, v2, p2)

            gu = (
                par_trans(u3, p3, end_point=p2)
                - 2 * u2
                + par_trans(u1, p1, end_point=p2)
            ) / eps**2

            return -gs.stack([gp, gu], axis=1) * eps

        @delayed
        def _geodesic_discrete(initial_point, end_point, n_stps):
            """Calculate the discrete geodesic."""
            p0, u0 = initial_point[0], initial_point[1]
            pL, uL = end_point[0], end_point[1]

            v = metric.log(pL, p0)
            s = gs.linspace(0.0, 1.0, n_stps + 1)
            pu_ini = []
            for i in range(1, n_stps):
                p_ini = metric.exp(s[i] * v, p0)
                u_ini = (1.0 - s[i]) * par_trans(u0, p0, end_point=p_ini) + s[
                    i
                ] * par_trans(uL, pL, end_point=p_ini)
                pu_ini.append(gs.array([p_ini, u_ini]))

            pu_ini = gs.array(pu_ini)
            x = _gradient_descent(pu_ini, initial_point, end_point, _grad, self.exp)

            return gs.vstack(
                [
                    gs.expand_dims(initial_point, axis=0),
                    x,
                    gs.expand_dims(end_point, axis=0),
                ]
            )

        i_pts = gs.reshape(initial_points, (-1, 2) + self._space.shape)
        e_pts = gs.reshape(end_points, (-1, 2) + self._space.shape)

        with Parallel(n_jobs=min(self.n_jobs, len(e_pts)), verbose=0) as parallel:
            rslt = parallel(
                _geodesic_discrete(i_pts[i % len(i_pts)], e_pt, n_steps)
                for i, e_pt in enumerate(e_pts)
            )

        rslt_shape = (-1, 2) + self._space.shape
        rslt_shape = rslt_shape if len(e_pts) == 1 else (len(e_pts),) + rslt_shape
        return gs.reshape(gs.array(rslt), rslt_shape)

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
        vec_a = gs.reshape(tangent_vec_a, (-1, 2) + self._space.shape)
        vec_b = gs.reshape(tangent_vec_b, (-1, 2) + self._space.shape)
        pt = gs.reshape(base_point, (-1, 2) + self._space.shape)

        inner = self._space.metric.inner_product
        rslt = inner(vec_a[:, 0], vec_b[:, 0], pt[:, 0]) + inner(
            vec_a[:, 1], vec_b[:, 1], pt[:, 0]
        )
        return rslt if gs.prod(gs.array(rslt.shape)) != 1 else gs.sum(rslt)
