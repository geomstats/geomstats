"""Class for the Sasaki metric.

A class implementing the Sasaki metric: The natural metric on the tangent bundle TM
of a Riemannian manifold M.

Lead authors: E. Nava-Yazdani, F. Ambellan, M. Hanik and C. von Tycowicz.
"""
import os
from joblib import Parallel, delayed

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

N_STEPS = 3


def _gradient_descent(x_ini, grad, exp, lrate=0.1, max_iter=100, tol=1e-6):
    """Apply a gradient descent until max_iter or a given tolerance is reached."""
    x = x_ini
    for _ in range(max_iter):
        grad_x = grad(x)
        grad_norm = gs.linalg.norm(grad_x)
        if grad_norm < tol:
            break
        grad_x = -lrate * grad_x
        x = exp(grad_x, gs.array(x))
        x = [*x]
    return list(x)


class SasakiMetric(RiemannianMetric):
    """Implements of the Sasaki metric on the tangent bundle TM of a Riem. manifold M.

    The Sasaki metric is characterized by the following three properties:
     * the canonical projection of TM becomes a Riemannian submersion,
     * parallel vector fields along curves are orthogonal to their fibres, and
     * its restriction to any tangent space is Euclidean.

    Geodesic computations are realized via a discrete formulation of the geodesic
    equation on TM that involve geodesics, parallel translation, and the curvature
    tensor on the base manifold M (see [1] for details).

    Parameters
    ----------
    metric : RiemannianMetric
        Metric of the base manifold of the tangent bundle.

    References
    ----------
    .. [1] Muralidharan, P., & Fletcher, P. T. (2012, June).
    Sasaki metrics for analysis of longitudinal data on manifolds.
    In 2012 IEEE conference on computer vision and pattern recognition (pp. 1027-1034).
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4270017/
    """

    def __init__(self, metric: RiemannianMetric):
        self.metric = metric
        shape = (2, gs.prod(gs.array(metric.shape)))

        super(SasakiMetric, self).__init__(2 * metric.dim, shape=shape,
                                           default_point_type='matrix')

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
        bs_pts = gs.reshape(base_point, (-1, 2) + self.metric.shape)
        tngs = gs.reshape(tangent_vec, bs_pts.shape)

        metric = self.metric
        par_trans = metric.parallel_transport
        eps = 1 / n_steps

        v0, w0 = tngs[:, 0], tngs[:, 1]
        p0, u0 = bs_pts[:, 0], bs_pts[:, 1]
        for _ in range(n_steps):
            p = metric.exp(eps * v0, p0)
            u = par_trans(u0 + eps * w0, p0, None, p)
            v = par_trans(v0 - eps * (metric.curvature(u0, w0, v0, p0)), p0, None, p)
            w = par_trans(w0, p0, None, p)
            p0, u0 = p, u
            v0, w0 = v, w

        return gs.reshape(gs.array(list(zip(p, u))), base_point.shape)

    def log(self, point, base_point, n_steps=N_STEPS, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Logarithmic map at base_point of point computed by
        iteratively relaxing a discretized geodesic between base_point and point.

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
        pts = gs.reshape(point, (-1, 2) + self.metric.shape)
        bs_pts = gs.reshape(base_point, (-1, 2) + self.metric.shape)

        metric = self.metric
        par_trans = metric.parallel_transport
        n_steps = n_steps

        @delayed
        def do_log(pt, bs_pt):
            """Calculate the logarithm."""
            pu = self.geodesic_discrete(bs_pt, pt, n_steps)
            p1, u1 = pu[1][0], pu[1][1]
            p0, u0 = bs_pt[0], bs_pt[1]
            w = (par_trans(u1, p1, None, p0) - u0)
            v = metric.log(point=p1, base_point=p0)
            return n_steps * gs.array([v, w])

        n_jobs = min(os.cpu_count(), len(pts))
        with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
            rslt = parallel(do_log(pt, bs_pts[i % len(bs_pts)])
                            for i, pt in enumerate(pts))

        return gs.reshape(gs.array(rslt), point.shape)

    def geodesic_discrete(self, initial_point, end_point, n_steps=N_STEPS, **kwargs):
        """Compute Sakai geodesic employing a variational time discretization.

        Parameters
        ----------
        end_point : array-like, shape=[2, M.shape]
            Point in the tangent bundle TM of manifold M.
        initial_point : array-like, shape=[2, M.shape]
            Point in the tangent bundle TM of manifold M.
        n_steps : int
            n_steps - 1 is the number of intermediate points in the discretization
            of the geodesic from initial_point to end_point
            Optional, default: N_STEPS.

        Returns
        -------
        geodesic : array-like, shape=[n_steps + 1, 2, M.shape]
            Discrete geodesic x(s)=(p(s), u(s)) in Sasaki metric connecting
            initial_point = x(0) and end_point = x(1).
        """
        metric = self.metric
        par_trans = metric.parallel_transport
        p0, u0 = initial_point[0], initial_point[1]
        pL, uL = end_point[0], end_point[1]

        def grad(pu):
            """Gradient of discrete geodesic energy."""
            g = []
            # add boundary points to the list of points
            pu = [initial_point] + pu + [end_point]
            for j in range(n_steps - 1):
                p1, u1 = pu[j][0], pu[j][1]
                p2, u2 = pu[j + 1][0], pu[j + 1][1]
                p3, u3 = pu[j + 2][0], pu[j + 2][1]
                p4, u4 = (p3, u3) if j + 3 >= len(pu) else (pu[j + 3][0], pu[j + 3][1])

                v, w = metric.log(p3, p2), par_trans(u3, p3, None, p2) - u2
                w1 = par_trans(u2, p2, None, p1) - u1
                w3 = par_trans(u4, p4, None, p3) - u3
                gp = metric.log(p3, p2) + metric.log(p1, p2) + metric.curvature(
                    u2, w, v, p2)
                gu = par_trans(w3, p3, None, p2) - par_trans(w1, p1, None, p2)
                g.append([gp, gu])
            return -n_steps * gs.array(g)

        # Initial guess for gradient_descent
        v = metric.log(pL, p0)
        s = gs.linspace(0., 1., n_steps + 1)
        pu_ini = []
        for i in range(1, n_steps):
            p_ini = metric.exp(s[i] * v, p0)
            u_ini = (1 - s[i]) * par_trans(u0, p0, None, p_ini) + s[i] * par_trans(
                uL, pL, None, p_ini)
            pu_ini.append(gs.array([p_ini, u_ini]))

        # Minimization by gradient descent
        x = _gradient_descent(pu_ini, grad, self.exp)

        return gs.array([initial_point] + x + [end_point])

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
        # unflatten
        vec_a = gs.reshape(tangent_vec_a, (-1, 2) + self.metric.shape)
        vec_b = gs.reshape(tangent_vec_b, (-1, 2) + self.metric.shape)
        pt = gs.reshape(base_point, (-1, 2) + self.metric.shape)

        # compute Sasaki inner product via metric of underlying manifold
        inner = self.metric.inner_product
        return inner(vec_a[:, 0], vec_b[:, 0], pt[:, 0]) + inner(vec_a[:, 1],
                                                                 vec_b[:, 1], pt[:, 0])
