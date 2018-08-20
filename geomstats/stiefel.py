"""
Stiefel manifold St(n,p),
a set of all orthonormal p-frames in n-dimensional space,
where p <= n
"""

import geomstats.backend as gs

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-6


def matrix_f(sq_mat, f):

    sq_mat = gs.to_ndarray(sq_mat, to_ndim=3)

    [eigenvalues, vectors] = gs.linalg.eig(sq_mat)

    exp_eigenvalues = f(eigenvalues)

    aux = gs.einsum('ijk,ik->ijk', vectors, exp_eigenvalues)
    exp_mat = gs.einsum('ijk,ikl->ijl', aux, gs.linalg.inv(vectors))

    return exp_mat.real


class Stiefel(EmbeddedManifold):

    def __init__(self, n, p):
        assert(n >= p)

        self.n = n
        self.p = p

        self.dimension = int(p * n - (p * (p + 1) / 2))

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to St(n,p),
        i.e. if it is a p-frame in n-dimensional space,
        and it is orthonormal.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, n, p = point.shape

        if (n, p) != (self.n, self.p):
            return gs.zeros((n_points,)).astype(bool)

        diff = gs.matmul(gs.transpose(point, axes=(0, 2, 1)), point) - gs.eye(p)
        point_norm = gs.norm(diff, axis=(1, 2))

        return gs.less_equal(point_norm, tolerance)

    def project(self):
        pass

    def random_uniform(self, n_samples=1):
        """
        Sample on St(n,p) with the uniform distribution.

        If Z(p,n) ~ N(0,1), then St(n,p) ~ U, according to Haar measure:
        St(n,p) := Z(Z^TZ)^{-1/2}
        """
        Z = gs.random.normal(shape=(n_samples, self.n, self.p))
        return gs.matmul(Z, gs.linalg.inv(
            matrix_f(gs.matmul(gs.transpose(Z, axes=(0, 2, 1)), Z), gs.sqrt)))


class StiefelMetric(RiemannianMetric):

    def __init__(self, dimension):
        self.dimension = dimension

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Compute the Frobenius inner product of tangent_vec_a and tangent_vec_b
        at base_point using the canonical Riemannian metric on St(n,p).
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape

        tangent_vec_b = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        assert(n_tangent_vecs_a == n_tangent_vecs_b)

        return gs.einsum("zij,zij->z", tangent_vec_a, tangent_vec_b)

    def mean(self, points,
             weights=None, n_max_iterations=32, epsilon=EPSILON):
        """
        Frechet mean of (weighted) points.
        """
        # TODO(nina): profile this code to study performance,
        # i.e. what to do with sq_dists_between_iterates.

        n_points = len(points)
        assert n_points > 0

        if weights is None:
            weights = gs.ones(n_points)

        n_weights = len(weights)
        assert n_points == n_weights
        sum_weights = gs.sum(weights)

        mean = points[0]
        if n_points == 1:
            return mean

        sq_dists_between_iterates = []
        iteration = 0
        while iteration < n_max_iterations:
            a_tangent_vector = self.log(mean, mean)
            tangent_mean = gs.zeros_like(a_tangent_vector)

            for i in range(n_points):
                # TODO(nina): abandon the for loop
                point_i = points[i]
                weight_i = weights[i]
                tangent_mean = tangent_mean + weight_i * self.log(
                                                    point=point_i,
                                                    base_point=mean)
            tangent_mean /= sum_weights

            mean_next = self.exp(
                tangent_vec=tangent_mean,
                base_point=mean)

            sq_dist = self.squared_dist(mean_next, mean)
            sq_dists_between_iterates.append(sq_dist)

            variance = self.variance(points=points,
                                     weights=weights,
                                     base_point=mean_next)
            if gs.isclose(variance, 0):
                break
            if sq_dist <= epsilon * variance:
                break

            mean = mean_next
            iteration += 1

        if iteration is n_max_iterations:
            print('Maximum number of iterations {} reached.'
                  'The mean may be inaccurate'.format(n_max_iterations))
        return mean

    def exp(self, tangent_vec, base_point):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        A = gs.matmul(gs.transpose(base_point, axes=(0, 2, 1)), tangent_vec)
        K = tangent_vec - gs.matmul(base_point, A)

        Q = gs.zeros((K.shape))
        R = gs.zeros((K.shape[0], K.shape[2], K.shape[2]))
        for i, k in enumerate(K):
            Q[i], R[i] = gs.linalg.qr(k)

        AR = gs.concatenate([A, -gs.transpose(R, axes=(0, 2, 1))], axis=2)
        RZ = gs.concatenate([R, gs.zeros((n_base_points, p, p))], axis=2)
        block = gs.concatenate([AR, RZ], axis=1)
        MNe = gs.expm(block)

        return gs.matmul(
            gs.concatenate([base_point, Q], axis=2), MNe[:, :, 0:p])

    def log(self, point, base_point, max_iter=100, tol=1e-6):
        """
        Riemannian logarithm of a point wrt a base point.

        Based on:
        Zimmermann, Ralf
        "A Matrix-Algebraic Algorithm for the Riemannian Logarithm
        on the Stiefel Manifold under the Canonical Metric"
        SIAM J. Matrix Anal. & Appl., 38(2), 322–342.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        M = gs.matmul(gs.transpose(base_point, (0, 2, 1)), point)

        # QR of normal component of a point
        K = point - gs.matmul(base_point, M)

        Q = gs.zeros((K.shape))
        N = gs.zeros((K.shape[0], K.shape[2], K.shape[2]))
        for i, k in enumerate(K):
            Q[i], N[i] = gs.linalg.qr(k)

        # orthogonal completion
        W = gs.concatenate([M, N], axis=1)

        V = gs.zeros((
            W.shape[0],
            max(W.shape[1], W.shape[2]),
            max(W.shape[1], W.shape[2])
            ))

        for i, w in enumerate(W):
            V[i], _ = gs.linalg.qr(w, mode="complete")

        # Procrustes preprocessing
        [D, S, R] = gs.linalg.svd(V[:, p:2*p, p:2*p])

        V[:, :, p:2*p] = gs.matmul(
            V[:, :, p:2*p], gs.matmul(R, gs.transpose(D, axes=(0, 2, 1))))
        V = gs.concatenate(
            [gs.concatenate([M, N], axis=1), V[:, :, p:2*p]], axis=2)

        for k in range(max_iter):

            LV = gs.logm(V)

            C = LV[:, p:2*p, p:2*p]
            normC = gs.linalg.norm(C, ord=2, axis=(1, 2))

            if normC < tol:
                # print("Converged in {} iterations".format(k+1))
                break

            Phi = gs.expm(-C)
            V[:, :, p:2*p] = gs.matmul(V[:, :, p:2*p], Phi)

        XV = gs.matmul(base_point, LV[:, 0:p, 0:p])
        QV = gs.matmul(Q, LV[:, p:2*p, 0:p])

        return XV + QV

    def retraction(self, tangent_vec, base_point):
        """
        Retraction map, based on QR-decomposion:
        P_x(V) = qf(X + V)
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        # Q, R = gs.linalg.qr(base_point + tangent_vec)
        # TODO: remove cycle, when qr will be vectorized
        Q = gs.zeros((base_point.shape))
        R = gs.zeros((
            base_point.shape[0], base_point.shape[2], base_point.shape[2]))
        for i, k in enumerate(base_point + tangent_vec):
            Q[i], R[i] = gs.linalg.qr(k)

        # flipping signs
        # Q = gs.matmul(Q, gs.diag(gs.sign(gs.sign(gs.diagonal(R)) + 0.5)))
        # TODO: remove cycle, when diag, diangonal will be vectorized
        result = gs.zeros((base_point.shape))
        for i, _ in enumerate(R):
            result[i] = gs.matmul(
                Q, gs.diag(gs.sign(gs.sign(gs.diagonal(R[i])) + 0.5)))

        return result

    def lifting(self, point, base_point):
        """
        Lifting map, based on QR-decomposion:
        P_x^{-1}(Q) = QR - X
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, p, n = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        def make_minor(i, M):
            return M[:i+1, :i+1]

        def make_r(i, M):
            if i == 0:
                if (M[0, 0] > 0):
                    return gs.array([1. / M[0, 0]])
                else:
                    raise Exception("M[0,0] <= 0")
            else:
                return M[:i+1, i]

        def make_b(i, M, r):

            b = gs.ones(i+1)

            for j in range(i):
                b[j] = - gs.matmul(M[i, :j+1], r[j])

            return b

        R = gs.zeros((n_base_points, n, n))
        M = gs.matmul(gs.transpose(base_point, axes=(0, 2, 1)), point)

        for k in range(n_base_points):
            r = []

            # construct r_0
            r.append(make_r(0, M[k]))

            for i in range(1, n):

                # get principal minor
                M_i = make_minor(i, M[k])

                if (gs.linalg.det(M_i) != 0):
                    b_i = make_b(i, M[k], r)
                    r_i = gs.matmul(gs.linalg.inv(M_i), b_i)

                    if r_i[i] <= 0:
                        raise Exception("(r_i)_i <= 0")
                    else:
                        r.append(r_i)
                else:
                    raise Exception("det(M_i) == 0, not invertible")

            for i, item in enumerate(r):
                R[k, :len(item), i] = gs.array(item)

        return gs.matmul(point, R) - base_point

    def dist(self):
        raise NotImplementedError('Geodesic distance is not implemented.')
