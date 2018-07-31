"""
Stiefel manifold St(p,n),
a set of all orthonormal p-frames in n-dimensional space,
where p <= n
"""

import logging
import random

import geomstats.backend as gs

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-6

class Stiefel(EmbeddedManifold):

    def __init__(self, p, n):
        self.p = p
        self.n = n

        self.dimension = int(p * n - (p * (p - 1) / 2))

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to St(p,n),
        i.e. if it is a p-frame in n-dimensional space,
        and it is orthonormal.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        (_, point_dim) = point.shape

        if point_dim != (self.n, self.p):
            return False
        
        point_norm = gs.norm(gs.dot(gs.transpose(point), point) - gs.eye(point_dim[1]))

        return gs.less_equal(point_norm, tolerance)

    def project(self):
        pass

    def random_uniform(self, n_samples=1):
        pass

class StiefelMetric(RiemannianMetric):

    def __init__(self, dimension):
        self.dimension = dimension

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the canonical Riemannian metric on St(p,n).
        """
        pass

    def norm(self, tangent_vec_a, tangent_vec_b):
        pass

    def exp(self):
        raise NotImplementedError('Exponential mapping is not implemented.')

    def log(self):
        raise NotImplementedError('Logarithmic mapping is not implemented.')

    def mean(self, points, epsilon=EPSILON):
        """
        Compute Riemannian mean on St(p,n), by applying retraction to
        the arithmetic average of the lifted points from the manifold to tangent space.
        """
        X = gs.asarray(points)

        i = 0
        converged = False

        # set random element as initial mean
        random_id = random.randrange(int(gs.shape(X)[0]))
        X_mean = X[random_id]
        
        while not converged:
            i += 1
            X_mean_prev = gs.copy(X_mean)

            X_mean = self.mean_iteration(X_mean, X)

            if gs.linalg.norm(X_mean - X_mean_prev) <= epsilon:
                converged = True

        return X_mean

    def mean_iteration(self, X, Qs):
        """
        Iteration of a fixed point algorithm
        for computing Riemannian mean on St(p,n).

        Based on:
        Kaneko, Tetsuya, Simone G. O. Fiori and Toshihisa Tanaka. 
        "Empirical Arithmetic Averaging Over the Compact Stiefel Manifold."
        IEEE Transactions on Signal Processing 61 (2013): 883-894.
        """

        # lift points to tangent space
        V = []
        for i, Q in enumerate(Qs):
            try:
                v = self.lifting(X, Q)
                V.append(v)
            except Exception as error:
                pass
                print("Q does not belong to the domain of the lifting map P_X^{-1}, because of {}".format(error))
                
        if len(V) <= 0:
            raise Exception("Nothing to average. Q do(es) not belong to the domain of the lifting map at point X.")
                
        V = gs.array(V)
        
        # average in tangent space
        V = gs.mean(V, axis=0)

        # retract computed mean to manifold
        X_mean = self.retraction(X, V)

        return X_mean

    def retraction(self, X, V):
        """
        Retraction map, based on QR-decomposion:
        P_x(V) = qf(X + V)
        """
        Q, R = gs.linalg.qr(X + V)
        
        # flipping signs
        Q = gs.matmul(Q, gs.diag(gs.sign(gs.sign(gs.diagonal(R)) + 0.5)))
        
        return Q

    def lifting(self, X, Q):
        """
        Lifting map, based on QR-decomposion:
        P_x^{-1}(Q) = QR - X
        """
        p, n = gs.shape(X)
        
        R = gs.zeros((n, n))
        r = []
        
        M = gs.matmul(X.T, Q)
        
        # construct r_0
        r.append(make_r(0, M))
            
        for i in range(1, n):
            
            # get principal minor
            M_i = make_minor(i, M)

            if (gs.linalg.det(M_i) != 0):
                b_i = make_b(i, M, r)
                r_i = gs.matmul(gs.linalg.inv(M_i), b_i)
                
                if r_i[i] <= 0:
                    raise Exception("(r_i)_i <= 0")
                else:
                    r.append(r_i)
            else:
                raise Exception("det(M_i) == 0, not invertible")
        
        for i, item in enumerate(r):
            R[:len(item),i] = gs.array(item)
            
        return gs.matmul(Q, R) - X

    def dist(self):
        raise NotImplementedError('Geodesic distance is not implemented.')

def make_minor(i, M):
    return M[:i+1,:i+1]

def make_r(i, M):
    if i == 0:
        if (M[0,0] > 0):
            return gs.array([1. / M[0,0]])
        else:
            raise Exception("M[0,0] <= 0")
    else:
        return M[:i+1,i]

def make_b(i, M, r):

    b = gs.ones(i+1)

    for j in range(i):
        b[j] = - gs.matmul(M[i,:j+1], r[j])

    return b