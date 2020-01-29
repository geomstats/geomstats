"""Manifold for sets of landmarks that belong to any given manifold."""

import math

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Landmarks(Manifold):
    """Class for landmarks."""

    def __init__(self, ambient_manifold, n_landmarks=None):
        """Construct an instance of the LandmarksSpace class.

        Parameters
        ----------
        ambient_manifold : object from the class Manifold
        n_landmarks
        """
        self.dimension = None
        if n_landmarks:
            self.dimension = n_landmarks * ambient_manifold.dimension
        self.ambient_manifold = ambient_manifold
        self.l2_metric = L2Metric(self.ambient_manifold)
        self.n_landmarks = n_landmarks

        super(Landmarks, self).__init__(dimension=self.dimension)

    def belongs(self, point):
        """Compute whether or not a point belongs to the manifold.

        Parameters
        ----------
        point

        Returns
        -------
        belongs : bool
        """
        belongs = gs.all(self.ambient_manifold.belongs(point))
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
        return belongs


class L2Metric(RiemannianMetric):
    """L2 Riemannian metric on the space of landmarks."""

    def __init__(self, ambient_manifold):
        super(L2Metric, self).__init__(
            dimension=math.inf,
            signature=(math.inf, 0, 0))
        self.ambient_manifold = ambient_manifold
        self.ambient_metric = ambient_manifold.metric

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_landmarks):
        """Compute inner product between tangent vectors at base landmark set.

        Parameters
        ----------
        tangent_vec_a
        tangent_vec_b
        base_landmarks

        Returns
        -------
        inner_prod
        """
        assert tangent_vec_a.shape == tangent_vec_b.shape
        assert tangent_vec_a.shape == base_landmarks.shape
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        base_landmarks = gs.to_ndarray(base_landmarks, to_ndim=3)

        n_landmark_sets, n_landmarks_per_set, n_coords = tangent_vec_a.shape

        new_dim = n_landmark_sets * n_landmarks_per_set
        tangent_vec_a = gs.reshape(tangent_vec_a, (new_dim, n_coords))
        tangent_vec_b = gs.reshape(tangent_vec_b, (new_dim, n_coords))
        base_landmarks = gs.reshape(base_landmarks, (new_dim, n_coords))

        inner_prod = self.ambient_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_landmarks)
        inner_prod = gs.reshape(
            inner_prod, (n_landmark_sets, n_landmarks_per_set))
        inner_prod = gs.sum(inner_prod, -1)

        n_landmarks_per_set_float = gs.array(n_landmarks_per_set)
        n_landmarks_per_set_float = gs.cast(
            n_landmarks_per_set_float, gs.float32)
        inner_prod = inner_prod / n_landmarks_per_set_float
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=1)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)

        return inner_prod

    def dist(self, landmarks_a, landmarks_b):
        """Compute geodesic distance between two landmark sets.

        Parameters
        ----------
        landmarks_a
        landmarks_b

        Returns
        -------
        dist
        """
        assert landmarks_a.shape == landmarks_b.shape
        landmarks_a = gs.to_ndarray(landmarks_a, to_ndim=3)
        landmarks_b = gs.to_ndarray(landmarks_b, to_ndim=3)

        n_landmark_sets, n_landmarks_per_set, n_coords = landmarks_a.shape

        landmarks_a = gs.reshape(
            landmarks_a, (n_landmark_sets * n_landmarks_per_set, n_coords))
        landmarks_b = gs.reshape(
            landmarks_b, (n_landmark_sets * n_landmarks_per_set, n_coords))

        dist = self.ambient_metric.dist(landmarks_a, landmarks_b)
        dist = gs.reshape(dist, (n_landmark_sets, n_landmarks_per_set))
        n_landmarks_per_set_float = gs.array(n_landmarks_per_set)
        n_landmarks_per_set_float = gs.cast(
            n_landmarks_per_set_float, gs.float32)
        dist = gs.sqrt(gs.sum(dist ** 2, -1) / n_landmarks_per_set_float)
        dist = gs.to_ndarray(dist, to_ndim=1)
        dist = gs.to_ndarray(dist, to_ndim=2, axis=1)

        return dist

    def exp(self, tangent_vec, base_landmarks):
        """Compute Riemannian exponential of tan vector wrt base landmark set.

        Parameters
        ----------
        tangent_vec
        base_landmarks

        Returns
        -------
        exp
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        base_landmarks = gs.to_ndarray(base_landmarks, to_ndim=3)

        n_landmark_sets, n_landmarks_per_set, n_coords = base_landmarks.shape
        n_tangent_vecs = tangent_vec.shape[0]

        new_dim = n_landmark_sets * n_landmarks_per_set
        new_base_landmarks = gs.reshape(base_landmarks, (new_dim, n_coords))
        new_tangent_vec = gs.reshape(tangent_vec, (new_dim, n_coords))

        exp = self.ambient_metric.exp(new_tangent_vec, new_base_landmarks)
        exp = gs.reshape(exp, (n_tangent_vecs, n_landmarks_per_set, n_coords))
        exp = gs.squeeze(exp)

        return exp

    def log(self, landmarks, base_landmarks):
        """Compute Riemannian log of a set of landmarks wrt base landmark set.

        Parameters
        ----------
        landmarks
        base_landmarks

        Returns
        -------
        log
        """
        assert landmarks.shape == base_landmarks.shape
        landmarks = gs.to_ndarray(landmarks, to_ndim=3)
        base_landmarks = gs.to_ndarray(base_landmarks, to_ndim=3)

        n_landmark_sets, n_landmarks_per_set, n_coords = landmarks.shape

        landmarks = gs.reshape(
            landmarks, (n_landmark_sets * n_landmarks_per_set, n_coords))
        base_landmarks = gs.reshape(
            base_landmarks, (n_landmark_sets * n_landmarks_per_set, n_coords))
        log = self.ambient_metric.log(landmarks, base_landmarks)
        log = gs.reshape(log, (n_landmark_sets, n_landmarks_per_set, n_coords))
        log = gs.squeeze(log)

        return log

    def geodesic(self, initial_landmarks,
                 end_landmarks=None, initial_tangent_vec=None):
        """Compute geodesic from initial & end landmark set (or init. tan vec).

        Compute the geodesic specified either by an initial landmark set and
        an end landmark set, or by an initial landmark set and an initial
        tangent vector.

        Parameters
        ----------
        initial_landmarks
        end_landmarks
        initial_tangent_vec

        Returns
        -------
        landmarks_on_geodesic
        """
        landmarks_ndim = 2
        initial_landmarks = gs.to_ndarray(
            initial_landmarks, to_ndim=landmarks_ndim + 1)

        if end_landmarks is None and initial_tangent_vec is None:
            raise ValueError(
                'Specify an end landmark set or an initial tangent'
                'vector to define the geodesic.')
        if end_landmarks is not None:
            end_landmarks = gs.to_ndarray(
                end_landmarks, to_ndim=landmarks_ndim + 1)
            shooting_tangent_vec = self.log(landmarks=end_landmarks,
                                            base_landmarks=initial_landmarks)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=landmarks_ndim + 1)

        def landmarks_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_landmarks = gs.to_ndarray(
                initial_landmarks, to_ndim=landmarks_ndim + 1)
            new_initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                                    to_ndim=landmarks_ndim + 1)

            tangent_vecs = gs.einsum('il,nkm->ikm', t, new_initial_tangent_vec)

            def point_on_landmarks(tangent_vec):
                assert gs.ndim(tangent_vec) >= 2
                exp = self.exp(
                    tangent_vec=tangent_vec,
                    base_landmarks=new_initial_landmarks)
                return exp

            landmarks_at_time_t = gs.vectorize(
                tangent_vecs,
                point_on_landmarks,
                signature='(i,j)->(i,j)')

            return landmarks_at_time_t

        return landmarks_on_geodesic
