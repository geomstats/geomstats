"""New way to compute the log by "iterative tree-growing"."""

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions

space = BetaDistributions()


class NewLog:
    """Class for the NewLog algorithm."""

    def __init__(self, point_a, point_b, epsilon, n_rays, error_threhsold, space):
        """Instantiate the algorithm.

        Parameters
        ----------
        point_a : array-like, shape=[2]
            Base point of the log, point on the manifold.
        point_b : array-like, shape=[2]
            Target point of the log, point on the manifold.
        epsilon : positive float
            Growth rate of the trees.
        n_rays : int
            Number of leaves.
        error_threshold : positive float
            Euclidean error threshold for the log.
        space : InformationManifold
            Manifold.
        """
        self.point_a = point_a
        self.point_b = point_b
        self.epsilon = epsilon
        self.n_rays = n_rays
        self.error_threshold = error_threhsold
        self.space = space
        self.theta = gs.linspace(0, 2 * gs.pi, n_rays + 1)[:-1]
        self.directions_a = gs.transpose(
            gs.array([gs.cos(self.theta), gs.sin(self.theta)])
        )
        self.directions_b = gs.transpose(
            gs.array([gs.cos(self.theta), gs.sin(self.theta)])
        )

        self.init_tangent_vec_a = self.epsilon * self.space.metric.normalize(
            self.directions_a, self.point_a
        )
        self.init_tangent_vec_b = self.epsilon * self.space.metric.normalize(
            self.directions_b, self.point_b
        )

        self.tangent_vec_a = self.init_tangent_vec_a
        self.tangent_vec_b = self.init_tangent_vec_b

        self.outer_a = self.space.metric.exp(
            tangent_vec=self.tangent_vec_a, base_point=self.point_a
        )
        self.outer_b = self.space.metric.exp(
            tangent_vec=self.tangent_vec_b, base_point=self.point_b
        )

    def grow(self):
        """Increase the length of the branches of the trees."""
        self.tangent_vec_a *= 1 + self.epsilon
        self.tangent_vec_b *= 1 + self.epsilon

        self.outer_a = self.space.metric.exp(
            tangent_vec=self.tangent_vec_a, base_point=self.point_a
        )
        self.outer_b = self.space.metric.exp(
            tangent_vec=self.tangent_vec_b, base_point=self.point_b
        )

    def reaim(self, directions_a, directions_b):
        """Precise the aim once trees have touched."""
        self.directions_a, self.directions_b = directions_a, directions_b

        self.init_tangent_vec_a = self.epsilon * self.space.metric.normalize(
            self.directions_a, self.point_a
        )
        self.init_tangent_vec_b = self.epsilon * self.space.metric.normalize(
            self.directions_b, self.point_b
        )

        self.tangent_vec_a = self.init_tangent_vec_a
        self.tangent_vec_b = self.init_tangent_vec_b

        self.outer_a = self.space.metric.exp(
            tangent_vec=self.tangent_vec_a, base_point=self.point_a
        )
        self.outer_b = self.space.metric.exp(
            tangent_vec=self.tangent_vec_b, base_point=self.point_b
        )

    def contact_b_in_a(self):
        """Check if a leaf of the b-tree is in a_tree."""
        for i in range(self.n_rays - 1):
            polygon_a = Polygon(gs.vstack([self.point_a, self.outer_a[[i, i + 1]]]))
            for j in range(self.n_rays):
                pt_b = Point(self.outer_b[j])
                if polygon_a.contains(pt_b):
                    return True, {"a": [i, i + 1], "b": [j]}
        return False, {"a": None, "b": None}

    def contact_a_in_b(self):
        """Check if a leaf of the a-tree is in b_tree."""
        for i in range(self.n_rays - 1):
            polygon_b = Polygon(gs.vstack([self.point_b, self.outer_b[[i, i + 1]]]))
            for j in range(self.n_rays):
                pt_a = Point(self.outer_a[j])
                if polygon_b.contains(pt_a):
                    return True, {"b": [i, i + 1], "a": [j]}
        return False, {"a": None, "b": None}

    def contact(self):
        """Check if there is contact between the trees."""
        contact_a_in_b, contact_b_in_a = self.contact_a_in_b(), self.contact_b_in_a()
        contact_bool, indices = contact_b_in_a if contact_b_in_a[0] else contact_a_in_b
        return contact_bool, indices

    def update(self):
        """Update the algorithm, either reaim if contact or grow the trees."""
        contact_bool, indices = self.contact()
        if contact_bool:
            if len(indices["a"]) == 2:
                theta_a = gs.linspace(*self.theta[indices["a"]], self.n_rays)
                directions_a = gs.transpose(
                    gs.array([gs.cos(theta_a), gs.sin(theta_a)])
                )

                ind_b = indices["b"][0]
                ind_a_previous = ind_b - 1
                ind_a_next = ind_b + 1 if ind_b < self.n_rays - 1 else 0
                theta_b = gs.linspace(
                    (self.theta[ind_a_previous] + self.theta[ind_b]) / 2,
                    (self.theta[ind_a_next] + self.theta[ind_b]) / 2,
                    self.n_rays,
                )
                directions_b = gs.transpose(
                    gs.array([gs.cos(theta_b), gs.sin(theta_b)])
                )

            else:
                theta_b = gs.linspace(*self.theta[indices["b"]], self.n_rays)
                directions_b = gs.transpose(
                    gs.array([gs.cos(theta_b), gs.sin(theta_b)])
                )

                ind_a = indices["a"][0]
                ind_a_previous = ind_a - 1
                ind_a_next = ind_a + 1 if ind_a < self.n_rays - 1 else 0
                theta_a = gs.linspace(
                    (self.theta[ind_a_previous] + self.theta[ind_a]) / 2,
                    (self.theta[ind_a_next] + self.theta[ind_a]) / 2,
                    self.n_rays,
                )
                directions_a = gs.transpose(
                    gs.array([gs.cos(theta_a), gs.sin(theta_a)])
                )

            self.reaim(directions_a, directions_b)
        else:
            self.grow()

    def shooting(self):
        """Give current estimation for log."""
        self.shooting_vec = self.directions_a[(self.n_rays - 1) // 2]
        return self.space.metric.exp(
            tangent_vec=self.shooting_vec, base_point=self.point_a
        )

    def error(self):
        """Give current euclidean error."""
        return gs.sqrt(gs.sum((self.shooting() - self.point_b) ** 2))

    def run(self):
        """Run the algorithm."""
        while self.error() > self.error_threshold:
            self.update()
        return self.shooting_vec
