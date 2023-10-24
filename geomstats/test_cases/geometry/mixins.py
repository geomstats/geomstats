import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data

# TODO: missing data


class ProjectionTestCaseMixins:
    # TODO: should projection be part of manifold? (not only in tests)

    def test_projection(self, point, expected, atol):
        proj_point = self.space.projection(point)
        self.assertAllClose(proj_point, expected, atol=atol)

    @pytest.mark.vec
    def test_projection_vec(self, n_reps, atol):
        point = self.data_generator.point_to_project()
        expected = self.space.projection(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_projection_belongs(self, n_points, atol):
        """Check projection belongs to manifold.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point = self.data_generator.point_to_project(n_points)
        proj_point = self.space.projection(point)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(proj_point, expected, atol)


class GroupExpTestCaseMixins:
    def test_exp(self, tangent_vec, base_point, expected, atol):
        point = self.space.exp(tangent_vec, base_point)
        self.assertAllClose(point, expected, atol=atol)
