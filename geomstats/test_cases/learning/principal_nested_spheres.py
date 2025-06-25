import pytest
import geomstats.backend as gs
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class PrincipalNestedSpheresTestCase(BaseEstimatorTestCase):
    """
    TestCase for PrincipalNestedSpheres estimator, providing common tests.
    """

    @pytest.mark.random
    def test_fit_transform_consistency(self, n_samples, atol):
        """
        Fit followed by transform should match fit_transform.
        """
        X = self.data_generator.random_point(n_samples)
        projected1 = self.estimator.fit_transform(X)
        projected2 = self.estimator.fit(X).transform(X)
        self.assertAllClose(projected1, projected2, atol=atol)

    @pytest.mark.random
    def test_output_on_circle(self, n_samples, atol):
        """
        The final embedding must lie on the unit circle S^1.
        """
        X = self.data_generator.random_point(n_samples)
        projected = self.estimator.fit_transform(X)
        norms = gs.linalg.norm(projected, axis=1)
        self.assertAllClose(norms, gs.ones_like(norms), atol=atol)

    @pytest.mark.random
    def test_nested_levels(self, n_samples):
        """
        The number of nested spheres equals original sphere dimension minus one.
        """
        X = self.data_generator.random_point(n_samples)
        self.estimator.fit(X)
        expected = self.estimator.space.dim - 1
        self.assertEqual(len(self.estimator.nested_spheres_), expected)
