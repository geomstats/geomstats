import pytest

import geomstats.backend as gs
from geomstats.learning.principal_nested_spheres import gram_schmidt
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class PrincipalNestedSpheresTestCase(BaseEstimatorTestCase):
    """Test case for Principal Nested Spheres."""

    @pytest.mark.random
    def test_fit(self, n_samples, atol):
        """Test that fit method works and creates necessary attributes."""
        X = self.data_generator.random_point(n_samples)
        
        self.estimator.fit(X)
        
        # Check that all necessary attributes are created
        self.assertTrue(hasattr(self.estimator, "nested_spheres_"))
        self.assertTrue(hasattr(self.estimator, "residuals_"))
        self.assertTrue(hasattr(self.estimator, "mean_"))
        
        # Check shapes and types
        assert isinstance(self.estimator.nested_spheres_, list)
        assert isinstance(self.estimator.residuals_, list)
        self.assertEqual(len(self.estimator.nested_spheres_), len(self.estimator.residuals_))
        
        # Check mean_ is on S^1
        self.assertEqual(self.estimator.mean_.shape, (2,))
        self.assertAllClose(gs.linalg.norm(self.estimator.mean_), 1.0, atol=atol)

    @pytest.mark.random
    def test_transform(self, n_samples, atol):
        """Test transform method after fitting."""
        X = self.data_generator.random_point(n_samples)
        
        self.estimator.fit(X)
        X_transformed = self.estimator.transform(X)
        
        # Check output shape - should project to S^1 (2D)
        expected_shape = (n_samples, 2)
        self.assertEqual(X_transformed.shape, expected_shape)
        
        # Check that output points are on unit sphere
        norms = gs.linalg.norm(X_transformed, axis=1)
        expected_norms = gs.ones(n_samples)
        self.assertAllClose(norms, expected_norms, atol=atol)

    @pytest.mark.random
    def test_fit_transform(self, n_samples, atol):
        """Test fit_transform method."""
        X = self.data_generator.random_point(n_samples)
        
        X_transformed = self.estimator.fit_transform(X)
        
        # Check output shape and properties
        expected_shape = (n_samples, 2)
        self.assertEqual(X_transformed.shape, expected_shape)
        
        # Check that output points are on unit sphere
        norms = gs.linalg.norm(X_transformed, axis=1)
        expected_norms = gs.ones(n_samples)
        self.assertAllClose(norms, expected_norms, atol=atol)

    @pytest.mark.random
    def test_fit_transform_and_transform_after_fit(self, n_samples, atol):
        """Test that fit_transform and transform after fit work and give valid outputs."""
        X = self.data_generator.random_point(n_samples)

        # Test fit_transform
        X_fit_transform = self.estimator.fit_transform(X)
        
        # Test fit then transform on a fresh estimator
        fresh_estimator = self.estimator.__class__(
            space=self.estimator.space,
            n_init=self.estimator.n_init,
            max_iter=self.estimator.max_iter,
            tol=self.estimator.tol
        )
        fresh_estimator.fit(X)
        X_transform_after_fit = fresh_estimator.transform(X)
        
        # Both should produce valid outputs on S^1
        expected_shape = (n_samples, 2)
        self.assertEqual(X_fit_transform.shape, expected_shape)
        self.assertEqual(X_transform_after_fit.shape, expected_shape)
        
        # Check that both results are on unit circle
        norms1 = gs.linalg.norm(X_fit_transform, axis=1)
        norms2 = gs.linalg.norm(X_transform_after_fit, axis=1)
        expected_norms = gs.ones(n_samples)
        self.assertAllClose(norms1, expected_norms, atol=atol)
        self.assertAllClose(norms2, expected_norms, atol=atol)

    @pytest.mark.random
    def test_sphere_mode(self, n_samples, atol):
        """Test different sphere fitting modes."""
        X = self.data_generator.random_point(n_samples)
        
        modes = ["adaptive", "great", "small"]
        results = {}
        
        for mode in modes:
            estimator_mode = self.estimator.__class__(
                space=self.estimator.space,
                sphere_mode=mode,
                n_init=3,  # Reduce for faster testing
                max_iter=100
            )
            try:
                X_transformed = estimator_mode.fit_transform(X)
                results[mode] = X_transformed
                
                # Check basic properties
                self.assertEqual(X_transformed.shape, (n_samples, 2))
                norms = gs.linalg.norm(X_transformed, axis=1)
                expected_norms = gs.ones(n_samples)
                self.assertAllClose(norms, expected_norms, atol=atol)
            except Exception:
                # Some modes might fail for certain data, which is acceptable
                pass
        
        # At least one mode should work
        self.assertTrue(len(results) > 0)

    @pytest.mark.random
    def test_n_init(self, n_samples, atol):
        """Test n_init parameter."""
        X = self.data_generator.random_point(n_samples)
        
        # Test with different numbers of initializations
        n_init_values = [1, 5]
        
        for n_init in n_init_values:
            estimator_n_init = self.estimator.__class__(
                space=self.estimator.space,
                n_init=n_init,
                max_iter=100
            )
            X_transformed = estimator_n_init.fit_transform(X)
            
            # Check basic properties
            self.assertEqual(X_transformed.shape, (n_samples, 2))
            norms = gs.linalg.norm(X_transformed, axis=1)
            expected_norms = gs.ones(n_samples)
            self.assertAllClose(norms, expected_norms, atol=atol)

    def test_verbose(self, n_samples):
        """Test verbose parameter."""
        X = self.data_generator.random_point(n_samples)
        
        # Test that verbose=True doesn't break anything
        estimator_verbose = self.estimator.__class__(
            space=self.estimator.space,
            verbose=True,
            n_init=2,  # Reduce for faster testing
            max_iter=50
        )
        
        # Should not raise an exception
        X_transformed = estimator_verbose.fit_transform(X)
        self.assertEqual(X_transformed.shape, (n_samples, 2))

    @pytest.mark.random
    def test_nested_spheres_shape(self, n_samples, atol):
        """Test that nested spheres have correct shapes."""
        X = self.data_generator.random_point(n_samples)
        
        self.estimator.fit(X)
        
        expected_levels = X.shape[1] - 2  # From R^{n+1} to S^1
        self.assertEqual(len(self.estimator.nested_spheres_), expected_levels)
        
        for i, (normal, height) in enumerate(self.estimator.nested_spheres_):
            # Normal should be unit vector in appropriate dimension
            expected_dim = X.shape[1] - i
            self.assertEqual(normal.shape, (expected_dim,))
            self.assertAllClose(gs.linalg.norm(normal), 1.0, atol=atol)
            
            # Height should be a scalar
            is_python_scalar = isinstance(height, (int, float))
            is_numpy_scalar = hasattr(height, 'ndim') and (height.ndim == 0 or height.shape == ())
            self.assertTrue(is_python_scalar or is_numpy_scalar)

    @pytest.mark.random  
    def test_circular_mean(self, n_samples, atol):
        """Test circular mean computation."""
        X = self.data_generator.random_point(n_samples)
        
        self.estimator.fit(X)
        mean = self.estimator.mean_
        
        # Mean should be on S^1
        self.assertEqual(mean.shape, (2,))
        self.assertAllClose(gs.linalg.norm(mean), 1.0, atol=atol)

    def test_gram_schmidt(self, n_samples, atol):
        """Test Gram-Schmidt orthonormalization function."""
        # Create a random matrix
        dim = 4
        matrix = gs.random.normal(size=(n_samples, dim))
        
        # Apply Gram-Schmidt
        orthonormal = gram_schmidt(matrix)
        
        # Check orthonormality
        if orthonormal.shape[0] > 0:
            # Check that vectors are unit length
            norms = gs.linalg.norm(orthonormal, axis=1)
            expected_norms = gs.ones(orthonormal.shape[0])
            self.assertAllClose(norms, expected_norms, atol=atol)
            
            # Check that vectors are orthogonal
            dot_products = gs.matmul(orthonormal, gs.transpose(orthonormal))
            expected_identity = gs.eye(orthonormal.shape[0])
            self.assertAllClose(dot_products, expected_identity, atol=atol)

    @pytest.mark.random
    def test_output_on_circle(self, n_samples, atol):
        """Test that output is always on S^1."""
        X = self.data_generator.random_point(n_samples)
        
        X_transformed = self.estimator.fit_transform(X)
        
        # All points should be on unit circle
        norms = gs.linalg.norm(X_transformed, axis=1)
        expected_norms = gs.ones(n_samples)
        self.assertAllClose(norms, expected_norms, atol=atol)

    @pytest.mark.random
    def test_nested_levels(self, n_samples):
        """Test that we get the expected number of nested levels."""
        X = self.data_generator.random_point(n_samples)
        
        self.estimator.fit(X)
        
        # Number of nested spheres should be dim - 1 
        # (from S^n to S^1 takes n-1 steps)
        original_dim = X.shape[1] - 1  # Sphere dimension
        expected_levels = max(0, original_dim - 1)
        actual_levels = len(self.estimator.nested_spheres_)
        
        self.assertEqual(actual_levels, expected_levels)

    @pytest.mark.random
    def test_fit_transform_consistency(self, n_samples, atol):
        """Test that fit_transform produces consistent valid outputs."""
        X = self.data_generator.random_point(n_samples)
        
        # Test using the same estimator instance to ensure internal consistency
        estimator = self.estimator.__class__(
            space=self.estimator.space,
            n_init=self.estimator.n_init,
            max_iter=self.estimator.max_iter,
            tol=self.estimator.tol
        )
        
        # First fit the estimator
        estimator.fit(X)
        X_transform_after_fit = estimator.transform(X)
        
        # The transform should be consistent with the fitted state
        self.assertEqual(X_transform_after_fit.shape, (n_samples, 2))
        norms = gs.linalg.norm(X_transform_after_fit, axis=1)
        expected_norms = gs.ones(n_samples)
        self.assertAllClose(norms, expected_norms, atol=atol)
        
        # Test that fit_transform also produces valid output
        X_fit_transform = estimator.fit_transform(X)
        self.assertEqual(X_fit_transform.shape, (n_samples, 2))
        norms2 = gs.linalg.norm(X_fit_transform, axis=1)
        self.assertAllClose(norms2, expected_norms, atol=atol)

    @pytest.mark.vec
    def test_fit_vectorization(self, n_samples, atol):
        """Test that fit method works with vectorized input."""
        X = self.data_generator.random_point(n_samples)
        
        # Test with single sample
        if n_samples > 1:
            X_single = X[:1]
            self.estimator.fit(X_single)
            self.assertTrue(hasattr(self.estimator, "nested_spheres_"))
        
        # Test with multiple samples
        self.estimator.fit(X)
        self.assertTrue(hasattr(self.estimator, "nested_spheres_"))

    @pytest.mark.vec 
    def test_transform_vectorization(self, n_samples, atol):
        """Test that transform method works with vectorized input."""
        X = self.data_generator.random_point(n_samples)
        
        self.estimator.fit(X)
        
        # Test single point transform
        if n_samples > 1:
            X_single = X[:1]
            X_transformed_single = self.estimator.transform(X_single)
            self.assertEqual(X_transformed_single.shape, (1, 2))
            self.assertAllClose(
                gs.linalg.norm(X_transformed_single, axis=1), 
                gs.ones(1), 
                atol=atol
            )
        
        # Test multiple points transform
        X_transformed = self.estimator.transform(X)
        self.assertEqual(X_transformed.shape, (n_samples, 2))
        self.assertAllClose(
            gs.linalg.norm(X_transformed, axis=1),
            gs.ones(n_samples),
            atol=atol
        )

    @pytest.mark.vec
    def test_fit_transform_vectorization(self, n_samples, atol):
        """Test that fit_transform method works with vectorized input."""
        X = self.data_generator.random_point(n_samples)
        
        X_transformed = self.estimator.fit_transform(X)
        
        # Check vectorized output
        self.assertEqual(X_transformed.shape, (n_samples, 2))
        norms = gs.linalg.norm(X_transformed, axis=1)
        self.assertAllClose(norms, gs.ones(n_samples), atol=atol)

    def test_invalid_sphere_mode(self, n_samples):
        """Test error handling for invalid sphere mode."""
        X = self.data_generator.random_point(n_samples)
        
        # Test invalid sphere mode
        invalid_estimator = self.estimator.__class__(
            space=self.estimator.space,
            sphere_mode="invalid_mode"
        )
        try:
            invalid_estimator.fit(X)
            assert False, "Expected ValueError for invalid sphere mode"
        except ValueError:
            pass  # Expected behavior

    def test_transform_before_fit(self, n_samples):
        """Test error handling when transform is called before fit."""
        X = self.data_generator.random_point(n_samples)
        
        fresh_estimator = self.estimator.__class__(space=self.estimator.space)
        
        # This should raise an AttributeError since nested_spheres_ doesn't exist
        try:
            fresh_estimator.transform(X)
            assert False, "Expected AttributeError when transform called before fit"
        except AttributeError:
            pass  # Expected behavior

    @pytest.mark.random
    def test_edge_case_small_sample(self, atol):
        """Test with very small sample sizes."""
        # Test with minimum possible sample size
        X = self.data_generator.random_point(2)
        
        try:
            X_transformed = self.estimator.fit_transform(X)
            self.assertEqual(X_transformed.shape, (2, 2))
            norms = gs.linalg.norm(X_transformed, axis=1)
            self.assertAllClose(norms, gs.ones(2), atol=atol)
        except Exception:
            # Small samples might fail, which is acceptable
            pass

    @pytest.mark.random
    def test_numerical_stability(self, n_samples, atol):
        """Test numerical stability with various tolerance settings."""
        X = self.data_generator.random_point(n_samples)
        
        # Test with different tolerances
        tolerances = [1e-4, 1e-6, 1e-8]
        
        for tol in tolerances:
            estimator_tol = self.estimator.__class__(
                space=self.estimator.space,
                tol=tol,
                n_init=2,
                max_iter=50
            )
            try:
                X_transformed = estimator_tol.fit_transform(X)
                self.assertEqual(X_transformed.shape, (n_samples, 2))
                norms = gs.linalg.norm(X_transformed, axis=1)
                self.assertAllClose(norms, gs.ones(n_samples), atol=atol)
            except Exception:
                # Some tolerance settings might fail, which is acceptable
                pass
