from ._base import BaseEstimatorTestData


class PrincipalNestedSpheresTestData(BaseEstimatorTestData):
    """Test data for Principal Nested Spheres."""
    
    MIN_RANDOM = 5
    MAX_RANDOM = 20

    def fit_test_data(self):
        """Test data for fit method."""
        return self.generate_random_data()

    def transform_test_data(self):
        """Test data for transform method."""
        return self.generate_random_data()

    def fit_transform_test_data(self):
        """Test data for fit_transform method."""
        return self.generate_random_data()

    def fit_transform_and_transform_after_fit_test_data(self):
        """Test data for comparing fit_transform and transform after fit."""
        return self.generate_random_data()

    def sphere_mode_test_data(self):
        """Test data for different sphere modes."""
        return self.generate_random_data()

    def n_init_test_data(self):
        """Test data for n_init parameter."""
        return self.generate_random_data()

    def verbose_test_data(self):
        """Test data for verbose parameter."""
        return self.generate_random_data()

    def nested_spheres_shape_test_data(self):
        """Test data for nested spheres attributes shape."""
        return self.generate_random_data()

    def circular_mean_test_data(self):
        """Test data for circular mean computation."""
        return self.generate_random_data()

    def gram_schmidt_test_data(self):
        """Test data for Gram-Schmidt orthonormalization."""
        return self.generate_random_data()

    def output_on_circle_test_data(self):
        """Test data for output dimension verification."""
        return self.generate_random_data()

    def nested_levels_test_data(self):
        """Test data for nested sphere levels."""
        return self.generate_random_data()

    def fit_transform_consistency_test_data(self):
        """Test data for fit_transform consistency."""
        return self.generate_random_data()
